import hashlib, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("LATENTSYNC_DIR", "/content/LatentSync"))
WORK = Path(os.environ.get("GHX_WORK_DIR", "/content/drive/MyDrive/GHX_Lip_Sync_Studio"))
CHUNK_SECONDS = 8
MAX_RETRIES = 2

def run(cmd, check=True, capture=False):
    p = subprocess.run([str(x) for x in cmd], text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None)
    if check and p.returncode:
        raise RuntimeError((p.stdout or "")[-4000:] if capture else f"Command failed ({p.returncode}): {' '.join(map(str,cmd))}")
    return p.stdout if capture else p.returncode

def require_tools():
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None: raise RuntimeError(f"{name} missing. Run the notebook setup cell.")

def duration(path):
    raw=run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path],capture=True)
    try: return float(raw.strip())
    except Exception: raise RuntimeError(f"Cannot read duration for {path}")

def valid_video(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size<1024: return False
    return bool(run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name","-of","csv=p=0",p],check=False,capture=True).strip())

def probe_video(path):
    raw=run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate","-of","json",path],capture=True)
    s=json.loads(raw)["streams"][0]
    a,b=s.get("r_frame_rate","25/1").split("/")
    return int(s["width"]),int(s["height"]),float(a)/float(b)

def fingerprint(paths, quality):
    h=hashlib.sha256(quality.encode())
    for name in paths:
        p=Path(name); h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
        with p.open("rb") as f:
            while True:
                block=f.read(4*1024*1024)
                if not block: break
                h.update(block)
    return h.hexdigest()[:18]

def sync_long_video(video_file, audio_file, quality="Auto", inference_steps=20, guidance=1.5, progress=None):
    require_tools()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not detected. Select Runtime > Change runtime type > T4 GPU.")
    video_file=str(video_file); audio_file=str(audio_file)
    if not Path(video_file).is_file() or not Path(audio_file).is_file(): raise RuntimeError("Upload both source video and replacement audio.")
    vd,ad=duration(video_file),duration(audio_file)
    if abs(vd-ad)>0.5:
        raise RuntimeError(f"Media durations differ (video {vd/60:.2f} min, audio {ad/60:.2f} min). Please prepare matching durations. No source audio will be reused.")
    target=min(vd,ad)
    if target<2: raise RuntimeError("Media is too short for processing.")
    if quality not in ("720p","1080p","Auto"): raise RuntimeError("Unsupported output quality.")
    job=fingerprint([video_file,audio_file],quality)
    d=WORK/"jobs"/job
    for folder in ("chunks","results","final","temp"): (d/folder).mkdir(parents=True,exist_ok=True)
    src=d/"source_video.mp4"; aud=d/"replacement_audio.wav"
    if not src.exists(): shutil.copy2(video_file,src)
    if not aud.exists(): run(["ffmpeg","-y","-i",audio_file,"-vn","-ac","2","-ar","16000","-c:a","pcm_s16le",aud])
    width,height,fps=probe_video(src)
    if quality=="720p": ow,oh=1280,720
    elif quality=="1080p": ow,oh=1920,1080
    else: ow,oh=width,height
    total=max(1,int((target+CHUNK_SECONDS-0.001)//CHUNK_SECONDS))
    manifest=d/"manifest.json"
    state=json.loads(manifest.read_text()) if manifest.exists() else {"completed":[]}
    state.update({"job":job,"total_chunks":total,"quality":quality,"duration":target})
    def report(frac,msg):
        if progress: progress(max(0,min(1,frac)),msg)
    inference=UPSTREAM/"scripts"/"inference.py"
    checkpoint=UPSTREAM/"checkpoints"/"latentsync_unet.pt"
    config=UPSTREAM/"configs"/"unet.yaml"
    whisper=UPSTREAM/"checkpoints"/"whisper"/"tiny.pt"
    for p in (inference,checkpoint,config,whisper):
        if not p.exists(): raise RuntimeError(f"Required LatentSync file is missing: {p}. Re-run the setup cell.")
    report(0.02,f"Job {job}: {total} chunks · {target/60:.1f} minutes")
    for i in range(total):
        start=i*CHUNK_SECONDS; seconds=min(CHUNK_SECONDS,target-start)
        vchunk=d/"chunks"/f"v_{i:05d}.mp4"
        achunk=d/"chunks"/f"a_{i:05d}.wav"
        result=d/"results"/f"r_{i:05d}.mp4"
        if str(i) in state["completed"] and valid_video(result): continue
        report(0.04+0.90*i/total,f"Preparing and processing chunk {i+1}/{total}")
        run(["ffmpeg","-y","-ss",start,"-i",src,"-t",seconds,"anullsrc=channel_layout=stereo:sample_rate=16000","-shortest","-an","-vf",f"fps={fps:.6f}","-c:v","libx264","-preset","veryfast","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",vchunk])
        run(["ffmpeg","-y","-ss",start,"-i",aud,"-t",seconds,"-vn","-ac","2","-ar","16000","-c:a","pcm_s16le",achunk])
        ok=False; error=""
        for attempt in range(MAX_RETRIES+1):
            try:
                cmd=["python",str(inference),"--unet_config_path",str(config),"--inference_ckpt_path",str(checkpoint),
                     "--video_path",str(vchunk),"--audio_path",str(achunk),"--video_out_path",str(result),
                     "--inference_steps",str(int(inference_steps)),"--guidance_scale",str(float(guidance)),
                     "--temp_dir",str(d/"temp"/f"{i:05d}"),"--seed","1247"]
                run(cmd)
                if not valid_video(result): raise RuntimeError("Model did not produce a valid video.")
                ok=True; break
            except Exception as exc:
                error=str(exc)
                try: torch.cuda.empty_cache()
                except Exception: pass
                if attempt<MAX_RETRIES: time.sleep(3)
        if not ok: raise RuntimeError(f"Chunk {i+1}/{total} failed after {MAX_RETRIES+1} attempts. Completed chunks are saved; reconnect and Resume. Details: {error[-1800:]}")
        state["completed"]=sorted(set(state.get("completed",[])+[str(i)]),key=int)
        manifest.write_text(json.dumps(state,indent=2))
        report(0.04+0.90*(i+1)/total,f"Completed {i+1}/{total} chunks")
    report(0.96,"Merging chunks and replacing original audio")
    concat=d/"concat.txt"
    concat.write_text("".join("file '"+(d/"results"/f"r_{i:05d}.mp4").as_posix()+"'\n" for i in range(total)))
    merged=d/"final"/"merged_video.mp4"
    vf=f"scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2"
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat,"-an","-vf",vf,"-c:v","libx264","-preset","slow","-crf","17","-pix_fmt","yuv420p","-r",f"{fps:.6f}","-movflags","+faststart",merged])
    final=d/"final"/f"GHX_LipSync_{quality}.mp4"
    run(["ffmpeg","-y","-i",merged,"-i",aud,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart",final])
    report(1.0,"Finished · replacement audio applied")
    return str(final),f"Done. {total} chunks processed/resumed. Source audio removed. Final: {ow}×{oh} MP4."
