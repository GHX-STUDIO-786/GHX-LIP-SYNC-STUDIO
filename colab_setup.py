import os, pathlib, subprocess, sys, shutil
def sh(cmd): subprocess.run(cmd,check=True)
if not shutil.which("ffmpeg"):
    sh(["apt-get","update","-qq"]); sh(["apt-get","install","-y","-qq","ffmpeg"])
repo=pathlib.Path("/content/GHX-LIP-SYNC-STUDIO")
if not repo.exists(): sh(["git","clone","https://github.com/GHX-STUDIO-786/GHX-LIP-SYNC-STUDIO.git",str(repo)])
up=pathlib.Path("/content/LatentSync")
if not up.exists(): sh(["git","clone","https://github.com/bytedance/LatentSync.git",str(up)])
os.chdir(up)
sh(["bash","-lc","pip install -q -r requirements.txt"])
sh([sys.executable,"-m","pip","install","-q","huggingface_hub==0.30.2"])
# Persist the large model weights in Drive so a Colab reconnect does not download 5+ GB again.
cache=pathlib.Path("/content/drive/MyDrive/GHX_Lip_Sync_Studio/model_cache")
(cache/"whisper").mkdir(parents=True,exist_ok=True)
sh(["bash","-lc",f"huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir '{cache}'"])
sh(["bash","-lc",f"huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir '{cache}'"])
(up/"checkpoints").mkdir(exist_ok=True)
(up/"checkpoints"/"whisper").mkdir(exist_ok=True)
for source,target in [
    (cache/"latentsync_unet.pt",up/"checkpoints"/"latentsync_unet.pt"),
    (cache/"whisper"/"tiny.pt",up/"checkpoints"/"whisper"/"tiny.pt")
]:
    if target.exists() or target.is_symlink(): target.unlink()
    target.symlink_to(source)
os.chdir(repo)
print("Setup complete. Persistent Drive model cache:",cache)
print("LatentSync 1.5 checkpoints:", (up/"checkpoints"/"latentsync_unet.pt").exists(), (up/"checkpoints"/"whisper"/"tiny.pt").exists())
