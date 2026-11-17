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
# Model weights are fetched from the official ByteDance LatentSync 1.5 repository.
sh(["bash","-lc","huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints"])
sh(["bash","-lc","mkdir -p checkpoints/whisper && huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints"])
os.chdir(repo)
print("Setup finished. Confirm the checkpoint paths exist before launching.")
