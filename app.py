import json, gradio as gr
from pathlib import Path
from engine import sync_long_video
ROOT=Path(__file__).resolve().parent
BRAND=json.loads((ROOT/"branding.json").read_text())
NAME=BRAND["product_name"]; CEO=BRAND["ceo"]
CSS="""
.gradio-container{max-width:1160px!important;margin:auto!important}
#hero{padding:26px;border-radius:18px;background:linear-gradient(120deg,#101b35,#164b76);color:white;border:1px solid #294d78}
#hero h1{font-size:30px;margin:0 0 8px}
#hero p{opacity:.86}
"""
def process(video,audio,quality,steps,guidance,progress=gr.Progress()):
    if not video or not audio: raise gr.Error("Please upload both a source video and a replacement audio track.")
    def cb(frac,msg): progress(frac,desc=msg)
    try: return (*sync_long_video(video,audio,quality,int(steps),float(guidance),cb),)
    except Exception as exc: raise gr.Error(str(exc))
def gpu_status():
    import torch
    return f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "GPU not detected — choose T4 in Runtime settings."
with gr.Blocks(title=NAME,theme=gr.themes.Soft(primary_hue="blue",secondary_hue="slate"),css=CSS) as demo:
    gr.HTML(f'<div id="hero"><h1>◈ {NAME}</h1><p>AI VIDEO LIP-SYNC WORKSPACE · LONG-FORM CHUNK ENGINE</p><b>CEO: {CEO}</b></div>')
    gr.Markdown("### Create a synchronized video\nUpload the video first, then the audio/voice track that should replace its original sound.")
    with gr.Row():
        video=gr.Video(label="01 · Source video",sources=["upload"],format="mp4")
        audio=gr.Audio(label="02 · Replacement audio",sources=["upload"],type="filepath")
    with gr.Row():
        quality=gr.Radio(["720p","1080p","Auto"],value="Auto",label="Output resolution")
        steps=gr.Slider(20,30,value=20,step=1,label="Inference steps")
        guidance=gr.Slider(1.0,2.0,value=1.5,step=0.1,label="Lip-sync guidance")
    gr.Markdown("**Long video engine:** 8-second chunks · automatic retries · completed-chunk checkpoints · resume from Google Drive. Video and audio durations must match.")
    start=gr.Button("START LIP SYNC",variant="primary")
    result=gr.File(label="Final synchronized MP4")
    status=gr.Markdown()
    start.click(process,[video,audio,quality,steps,guidance],[result,status])
    with gr.Accordion("System & contact",open=False):
        gr.Markdown(gpu_status())
        gr.Markdown(f"**CEO:** {CEO}\n\n**WhatsApp number:** {BRAND.get('whatsapp_number','ADD_OFFICIAL_NUMBER_HERE')}\n\n**WhatsApp channel:** {BRAND.get('whatsapp_channel','')}")
        gr.Markdown("To set the correct contact number, edit branding.json. No phone number is guessed or fabricated.")
    gr.Markdown("Use only footage and audio you own or are authorized to edit. AI output quality depends on the source face, lighting, audio, model and GPU.")
demo.queue(default_concurrency_limit=1).launch(share=True,debug=True)
