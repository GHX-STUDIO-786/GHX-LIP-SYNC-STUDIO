# GHX LIP SYNC STUDIO
**Professional AI Lip-Sync Workspace**  
CEO: **GUL HASSAN**

Google Colab-first workflow using the official ByteDance LatentSync inference project, with chunk-based processing, retries, resume checkpoints and a Gradio interface.

## Features
- Upload a source video and a separate replacement audio track.
- Source video's original audio is discarded; uploaded audio is muxed into final MP4.
- 8-second chunks, retry handling, Drive checkpointing and resume support.
- Output options: 720p, 1080p or Auto (preserve source dimensions).
- H.264/AAC MP4 at the model's 25 FPS working rate, and Open in Colab notebook.
- Designed for Google Colab T4 with LatentSync 1.5.

## Open in Google Colab
[Open GHX Lip Sync Studio](https://colab.research.google.com/github/GHX-STUDIO-786/GHX-LIP-SYNC-STUDIO/blob/main/GHX_Lip_Sync_Studio.ipynb)

Choose Runtime → Change runtime type → T4 GPU (if available), then run notebook cells from top to bottom. Upload the source video and replacement audio in the Studio interface.

## T4 and quality limitations
The upstream LatentSync README lists 8 GB minimum inference VRAM for version 1.5 and 18 GB for version 1.6. This project targets 1.5 for T4. Actual availability and memory headroom vary by Colab session.

LatentSync's face-generation stage works at its model resolution. 720p/1080p controls final delivery dimensions; it does not make the model synthesize native 720p/1080p facial detail. No model can guarantee perfect identity preservation, zero texture changes or error-free results on every input. Use a clear, front-facing face, stable lighting and clean speech audio.

The long-video engine uses 8-second segments (the upstream documentation recommends 5–10 second segments in its data pipeline). A 20-minute job can require many hours and multiple Colab sessions. Keep the workspace on Google Drive and resume after reconnecting. Video and audio durations should match; this version stops safely and asks for matching media if they differ by more than 0.5 seconds.

## Branding and contact
- CEO: GUL HASSAN
- WhatsApp Channel: https://whatsapp.com/channel/0029VbA5Z6UCsU9KEtgmrN2m
- WhatsApp Number: Add the official number in branding.json. It is intentionally not fabricated.

## Upstream
- Code: https://github.com/bytedance/LatentSync
- Model: https://huggingface.co/ByteDance/LatentSync-1.5

Review upstream code and model licenses before commercial use. This repository is an integration layer and does not relicense third-party weights.
