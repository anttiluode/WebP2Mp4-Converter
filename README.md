# WebP / WebM → MP4 Converter (ComfyUI Compatible)

A simple Python GUI tool (Tkinter) to convert **animated WebP** (including those exported by ComfyUI) and WebM files into **MP4**.

✅ Handles problematic WebP animations that don’t convert with standard FFmpeg alone.  
✅ Offers three conversion modes:  
- **Auto**: Try direct FFmpeg, fallback to Pillow frame extraction if needed.  
- **Direct**: Use FFmpeg directly (fast, but less reliable on some files).  
- **Frames**: Decode each frame with Pillow, then re-encode with FFmpeg (slower, but works on almost everything).

---

## ✨ Features
- Batch conversion of multiple `.webp` files.
- Detects if a WebP is animated or static.
- Progress bars for file and overall conversion.
- Cancel support.
- GUI built with Tkinter.

---

## 📦 Requirements

Install Python dependencies:

pip install -r pillow

You also need FFmpeg installed and available in your system PATH.

Download FFmpeg

On Windows, you can install via Chocolatey or Winget.

On macOS: brew install ffmpeg

On Linux: sudo apt install ffmpeg

🚀 Usage

Run the app:

python WebM2MP4.py

# Steps:

Click Add WebP Files… and select one or more .webp animations.

Optionally choose an output folder (defaults to same as source).

Select conversion method: Auto, Direct, or Frames.

Press Convert All.

Converted .mp4 files will appear in the output folder.

# ⚠️ Notes
Static WebP images will be listed as "Static image" and skipped (since they’re not animations).

If you want to force static → MP4 (e.g. looped stills), that would require a small modification.

The “Frames” method creates temporary PNGs per frame and re-encodes them with FFmpeg, so it’s slower but reliable.

# 📜 License

Open-source and free to use for research, creative, or personal projects.
