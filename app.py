# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from transcriber import run_whisperx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the HF token
hf_token = os.getenv('HF_TOKEN')

ALLOWED_EXT = {"wav", "mp3", "m4a", "mp4", "mkv", "flac", "ogg"}
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "dev-secret"  # change this for production!

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _extract_transcript_text(absolute_file_paths):
    """
    Best-effort extraction of transcript text from generated files.
    Preference order:
    1) .timestamped.txt
    2) .txt
    3) .json (concatenate segment texts)
    4) .srt (concatenate subtitle text lines)
    Returns a string (may be empty if nothing usable found).
    """
    # Prefer timestamped txt
    try_order = [
        lambda p: p.lower().endswith('.timestamped.txt'),
        lambda p: p.lower().endswith('.txt') and not p.lower().endswith('.timestamped.txt'),
        lambda p: p.lower().endswith('.json'),
        lambda p: p.lower().endswith('.srt'),
    ]

    # First pass for txt files
    for predicate in try_order[:2]:
        for p in absolute_file_paths:
            if predicate(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception:
                    continue

    # JSON: build from segments
    for p in absolute_file_paths:
        if p.lower().endswith('.json'):
            try:
                import json
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                segments = data.get('segments') or []
                lines = []
                for seg in segments:
                    text = seg.get('text')
                    if text:
                        lines.append(text.strip())
                if lines:
                    return "\n".join(lines).strip()
            except Exception:
                continue

    # SRT: extract non-timestamp, non-index lines
    for p in absolute_file_paths:
        if p.lower().endswith('.srt'):
            try:
                lines = []
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        # Skip index lines (integers) and timestamp lines
                        if s.isdigit() or ('-->' in s):
                            continue
                        lines.append(s)
                if lines:
                    return "\n".join(lines).strip()
            except Exception:
                continue

    return ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("File type not allowed")
        return redirect(url_for("index"))

    # read options from the form
    model = request.form.get("model", "medium")
    output_formats = request.form.getlist("format")  # supports multiple
    device = request.form.get("device", "cpu")
    compute_type = request.form.get("compute", "float32")
    diarize = request.form.get("diarize") == "on"
    timestamped_txt = request.form.get("timestamped_txt") == "on"

    # create safe unique filename
    uid = uuid.uuid4().hex
    original = secure_filename(f.filename)
    saved_name = f"{uid}_{original}"
    filepath = os.path.join(UPLOAD_DIR, saved_name)
    f.save(filepath)

    # call the transcriber
    result = run_whisperx(
        infile=filepath,
        outdir=OUTPUT_DIR,
        model=model,
        output_formats=output_formats,
        compute_type=compute_type,
        device=device,
        diarize=diarize,
        hf_token=hf_token,
        timestamped_txt=timestamped_txt
    )

    if not result.get("success"):
        return render_template("index.html", success=False, log=result.get("log", ""), files=[], transcript_text="")

    # show results (files are absolute paths; convert to relative filenames)
    abs_files = result.get("files", [])
    files = []
    for p in abs_files:
        if os.path.commonpath([os.path.abspath(p), os.path.abspath(OUTPUT_DIR)]) == os.path.abspath(OUTPUT_DIR):
            files.append(os.path.basename(p))

    transcript_text = _extract_transcript_text(abs_files)

    return render_template("index.html", success=True, files=files, transcript_text=transcript_text)

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
