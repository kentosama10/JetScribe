# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from transcriber import run_whisperx

ALLOWED_EXT = {"wav", "mp3", "m4a", "mp4", "mkv", "flac", "ogg"}
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "dev-secret"  # change this for production!

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

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
    hf_token = request.form.get("hf_token", None)
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
        return render_template("result.html", success=False, log=result.get("log", ""), files=[])

    # show results (files are absolute paths; convert to relative filenames)
    files = []
    for p in result.get("files", []):
        if os.path.commonpath([os.path.abspath(p), os.path.abspath(OUTPUT_DIR)]) == os.path.abspath(OUTPUT_DIR):
            files.append(os.path.basename(p))

    return render_template("result.html", success=True, files=files, log=result.get("log", ""))

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
