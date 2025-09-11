# transcriber.py
import os
import sys
import subprocess
import glob
import json
from pathlib import Path
from datetime import timedelta

def run_whisperx(infile, outdir="outputs", model="medium", output_formats=None,
                 compute_type="float32", device="cpu", diarize=False, hf_token=None,
                 timestamped_txt=False):
    """
    Run whisperx as subprocess.
    Supports multiple output formats (list), or "all".
    """
    os.makedirs(outdir, exist_ok=True)
    if not output_formats:
        output_formats = ["txt"]

    log_output = ""
    generated_files = []
    success = True

    try:
        for fmt in output_formats:
            cmd = [
                sys.executable, "-m", "whisperx",
                infile,
                "--model", model,
                "--output_dir", outdir,
                "--output_format", fmt,
                "--compute_type", compute_type,
                "--device", device
            ]

            if diarize:
                if not hf_token:
                    return {"success": False, "files": [], "log": "Diarization requested but no Hugging Face token provided."}
                cmd += ["--diarize", "--hf_token", hf_token]

            # always add JSON if timestamped txt requested
            if timestamped_txt and fmt != "json":
                cmd_with_json = cmd.copy()
                cmd_with_json[cmd_with_json.index("--output_format") + 1] = "json"
                proc = subprocess.run(cmd_with_json, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                log_output += f"\n\n=== JSON for timestamped TXT ===\n{proc.stdout}"
                if proc.returncode != 0:
                    success = False

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log_output += f"\n\n=== Format: {fmt} ===\n{proc.stdout}"
            if proc.returncode != 0:
                success = False

        # collect output files
        base = Path(infile).stem
        matches = sorted(glob.glob(os.path.join(outdir, base + "*")))
        generated_files.extend(matches)

        # generate timestamped TXT if requested
        if timestamped_txt:
            json_candidates = [p for p in matches if p.lower().endswith(".json")]
            if json_candidates:
                json_path = json_candidates[0]
                try:
                    txt_path = os.path.join(outdir, base + ".timestamped.txt")
                    _create_timestamped_txt(json_path, txt_path)
                    generated_files.append(txt_path)
                except Exception as e:
                    log_output += f"\n[WARN] Failed to create timestamped TXT: {e}"

    except Exception as e:
        log_output += f"\nException: {str(e)}"
        success = False

    return {"success": success, "files": list(set(generated_files)), "log": log_output}


def _create_timestamped_txt(json_path, txt_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("JSON missing 'segments' list")

    lines = []
    for seg in segments:
        start = seg.get("start")
        text = seg.get("text")
        if start is None or text is None:
            continue
        start_td = timedelta(seconds=float(start))
        start_str = str(start_td).split(".")[0]
        if len(start_str.split(":")) == 2:
            start_str = "0:" + start_str
        lines.append(f"[{start_str}] {text.strip()}")

    if not lines:
        raise ValueError("No valid segments to write")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
