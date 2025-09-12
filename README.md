# JetScribe

JetScribe is a web-based application that provides audio and video transcription services using WhisperX.

## Features

- Support for multiple audio/video formats
- Multiple transcription model options
- Multiple output format options (TXT, SRT, JSON)
- Speaker diarization support
- CPU/GPU processing support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jetscribe.git
cd jetscribe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:8080`

3. Upload your audio/video file:
   - Select your file using the file picker
   - Choose your preferred model:
     - tiny: Fastest, lowest accuracy
     - base: Fast with decent accuracy
     - small: Balanced speed and accuracy (~2GB RAM)
     - medium: Good accuracy (recommended)
     - large-v2: High accuracy, slower processing
     - large-v3: Highest accuracy, requires more resources

4. Select output format:
   - txt: Plain text transcript
   - srt: Subtitles format
   - json: Detailed transcript with timestamps
   - all: Generate all formats

5. Additional options:
   - Device: Choose between CPU/GPU processing
   - Speaker diarization: Enable to identify different speakers

## System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM (8GB+ recommended for larger models)
- GPU support (optional) requires CUDA-compatible graphics card

## Notes

- Large models require more processing time and memory
- For testing purposes, start with small/medium models
- Processing time depends on file size and chosen model
