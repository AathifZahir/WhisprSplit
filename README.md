# Whisper + pyannote.audio Transcription System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://github.com/openai/whisper)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)](https://pytorch.org/)
[![Speaker Diarization](https://img.shields.io/badge/pyannote-Speaker%20Diarization-purple.svg)](https://github.com/pyannote/pyannote-audio)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Audio%20Processing-red.svg)](https://ffmpeg.org/)
[![Local Processing](https://img.shields.io/badge/Local-Processing%20Only-brightgreen.svg)](https://github.com/)
[![Multi Format](https://img.shields.io/badge/Multi-Format%20Output-yellow.svg)](https://github.com/)
[![GPU Optimized](https://img.shields.io/badge/GPU-Optimized-lightblue.svg)](https://github.com/)

A powerful, local speech-to-text transcription system that combines OpenAI's Whisper for accurate transcription with pyannote.audio for speaker diarization (identifying who spoke when). Perfect for meetings, interviews, podcasts, and any audio/video content that needs accurate transcription with speaker identification.

## 🚀 Features

- **High-Quality Transcription**: Uses OpenAI's Whisper models (tiny to large) for accurate speech recognition
- **Speaker Diarization**: Identifies different speakers by voice patterns using pyannote.audio
- **Video Support**: Extract audio from video files and run complete video-to-text pipelines
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs (RTX series) with 5-10x speed improvement
- **Multiple Output Formats**: JSON, TXT, SRT, VTT for different use cases
- **Batch Processing**: Process multiple files at once for efficiency
- **Multi-language Support**: Auto-detects language with excellent support for English and other languages
- **Interactive Workflow**: User-friendly guided workflow for beginners
- **Flexible Audio Formats**: Support for MP3, WAV, M4A, FLAC, OGG, WMA input/output

## 📋 Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- CUDA-compatible GPU (optional, for acceleration)
- HuggingFace account and token (for speaker diarization)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd transcriptor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with PyTorch, install it separately:

```bash
# For CUDA support (recommended)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchaudio
```

### 3. Install FFmpeg

**Windows:**

- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add to PATH or place in project directory

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt update && sudo apt install ffmpeg
```

### 4. Get HuggingFace Token (Required for Speaker Diarization)

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Accept terms at [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
4. Set environment variable using one of these methods:

   **Method 1: Set environment variable directly**

   ```bash
   # Windows (Command Prompt)
   set HF_TOKEN=your_token_here

   # Windows (PowerShell)
   $env:HF_TOKEN="your_token_here"

   # Linux/Mac
   export HF_TOKEN=your_token_here
   ```

   **Method 2: Create .env file (Recommended)**

   ```bash
   # Create .env file in project root
   echo HF_TOKEN=your_token_here > .env

   # Or manually create .env file with:
   # HF_TOKEN=your_token_here
   ```

   **Note**: The .env file method is recommended as it persists across terminal sessions and is automatically loaded by the application.

## 🚀 Quick Start

### Recommended: Interactive Workflow

The easiest way to get started is using the interactive workflow:

```bash
python transcribe_workflow.py
```

This will guide you through the entire process with prompts, helping you choose:

- Input type (video or audio)
- Processing options
- Output preferences

### Alternative: Direct Commands

```bash
# Basic transcription
python transcribe.py "path/to/audio.mp3"

# With speaker diarization
python transcribe.py "path/to/audio.mp3" --model small --device cuda

# Video to text (extracts audio first, then transcribes)
python video_to_text.py "path/to/video.mp4"

# Audio extraction only
python extract_audio.py "path/to/video.mp4" --format wav --quality high
```

## 📖 Detailed Usage

### 1. Audio Transcription (`transcribe.py`)

The core transcription script with speaker diarization:

```bash
python transcribe.py "audio/meeting.wav" --model small --device cuda --output results/
```

**Options:**

- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--device`: Device to use (cpu, cuda, auto)
- `--output`: Output directory (default: "output")

**Model Selection Guide:**

- `tiny`: Fastest, good for English (39M parameters)
- `base`: Good balance of speed/accuracy (74M parameters)
- `small`: Better accuracy, moderate speed (244M parameters)
- `medium`: High accuracy, slower (769M parameters)
- `large`: Best accuracy, slowest (1550M parameters)

### 2. Video Processing (`video_to_text.py`)

Complete pipeline from video to transcribed text:

```bash
python video_to_text.py "video/presentation.mp4" --whisper-model small --device cuda
```

**Options:**

- `--audio-format`: Audio format (mp3, wav, m4a, flac, ogg)
- `--audio-quality`: Quality (low, medium, high)
- `--whisper-model`: Whisper model size
- `--device`: Device to use
- `--keep-audio`: Keep extracted audio file

### 3. Audio Extraction (`extract_audio.py`)

Extract audio from video files:

```bash
python extract_audio.py "video.mp4" --format wav --quality high --output audio/
```

**Options:**

- `--format`: Output audio format
- `--quality`: Audio quality (affects bitrate/sample rate)
- `--output`: Output directory
- `--batch`: Process all videos in directory

### 4. Interactive Workflow (`transcribe_workflow.py`)

User-friendly interface for all transcription tasks:

```bash
python transcribe_workflow.py
```

**Features:**

- Guided file selection
- Interactive option configuration
- Progress tracking
- Error handling and suggestions

## 📁 Output Formats

The system generates multiple output formats for different use cases:

- **JSON**: Detailed transcription with timestamps, speaker info, and confidence scores
- **TXT**: Plain text transcription for easy reading
- **SRT**: Subtitle format with speaker labels for video players
- **VTT**: Web video subtitle format for web applications

**Example JSON Output:**

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "speaker": "Speaker 1",
      "text": "Hello, welcome to our meeting.",
      "confidence": 0.95
    }
  ]
}
```

## 🎯 Supported Formats

### Video Input

- **Common**: MP4, AVI, MOV, MKV, WMV, FLV
- **Web**: WebM, M4V
- **Mobile**: 3GP

### Audio Input/Output

- **Lossy**: MP3, M4A, OGG, WMA
- **Lossless**: WAV, FLAC

## 🌍 Language Support

Whisper automatically detects the language. For best results:

- **English**: All models work excellently
- **Other Languages**: Use `medium` or `large` models for better accuracy
- **Mixed Language**: Large models handle code-switching well

## ⚡ Performance Optimization

### GPU Acceleration

- **CUDA Users**: Use `--device cuda` for 5-10x faster processing
- **Memory Management**: Close other GPU applications to avoid CUDA out of memory errors
- **Model Selection**: Balance between speed and accuracy based on your needs

### Processing Tips

- **Short Files**: Use `tiny` or `base` models for quick results
- **Long Files**: Use `small` or `medium` for better accuracy
- **Batch Processing**: Process multiple files overnight for efficiency

## 🔧 Configuration

### Environment Variables

- `HF_TOKEN`: HuggingFace authentication token for speaker diarization
- `CUDA_VISIBLE_DEVICES`: Specify which GPU to use (if multiple)

### Custom Settings

Edit the Python files to customize:

- Default model sizes
- Output directory structure
- Audio quality preferences
- Speaker diarization parameters

## 🚨 Troubleshooting

### Common Issues

#### 1. "No module named 'torch'"

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Speaker diarization not working

- Ensure you have a valid HuggingFace token
- Accept the pyannote.audio model terms
- Set `HF_TOKEN` environment variable correctly
- Check token permissions

#### 3. CUDA out of memory

- Use smaller Whisper model (`tiny` or `base`)
- Close other GPU applications
- Process shorter audio segments
- Use CPU if GPU memory is insufficient

#### 4. Audio extraction fails

- Install FFmpeg: `conda install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/)
- Check video file integrity
- Ensure video has audio track

#### 5. Poor transcription quality

- Use larger models for better accuracy
- Ensure clear audio input
- Check audio format compatibility
- Consider audio preprocessing for noisy files

### Getting Help

1. Check the output files for detailed error information
2. Ensure all dependencies are installed correctly
3. Verify your HuggingFace token is valid
4. Check system requirements (Python version, FFmpeg, etc.)

## 📚 Example Workflows

### Meeting Transcription

1. **Extract audio from meeting video:**

   ```bash
   python extract_audio.py "meeting.mp4" --format wav --quality high
   ```

2. **Transcribe with speaker identification:**

   ```bash
   python transcribe.py "meeting_audio.wav" --model small --device cuda
   ```

3. **View results:**
   - Check `output/` folder for all formats
   - Open SRT file in video player for subtitles
   - Use JSON for detailed analysis

### Podcast Processing

1. **Batch process multiple episodes:**

   ```bash
   python extract_audio.py "podcasts/" --batch --format mp3 --quality high
   ```

2. **Transcribe all episodes:**
   ```bash
   for file in audio/*.mp3; do
     python transcribe.py "$file" --model medium --device cuda
   done
   ```

### Video Content Creation

1. **Extract audio for editing:**

   ```bash
   python extract_audio.py "content.mp4" --format wav --quality high
   ```

2. **Generate subtitles:**
   ```bash
   python transcribe.py "content_audio.wav" --model small --device cuda
   ```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Use GitHub issues for bug reports and feature requests
2. **Submit PRs**: Fork the repository and submit pull requests
3. **Improve Documentation**: Help make the setup and usage clearer
4. **Add Features**: Implement new output formats or processing options

### Development Setup

```bash
git clone <your-fork-url>
cd transcriptor
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenAI Whisper**: For the excellent speech recognition models
- **pyannote.audio**: For speaker diarization capabilities
- **MoviePy**: For video processing and audio extraction
- **FFmpeg**: For multimedia processing

---

**Made with ❤️ for easy, local transcription**

_Transform your audio and video content into searchable, accessible text with professional-grade accuracy._
