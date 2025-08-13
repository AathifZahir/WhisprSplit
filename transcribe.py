#!/usr/bin/env python3
"""
Whisper + pyannote.audio Transcription System
Combines OpenAI's Whisper for speech-to-text with pyannote.audio for speaker diarization
"""

import argparse
import os
import json
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import warnings

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")


class TranscriptionSystem:
    def __init__(self, model_size="base", device=None):
        """
        Initialize the transcription system

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto-detected if None)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing transcription system...")
        print(f"Using device: {self.device}")
        print(f"Loading Whisper model: {model_size}")

        # Load Whisper model
        self.whisper_model = whisper.load_model(model_size, device=self.device)

        # Initialize pyannote.audio pipeline
        print("Loading pyannote.audio pipeline...")
        try:
            # Try to load pipeline with token from environment variable
            import os

            auth_token = os.getenv("HF_TOKEN")

            if auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1", use_auth_token=auth_token
                )
                self.pipeline.to(torch.device(self.device))
                print("pyannote.audio pipeline loaded successfully with token")
            else:
                print("No HuggingFace token found. Set HF_TOKEN environment variable.")
                print("Speaker diarization will be disabled")
                self.pipeline = None
        except Exception as e:
            print(f"pyannote.audio pipeline loading failed: {e}")
            print("Speaker diarization will be disabled")
            self.pipeline = None

    def transcribe_audio(self, audio_path, output_dir="output"):
        """
        Transcribe audio file with speaker diarization

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save output files
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        print(f"\nProcessing audio: {audio_path}")

        # Step 1: Transcribe with Whisper
        print("Running Whisper transcription...")
        result = self.whisper_model.transcribe(
            audio_path,
            verbose=False,  # Disable verbose output to avoid printing text
            language=None,  # Auto-detect language
            task="transcribe",
        )

        # Step 2: Speaker diarization (if available)
        speaker_segments = []
        if self.pipeline:
            print("Running speaker diarization...")
            try:
                with ProgressHook() as hook:
                    diarization = self.pipeline(audio_path, hook=hook)

                # Extract speaker segments
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append(
                        {"start": turn.start, "end": turn.end, "speaker": speaker}
                    )
                print(f"Identified {len(speaker_segments)} speaker segments")
            except Exception as e:
                print(f"Speaker diarization failed: {e}")
                speaker_segments = []
        else:
            print("Speaker diarization not available")

        # Step 3: Combine transcription with speaker information
        final_result = self._combine_transcription_and_speakers(
            result, speaker_segments
        )

        # Step 4: Save results
        self._save_results(final_result, base_name, output_dir)

        return final_result

    def _combine_transcription_and_speakers(self, whisper_result, speaker_segments):
        """Combine Whisper transcription with speaker diarization"""
        segments = whisper_result.get("segments", [])

        # If no speaker segments, return original result
        if not speaker_segments:
            return whisper_result

        # Assign speakers to segments based on timing
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]

            # Find overlapping speaker segments
            overlapping_speakers = []
            for speaker_seg in speaker_segments:
                if (
                    speaker_seg["start"] <= segment_end
                    and speaker_seg["end"] >= segment_start
                ):
                    overlap_start = max(segment_start, speaker_seg["start"])
                    overlap_end = min(segment_end, speaker_seg["end"])
                    overlap_duration = overlap_end - overlap_start
                    overlapping_speakers.append(
                        {"speaker": speaker_seg["speaker"], "overlap": overlap_duration}
                    )

            # Assign the speaker with the most overlap
            if overlapping_speakers:
                best_speaker = max(overlapping_speakers, key=lambda x: x["overlap"])
                segment["speaker"] = best_speaker["speaker"]
            else:
                segment["speaker"] = "unknown"

        return whisper_result

    def _save_results(self, result, base_name, output_dir):
        """Save transcription results to various formats"""
        # Save detailed JSON
        json_path = os.path.join(output_dir, f"{base_name}_transcription.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results: {json_path}")

        # Save plain text
        txt_path = os.path.join(output_dir, f"{base_name}_transcription.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Saved plain text: {txt_path}")

        # Save SRT subtitle format
        srt_path = os.path.join(output_dir, f"{base_name}_transcription.srt")
        self._save_srt(result, srt_path)
        print(f"Saved SRT subtitles: {srt_path}")

        # Save VTT format
        vtt_path = os.path.join(output_dir, f"{base_name}_transcription.vtt")
        self._save_vtt(result, vtt_path)
        print(f"Saved VTT subtitles: {vtt_path}")

    def _save_srt(self, result, filepath):
        """Save transcription in SRT subtitle format"""
        segments = result.get("segments", [])

        with open(filepath, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_time_srt(segment["start"])
                end_time = self._format_time_srt(segment["end"])
                speaker = segment.get("speaker", "unknown")
                text = segment["text"].strip()

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{speaker}] {text}\n\n")

    def _save_vtt(self, result, filepath):
        """Save transcription in VTT subtitle format"""
        segments = result.get("segments", [])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for segment in segments:
                start_time = self._format_time_vtt(segment["start"])
                end_time = self._format_time_vtt(segment["end"])
                speaker = segment.get("speaker", "unknown")
                text = segment["text"].strip()

                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{speaker}] {text}\n\n")

    def _format_time_srt(self, seconds):
        """Format time in SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _format_time_vtt(self, seconds):
        """Format time in VTT format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Whisper + pyannote.audio speaker diarization"
    )
    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument(
        "--model",
        "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output", "-o", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda"],
        help="Device to use (auto-detected if not specified)",
    )

    args = parser.parse_args()

    try:
        # Initialize transcription system
        system = TranscriptionSystem(model_size=args.model, device=args.device)

        # Transcribe audio
        result = system.transcribe_audio(args.audio_file, args.output)

        print(f"\nTranscription completed successfully!")
        print(f"Total duration: {result['segments'][-1]['end']:.2f} seconds")
        print(f"Total text length: {len(result['text'])} characters")

        if result.get("segments"):
            speakers = set(seg.get("speaker", "unknown") for seg in result["segments"])
            print(f"Speakers identified: {', '.join(speakers)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
