#!/usr/bin/env python3
"""
Video to Text Pipeline
Combines audio extraction and transcription in one workflow
"""

import argparse
import os
import sys
from pathlib import Path
from extract_audio import AudioExtractor
from transcribe import TranscriptionSystem


class VideoToTextPipeline:
    def __init__(
        self,
        audio_format="mp3",
        audio_quality="high",
        whisper_model="base",
        device=None,
    ):
        """
        Initialize the video-to-text pipeline

        Args:
            audio_format: Audio format for extraction (mp3, wav, m4a, flac, ogg)
            audio_quality: Audio quality (low, medium, high)
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto-detected if None)
        """
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.whisper_model = whisper_model
        self.device = device

        print(f"Video to Text Pipeline initialized")
        print(f"Audio format: {audio_format.upper()}")
        print(f"Audio quality: {audio_quality}")
        print(f"Whisper model: {whisper_model}")

        # Initialize components
        self.audio_extractor = AudioExtractor(audio_format, audio_quality)
        self.transcription_system = TranscriptionSystem(whisper_model, device)

    def process_video(
        self, video_path, output_dir="output", keep_audio=True, custom_filename=None
    ):
        """
        Process video: extract audio and transcribe

        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            keep_audio: Whether to keep extracted audio file
            custom_filename: Custom filename for output files
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate base filename
        if custom_filename:
            base_name = custom_filename
        else:
            base_name = Path(video_path).stem

        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print("=" * 60)

        # Step 1: Extract audio
        print("Step 1: Extracting audio from video...")
        try:
            audio_path = self.audio_extractor.extract_audio(
                video_path, output_dir, base_name
            )
            print(f"Audio extraction completed: {os.path.basename(audio_path)}")
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            raise

        # Step 2: Transcribe audio
        print(f"\nStep 2: Transcribing audio...")
        try:
            result = self.transcription_system.transcribe_audio(audio_path, output_dir)
            print(f"Transcription completed!")

            # Display results
            print(f"\nTranscription Results:")
            print(f"Total text: {result['text'][:200]}...")
            print(f"Duration: {result['segments'][-1]['end']:.2f} seconds")

            if result.get("segments"):
                speakers = set(
                    seg.get("speaker", "unknown") for seg in result["segments"]
                )
                print(f"Speakers identified: {', '.join(speakers)}")

        except Exception as e:
            print(f"Transcription failed: {e}")
            raise

        # Step 3: Clean up (optional)
        if not keep_audio:
            try:
                os.remove(audio_path)
                print(f"Removed temporary audio file: {os.path.basename(audio_path)}")
            except Exception as e:
                print(f"Warning: Could not remove audio file: {e}")

        print(f"\nVideo to text pipeline completed successfully!")
        print(f"Results saved in: {output_dir}")

        return result

    def batch_process(self, input_dir, output_dir="output", keep_audio=False):
        """Process all video files in a directory"""
        video_extensions = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".3gp",
        ]

        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(input_dir).glob(f"*{ext}"))
            video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

        if not video_files:
            print(f"No video files found in: {input_dir}")
            return

        print(f"Found {len(video_files)} video files:")
        for file in video_files:
            print(f"  - {file.name}")

        # Process each file
        successful = 0
        failed = 0

        for i, video_file in enumerate(video_files, 1):
            print(f"\nProcessing {i}/{len(video_files)}: {video_file.name}")
            print("=" * 60)

            try:
                self.process_video(str(video_file), output_dir, keep_audio)
                successful += 1
                print(f"Successfully processed: {video_file.name}")
            except Exception as e:
                failed += 1
                print(f"Failed to process {video_file.name}: {e}")

        # Summary
        print(f"\nBatch processing completed!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert video files to text using audio extraction + transcription"
    )
    parser.add_argument("input", help="Video file or directory to process")
    parser.add_argument(
        "--output", "-o", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "--audio-format",
        "-af",
        default="mp3",
        choices=["mp3", "wav", "m4a", "flac", "ogg"],
        help="Audio format for extraction (default: mp3)",
    )
    parser.add_argument(
        "--audio-quality",
        "-aq",
        default="high",
        choices=["low", "medium", "high"],
        help="Audio quality (default: high)",
    )
    parser.add_argument(
        "--whisper-model",
        "-wm",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda"],
        help="Device to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all video files in directory",
    )
    parser.add_argument(
        "--keep-audio",
        "-ka",
        action="store_true",
        help="Keep extracted audio files (default: remove after transcription)",
    )
    parser.add_argument(
        "--filename", "-n", help="Custom filename for output files (without extension)"
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = VideoToTextPipeline(
            audio_format=args.audio_format,
            audio_quality=args.audio_quality,
            whisper_model=args.whisper_model,
            device=args.device,
        )

        if args.batch or os.path.isdir(args.input):
            # Batch processing
            pipeline.batch_process(args.input, args.output, args.keep_audio)
        else:
            # Single file processing
            pipeline.process_video(
                args.input, args.output, args.keep_audio, args.filename
            )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
