#!/usr/bin/env python3
"""
Centralized Transcription Workflow
Single entry point for all transcription tasks with interactive prompts
"""

import os
import sys
import argparse
from pathlib import Path
from extract_audio import AudioExtractor
from transcribe import TranscriptionSystem


class TranscriptionWorkflow:
    def __init__(self):
        """Initialize the centralized transcription workflow"""
        self.audio_extractor = None
        self.transcription_system = None

    def setup_components(
        self,
        whisper_model="base",
        device=None,
        audio_format="mp3",
        audio_quality="high",
    ):
        """Setup audio extractor and transcription system"""
        print("Setting up transcription components...")

        # Initialize audio extractor for video processing
        self.audio_extractor = AudioExtractor(audio_format, audio_quality)

        # Initialize transcription system
        self.transcription_system = TranscriptionSystem(whisper_model, device)

        print("Components initialized successfully!")

    def get_user_choice(self):
        """Get user choice between video and audio processing"""
        print("\n" + "=" * 60)
        print("TRANSCRIPTION WORKFLOW")
        print("=" * 60)
        print("Choose your input type:")
        print("1. Video file (extract audio + transcribe)")
        print("2. Audio file (direct transcription)")
        print("3. Exit")

        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice in ["1", "2", "3"]:
                    return choice
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                sys.exit(0)

    def get_file_path(self, file_type):
        """Get file path from user"""
        while True:
            if file_type == "video":
                print(
                    f"\nSupported video formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP"
                )
                file_path = input("Enter video file path: ").strip().strip('"')
            else:
                print(f"\nSupported audio formats: MP3, WAV, M4A, FLAC, OGG, WMA")
                file_path = input("Enter audio file path: ").strip().strip('"')

            # Remove quotes if user added them
            file_path = file_path.strip('"').strip("'")

            if not file_path:
                print("Please enter a valid file path.")
                continue

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            return file_path

    def get_processing_options(self):
        """Get processing options from user"""
        print("\n" + "=" * 40)
        print("PROCESSING OPTIONS")
        print("=" * 40)

        # Whisper model selection
        print("\nWhisper Model Options:")
        print("1. tiny (fastest, good for English)")
        print("2. base (good balance)")
        print("3. small (better accuracy)")
        print("4. medium (high accuracy)")
        print("5. large (best accuracy, slowest)")

        while True:
            model_choice = input("Select model (1-5, default: 2): ").strip()
            if not model_choice:
                model_choice = "2"

            model_map = {
                "1": "tiny",
                "2": "base",
                "3": "small",
                "4": "medium",
                "5": "large",
            }
            if model_choice in model_map:
                whisper_model = model_map[model_choice]
                break
            else:
                print("Invalid choice. Please enter 1-5.")

        # Device selection
        print("\nDevice Options:")
        print("1. Auto-detect (recommended)")
        print("2. CPU")
        print("3. CUDA (GPU)")

        while True:
            device_choice = input("Select device (1-3, default: 1): ").strip()
            if not device_choice:
                device_choice = "1"

            device_map = {"1": None, "2": "cpu", "3": "cuda"}
            if device_choice in device_map:
                device = device_map[device_choice]
                break
            else:
                print("Invalid choice. Please enter 1-5.")

        # Audio format (for video processing)
        print("\nAudio Format Options (for video processing):")
        print("1. MP3 (recommended)")
        print("2. WAV")
        print("3. M4A")
        print("4. FLAC")
        print("5. OGG")

        while True:
            format_choice = input("Select audio format (1-5, default: 1): ").strip()
            if not format_choice:
                format_choice = "1"

            format_map = {"1": "mp3", "2": "wav", "3": "m4a", "4": "flac", "5": "ogg"}
            if format_choice in format_map:
                audio_format = format_map[format_choice]
                break
            else:
                print("Invalid choice. Please enter 1-5.")

        # Audio quality
        print("\nAudio Quality Options:")
        print("1. Low (faster, smaller files)")
        print("2. Medium (balanced)")
        print("3. High (slower, better quality)")

        while True:
            quality_choice = input("Select quality (1-3, default: 2): ").strip()
            if not quality_choice:
                quality_choice = "2"

            quality_map = {"1": "low", "2": "medium", "3": "high"}
            if quality_choice in format_map:
                audio_quality = quality_map[quality_choice]
                break
            else:
                print("Invalid choice. Please enter 1-3.")

        return {
            "whisper_model": whisper_model,
            "device": device,
            "audio_format": audio_format,
            "audio_quality": audio_quality,
        }

    def process_video(self, video_path, options):
        """Process video: extract audio and transcribe"""
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print("=" * 60)

        # Step 1: Extract audio
        print("Step 1: Extracting audio from video...")
        try:
            # Create temporary output directory
            temp_output = "temp_audio"
            os.makedirs(temp_output, exist_ok=True)

            # Extract audio
            audio_path = self.audio_extractor.extract_audio(
                video_path, temp_output, Path(video_path).stem
            )
            print(f"Audio extraction completed: {os.path.basename(audio_path)}")

        except Exception as e:
            print(f"Audio extraction failed: {e}")
            raise

        # Step 2: Transcribe audio
        print(f"\nStep 2: Transcribing audio...")
        try:
            result = self.transcription_system.transcribe_audio(audio_path, "output")
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
        finally:
            # Clean up temporary audio file
            try:
                os.remove(audio_path)
                os.rmdir(temp_output)
            except:
                pass

        return result

    def process_audio(self, audio_path, options):
        """Process audio: direct transcription"""
        print(f"\nProcessing audio: {os.path.basename(audio_path)}")
        print("=" * 60)

        try:
            result = self.transcription_system.transcribe_audio(audio_path, "output")
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

        return result

    def run_workflow(self):
        """Run the main transcription workflow"""
        try:
            # Get user choice
            choice = self.get_user_choice()

            if choice == "3":
                print("Goodbye!")
                return

            # Get processing options
            options = self.get_processing_options()

            # Setup components
            self.setup_components(
                whisper_model=options["whisper_model"],
                device=options["device"],
                audio_format=options["audio_format"],
                audio_quality=options["audio_quality"],
            )

            # Process based on choice
            if choice == "1":
                # Video processing
                video_path = self.get_file_path("video")
                result = self.process_video(video_path, options)
            else:
                # Audio processing
                audio_path = self.get_file_path("audio")
                result = self.process_audio(audio_path, options)

            # Final summary
            print(f"\n" + "=" * 60)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Results saved in: output/")
            print(f"Files generated:")
            base_name = Path(video_path if choice == "1" else audio_path).stem
            print(f"  - {base_name}_transcription.json")
            print(f"  - {base_name}_transcription.txt")
            print(f"  - {base_name}_transcription.srt")
            print(f"  - {base_name}_transcription.vtt")

        except KeyboardInterrupt:
            print("\n\nWorkflow interrupted by user.")
        except Exception as e:
            print(f"\nWorkflow failed: {e}")
            print("Please check your input and try again.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Centralized Transcription Workflow - Interactive transcription system"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (for scripting)",
    )

    args = parser.parse_args()

    if args.non_interactive:
        print("Non-interactive mode not yet implemented. Use interactive mode.")
        return

    # Run interactive workflow
    workflow = TranscriptionWorkflow()
    workflow.run_workflow()


if __name__ == "__main__":
    main()
