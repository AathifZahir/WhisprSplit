#!/usr/bin/env python3
"""
Audio Extraction from Video Files
Extracts audio from video files in various formats using moviepy
"""

import argparse
import os
import sys
from pathlib import Path
from moviepy import VideoFileClip
import ffmpeg


class AudioExtractor:
    def __init__(self, output_format="mp3", quality="high"):
        """
        Initialize the audio extractor

        Args:
            output_format: Audio format (mp3, wav, m4a, flac, ogg)
            quality: Audio quality (low, medium, high)
        """
        self.output_format = output_format.lower()
        self.quality = quality.lower()

        # Quality settings for different formats
        self.quality_settings = {
            "mp3": {"low": 128, "medium": 192, "high": 320},
            "wav": {"low": 16000, "medium": 44100, "high": 48000},
            "m4a": {"low": 128, "medium": 192, "high": 256},
            "flac": {"low": 16000, "medium": 44100, "high": 48000},
            "ogg": {"low": 128, "medium": 192, "high": 256},
        }

        print(f"Audio Extractor initialized")
        print(f"Output format: {self.output_format.upper()}")
        print(f"Quality: {self.quality}")

    def extract_audio(self, video_path, output_dir="audio", custom_filename=None):
        """
        Extract audio from video file

        Args:
            video_path: Path to video file
            output_dir: Directory to save audio file
            custom_filename: Custom filename (without extension)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get video info
        video_info = self._get_video_info(video_path)

        # Generate output filename
        if custom_filename:
            output_filename = f"{custom_filename}.{self.output_format}"
        else:
            video_name = Path(video_path).stem
            output_filename = f"{video_name}_audio.{self.output_format}"

        output_path = os.path.join(output_dir, output_filename)

        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print(f"Video duration: {video_info['duration']:.2f} seconds")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"Audio codec: {video_info['audio_codec']}")
        print(f"Audio channels: {video_info['audio_channels']}")
        print(f"Output: {output_path}")

        try:
            # Extract audio using moviepy
            with VideoFileClip(video_path) as video:
                # Get audio
                audio = video.audio

                if audio is None:
                    raise ValueError("No audio track found in video file")

                # Apply quality settings
                audio_params = self._get_audio_params()

                # Extract audio
                print("Extracting audio...")
                audio.write_audiofile(output_path, logger=None, **audio_params)

                print(f"Audio extracted successfully!")
                print(f"Saved to: {output_path}")

                # Get output file info
                output_info = self._get_audio_info(output_path)
                print(f"Output file size: {output_info['size_mb']:.2f} MB")
                print(f"Output bitrate: {output_info['bitrate']} kbps")

                return output_path

        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise

    def _get_video_info(self, video_path):
        """Get video file information"""
        try:
            with VideoFileClip(video_path) as video:
                return {
                    "duration": video.duration,
                    "width": video.w,
                    "height": video.h,
                    "fps": video.fps,
                    "audio_codec": getattr(video.audio, "codec", "Unknown")
                    if video.audio
                    else "None",
                    "audio_channels": getattr(video.audio, "nchannels", 0)
                    if video.audio
                    else 0,
                }
        except Exception as e:
            print(f"Warning: Could not get video info: {e}")
            return {
                "duration": 0,
                "width": 0,
                "height": 0,
                "fps": 0,
                "audio_codec": "Unknown",
                "audio_channels": 0,
            }

    def _get_audio_params(self):
        """Get audio parameters based on format and quality"""
        if self.output_format == "mp3":
            return {"bitrate": f"{self.quality_settings['mp3'][self.quality]}k"}
        elif self.output_format == "wav":
            return {"fps": self.quality_settings["wav"][self.quality]}
        elif self.output_format == "m4a":
            return {"bitrate": f"{self.quality_settings['m4a'][self.quality]}k"}
        elif self.output_format == "flac":
            return {"fps": self.quality_settings["flac"][self.quality]}
        elif self.output_format == "ogg":
            return {"bitrate": f"{self.quality_settings['ogg'][self.quality]}k"}
        else:
            return {}

    def _get_audio_info(self, audio_path):
        """Get extracted audio file information"""
        try:
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB

            # Try to get bitrate using ffmpeg
            try:
                probe = ffmpeg.probe(audio_path)
                audio_stream = next(
                    (
                        stream
                        for stream in probe["streams"]
                        if stream["codec_type"] == "audio"
                    ),
                    None,
                )
                bitrate = (
                    int(audio_stream.get("bit_rate", 0)) / 1000 if audio_stream else 0
                )
            except:
                bitrate = 0

            return {
                "size_mb": file_size,
                "bitrate": f"{bitrate:.0f}" if bitrate > 0 else "Unknown",
            }
        except Exception as e:
            return {"size_mb": 0, "bitrate": "Unknown"}


def batch_extract_audio(
    input_dir, output_dir="audio", output_format="mp3", quality="high"
):
    """Extract audio from all video files in a directory"""
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

    # Initialize extractor
    extractor = AudioExtractor(output_format=output_format, quality=quality)

    # Process each file
    successful = 0
    failed = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing {i}/{len(video_files)}: {video_file.name}")

        try:
            extractor.extract_audio(str(video_file), output_dir)
            successful += 1
            print(f"Successfully extracted: {video_file.name}")
        except Exception as e:
            failed += 1
            print(f"Failed to extract {video_file.name}: {e}")

    # Summary
    print(f"\nBatch extraction completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio from video files")
    parser.add_argument("input", help="Video file or directory to process")
    parser.add_argument(
        "--output", "-o", default="audio", help="Output directory (default: audio)"
    )
    parser.add_argument(
        "--format",
        "-f",
        default="mp3",
        choices=["mp3", "wav", "m4a", "flac", "ogg"],
        help="Output audio format (default: mp3)",
    )
    parser.add_argument(
        "--quality",
        "-q",
        default="high",
        choices=["low", "medium", "high"],
        help="Audio quality (default: high)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all video files in directory",
    )
    parser.add_argument(
        "--filename", "-n", help="Custom filename for output (without extension)"
    )

    args = parser.parse_args()

    try:
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            batch_extract_audio(args.input, args.output, args.format, args.quality)
        else:
            # Single file processing
            extractor = AudioExtractor(output_format=args.format, quality=args.quality)

            extractor.extract_audio(args.input, args.output, args.filename)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
