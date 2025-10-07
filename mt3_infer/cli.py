"""
Command-line interface for mt3_infer.

Provides CLI commands for common operations:
- download: Download model checkpoints
- list: List available models
- transcribe: Transcribe audio files to MIDI
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mt3_infer import __version__, download_model, list_models, transcribe
from mt3_infer.utils.audio import load_audio


def cmd_download(args: argparse.Namespace) -> int:
    """Download model checkpoints."""
    # Resolve model list
    if args.all:
        all_models = list_models()
        models_to_download = list(all_models.keys())
    elif args.models:
        models_to_download = args.models
    else:
        # Default: download default model
        models_to_download = ["default"]

    print(f"\nDownloading {len(models_to_download)} model(s)...\n")

    # Download each model
    success_count = 0
    failed_models = []

    for i, model_id in enumerate(models_to_download, 1):
        print(f"[{i}/{len(models_to_download)}] {model_id}")

        try:
            checkpoint_path = download_model(model_id)
            success_count += 1

            if checkpoint_path:
                print(f"✓ Downloaded to: {checkpoint_path}\n")
            else:
                print(f"✓ Uses built-in resolution\n")

        except Exception as e:
            print(f"✗ Failed: {e}\n")
            failed_models.append(model_id)

    # Summary
    print("=" * 70)
    print(f"Downloaded: {success_count}/{len(models_to_download)}")
    if failed_models:
        print(f"Failed: {', '.join(failed_models)}")
    print("=" * 70)

    return 0 if success_count > 0 else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available models."""
    all_models = list_models()

    print("\nAvailable MT3 Models:")
    print("=" * 70)

    for model_id, model_info in all_models.items():
        print(f"\n{model_id}:")
        print(f"  Name: {model_info['name']}")
        print(f"  Description: {model_info['description']}")
        print(f"  Framework: {model_info['framework']}")
        print(f"  Checkpoint size: {model_info['checkpoint']['size_mb']} MB")

        if args.verbose and "metadata" in model_info:
            metadata = model_info["metadata"]
            if "performance" in metadata:
                perf = metadata["performance"]
                print(f"  Speed: {perf.get('speed_x_realtime', 'N/A')}x real-time")
                print(f"  GPU memory: {perf.get('peak_gpu_memory_mb', 'N/A')} MB")

    print("\nAliases:")
    registry = {"aliases": {"fast": "mr_mt3", "accurate": "mt3_pytorch", "multitask": "yourmt3"}}
    for alias, model_id in registry["aliases"].items():
        print(f"  {alias} → {model_id}")

    print()
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio file to MIDI."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".mid")

    # Validate input
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return 1

    print(f"\nTranscribing: {input_path}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    try:
        # Load audio
        print("\n[1/3] Loading audio...")
        audio, sr = load_audio(str(input_path))
        print(f"✓ Loaded {len(audio)/sr:.2f}s audio at {sr} Hz")

        # Transcribe
        print(f"\n[2/3] Transcribing with {args.model}...")
        midi = transcribe(
            audio,
            model=args.model,
            sr=sr,
            device=args.device,
            auto_download=not args.no_download
        )

        # Count notes
        note_count = sum(1 for track in midi.tracks for msg in track if msg.type == "note_on")
        print(f"✓ Detected {note_count} notes")

        # Save MIDI
        print(f"\n[3/3] Saving MIDI...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.save(str(output_path))
        print(f"✓ Saved to: {output_path}")

        print(f"\n✓ Transcription complete!")
        return 0

    except Exception as e:
        print(f"\n✗ Transcription failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mt3-infer",
        description="MT3-Infer: Unified toolkit for MT3 music transcription models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    parser_download = subparsers.add_parser(
        "download",
        help="Download model checkpoints",
        description="Download MT3 model checkpoints"
    )
    parser_download.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Model IDs to download (e.g., mr_mt3, mt3_pytorch, yourmt3)"
    )
    parser_download.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser_download.set_defaults(func=cmd_download)

    # List command
    parser_list = subparsers.add_parser(
        "list",
        help="List available models",
        description="List all available MT3 models"
    )
    parser_list.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    parser_list.set_defaults(func=cmd_list)

    # Transcribe command
    parser_transcribe = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio to MIDI",
        description="Transcribe audio file to MIDI using MT3 models"
    )
    parser_transcribe.add_argument(
        "input",
        help="Input audio file (WAV, MP3, FLAC, etc.)"
    )
    parser_transcribe.add_argument(
        "-o", "--output",
        help="Output MIDI file (default: same as input with .mid extension)"
    )
    parser_transcribe.add_argument(
        "-m", "--model",
        default="default",
        help="Model to use (default, fast, accurate, multitask, mr_mt3, mt3_pytorch, yourmt3)"
    )
    parser_transcribe.add_argument(
        "-d", "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser_transcribe.add_argument(
        "--no-download",
        action="store_true",
        help="Disable automatic checkpoint download"
    )
    parser_transcribe.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed error messages"
    )
    parser_transcribe.set_defaults(func=cmd_transcribe)

    # Parse arguments
    args = parser.parse_args(argv)

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
