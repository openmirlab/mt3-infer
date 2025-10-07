#!/usr/bin/env python3
"""
Download all MT3 model checkpoints.

This script downloads checkpoints for all available models (MR-MT3, MT3-PyTorch, YourMT3).
It can be run standalone or imported as a module.

Usage:
    python tools/download_all_checkpoints.py [--models MODEL1 MODEL2 ...]

Examples:
    # Download all models
    python tools/download_all_checkpoints.py

    # Download specific models
    python tools/download_all_checkpoints.py --models mr_mt3 mt3_pytorch

    # From package root
    uv run python tools/download_all_checkpoints.py
"""

import argparse
import sys
from pathlib import Path

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from mt3_infer import download_model, list_models


def download_all(model_ids=None):
    """
    Download all or selected model checkpoints.

    Args:
        model_ids: List of model IDs to download, or None for all models.

    Returns:
        Number of successfully downloaded models.
    """
    # Get all available models
    all_models = list_models()

    # Filter to requested models or use all
    if model_ids:
        models_to_download = {k: v for k, v in all_models.items() if k in model_ids}
        if not models_to_download:
            print(f"✗ No valid models found in: {model_ids}")
            print(f"  Available models: {list(all_models.keys())}")
            return 0
    else:
        models_to_download = all_models

    print("\n" + "=" * 70)
    print(f"MT3-Infer Checkpoint Downloader")
    print("=" * 70)
    print(f"\nDownloading {len(models_to_download)} model(s):")
    for model_id in models_to_download.keys():
        print(f"  - {model_id}")
    print()

    # Download each model
    success_count = 0
    failed_models = []

    for i, (model_id, model_info) in enumerate(models_to_download.items(), 1):
        print(f"\n[{i}/{len(models_to_download)}] Downloading {model_id}...")
        print(f"  Name: {model_info['name']}")
        print(f"  Size: {model_info['checkpoint']['size_mb']} MB")

        try:
            checkpoint_path = download_model(model_id)
            success_count += 1

            if checkpoint_path:
                print(f"  ✓ Downloaded to: {checkpoint_path}")
            else:
                print(f"  ✓ Uses built-in resolution (no download needed)")

        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            failed_models.append(model_id)

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"  Successful: {success_count}/{len(models_to_download)}")

    if failed_models:
        print(f"  Failed: {len(failed_models)}")
        print(f"  Failed models: {', '.join(failed_models)}")
    else:
        print("  ✓ All checkpoints downloaded successfully!")

    print("=" * 70 + "\n")

    return success_count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download MT3 model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  %(prog)s

  # Download specific models
  %(prog)s --models mr_mt3 mt3_pytorch

  # Download using aliases
  %(prog)s --models fast accurate

  # List available models
  %(prog)s --list
        """
    )

    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Model IDs to download (default: all models)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    # List models if requested
    if args.list:
        all_models = list_models()
        print("\nAvailable models:")
        for model_id, model_info in all_models.items():
            print(f"\n  {model_id}:")
            print(f"    Name: {model_info['name']}")
            print(f"    Description: {model_info['description']}")
            print(f"    Size: {model_info['checkpoint']['size_mb']} MB")
        print()
        return 0

    # Download models
    try:
        success_count = download_all(args.models)
        return 0 if success_count > 0 else 1
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
