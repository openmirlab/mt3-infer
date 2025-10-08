"""
MT3-PyTorch Download Examples

This script demonstrates two ways to download the MT3-PyTorch checkpoint:
1. Automatic download (happens when you load the model)
2. Manual download (download first, then load)

Both methods download the checkpoint to: .mt3_checkpoints/mt3_pytorch/
"""

from mt3_infer import load_model, download_model


def example_automatic_download():
    """
    Example 1: Automatic Download

    The checkpoint will be automatically downloaded when you load the model
    if it doesn't exist yet.
    """
    print("="*60)
    print("Example 1: Automatic Download")
    print("="*60)

    # Load model with auto_download=True (default)
    print("Loading MT3-PyTorch model...")
    print("(Checkpoint will be downloaded automatically if missing)")

    model = load_model('mt3_pytorch', auto_download=True)

    print("✅ Model loaded successfully!")
    print(f"   Checkpoint location: .mt3_checkpoints/mt3_pytorch/")
    print()


def example_manual_download():
    """
    Example 2: Manual Download

    Download the checkpoint first using download_model(), then load it.
    This is useful if you want to pre-download all models before using them.
    """
    print("="*60)
    print("Example 2: Manual Download")
    print("="*60)

    # Step 1: Download checkpoint manually
    print("Downloading MT3-PyTorch checkpoint...")
    checkpoint_path = download_model('mt3_pytorch')

    print(f"✅ Checkpoint downloaded to: {checkpoint_path}")
    print()

    # Step 2: Load model (will use the downloaded checkpoint)
    print("Loading model...")
    model = load_model('mt3_pytorch')

    print("✅ Model loaded successfully!")
    print()


def example_download_all_models():
    """
    Example 3: Download All Models

    Pre-download all available models at once.
    """
    print("="*60)
    print("Example 3: Download All Models")
    print("="*60)

    models = ["mr_mt3", "mt3_pytorch", "yourmt3"]

    for model_id in models:
        print(f"\nDownloading {model_id}...")
        try:
            checkpoint_path = download_model(model_id)
            print(f"✅ {model_id}: Downloaded to {checkpoint_path}")
        except Exception as e:
            print(f"❌ {model_id}: {e}")

    print("\n✅ All downloads complete!")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python examples/download_mt3_pytorch.py auto     # Automatic download")
        print("  python examples/download_mt3_pytorch.py manual   # Manual download")
        print("  python examples/download_mt3_pytorch.py all      # Download all models")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "auto":
        example_automatic_download()
    elif mode == "manual":
        example_manual_download()
    elif mode == "all":
        example_download_all_models()
    else:
        print(f"Unknown mode: {mode}")
        print("Use: auto, manual, or all")
        sys.exit(1)
