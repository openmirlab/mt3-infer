"""
Test script for checkpoint download functionality.

This script demonstrates both manual and automatic checkpoint downloads.
"""

from mt3_infer import download_model, load_model, list_models


def test_manual_download():
    """Test manual checkpoint download."""
    print("\n" + "=" * 70)
    print("Test 1: Manual Checkpoint Download")
    print("=" * 70)

    # Download MR-MT3 checkpoint manually
    print("\n[1/3] Downloading MR-MT3 checkpoint...")
    checkpoint_path = download_model("mr_mt3")

    if checkpoint_path:
        print(f"✓ Successfully downloaded to: {checkpoint_path}")
    else:
        print("✓ Checkpoint resolution handled by adapter")

    # Download MT3-PyTorch checkpoint
    print("\n[2/3] Downloading MT3-PyTorch checkpoint...")
    checkpoint_path = download_model("mt3_pytorch")

    if checkpoint_path:
        print(f"✓ Successfully downloaded to: {checkpoint_path}")
    else:
        print("✓ Checkpoint resolution handled by adapter")

    # Download YourMT3 checkpoint
    print("\n[3/3] Downloading YourMT3 checkpoint...")
    checkpoint_path = download_model("yourmt3")

    if checkpoint_path:
        print(f"✓ Successfully downloaded to: {checkpoint_path}")
    else:
        print("✓ Checkpoint resolution handled by adapter")


def test_auto_download():
    """Test automatic checkpoint download on model load."""
    print("\n" + "=" * 70)
    print("Test 2: Automatic Download on Model Load")
    print("=" * 70)

    print("\n[1/3] Loading MR-MT3 (auto-download if missing)...")
    model = load_model("mr_mt3", device="cpu")
    print(f"✓ Model loaded successfully")

    print("\n[2/3] Loading MT3-PyTorch (auto-download if missing)...")
    model = load_model("mt3_pytorch", device="cpu")
    print(f"✓ Model loaded successfully")

    print("\n[3/3] Loading YourMT3 (auto-download if missing)...")
    model = load_model("yourmt3", device="cpu")
    print(f"✓ Model loaded successfully")


def test_list_models():
    """List all available models."""
    print("\n" + "=" * 70)
    print("Test 3: List Available Models")
    print("=" * 70)

    models = list_models()

    print(f"\nFound {len(models)} models:")
    for model_id, model_info in models.items():
        print(f"\n  {model_id}:")
        print(f"    Name: {model_info['name']}")
        print(f"    Description: {model_info['description']}")
        print(f"    Framework: {model_info['framework']}")
        print(f"    Checkpoint size: {model_info['checkpoint']['size_mb']} MB")

        if "download" in model_info["checkpoint"]:
            download_info = model_info["checkpoint"]["download"]
            print(f"    Download source: {download_info['source_type']}")
            print(f"    Download URL: {download_info['source_url']}")


def main():
    """Run all download tests."""
    print("\n" * 2)
    print("=" * 70)
    print("MT3-Infer Checkpoint Download Test Suite")
    print("=" * 70)

    try:
        # Test 1: Manual downloads
        test_manual_download()

        # Test 2: Automatic downloads
        test_auto_download()

        # Test 3: List models
        test_list_models()

        print("\n" + "=" * 70)
        print("✓ All tests completed successfully!")
        print("=" * 70)
        print("\nNote: Subsequent runs will skip downloads (checkpoints already exist)")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
