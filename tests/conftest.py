"""
Pytest configuration and shared fixtures for mt3_infer tests.
"""

import numpy as np
import pytest


@pytest.fixture
def random_audio():
    """Generate random audio for testing."""
    np.random.seed(42)  # Reproducible
    return np.random.randn(16000).astype(np.float32)


@pytest.fixture
def stereo_audio():
    """Generate random stereo audio for testing."""
    np.random.seed(42)
    return np.random.randn(16000, 2).astype(np.float32)


@pytest.fixture
def empty_audio():
    """Generate empty audio array for testing."""
    return np.array([], dtype=np.float32)


@pytest.fixture
def sample_rate():
    """Standard sample rate for testing."""
    return 16000
