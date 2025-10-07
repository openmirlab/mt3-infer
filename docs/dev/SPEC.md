# MT3-Infer Formal Specification

**Version:** 0.1.0-alpha
**Specification Date:** 2025-10-05
**Status:** Draft

---

## 1. Overview

### 1.1 Purpose
This specification defines the public API, behavior contracts, and versioning policies for the `mt3_infer` package.

### 1.2 Scope
- Public API interfaces (classes, functions, types)
- Model adapter contracts
- Configuration formats (YAML schemas)
- Error handling guarantees
- Versioning and stability promises

### 1.3 Conformance
An implementation conforms to this specification if:
1. All public APIs match the signatures defined in §3
2. All behavior contracts in §4 are satisfied
3. All configuration schemas in §5 validate correctly
4. Error handling follows §6 conventions

---

## 2. Terminology

| Term | Definition |
|------|------------|
| **Adapter** | Framework-specific implementation of MT3Base |
| **Backend** | Underlying ML framework (PyTorch, TensorFlow, JAX) |
| **Checkpoint** | Pre-trained model weights file |
| **Transcription** | Process of converting audio to MIDI representation |
| **Model Registry** | `checkpoints.yaml` file containing model metadata |

---

## 3. Public API

### 3.1 High-Level Interface

#### 3.1.1 `transcribe()`
```python
def transcribe(
    audio: np.ndarray,
    model: str,
    sr: int = 16000,
    **kwargs
) -> mido.MidiFile
```

**Parameters:**
- `audio` (np.ndarray): Audio waveform, shape `(n_samples,)` or `(n_samples, n_channels)`
  - dtype: `float32` or `float64`
  - Range: `[-1.0, 1.0]`
- `model` (str): Model identifier from registry (e.g., `"mr_mt3"`, `"mt3_pytorch"`)
- `sr` (int): Sample rate in Hz (default: 16000)
- `**kwargs`: Model-specific parameters (see §3.3)

**Returns:**
- `mido.MidiFile`: MIDI object with transcribed notes

**Raises:**
- `ValueError`: Invalid audio shape, sample rate, or model name
- `RuntimeError`: Model loading or inference failure
- `ImportError`: Required framework not installed

**Behavior:**
1. Validates input audio format
2. Loads model from registry (cached after first call)
3. Preprocesses audio to model-specific format
4. Runs inference
5. Decodes output to MIDI
6. Returns `mido.MidiFile` object

**Example:**
```python
import numpy as np
from mt3_infer import transcribe

audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
midi = transcribe(audio, model="mr_mt3", sr=16000)
midi.save("output.mid")
```

---

#### 3.1.2 `load_model()`
```python
def load_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> MT3Base
```

**Parameters:**
- `model_name` (str): Model identifier from registry
- `checkpoint_path` (Optional[str]): Override default checkpoint path
- `device` (Optional[str]): Device placement (`"cuda"`, `"cpu"`, `"auto"`)

**Returns:**
- `MT3Base`: Initialized model adapter

**Raises:**
- `ValueError`: Unknown model name
- `FileNotFoundError`: Checkpoint not found
- `RuntimeError`: Model initialization failure

**Example:**
```python
from mt3_infer import load_model

model = load_model("mr_mt3", device="cuda")
midi = model.transcribe(audio, sr=16000)
```

---

### 3.2 Base Adapter Interface

#### 3.2.1 `MT3Base` Abstract Class
```python
from abc import ABC, abstractmethod
import numpy as np
import mido

class MT3Base(ABC):
    """Abstract base class for all MT3 model adapters."""

    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str = "auto") -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint file
            device: Target device ("cuda", "cpu", "auto")

        Raises:
            FileNotFoundError: Checkpoint does not exist
            RuntimeError: Model loading failed
        """
        pass

    @abstractmethod
    def preprocess(self, audio: np.ndarray, sr: int) -> Any:
        """
        Convert raw audio to model input format.

        Args:
            audio: Audio waveform, shape (n_samples,), dtype float32
            sr: Sample rate in Hz

        Returns:
            Model-specific input tensor/array

        Raises:
            ValueError: Invalid audio format or sample rate
        """
        pass

    @abstractmethod
    def forward(self, features: Any) -> Any:
        """
        Run model inference.

        Args:
            features: Preprocessed model input

        Returns:
            Model-specific output representation

        Raises:
            RuntimeError: Inference failed
        """
        pass

    @abstractmethod
    def decode(self, outputs: Any) -> mido.MidiFile:
        """
        Convert model outputs to MIDI.

        Args:
            outputs: Model output from forward()

        Returns:
            MIDI file object

        Raises:
            ValueError: Output format invalid
        """
        pass

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> mido.MidiFile:
        """
        End-to-end transcription pipeline.

        This method MUST NOT be overridden by adapters.

        Args:
            audio: Audio waveform
            sr: Sample rate in Hz

        Returns:
            MIDI file object

        Raises:
            ValueError: Invalid inputs
            RuntimeError: Transcription failed
        """
        features = self.preprocess(audio, sr)
        outputs = self.forward(features)
        midi = self.decode(outputs)
        return midi
```

---

### 3.3 Adapter-Specific Parameters

#### 3.3.1 PyTorch Adapters (MR-MT3, MT3-PyTorch)
```python
transcribe(
    audio,
    model="mr_mt3",
    sr=16000,
    batch_size: int = 1,
    use_amp: bool = True,  # Automatic mixed precision
    compile: bool = False   # torch.compile() optimization
)
```

#### 3.3.2 TensorFlow Adapter (YourMT3)
```python
transcribe(
    audio,
    model="yourmt3",
    sr=16000,
    use_xla: bool = False  # XLA compilation
)
```

#### 3.3.3 JAX Adapter (Magenta MT3)
```python
transcribe(
    audio,
    model="magenta",
    sr=16000,
    jit: bool = True  # JIT compilation
)
```

---

## 4. Behavior Contracts

### 4.1 Input Validation
**Contract:** All public functions MUST validate inputs before processing.

**Rules:**
1. `audio` must be 1D or 2D numpy array
2. `audio.dtype` must be `float32` or `float64`
3. `audio` values SHOULD be in `[-1.0, 1.0]` (warn if exceeded)
4. `sr` must be positive integer
5. `model` must exist in registry

---

### 4.2 Output Guarantees
**Contract:** Transcription output MUST be valid MIDI.

**Rules:**
1. Return type is `mido.MidiFile`
2. MIDI contains at least one track
3. Note on/off events are properly paired
4. Timestamps are monotonically increasing
5. All note velocities in `[0, 127]`

---

### 4.3 Framework Isolation
**Contract:** Importing one backend MUST NOT trigger imports of other backends.

**Test:**
```python
# This MUST work without TensorFlow installed
from mt3_infer import load_model
model = load_model("mr_mt3")  # PyTorch-only
```

**Implementation:**
- Use lazy imports (`importlib.import_module`)
- Check framework availability at runtime
- Raise `ImportError` with helpful message if missing

---

### 4.4 Reproducibility
**Contract:** Given identical inputs and fixed random seed, output MUST be deterministic.

**Requirements:**
1. Set all framework RNG seeds in `load_model()`
2. Use deterministic algorithms (e.g., `torch.use_deterministic_algorithms(True)`)
3. Document any non-deterministic operations (e.g., GPU atomics)

---

### 4.5 Version Compatibility
**Contract:** Framework versions MUST match §7.1 requirements.

**Enforcement:**
```python
# In mt3_infer/utils/framework.py
def check_torch_version():
    import torch
    required = "2.7.1"
    actual = torch.__version__
    if not actual.startswith(required):
        raise RuntimeError(
            f"torch=={required} required, found {actual}. "
            "See docs/dev/PRINCIPLES.md for version alignment."
        )
```

---

## 5. Configuration Schemas

### 5.1 Model Registry (`checkpoints.yaml`)

**Schema:**
```yaml
models:
  <model_id>:
    name: str                    # Human-readable name
    framework: str               # "pytorch" | "tensorflow" | "jax"
    adapter_class: str           # Python import path
    checkpoint:
      url: str                   # Download URL
      sha256: str                # Checksum for verification
      size_mb: float             # File size
    metadata:
      vocab_size: int
      max_seq_len: int
      sample_rate: int
      verified_on: str           # ISO 8601 date
```

**Example:**
```yaml
models:
  mr_mt3:
    name: "MR-MT3"
    framework: "pytorch"
    adapter_class: "mt3_infer.adapters.mr_mt3.MRMT3Adapter"
    checkpoint:
      url: "https://example.com/mr_mt3.pth"
      sha256: "a1b2c3d4..."
      size_mb: 245.3
    metadata:
      vocab_size: 388
      max_seq_len: 1024
      sample_rate: 16000
      verified_on: "2025-10-05"
```

**Validation Rules:**
1. All required fields MUST be present
2. `framework` MUST be one of allowed values
3. `adapter_class` MUST be importable
4. `sha256` MUST be 64 hexadecimal characters
5. `verified_on` MUST be valid ISO 8601 date

---

### 5.2 External Integrations (`external_integrations.yaml`)

**Schema:**
```yaml
<model_id>:
  source_repo: str               # Git repository URL
  commit: str                    # Git commit hash (7+ chars)
  license: str                   # SPDX license identifier
  migrated_to: str               # File path in mt3_infer
  framework_upgrade:
    from: str                    # Original version
    to: str                      # Upgraded version
  verified_on: str               # ISO 8601 date
  notes: str                     # Migration notes
```

**Example:**
```yaml
mr_mt3:
  source_repo: "https://github.com/gudgud96/MR-MT3"
  commit: "2e31b4c"
  license: "MIT"
  migrated_to: "mt3_infer/adapters/mr_mt3.py"
  framework_upgrade:
    from: "torch==1.13"
    to: "torch==2.7.1"
  verified_on: "2025-10-05"
  notes: "Removed training code, updated to inference_mode()"
```

---

## 6. Error Handling

### 6.1 Exception Hierarchy
```
Exception
└── MT3InferError (base for all package errors)
    ├── ModelNotFoundError (unknown model ID)
    ├── CheckpointError (download/validation failure)
    ├── FrameworkError (version mismatch, import failure)
    ├── AudioError (invalid input format)
    └── InferenceError (model forward pass failure)
```

### 6.2 Error Messages
**Contract:** All exceptions MUST include actionable error messages.

**Format:**
```
{Error type}: {What went wrong}
{Why it happened}
{How to fix it}
```

**Example:**
```python
raise FrameworkError(
    "torch==2.7.1 required, found torch==2.4.0\n"
    "Version mismatch detected to prevent conflicts with worzpro-demo.\n"
    "Run: uv sync --reinstall-package torch"
)
```

---

## 7. Versioning Policy

### 7.1 Framework Versions
**Contract:** Framework versions are LOCKED to worzpro-demo.

| Framework | Required Version | Source |
|-----------|------------------|--------|
| PyTorch | `==2.7.1` | worzpro-demo/pyproject.toml:17 |
| torchvision | `==0.22.1` | worzpro-demo/pyproject.toml:18 |
| torchaudio | `==2.7.1` | worzpro-demo/pyproject.toml:19 |
| TensorFlow | `>=2.13.0` | worzpro-demo/pyproject.toml:42 |
| JAX | `==0.4.28` | mt3-infer (independent) |
| Flax | `==0.8.2` | mt3-infer (independent) |

**Update Process:**
1. worzpro-demo updates PyTorch/TensorFlow
2. mt3-infer updates matching versions
3. Run full regression test suite
4. Update `verified_on` in checkpoints.yaml
5. Bump MINOR version

---

### 7.2 Semantic Versioning
**Contract:** Package version follows SemVer 2.0.

**Rules:**
- `MAJOR`: Breaking API changes
- `MINOR`: New models, framework upgrades
- `PATCH`: Bug fixes, checkpoint updates

**Examples:**
- Add new model → `0.1.0` → `0.2.0`
- Fix MIDI decoding bug → `0.1.0` → `0.1.1`
- Change `transcribe()` signature → `0.1.0` → `1.0.0`

---

### 7.3 API Stability
**Contract:** Public API stability by version range.

| Version Range | Stability | Breaking Changes Allowed |
|---------------|-----------|--------------------------|
| 0.x.x | Alpha | Yes (with deprecation notice) |
| 1.0.0 - 1.x.x | Stable | No (only deprecations) |
| 2.0.0+ | Stable | Only at MAJOR bumps |

**Deprecation Process:**
1. Add `warnings.warn()` with removal version
2. Update docs with migration guide
3. Wait at least 2 MINOR versions
4. Remove in next MAJOR version

---

## 8. Testing Requirements

### 8.1 Conformance Tests
**Contract:** All implementations MUST pass the conformance test suite.

**Test Categories:**

**Smoke Tests** (required for all adapters):
```python
def test_import():
    """Test adapter can be imported."""
    from mt3_infer.adapters.mr_mt3 import MRMT3Adapter

def test_load_model():
    """Test model loads without error."""
    model = load_model("mr_mt3")
    assert model is not None

def test_transcribe_basic():
    """Test basic transcription produces MIDI."""
    audio = np.random.randn(16000).astype(np.float32)
    midi = transcribe(audio, model="mr_mt3", sr=16000)
    assert isinstance(midi, mido.MidiFile)
    assert len(midi.tracks) > 0
```

**Regression Tests** (required for v1.0.0):
```python
def test_output_deterministic():
    """Test same input produces same output."""
    audio = load_test_audio("piano_C4.wav")
    midi1 = transcribe(audio, model="mr_mt3", sr=16000)
    midi2 = transcribe(audio, model="mr_mt3", sr=16000)
    assert midi_hash(midi1) == midi_hash(midi2)

def test_upstream_equivalence():
    """Test output matches original repository."""
    audio = load_test_audio("piano_C4.wav")
    midi = transcribe(audio, model="mr_mt3", sr=16000)
    expected_hash = "a1b2c3d4..."  # From baseline
    assert midi_hash(midi) == expected_hash
```

---

### 8.2 Performance Requirements
**Contract:** Inference time MUST NOT exceed 1.2× upstream baseline.

**Measurement:**
```python
import time

audio = np.random.randn(16000 * 30).astype(np.float32)  # 30 seconds

start = time.perf_counter()
midi = transcribe(audio, model="mr_mt3", sr=16000)
elapsed = time.perf_counter() - start

assert elapsed < 5.0  # 30s audio transcribed in <5s
```

---

## 9. Documentation Requirements

### 9.1 Docstring Format
**Contract:** All public APIs MUST have Google-style docstrings.

**Example:**
```python
def transcribe(audio: np.ndarray, model: str, sr: int = 16000) -> mido.MidiFile:
    """
    Transcribe audio to MIDI using specified MT3 model.

    Args:
        audio: Audio waveform with shape (n_samples,) or (n_samples, n_channels).
            Values should be in range [-1.0, 1.0].
        model: Model identifier from registry (e.g., "mr_mt3", "mt3_pytorch").
        sr: Sample rate in Hz. Default is 16000.

    Returns:
        MIDI file object with transcribed notes.

    Raises:
        ValueError: If audio format or model name is invalid.
        RuntimeError: If model loading or inference fails.
        ImportError: If required framework is not installed.

    Examples:
        >>> import numpy as np
        >>> from mt3_infer import transcribe
        >>> audio = np.random.randn(16000 * 5).astype(np.float32)
        >>> midi = transcribe(audio, model="mr_mt3", sr=16000)
        >>> midi.save("output.mid")
    """
```

---

### 9.2 Type Annotations
**Contract:** All public functions MUST have type annotations.

**Requirements:**
- Use `typing` module for complex types
- Use `Optional[T]` for nullable parameters
- Use `Union[T1, T2]` for multiple types
- Use `Any` only when framework-specific (with comment)

---

## 10. Migration Path from Upstream

### 10.1 Code Extraction Guidelines
**Contract:** Migrated code MUST preserve original licensing and attribution.

**Required Header:**
```python
"""
Adapted from {original_repo} ({commit})
Original license: {license}
Migration date: {date}
Changes: {summary}
"""
```

### 10.2 Verification Process
**Contract:** All migrated models MUST pass equivalence tests.

**Steps:**
1. Run upstream code on test audio → save MIDI hash
2. Run mt3_infer adapter on same audio → save MIDI hash
3. Assert hashes match (or difference < 0.01s timing)
4. Record baseline in `tests/baselines/{model_id}.json`

---

## 11. Security Considerations

### 11.1 Checkpoint Verification
**Contract:** All downloaded checkpoints MUST be verified.

**Implementation:**
```python
import hashlib

def verify_checkpoint(path: str, expected_sha256: str) -> bool:
    """Verify checkpoint integrity."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        raise CheckpointError(
            f"Checksum mismatch: expected {expected_sha256}, got {actual}\n"
            "File may be corrupted or tampered with."
        )
    return True
```

### 11.2 Safe Deserialization
**Contract:** Model loading MUST use safe deserialization.

**Requirements:**
- PyTorch: `torch.load(..., weights_only=True)`
- TensorFlow: Use SavedModel format (not pickle)
- JAX: Use `msgpack` or `orbax` (not pickle)

---

## 12. Compliance Checklist

An implementation is specification-compliant if:

- [ ] All public APIs in §3 are implemented
- [ ] All behavior contracts in §4 are satisfied
- [ ] Configuration schemas in §5 validate correctly
- [ ] Error handling follows §6 conventions
- [ ] Framework versions match §7.1
- [ ] SemVer policy in §7.2 is followed
- [ ] All conformance tests in §8.1 pass
- [ ] Performance requirements in §8.2 are met
- [ ] Documentation requirements in §9 are satisfied
- [ ] Migration guidelines in §10 are followed
- [ ] Security requirements in §11 are enforced

---

## 13. Change Log

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0-alpha | 2025-10-05 | Initial specification draft |

---

## 14. References

1. [Semantic Versioning 2.0](https://semver.org/)
2. [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
3. [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
4. [SPDX License List](https://spdx.org/licenses/)

---

**End of Specification**
