# MT3-Infer Implementation Plan

**Version:** 0.1.0
**Last Updated:** 2025-10-05
**Status:** Planning Phase

---

## 🎯 Mission Statement

Create a **lightweight, framework-neutral, inference-only package** that unifies all MT3-family transcription models under a single consistent API, designed for integration into the worzpro-demo ecosystem.

---

## 📋 Goals

### Primary Goals
1. **Unified Interface**: Single API for all MT3 variants (Magenta MT3, MR-MT3, MT3-PyTorch, YourMT3)
2. **Framework Isolation**: Clean separation of PyTorch, TensorFlow, and JAX dependencies
3. **Inference-Only**: Remove all training code, focus on production transcription
4. **Version Compatibility**: Align with worzpro-demo dependencies to prevent conflicts
5. **Reproducibility**: Pinned versions, lockfiles, and checkpoint verification

### Secondary Goals
- **Integration-Ready**: Easy to use in rhythm-kit and other MIR pipelines
- **Maintainability**: Spec-driven development with clear migration provenance
- **Testing**: Regression tests ensuring output equivalence with upstream models

---

## 🏗️ Architecture Overview

### Package Structure
```
mt3_infer/
├── __init__.py                    # Public API + model registry
├── base.py                        # MT3Base abstract interface
├── adapters/
│   ├── __init__.py
│   ├── mr_mt3.py                  # MR-MT3 (PyTorch)
│   ├── mt3_pytorch.py             # MT3-PyTorch (PyTorch)
│   ├── magenta.py                 # Magenta MT3 (JAX/Flax)
│   └── yourmt3.py                 # YourMT3 (TensorFlow)
├── config/
│   ├── checkpoints.yaml           # Model registry
│   └── external_integrations.yaml # Migration provenance
├── utils/
│   ├── audio.py                   # Audio preprocessing utilities
│   ├── midi.py                    # MIDI post-processing utilities
│   └── framework.py               # Framework version checks
└── tests/
    ├── test_base.py
    ├── test_mr_mt3.py
    ├── test_mt3_pytorch.py
    ├── test_magenta.py
    └── test_yourmt3.py
```

### Core Abstractions

**MT3Base** - Abstract interface defining the inference lifecycle:
```python
class MT3Base:
    def load_model(self, checkpoint_path: str) -> None
    def preprocess(self, audio: np.ndarray, sr: int) -> Any
    def forward(self, features: Any) -> Any
    def decode(self, outputs: Any) -> mido.MidiFile
    def transcribe(self, audio: np.ndarray, sr: int) -> mido.MidiFile
```

**Unified API** - Framework-agnostic entry point:
```python
from mt3_infer import transcribe, load_model

# Simple transcription
midi = transcribe(audio, model="mr_mt3", sr=16000)

# Advanced usage
model = load_model("mr_mt3")
midi = model.transcribe(audio, sr=16000)
```

---

## 🔧 Framework Version Strategy

### Alignment with worzpro-demo

**Critical Dependencies** (must match worzpro-demo):
- `torch==2.7.1`
- `torchvision==0.22.1`
- `torchaudio==2.7.1`
- `tensorflow>=2.13.0`
- `nvidia-cuda-runtime-cu12>=12.6.77`
- `nvidia-cudnn-cu12>=9.5.1.17`

**Framework-Specific** (mt3-infer controlled):
- `jax==0.4.28`
- `flax==0.8.2`

**Rationale**: mt3-infer will be installed as a patched module in worzpro-demo. Version alignment prevents:
- Binary incompatibilities in CUDA libraries
- Conflicting PyTorch/TensorFlow GPU memory management
- Protobuf version conflicts

### Optional Extras Strategy
```toml
[project.optional-dependencies]
torch = ["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"]
tensorflow = ["tensorflow>=2.13.0"]
jax = ["jax==0.4.28", "flax==0.8.2"]
all = ["mt3-infer[torch,tensorflow,jax]"]
```

---

## 🗺️ Development Roadmap

### Phase 1: Foundation (v0.1.0) - **Current Phase**
**Timeline**: 2 weeks
**Goal**: Establish core architecture with PyTorch adapters

#### Deliverables
- [ ] `base.py` - MT3Base abstract class
- [ ] `__init__.py` - Model registry + unified API
- [ ] `config/checkpoints.yaml` - Model metadata
- [ ] `config/external_integrations.yaml` - Provenance tracking
- [ ] `adapters/mr_mt3.py` - MR-MT3 adapter (PyTorch)
- [ ] `adapters/mt3_pytorch.py` - MT3-PyTorch adapter (PyTorch)
- [ ] `tests/test_mr_mt3.py` - Regression tests
- [ ] `tests/test_mt3_pytorch.py` - Regression tests
- [ ] `utils/framework.py` - Version verification
- [ ] `pyproject.toml` - Dependencies + optional extras
- [ ] `README.md` - Basic usage documentation
- [ ] `SPEC.md` - Formal specification

#### Success Criteria
- ✅ Both PyTorch adapters pass smoke tests (non-empty MIDI output)
- ✅ Framework version verification works
- ✅ Can be imported without triggering unused framework dependencies
- ✅ Installation in worzpro-demo succeeds without conflicts

---

### Phase 2: JAX/TF Expansion (v0.2.0)
**Timeline**: 2 weeks
**Goal**: Complete adapter coverage for all MT3 variants

#### Deliverables
- [ ] `adapters/magenta.py` - Magenta MT3 (JAX/Flax)
- [ ] `adapters/yourmt3.py` - YourMT3 (TensorFlow)
- [ ] Tests for both new adapters
- [ ] Regression baselines (MIDI hash comparison)

#### Success Criteria
- ✅ All 4 adapters produce consistent MIDI output
- ✅ Framework isolation maintained
- ✅ CI matrix tests all backends

---

### Phase 3: Usability (v0.3.0)
**Timeline**: 1 week
**Goal**: CLI and deployment-friendly features

#### Deliverables
- [ ] CLI tool: `mt3-infer transcribe audio.wav --model mr_mt3`
- [ ] ONNX export support for PyTorch models
- [ ] Batch processing utilities
- [ ] Enhanced documentation + examples

#### Success Criteria
- ✅ CLI can transcribe audio without Python code
- ✅ ONNX models run 1.5× faster than PyTorch
- ✅ Batch processing handles 100+ files

---

### Phase 4: Validation (v0.4.0)
**Timeline**: 1 week
**Goal**: Benchmarking and quality assurance

#### Deliverables
- [ ] Benchmark suite (speed + accuracy)
- [ ] Gradio demo for model comparison
- [ ] Performance regression tests
- [ ] Accuracy comparison with upstream repos

#### Success Criteria
- ✅ Runtime within 1.2× of upstream models
- ✅ MIDI output matches upstream (±0.01s timing)
- ✅ Interactive demo deployed

---

### Phase 5: Production (v1.0.0)
**Timeline**: 2 weeks
**Goal**: Production-ready release

#### Deliverables
- [ ] Full CI/CD pipeline (GitHub Actions)
- [ ] Comprehensive documentation
- [ ] Migration guide from legacy repos
- [ ] Security audit + dependency scanning
- [ ] Release artifacts (wheels, conda packages)

#### Success Criteria
- ✅ 90%+ test coverage
- ✅ All linters pass (ruff, mypy, black)
- ✅ Documentation complete
- ✅ Published to PyPI

---

## 🔄 Migration Strategy

### Upstream Repository Handling

For each MT3 variant:
1. **Analysis**: Identify inference-critical code paths
2. **Extraction**: Isolate model definition + forward pass
3. **Modernization**: Update to aligned framework versions
4. **Adaptation**: Wrap in MT3Base interface
5. **Validation**: Compare MIDI output to upstream
6. **Documentation**: Record in `external_integrations.yaml`

### Provenance Tracking
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

## 🧪 Testing Strategy

### Test Levels

**1. Smoke Tests** (Fast, always run)
- Import succeeds
- Model loads
- Inference returns non-empty MIDI

**2. Regression Tests** (Medium, PR-gated)
- MIDI output matches baseline hash
- Framework version matches expected
- No dependency leakage across extras

**3. Integration Tests** (Slow, nightly)
- End-to-end audio → MIDI pipeline
- Performance benchmarks
- Memory usage profiling

### Benchmark Strategy
- Prioritize building adapters; use upstream repositories for targeted comparisons only when behaviour looks suspect.
- Keep shared benchmark audio in `assets/` (starting with `HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav`; see `docs/dev/BENCHMARKS.md`) to reproduce issues quickly.
- Set up per-reference `uv` environments in `refs/<repo>/` only as-needed for debugging or regression investigations.
- During Phase 6, run focused comparisons on-demand rather than maintaining always-up-to-date baseline hashes.

### Test Matrix
| Backend     | Python | Framework Version | CUDA |
|-------------|--------|-------------------|------|
| PyTorch     | 3.10   | torch==2.7.1      | 12.6 |
| TensorFlow  | 3.10   | tf>=2.13.0        | 12.6 |
| JAX         | 3.10   | jax==0.4.28       | 12.6 |

---

## 📦 Integration with worzpro-demo

### Installation Path
```toml
# In worzpro-demo/pyproject.toml
[tool.uv.sources]
mt3-infer = { path = "../patched_modules/mt3-infer", extras = ["torch"] }
```

### Usage in rhythm-kit
```python
from mt3_infer import transcribe

class MT3Stage(BaseStage):
    requires = ["audio.mix"]
    produces = ["symbolic.midi"]

    def _process_impl(self, data):
        midi = transcribe(
            data.audio["mix"],
            model="mr_mt3",
            sr=data.metadata["sample_rate"]
        )
        data.symbolic["midi"] = midi
        return data
```

---

## 🚨 Risk Assessment

### High Risk
| Risk | Mitigation |
|------|------------|
| Framework version conflicts | Pin to worzpro-demo versions, test in integration |
| Upstream model changes | Lock to specific git commits, track in YAML |
| CUDA compatibility | Match CUDA runtime versions exactly |

### Medium Risk
| Risk | Mitigation |
|------|------------|
| Performance regression | Establish benchmarks early, monitor in CI |
| API instability | Version with SemVer strictly |
| Checkpoint availability | Mirror checkpoints, provide fallback URLs |

### Low Risk
| Risk | Mitigation |
|------|------------|
| Documentation drift | Auto-generate from docstrings |
| Test flakiness | Use deterministic seeds, fixed test data |

---

## 📊 Success Metrics

### v0.1.0 Metrics
- [ ] 2 PyTorch adapters functional
- [ ] 100% smoke test pass rate
- [ ] Zero dependency conflicts with worzpro-demo
- [ ] <5 seconds cold-start transcription time

### v1.0.0 Metrics
- [ ] 4 backends supported
- [ ] 90%+ test coverage
- [ ] <3 seconds inference time (average)
- [ ] 10+ GitHub stars (community validation)

---

## 📚 Documentation Deliverables

- [x] `PLAN.md` (this document)
- [ ] `SPEC.md` - Formal API specification
- [ ] `TODO.md` - Concrete task checklist
- [ ] `README.md` - User-facing quick start
- [ ] `CONTRIBUTING.md` - Developer guidelines
- [ ] `CHANGELOG.md` - Version history
- [ ] `docs/MIGRATION.md` - Transition from legacy repos
- [ ] `docs/ARCHITECTURE.md` - Deep-dive design docs

---

## 🔗 References

- [worzpro-demo pyproject.toml](../../worzpro-demo/pyproject.toml)
- [MR-MT3 Repository](https://github.com/gudgud96/MR-MT3)
- [MT3 Original Paper](https://arxiv.org/abs/2111.03017)
- [UV Documentation](https://github.com/astral-sh/uv)

---

**Next Steps**: Proceed to SPEC.md for formal interface definitions.
