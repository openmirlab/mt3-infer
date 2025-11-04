# MT3-Infer Documentation

Complete documentation for the MT3-Infer package.

---

## User Documentation

### Getting Started
- [Main README](../README.md) - Installation, quick start, and usage examples
- [Examples](../examples/README.md) - Usage examples and diagnostics

### Technical Guides
- [Preprocessing & Postprocessing](PREPROCESSING_POSTPROCESSING.md) - Custom preprocessing and postprocessing implementations
  - MT3-PyTorch automatic instrument leakage filtering
  - YourMT3 adaptive time-stretching for dense patterns
- [Adaptive Preprocessing](ADAPTIVE_PREPROCESSING.md) - Detailed guide to YourMT3's adaptive mode
- [Instrument Leakage Investigation](INSTRUMENT_LEAKAGE_INVESTIGATION.md) - MT3-PyTorch auto-filter analysis
- [Audio Requirements](AUDIO_REQUIREMENTS.md) - Audio format requirements and best practices
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Benchmarks](BENCHMARKS.md) - Performance comparisons across models

---

## Developer Documentation

Developer-focused documentation is in the [`dev/`](dev/) directory:

- [API Specification](dev/SPEC.md) - Formal API specification
- [Design Principles](dev/PRINCIPLES.md) - Core design principles (UV, version alignment)
- [Download System](dev/DOWNLOAD.md) - Checkpoint download internals

### Integration
- [Claude Code Guide](../CLAUDE.md) - Development with Claude Code

---

## Directory Structure

```
docs/
├── README.md                              # This file - documentation index
├── PREPROCESSING_POSTPROCESSING.md        # Custom preprocessing/postprocessing
├── ADAPTIVE_PREPROCESSING.md              # YourMT3 adaptive mode
├── INSTRUMENT_LEAKAGE_INVESTIGATION.md    # MT3-PyTorch auto-filter
├── AUDIO_REQUIREMENTS.md                  # Audio format requirements
├── TROUBLESHOOTING.md                     # Common issues
├── BENCHMARKS.md                          # Performance benchmarks
└── dev/                                   # Developer documentation
    ├── SPEC.md                            # API specification
    ├── PRINCIPLES.md                      # Design principles
    └── DOWNLOAD.md                        # Download system internals
```

---

## Contributing Documentation

When adding new documentation:

- **User guides** → `docs/` (root level)
- **Developer documentation** → `docs/dev/`
- **Code examples** → `examples/`

---

## Quick Links

### For Users
- [Quick Start](../README.md#quick-start)
- [Preprocessing & Postprocessing Guide](PREPROCESSING_POSTPROCESSING.md)
- [Audio Requirements](AUDIO_REQUIREMENTS.md)
- [Troubleshooting](TROUBLESHOOTING.md)

### For Developers
- [API Specification](dev/SPEC.md)
- [Design Principles](dev/PRINCIPLES.md)
- [Contributing Guidelines](../CLAUDE.md)

### Technical Deep Dives
- [MT3-PyTorch Auto-Filter](INSTRUMENT_LEAKAGE_INVESTIGATION.md) - How automatic instrument leakage detection works
- [YourMT3 Adaptive Mode](ADAPTIVE_PREPROCESSING.md) - Time-stretching strategy for dense patterns
- [Performance Benchmarks](BENCHMARKS.md) - Speed and accuracy comparisons
