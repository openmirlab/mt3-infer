# MT3-Infer Documentation

Complete documentation for the MT3-Infer package.

---

## User Documentation

### Getting Started
- [Main README](../README.md) - Installation, quick start, and usage examples
- [Download Guide](DOWNLOAD.md) - Checkpoint download methods and troubleshooting

### Integration
- [Claude Code Guide](../CLAUDE.md) - Development with Claude Code

---

## Development Documentation

All development-related documentation is in the [`dev/`](dev/) directory:

### Core Development Docs
- [API Specification](dev/SPEC.md) - Formal API specification
- [Design Principles](dev/PRINCIPLES.md) - Core design principles (UV, version alignment)
- [Preprocessing & Postprocessing](dev/PREPROCESSING_POSTPROCESSING.md) - Custom preprocessing and postprocessing implementations
- [Adaptive Preprocessing](dev/ADAPTIVE_PREPROCESSING.md) - YourMT3 adaptive time-stretching for dense patterns
- [Instrument Leakage Investigation](dev/INSTRUMENT_LEAKAGE_INVESTIGATION.md) - MT3-PyTorch auto-filter analysis
- [Audio Requirements](dev/AUDIO_REQUIREMENTS.md) - Audio format requirements
- [Troubleshooting](dev/TROUBLESHOOTING.md) - Common issues and solutions
- [Benchmarks](dev/BENCHMARKS.md) - Performance benchmarks
- [Download Guide](dev/DOWNLOAD.md) - Checkpoint download system


---

## Directory Structure

```
docs/
├── README.md                              # This file - documentation index
└── dev/                                   # Development documentation
    ├── SPEC.md                           # API specification
    ├── PRINCIPLES.md                     # Design principles
    ├── PREPROCESSING_POSTPROCESSING.md   # Custom preprocessing/postprocessing
    ├── ADAPTIVE_PREPROCESSING.md         # YourMT3 adaptive mode
    ├── INSTRUMENT_LEAKAGE_INVESTIGATION.md  # MT3-PyTorch auto-filter
    ├── AUDIO_REQUIREMENTS.md             # Audio format requirements
    ├── TROUBLESHOOTING.md                # Common issues
    ├── BENCHMARKS.md                     # Performance benchmarks
    └── DOWNLOAD.md                       # Download system
```

---

## Contributing Documentation

When adding new documentation:

- **User guides** → Main README or examples/
- **Technical documentation** → `docs/dev/`
- **Code examples** → `examples/`

---

## Quick Links

### For Users
- [Quick Start](../README.md#quick-start)
- [Examples](../examples/README.md)

### For Contributors
- [Development Principles](dev/PRINCIPLES.md)
- [API Specification](dev/SPEC.md)
- [Preprocessing & Postprocessing](dev/PREPROCESSING_POSTPROCESSING.md)

### Technical Deep Dives
- [Preprocessing & Postprocessing Customizations](dev/PREPROCESSING_POSTPROCESSING.md) - MT3-PyTorch auto-filter and YourMT3 adaptive mode
- [Adaptive Preprocessing](dev/ADAPTIVE_PREPROCESSING.md) - YourMT3 time-stretching for dense patterns
- [Instrument Leakage Investigation](dev/INSTRUMENT_LEAKAGE_INVESTIGATION.md) - MT3-PyTorch auto-filter analysis
- [Benchmarks](dev/BENCHMARKS.md) - Performance comparisons
