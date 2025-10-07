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
- [Development Roadmap](dev/PLAN.md) - Phased development plan
- [Design Principles](dev/PRINCIPLES.md) - Core design principles (UV, version alignment)
- [Repository Organization](dev/REPOSITORY_ORGANIZATION.md) - Codebase structure

### Development Reports
Implementation reports and analyses are in [`dev/reports/`](dev/reports/):

#### Project Summaries
- [Cleanup Summary](dev/reports/CLEANUP_SUMMARY.md) - Production cleanup actions
- [Final Cleanup](dev/reports/FINAL_CLEANUP.md) - Final production-ready cleanup
- [Download Summary](dev/reports/DOWNLOAD_SUMMARY.md) - Download system implementation
- [Session Summary](dev/reports/SESSION_SUMMARY.md) - Development session notes

#### Package & Architecture
- [Public Package Analysis](dev/reports/PUBLIC_PACKAGE_ANALYSIS.md) - PyPI distribution analysis
- [Vendoring Success](dev/reports/VENDORING_SUCCESS.md) - Vendoring summary
- [Phase 5 Public API](dev/reports/PHASE5_PUBLIC_API.md) - Public API implementation

#### Adapter Implementation
- [YourMT3 Vendoring](dev/reports/YOURMT3_VENDORING.md) - YourMT3 vendoring details
- [YourMT3 Complete](dev/reports/YOURMT3_COMPLETE.md) - YourMT3 implementation guide
- [Kunato MT3 Analysis](dev/reports/KUNATO_MT3_ANALYSIS.md) - MT3-PyTorch adapter analysis
- [Kunato PR Analysis](dev/reports/KUNATO_PR_ANALYSIS.md) - PR review analysis

#### Performance & Testing
- [Model Comparison](dev/reports/MODEL_COMPARISON.md) - MR-MT3 vs YourMT3
- [Final Adapter Comparison](dev/reports/FINAL_ADAPTER_COMPARISON.md) - Comprehensive comparison
- [GPU Performance](dev/reports/GPU_PERFORMANCE.md) - GPU benchmarks
- [GPU Verification Complete](dev/reports/GPU_VERIFICATION_COMPLETE.md) - GPU verification
- [MT3 PyTorch GPU Test Results](dev/reports/MT3_PYTORCH_GPU_TEST_RESULTS.md) - GPU testing
- [CPU Speed Analysis](dev/reports/CPU_SPEED_ANALYSIS.md) - CPU performance analysis
- [Verification Report](dev/reports/VERIFICATION_REPORT.md) - YourMT3 verification

### Historical Development Docs
- [Progress Log](dev/PROGRESS.md) - Development progress notes
- [Success Log](dev/SUCCESS.md) - Implementation milestones
- [Review Notes](dev/REVIEW.md) - Code review notes
- [Benchmarks](dev/BENCHMARKS.md) - Performance benchmarks
- [TODO](dev/TODO.md) - Task tracking

---

## Directory Structure

```
docs/
├── README.md          # This file - documentation index
├── DOWNLOAD.md        # User-facing download guide
└── dev/               # Development documentation
    ├── SPEC.md
    ├── PLAN.md
    ├── PRINCIPLES.md
    ├── REPOSITORY_ORGANIZATION.md
    ├── PROGRESS.md
    ├── SUCCESS.md
    ├── REVIEW.md
    ├── BENCHMARKS.md
    ├── TODO.md
    └── reports/       # Implementation reports
        ├── CLEANUP_SUMMARY.md
        ├── FINAL_CLEANUP.md
        ├── DOWNLOAD_SUMMARY.md
        ├── SESSION_SUMMARY.md
        ├── PUBLIC_PACKAGE_ANALYSIS.md
        ├── VENDORING_SUCCESS.md
        ├── PHASE5_PUBLIC_API.md
        ├── YOURMT3_VENDORING.md
        ├── YOURMT3_COMPLETE.md
        ├── KUNATO_MT3_ANALYSIS.md
        ├── KUNATO_PR_ANALYSIS.md
        ├── MODEL_COMPARISON.md
        ├── FINAL_ADAPTER_COMPARISON.md
        ├── GPU_PERFORMANCE.md
        ├── GPU_VERIFICATION_COMPLETE.md
        ├── MT3_PYTORCH_GPU_TEST_RESULTS.md
        ├── CPU_SPEED_ANALYSIS.md
        └── VERIFICATION_REPORT.md
```

---

## Contributing Documentation

When adding new documentation:

- **User guides** → `docs/` (root) or `docs/guides/` (if created)
- **Development docs** → `docs/dev/`
- **Technical reports** → `docs/dev/reports/`
- **API reference** → Generate with tools, put in `docs/api/` (if created)

---

## Quick Links

### For Users
- [Quick Start](../README.md#quick-start)
- [Download Checkpoints](DOWNLOAD.md)
- [Examples](../examples/README.md)

### For Contributors
- [Development Principles](dev/PRINCIPLES.md)
- [API Specification](dev/SPEC.md)
- [Development Roadmap](dev/PLAN.md)

### Technical Deep Dives
- [Model Comparison](dev/reports/MODEL_COMPARISON.md) - MR-MT3 vs YourMT3
- [GPU Performance](dev/reports/GPU_PERFORMANCE.md) - GPU benchmarks
- [CPU Analysis](dev/reports/CPU_SPEED_ANALYSIS.md) - Why YourMT3 is faster on CPU
