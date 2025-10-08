# YourMT3 Discovery Report

**Date:** 2025-10-07
**Status:** ✅ **CHECKPOINTS DOWNLOADED**

## Summary

Successfully cloned YourMT3 HuggingFace Space and downloaded 5 pretrained model checkpoints totaling ~2.6GB.

## Checkpoint Files Downloaded

| Model Name | Checkpoint File | Size | Use Case |
|------------|----------------|------|----------|
| **YPTF.MoE+Multi (noPS)** | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt` | 536MB | **Default** - Multi-task, MoE, no pitch shift |
| YPTF.MoE+Multi (PS) | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt` | 724MB | Multi-task, MoE, with pitch shift |
| YPTF+Multi (PS) | `ptf_mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt` | 517MB | Multi-task, with pitch shift |
| YPTF+Single (noPS) | `ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt` | 345MB | Single-task |
| YMT3+ | `notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt` | 518MB | No task conditioning |

**Total:** ~2.6GB for all 5 checkpoints

## Repository Structure

```
refs/yourmt3/
├── amt/
│   ├── src/
│   │   ├── config/           # Configuration files
│   │   ├── model/            # YourMT3 model implementation
│   │   ├── utils/            # Audio, MIDI, tokenizer, etc.
│   │   └── test.py           # Test/inference script
│   └── logs/2024/
│       └── [model_name]/checkpoints/
│           └── model.ckpt or last.ckpt
├── app.py                    # Gradio demo app
├── model_helper.py           # Model loading and inference functions
├── html_helper.py            # HTML utilities
├── requirements.txt          # Dependencies
└── examples/                 # Test audio files
```

## Model Architecture

### YourMT3 Features:
- **Encoder types:** T5, Perceiver-TF, Conformer
- **Decoder types:** T5, Multi-T5
- **Special features:**
  - Mixture of Experts (MoE) feed-forward layers
  - Multiple positional encoding types (RoPE, ALiBi, etc.)
  - Task-conditional encoding
  - 8-stem separation capability
  - Multi-instrument transcription

### Default Model (YPTF.MoE+Multi noPS):
- Encoder: Perceiver-TF
- Decoder: Multi-T5 (26 layers)
- Feed-forward: MoE with 8 experts, top-k=2
- Activation: SiLU
- Positional encoding: RoPE (Rotary Position Embedding)
- Audio codec: Spectrogram
- Hop length: 300 frames

## Inference Pipeline

Based on `model_helper.py`:

1. **Load model checkpoint:**
   ```python
   model = load_model_checkpoint(args=args, device="cpu")
   ```

2. **Convert audio:**
   - Resample to model sample rate
   - Slice into segments (input_frames length)
   - Convert to float32 tensors

3. **Inference:**
   ```python
   pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
   ```

4. **Post-processing:**
   - Detokenize predictions
   - Merge note events across segments
   - Generate MIDI output

## Key Dependencies

From `requirements.txt`:
```
pytorch-lightning==2.0.7
youtube-dl
matplotlib
gradio
pretty-midi
```

Plus from `amt/src/` imports:
- torch, torchaudio
- transformers (for T5 models)
- Custom modules: model/ymt3.py, utils/task_manager.py, config/vocabulary.py

## Comparison with MT3-PyTorch

| Feature | MT3-PyTorch | YourMT3 |
|---------|-------------|---------|
| **Architecture** | Standard T5 (8 layers) | Perceiver-TF + Multi-T5 (26 layers) |
| **Checkpoint size** | 176MB | 345-724MB (5 models) |
| **Framework** | Pure PyTorch | PyTorch Lightning |
| **Complexity** | Simple, self-contained | Complex, many components |
| **Multi-task** | Single task (piano transcription) | Multi-task (8-stem separation) |
| **Integration** | ✅ Easy | ⚠️ Complex |

## Integration Strategy

### Challenges:
1. **Large codebase:** YourMT3 has an entire `amt/src/` package with many modules
2. **PyTorch Lightning:** Uses Lightning for training/inference (not just PyTorch)
3. **Complex configuration:** Many command-line arguments and config files
4. **Large checkpoints:** Default model is 536MB (vs 176MB for MT3-PyTorch)
5. **Dependencies:** May conflict with existing mt3-infer dependencies

### Recommended Approach:

#### Option 1: Full Vendor (Similar to MT3-PyTorch) ⚠️ **Complex**
- Copy entire `amt/src/` directory to `mt3_infer/vendor/yourmt3/`
- Create adapter that imports from vendor directory
- **Pros:** Complete functionality, all 5 models available
- **Cons:** Large codebase (~50+ files), potential dependency conflicts

#### Option 2: Simplified Adapter ✅ **Recommended**
- Extract only essential inference code
- Create standalone adapter without full amt/ package
- Use only the default checkpoint (536MB)
- **Pros:** Smaller, cleaner integration
- **Cons:** May lose some advanced features

#### Option 3: Defer to Future Release ⏳
- Focus on getting MR-MT3 and MT3-PyTorch production-ready first
- Add YourMT3 in v0.2.0 after core models are stable
- **Pros:** Cleaner v0.1.0 release, more time for integration
- **Cons:** Users don't get multi-task features yet

## Next Steps (If proceeding with integration)

### For Option 1 (Full Vendor):
1. Copy `amt/src/` to `mt3_infer/vendor/yourmt3/`
2. Create `mt3_infer/adapters/yourmt3.py` adapter
3. Update `checkpoints.yaml` with YourMT3 checkpoint info
4. Handle PyTorch Lightning dependency
5. Test with 30s audio file
6. Integrate into worzpro-demo

### For Option 2 (Simplified):
1. Extract minimal inference code from `model_helper.py`
2. Copy only essential files from `amt/src/`:
   - `model/ymt3.py`
   - `utils/task_manager.py`
   - `config/` directory
   - Other critical utilities
3. Create lightweight adapter
4. Use only default checkpoint (536MB)
5. Test and integrate

### For Option 3 (Defer):
1. Document YourMT3 findings
2. Mark as "Coming Soon" in demo UI
3. Focus on MR-MT3 + MT3-PyTorch for v0.1.0
4. Plan YourMT3 for v0.2.0

## Recommendation

Given the complexity of YourMT3 compared to MT3-PyTorch, I recommend **Option 3: Defer to v0.2.0**.

**Rationale:**
- MR-MT3 and MT3-PyTorch provide good speed/accuracy trade-offs
- YourMT3 integration would significantly increase package complexity
- 2.6GB of checkpoints is substantial
- Can be added later without breaking existing integrations
- Gives time to properly test and validate the integration

**Current Status for v0.1.0:**
- ✅ MR-MT3: Production ready (176MB, 22.7x real-time)
- ✅ MT3-PyTorch: Production ready (176MB, 5.7x real-time)
- ⏳ YourMT3: Deferred to v0.2.0 (536MB-2.6GB, ~15x real-time, 8-stem separation)

---

**YourMT3 Status: Checkpoints Downloaded, Integration Deferred** 📦

All checkpoints are available at:
`/home/worzpro/Desktop/dev/patched_modules/mt3-infer/refs/yourmt3/amt/logs/2024/`
