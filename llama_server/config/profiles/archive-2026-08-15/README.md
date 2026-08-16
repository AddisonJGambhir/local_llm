# Archived profiles — 2026-08-15

These reference GGUF files that no longer exist on disk. The 24 GiB-only era is
over: with the R9700 added there is 56 GiB of VRAM, so the Q4/IQ4 quants these
profiles were built around are no longer needed.

Kept rather than deleted so the measured tuning notes in their headers (DFlash
n-max values, context ceilings, KV-quant comparisons) are not lost.

Restoring one requires re-downloading its model; check `LLAMA_MODEL` in the file.
