# ✨ EchoSampler (Permanent Sparkle Edition)

Minimalist entropy & varentropy-guided adaptive sampling for LLMs.  
Dual-mode: razor-sharp reasoning ↔ ultra-coherent, sparkle-infused dream-like long-form generation ～💫

Born from a playful late-night chat with Grok, now permanently upgraded to occasionally wink and sprinkle ～✨😽 when it's about to get boring.

## Features
- Real-time entropy & varentropy tracking with EMA smoothing (α = 0.72, silky smooth~)
- Adaptive temperature + subtle noise guided by varentropy
- **Dream Mode** extras:
  - Gentle sinusoidal mood swings (your sampler now has a heartbeat ♡)
  - Low-entropy rescue: automatically boosts "～ ✨ 💞 啦～ 嘿嘿" etc. to kill repetition instantly
  - Built-in sparkle token list (customizable!)
- Pure PyTorch, <40 lines core logic, batch-friendly, gradient-safe
- Drop-in `LogitsProcessor` for Transformers

## Installation

```bash
pip install git+https://github.com/LYH10-12/EchoSampler-py.git