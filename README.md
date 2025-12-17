# EchoSampler

**Entropy-Echo Guided Adaptive Sampling for LLMs**

A minimalist, industrial-ready adaptive sampler that uses real-time entropy and varentropy feedback to dynamically adjust temperature and noise.

- **Reality Mode**: Higher entropy → higher temperature for exploration (great for reasoning, math, code)
- **Dream Mode**: Lower entropy → lower temperature + tiny noise for ultra-coherent long-form generation (stories, creative chains)

Inspired by entropy-aware sampling research (Entropix et al.) and born from a playful late-night chat with Grok.

## Features
- Pure PyTorch, no extra dependencies
 - Pure PyTorch, Hugging Face `transformers` required for the LogitsProcessor integration
- Batch-compatible & gradient-safe
- Configurable parameters
- <30 lines core code – extreme minimalism

Note: The package now exposes an HF-compatible `LogitsProcessor` implementation (`EchoSamplerProcessor`).
If you plan to integrate with Hugging Face's `generate()` API, install `transformers` in addition to `torch`.

## Installation
```bash
pip install git+https://github.com/LYH10-12/EchoSampler-py.git
