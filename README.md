# EchoSampler

**Entropy-Echo Guided Adaptive Sampling for LLMs**

A minimalist, industrial-ready adaptive sampler that uses real-time entropy and varentropy feedback to dynamically adjust temperature and noise.

- **Reality Mode**: Higher entropy → higher temperature for exploration (great for reasoning, math, code)
- **Dream Mode**: Lower entropy → lower temperature + tiny noise for ultra-coherent long-form generation (stories, creative chains)

Inspired by entropy-aware sampling research (Entropix et al.) and born from a playful late-night chat with Grok.

## Features
- Pure PyTorch, no extra dependencies
- Batch-compatible & gradient-safe
- Configurable parameters
- <30 lines core code – extreme minimalism

## Installation
```bash
pip install git+https://github.com/LYH10-12/EchoSampler-py.git
