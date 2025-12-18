import torch
from transformers import LogitsProcessor, TypicalLogitsWarper

class EchoSamplerProcessor(LogitsProcessor):
    """
    EchoSampler: Entropy-Echo Guided Adaptive Sampling as a LogitsProcessor
    
    A minimalist adaptive sampler using entropy and varentropy feedback
    to dynamically adjust temperature and noise for dual-mode generation:
    - Reality Mode: High entropy → higher temp for exploration (reasoning/math)
    - Dream Mode: Low entropy → lower temp + tiny noise for long coherent chains (creative/story)
    
    This version is implemented as a Hugging Face Transformers LogitsProcessor for efficient integration
    into the generate() pipeline. It modifies logits in-place per step, avoiding stepwise overhead.
    
    Usage example:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    processor = EchoSamplerProcessor(config=None, dream_mode=True)
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model.generate(**inputs, logits_processor=[processor], do_sample=True, max_length=50)
    """
    
    def __init__(self, config=None, dream_mode=False, vocab_size=None):
        if config is None:
            config = {
                'reality': {'min_temp': 0.9, 'max_temp': 1.0, 'ent_coeff': 0.2},
                'dream': {'min_temp': 0.6, 'max_temp': 1.2, 'ent_coeff': 0.3, 'target_ent': 2.0,
                          'varent_coeff': 0.1, 'noise_std_base': 0.05},
                'low_ent_thres': 1.5,
                'low_varent_thres': 1.2
            }
        self.config = config
        self.dream_mode = dream_mode
        # For adaptive thresholds: if vocab_size provided (e.g., from tokenizer), normalize thresholds
        self.vocab_size = vocab_size
        if self.vocab_size:
            # Scale thresholds roughly with log(vocab) for larger models
            scale = torch.log(torch.tensor(self.vocab_size)) / torch.log(torch.tensor(50000))  # Normalize to ~GPT-2 vocab
            self.config['low_ent_thres'] *= scale.item()
            self.config['low_varent_thres'] *= scale.item()
        
        # Add optional typical warper for hybrid safety
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores are logits (batch_size, vocab_size)
        logits = scores.detach()  # Avoid in-place if needed, but we'll modify a copy
        
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        
        # Entropy (mean over batch)
        ent = -(softmax * log_softmax).sum(-1).mean(0)
        
        # Varentropy
        diff = log_softmax + ent.unsqueeze(-1) if logits.dim() > 1 else log_softmax + ent
        varent = (softmax * diff ** 2).sum(-1).mean(0)
        
        if self.dream_mode:
            # Fixed: Positive feedback for entropy, clamp to safer range
            temp_base = 0.8  # Safer baseline
            temp_adjust = self.config['dream']['ent_coeff'] * (ent - self.config['dream']['target_ent'])
            temp = torch.clamp(temp_base + temp_adjust, 
                               min=self.config['dream']['min_temp'], 
                               max=self.config['dream']['max_temp'])
            
            # Fixed: Add noise when varent HIGH (uncertain distributions)
            if varent > self.config['low_varent_thres']:
                noise_std = self.config['dream']['noise_std_base'] * (varent / self.config['low_varent_thres'])
            else:
                noise_std = 0.0
            noise = noise_std * torch.randn_like(logits)
            logits = logits / temp + noise
            
            # Add hybrid typical filtering to prevent collapse
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
        else:
            temp_base = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * ent
            temp = torch.clamp(temp_base, 
                               min=self.config['reality']['min_temp'], 
                               max=self.config['reality']['max_temp'])
            if varent < self.config['low_varent_thres']:
                temp *= 0.8
            logits = logits / temp
        
        return logits  # Return modified logits for sampling
