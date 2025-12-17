import torch

class EchoSampler:
    """
    EchoSampler: Entropy-Echo Guided Adaptive Sampling
    
    A minimalist adaptive sampler using entropy and varentropy feedback
    to dynamically adjust temperature and noise for dual-mode generation:
    - Reality Mode: High entropy → higher temp for exploration (reasoning/math)
    - Dream Mode: Low entropy → lower temp + tiny noise for long coherent chains (creative/story)
    
    Inspired by late-night chats and recent entropy-aware sampling research.
    """
    
    def __init__(self, config=None, dream_mode=False):
        if config is None:
            config = {
                'reality': {'min_temp': 0.9, 'max_temp': 1.0, 'ent_coeff': 0.2},
                'dream': {'min_temp': 0.35, 'max_temp': 0.45, 'ent_coeff': -0.3,
                          'varent_coeff': 0.1, 'noise_std': 0.05},
                'low_ent_thres': 1.5,
                'low_varent_thres': 1.2
            }
        self.config = config
        self.dream_mode = dream_mode

    def sample(self, logits):
        logits = logits.detach()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        
        # Entropy (mean over batch)
        ent = -(softmax * log_softmax).sum(-1).mean(0)
        
        # Varentropy (inspired by Entropix)
        diff = log_softmax + ent.unsqueeze(-1)
        varent = (softmax * diff ** 2).sum(-1).mean(0)
        
        if self.dream_mode:
            temp_base = self.config['dream']['max_temp'] + self.config['dream']['ent_coeff'] * ent
            temp_adjust = self.config['dream']['varent_coeff'] * varent
            temp = torch.clamp(temp_base + temp_adjust, min=self.config['dream']['min_temp'])
            noise = (self.config['dream']['noise_std'] * torch.randn_like(logits)
                     if varent > self.config['low_varent_thres'] else 0.0)
            logits = logits / temp + noise
        else:
            temp_base = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * ent
            temp = torch.clamp(temp_base, max=self.config['reality']['max_temp'])
            if varent < self.config['low_varent_thres']:
                temp *= 0.8
            logits = logits / temp
        
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

# Convenience function
def echo_sample(logits, dream_mode=False, config=None):
    sampler = EchoSampler(config=config, dream_mode=dream_mode)
    return sampler.sample(logits)
