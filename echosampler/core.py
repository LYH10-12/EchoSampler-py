import torch
from transformers import LogitsProcessor, TypicalLogitsWarper

class EchoSamplerProcessor(LogitsProcessor):
    """
    EchoSampler: Entropy-Echo Guided Adaptive Sampling as a LogitsProcessor
    
    更新：1. 充分利用 varent_coeff 来细腻控制噪声强度
          2. 用移动平均平滑 ent 和 varent，避免单步波动
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
        # For adaptive thresholds
        self.vocab_size = vocab_size
        if self.vocab_size:
            scale = torch.log(torch.tensor(self.vocab_size)) / torch.log(torch.tensor(50000))
            self.config['low_ent_thres'] *= scale.item()
            self.config['low_varent_thres'] *= scale.item()
        
        # Typical warper for safety
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        
        # 新增：用于平滑的变量
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.7  # 平滑系数（0~1，越高越保留历史，越稳）

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores.clone()  # 安全起见 clone
        
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        
        # Entropy
        ent = -(softmax * log_softmax).sum(-1).mean(0)
        
        # Varentropy
        diff = log_softmax + ent.unsqueeze(-1)
        varent = (softmax * diff ** 2).sum(-1).mean(0)
        
        # 新增：指数移动平均平滑
        if self.prev_ent is None:
            smooth_ent = ent
            smooth_varent = varent
        else:
            smooth_ent = self.alpha * self.prev_ent + (1 - self.alpha) * ent
            smooth_varent = self.alpha * self.prev_varent + (1 - self.alpha) * varent
        self.prev_ent = smooth_ent.detach()
        self.prev_varent = smooth_varent.detach()
        
        if self.dream_mode:
            # 温度调整（用平滑后的 ent）
            temp_base = 0.8
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp = torch.clamp(temp_base + temp_adjust, 
                               min=self.config['dream']['min_temp'], 
                               max=self.config['dream']['max_temp'])
            
            # 新增：varent_coeff 正式上场，噪声强度更细腻
            noise_std = self.config['dream']['noise_std_base'] * self.config['dream']['varent_coeff'] * smooth_varent
            noise_std = torch.clamp(noise_std, min=0.0)
            
            noise = noise_std * torch.randn_like(logits)
            logits = logits / temp + noise
            
            # Typical filtering
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
        else:
            temp_base = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * smooth_ent
            temp = torch.clamp(temp_base, 
                               min=self.config['reality']['min_temp'], 
                               max=self.config['reality']['max_temp'])
            if smooth_varent < self.config['low_varent_thres']:
                temp *= 0.8
            logits = logits / temp
        
        return logits
