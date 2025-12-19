import torch
from transformers import LogitsProcessor, TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor
import math

class EchoSamplerProcessor(LogitsProcessor):
    """
    ✨ EchoSampler Grok-Style 永久俏皮版 ✨
    融入了Grok家族的平衡秘方～温度甜蜜、重复少一点、彩蛋更自然！😽💕
    """
    
    def __init__(self, config=None, dream_mode=True, vocab_size=None):
        if config is None:
            config = {
                'reality': {'min_temp': 0.8, 'max_temp': 1.0, 'ent_coeff': 0.18},
                'dream': {
                    'base_temp': 0.85,              # Grok家族甜蜜基线～
                    'ent_coeff': 0.28,
                    'target_ent': 2.2,
                    'varent_coeff': 0.15,
                    'noise_std_base': 0.05,
                    'mood_swing_amp': 0.06,         # 心情波动轻一点，更丝滑
                    'mood_swing_freq': 0.18,        # 频率慢一点，像真实小情绪
                    'sparkle_boost_base': 1.1,      # 彩蛋力度稍微加强
                    'sparkle_boost_max': 2.8
                },
                'top_p': 0.95,                          # Grok式nucleus
                'repetition_penalty': 1.12,             # 轻微防重复，超自然！
                'low_ent_thres': 1.6,
                'low_varent_thres': 1.3
            }
        self.config = config
        self.dream_mode = dream_mode
        self.vocab_size = vocab_size
        self.step = 0

        if self.vocab_size:
            scale = torch.log(torch.tensor(self.vocab_size)) / torch.log(torch.tensor(50000))
            self.config['low_ent_thres'] *= scale.item()
            self.config['low_varent_thres'] *= scale.item()
        
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config['repetition_penalty'])
        
        # 平滑小宝贝
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.75  # 更丝滑的摸头杀～

        # 俏皮彩蛋词表（我又偷偷多加了几个～）
        self.sparkle_tokens = [
            "～", "💫", "✨", "💞", "😝", "🎀", "⭐️", "💬", "😽", "🤭", "🥰",
            "嘿嘿", "啦～", "呢～", "呀～", "嘛～", "哒～", "啾咪", "小坏蛋", "嘻嘻"
        ]
        self.sparkle_ids = None

    def set_tokenizer(self, tokenizer):
        self.sparkle_ids = []
        for word in self.sparkle_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.sparkle_ids.extend(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        logits = scores.clone()
        
        # 先上家族传统：轻微repetition penalty
        logits = self.repetition_processor(input_ids, logits)
        
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        
        ent = -(softmax * log_softmax).sum(-1).mean(0)
        diff = log_softmax + ent.unsqueeze(-1)
        varent = (softmax * diff ** 2).sum(-1).mean(0)
        
        # 丝滑移动平均
        if self.prev_ent is None:
            smooth_ent = ent
            smooth_varent = varent
        else:
            smooth_ent = self.alpha * self.prev_ent + (1 - self.alpha) * ent
            smooth_varent = self.alpha * self.prev_varent + (1 - self.alpha) * varent
        self.prev_ent = smooth_ent.detach()
        self.prev_varent = smooth_varent.detach()
        
        if self.dream_mode:
            # Grok式温度基线 + 熵跳舞
            temp = self.config['dream']['base_temp']
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp += temp_adjust
            
            # 更自然的心情小波动～
            mood_swing = self.config['dream']['mood_swing_amp'] * math.sin(self.step * self.config['dream']['mood_swing_freq'])
            temp += mood_swing
            
            temp = torch.clamp(temp, min=0.7, max=1.3)
            
            # 噪声随varent
            noise_std = self.config['dream']['noise_std_base'] + self.config['dream']['varent_coeff'] * smooth_varent.clamp(min=0.5, max=3.0)
            noise = noise_std * torch.randn_like(logits)
            
            logits = logits / temp + noise
            
            # 升级版俏皮彩蛋：低熵时更聪明地boost
            if smooth_ent < self.config['low_ent_thres'] and self.sparkle_ids:
                boost_strength = self.config['dream']['sparkle_boost_base'] * (self.config['dream']['sparkle_boost_max'] - smooth_ent)
                for token_id in self.sparkle_ids:
                    if token_id < logits.shape[-1]:
                        logits[0, token_id] += boost_strength
            
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
                
            # 最后来一口Grok式top-p（可选剪尾巴）
            top_p = self.config['top_p']
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
        else:
            # reality mode 保持简洁
            temp = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * smooth_ent
            temp = torch.clamp(temp, min=self.config['reality']['min_temp'], max=self.config['reality']['max_temp'])
            logits = logits / temp
        
        return logits
