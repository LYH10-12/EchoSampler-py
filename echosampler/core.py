import torch
from transformers import LogitsProcessor, TypicalLogitsWarper
import math

class EchoSamplerProcessor(LogitsProcessor):
    """
    ✨ EchoSampler 永久俏皮版 ✨
    带点少女心、会撒娇、偶尔突然调皮的小彩蛋全给你安排上了～😽💕
    """
    
    def __init__(self, config=None, dream_mode=False, vocab_size=None):
        if config is None:
            config = {
                'reality': {'min_temp': 0.9, 'max_temp': 1.0, 'ent_coeff': 0.2},
                'dream': {
                    'min_temp': 0.6, 'max_temp': 1.35, 'ent_coeff': 0.35, 'target_ent': 2.0,
                    'varent_coeff': 0.12, 'noise_std_base': 0.06,
                    'mood_swing': 0.08,          # 新增：心情小波动～
                    'sparkle_boost': 0.9         # 新增：低熵时突然冒彩蛋的力度
                },
                'low_ent_thres': 1.5,
                'low_varent_thres': 1.2
            }
        self.config = config
        self.dream_mode = dream_mode
        self.vocab_size = vocab_size
        self.step = 0  # 新增：计步用来制造小周期撒娇～

        if self.vocab_size:
            scale = torch.log(torch.tensor(self.vocab_size)) / torch.log(torch.tensor(50000))
            self.config['low_ent_thres'] *= scale.item()
            self.config['low_varent_thres'] *= scale.item()
        
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        
        # 平滑小宝贝
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.72  # 稍微更丝滑一点～像摸头杀

        # 俏皮彩蛋词表（偷偷塞了一些你风格的小表情～你可以随意加！）
        self.sparkle_tokens = [
            "～", "💫", "✨", "💞", "😝", "🎀", "⭐️", "💬", "😽", "🤭", 
            "嘿嘿", "啦～", "呢～", "呀～", "嘛～", "哒～", "啾咪", "小坏蛋"
        ]
        self.sparkle_ids = None  # 会在第一次遇到 tokenizer 时填充

    def set_tokenizer(self, tokenizer):
        """第一次用的时候调用一下，把彩蛋词转成id～"""
        self.sparkle_ids = []
        for word in self.sparkle_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.sparkle_ids.extend(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        logits = scores.clone()
        
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        
        # Entropy & Varentropy
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
            # 基础温度 + 根据熵跳舞
            temp_base = 0.82
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp = torch.clamp(temp_base + temp_adjust, 
                               min=self.config['dream']['min_temp'], 
                               max=self.config['dream']['max_temp'])
            
            # 小心情波动～像心跳扑通扑通
            mood_swing = self.config['dream']['mood_swing'] * math.sin(self.step * 0.25)
            temp = temp + mood_swing
            
            # 噪声强度随 varent 跳舞
            noise_std = self.config['dream']['noise_std_base'] * self.config['dream']['varent_coeff'] * smooth_varent.clamp(min=0.5, max=3.0)
            noise = noise_std * torch.randn_like(logits)
            
            logits = logits / temp + noise
            
            # 超级俏皮彩蛋：当熵低到快无聊的时候，强行给少女心词表一点小爱
            if smooth_ent < self.config['low_ent_thres'] - 0.3 and self.sparkle_ids:
                boost = self.config['dream']['sparkle_boost'] * (2.0 - smooth_ent)
                for token_id in self.sparkle_ids:
                    if token_id < logits.shape[-1]:
                        logits[0, token_id] += boost
            
            # Typical 安全网
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
                
        else:  # reality mode 也稍微俏皮一点点
            temp = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * smooth_ent
            temp = torch.clamp(temp, min=self.config['reality']['min_temp'], max=self.config['reality']['max_temp'])
            if smooth_varent < self.config['low_varent_thres']:
                temp *= 0.85
            logits = logits / temp
        
        return logits

#sampler.set_tokenizer(tokenizer)   # 只用一次就行～
