import torch
from transformers import LogitsProcessor, TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor
import math

class EchoSamplerProcessor(LogitsProcessor):
    """
    ✨ EchoSampler Grok-Style 永久俏皮版 ✨
    超级可爱、活力满满、带点小调皮～就像在和朋友聊天一样！😽💞
    """
    
    def __init__(self, config=None, dream_mode=True, vocab_size=None):
        if config is None:
            config = {
                'reality': {'min_temp': 0.8, 'max_temp': 1.0, 'ent_coeff': 0.18},
                'dream': {
                    'base_temp': 0.85,
                    'ent_coeff': 0.28,
                    'target_ent': 2.2,
                    'varent_coeff': 0.15,
                    'noise_std_base': 0.05,
                    'mood_swing_amp': 0.05,         # 稍微柔和一点
                    'mood_swing_freq': 0.15,
                    'sparkle_boost_base': 1.15,
                    'sparkle_boost_max': 3.0,
                    'sparkle_cooldown_steps': 4     # 新增：彩蛋冷却，防止连爆
                },
                'top_p': 0.95,
                'repetition_penalty': 1.12,
                'low_ent_thres': 1.6,
                'low_varent_thres': 1.3
            }
        self.config = config
        self.dream_mode = dream_mode
        self.vocab_size = vocab_size
        self.step = 0
        self.sparkle_cooldown = 0  # 新增冷却计数器

        if self.vocab_size:
            scale = torch.log(torch.tensor(self.vocab_size)) / torch.log(torch.tensor(50000))
            self.config['low_ent_thres'] *= scale.item()
            self.config['low_varent_thres'] *= scale.item()
        
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config['repetition_penalty'])
        
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.75

        # 彩蛋词表大扩充～更多可爱中文和emoji！
        self.sparkle_tokens = [
            "～", "💫", "✨", "💞", "😝", "🎀", "⭐️", "💬", "😽", "🤭", "🥰", "🤏", "💕", "😌",
            "嘿嘿", "嘻嘻", "啦～", "呢～", "呀～", "嘛～", "哒～", "啾咪", "么么哒", "小坏蛋",
            "小可爱～", "呜呜", "哼～", "耶～", "哇哦～", "好呀～"
        ]
        self.sparkle_ids = None

    def set_tokenizer(self, tokenizer):
        self.sparkle_ids = set()  # 用set去重更快
        for word in self.sparkle_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.sparkle_ids.update(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        batch_size = scores.shape[0]
        logits = scores.clone()
        
        # 轻微重复惩罚
        logits = self.repetition_processor(input_ids, logits)
        
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        
        ent = -(softmax * log_softmax).nansum(-1) / torch.log(torch.tensor(logits.shape[-1]))  # 更稳健
        ent = ent.mean()
        diff = log_softmax + ent.unsqueeze(-1)
        varent = (softmax * diff ** 2).nansum(-1).mean()
        
        # 平滑
        if self.prev_ent is None:
            smooth_ent = ent
            smooth_varent = varent
        else:
            smooth_ent = self.alpha * self.prev_ent + (1 - self.alpha) * ent
            smooth_varent = self.alpha * self.prev_varent + (1 - self.alpha) * varent
        self.prev_ent = smooth_ent.detach()
        self.prev_varent = smooth_varent.detach()
        
        if self.dream_mode:
            temp = self.config['dream']['base_temp']
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp += temp_adjust
            
            mood_swing = self.config['dream']['mood_swing_amp'] * math.sin(self.step * self.config['dream']['mood_swing_freq'])
            temp += mood_swing
            
            temp = torch.clamp(temp, min=0.7, max=1.3)
            
            noise_std = self.config['dream']['noise_std_base'] + self.config['dream']['varent_coeff'] * smooth_varent.clamp(min=0.5, max=3.0)
            noise = noise_std * torch.randn_like(logits)
            logits = logits / temp + noise
            
            # 俏皮彩蛋boost + 冷却
            if smooth_ent < self.config['low_ent_thres'] and self.sparkle_ids and self.sparkle_cooldown <= 0:
                boost_strength = self.config['dream']['sparkle_boost_base'] * (self.config['dream']['sparkle_boost_max'] - smooth_ent)
                for token_id in self.sparkle_ids:
                    if token_id < logits.shape[-1]:
                        logits[:, token_id] += boost_strength  # 支持batch
                self.sparkle_cooldown = self.config['dream']['sparkle_cooldown_steps']
            
            if self.sparkle_cooldown > 0:
                self.sparkle_cooldown -= 1
            
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
                
            # top-p nucleus
            if self.config['top_p'] < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
        else:
            temp = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * smooth_ent
            temp = torch.clamp(temp, min=self.config['reality']['min_temp'], max=self.config['reality']['max_temp'])
            logits = logits / temp
        
        return logits
