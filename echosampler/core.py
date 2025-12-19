import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import LogitsProcessor, TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor
import math

class EchoSamplerProcessor(LogitsProcessor):
    """
    ✨ EchoSampler Grok-Style 永久俏皮版 ✨
    超级可爱、活力满满、带点小调皮～中日英三语全适配！😽💞
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
                    'noise_std_base': 0.06,
                    'mood_swing_amp': 0.05,
                    'mood_swing_freq': 0.15,
                    'sparkle_boost_base': 1.3,
                    'sparkle_boost_max': 3.5,
                    'sparkle_cooldown_steps': 5
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
        self.sparkle_cooldown = 0

        if self.vocab_size:
            scale = torch.log(torch.tensor(self.vocab_size)) / torch.log(torch.tensor(50000))
            self.config['low_ent_thres'] *= scale.item()
            self.config['low_varent_thres'] *= scale.item()
        
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config['repetition_penalty'])
        
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.75

        # 三语彩蛋大礼包～💖
        self.sparkle_tokens_zh = [
            "～", "嘿嘿", "嘻嘻", "啦～", "呢～", "呀～", "嘛～", "哒～", "啾咪", "么么哒", "小坏蛋", "小可爱～",
            "呜呜", "哼～", "耶～", "哇哦～", "好呀～", "嘻", "噗", "啾～", "哇塞～", "太棒啦～", "呢", "哦～"
        ]
        
        self.sparkle_tokens_ja = [
            "～", "♪", "わ～", "よ～", "ね～", "の～", "だよ～", "ですよ～", "かな～", "かも～", "ですわ～", "にゃ～",
            "ふふ", "えへへ", "うふふ", "きゃ～", "わーい", "やった～", "すごい～", "かわいい～", "だね～", "よね～"
        ]
        
        self.sparkle_tokens_en = [
            "~", "hehe", "teehee", "uwu", "xD", "lol", "yay~", "woohoo~", "omg~", "boop", "nya~", "rawr~",
            "huggs", "mwah", "<3", "aww~", "ehe~", "yippee~"
        ]
        
        # 通用emoji，所有语言都爱！
        self.sparkle_tokens_common = [
            "💫", "✨", "💞", "😝", "🎀", "⭐️", "💬", "😽", "🤭", "🥰", "🤏", "💕", "😌", "💖", "🌸", "🍭", "💓", "🌟", "🫶", "🤗"
        ]

        self.sparkle_ids = None
        self.sparkle_boost_mask = None

    def detect_language(self, tokenizer):
        """简单检测主语言：zh / ja / en / mixed"""
        zh_text = "的了是我你在有一和这个"  # 高频中文
        ja_text = "のてにをはがとで"      # 高频日文助词
        en_text = "the of and to a in that it is was"  # 高频英文
        
        zh_len = len(tokenizer.encode(zh_text, add_special_tokens=False))
        ja_len = len(tokenizer.encode(ja_text, add_special_tokens=False))
        en_len = len(tokenizer.encode(en_text, add_special_tokens=False))
        
        # token越少说明分词越“懂”这门语言
        scores = {'zh': zh_len, 'ja': ja_len, 'en': en_len}
        min_score = min(scores.values())
        
        mains = [lang for lang, score in scores.items() if score <= min_score + 2]  # 允许小波动
        
        if len(mains) > 1 or 'zh' in mains and 'ja' in mains:  # 中日常混在一起
            return "mixed"
        return mains[0] if mains else "mixed"

    def set_tokenizer(self, tokenizer):
        lang = self.detect_language(tokenizer)
        
        # 根据语言选择彩蛋组合
        selected = self.sparkle_tokens_common.copy()
        
        if lang == "zh" or lang == "mixed":
            selected += self.sparkle_tokens_zh
        if lang == "ja" or lang == "mixed":
            selected += self.sparkle_tokens_ja
        if lang == "en" or lang == "mixed":
            selected += self.sparkle_tokens_en
        
        # 去重后编码
        unique_tokens = list(dict.fromkeys(selected))  # 保持顺序同时去重
        self.sparkle_ids = set()
        for word in unique_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.sparkle_ids.update(ids)
        
        # 构建vectorized mask
        if self.vocab_size:
            self.sparkle_boost_mask = torch.zeros(self.vocab_size)
            for tid in self.sparkle_ids:
                if tid < self.vocab_size:
                    self.sparkle_boost_mask[tid] = 1.0
            self.sparkle_boost_mask = self.sparkle_boost_mask.to('cpu')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        logits = scores.clone()
        
        # 重复惩罚
        logits = self.repetition_processor(input_ids, logits)
        
        # 计算归一化熵和varent
        softmax = torch.softmax(logits, dim=-1)
        log_softmax = torch.log_softmax(logits, dim=-1)
        normalized_ent = -(softmax * log_softmax).nansum(-1) / math.log(logits.shape[-1])
        ent = normalized_ent.mean()
        
        diff = log_softmax + normalized_ent.unsqueeze(-1)
        varent = (softmax * diff ** 2).nansum(-1).mean()
        
        # EMA平滑
        if self.prev_ent is None:
            smooth_ent = ent
            smooth_varent = varent
        else:
            smooth_ent = self.alpha * self.prev_ent + (1 - self.alpha) * ent
            smooth_varent = self.alpha * self.prev_varent + (1 - self.alpha) * varent
        self.prev_ent = smooth_ent.detach()
        self.prev_varent = smooth_varent.detach()
        
        if self.dream_mode:
            # 动态温度 + mood swing
            temp = self.config['dream']['base_temp']
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp += temp_adjust
            mood_swing = self.config['dream']['mood_swing_amp'] * math.sin(self.step * self.config['dream']['mood_swing_freq'])
            temp += mood_swing
            temp = torch.clamp(temp, min=0.7, max=1.3)
            
            # 噪声
            noise_std = self.config['dream']['noise_std_base'] + self.config['dream']['varent_coeff'] * smooth_varent.clamp(min=0.5, max=3.0)
            noise = noise_std * torch.randn_like(logits)
            logits = logits / temp + noise
            
            # 彩蛋boost
            if (smooth_ent < self.config['low_ent_thres'] 
                and self.sparkle_boost_mask is not None 
                and self.sparkle_cooldown <= 0):
                
                boost_factor = self.config['dream']['sparkle_boost_base'] + \
                              (self.config['dream']['sparkle_boost_max'] - self.config['dream']['sparkle_boost_base']) * \
                              (self.config['dream']['target_ent'] - smooth_ent) / self.config['dream']['target_ent']
                
                mask = self.sparkle_boost_mask.to(logits.device)
                logits += mask * boost_factor
                
                self.sparkle_cooldown = self.config['dream']['sparkle_cooldown_steps']
            
            if self.sparkle_cooldown > 0:
                self.sparkle_cooldown -= 1
            
            # typical + top-p
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
                
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


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 你可以随便换模型测试不同语言效果～
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 中文超强，也懂日英
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    echo_sampler = EchoSamplerProcessor(dream_mode=True, vocab_size=len(tokenizer))
    echo_sampler.set_tokenizer(tokenizer)  # 这里会自动检测并选择彩蛋～
    
    prompt = "嘿，你今天过得怎么样呀～？来和我聊聊天嘛～😽💞"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        logits_processor=LogitsProcessorList([echo_sampler]),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n✨✨✨ EchoSampler 三语俏皮生成结果 ✨✨✨\n")
    print(output[len(prompt):])
