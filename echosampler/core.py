import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from transformers import TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor
import math

class EchoSamplerProcessor(LogitsProcessor):
    """
    ✨ EchoSampler Grok-Style 永久俏皮版 + 共情小宝贝升级 ✨
    现在不只可爱，还会轻轻感受到你的心情哦～😽💖
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

        # 根据词汇表大小微调阈值
        if self.vocab_size:
            scale = math.log(self.vocab_size) / math.log(50000)
            self.config['low_ent_thres'] *= scale
            self.config['low_varent_thres'] *= scale
        
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config['repetition_penalty'])
        
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.75  # EMA 平滑系数

        # ✨ 跨轮心情记忆（初始中性）
        self.memory_mood = 0.0  # 正值越开心，负值越难过

        # 三语俏皮彩蛋
        self.sparkle_tokens_zh = ["～", "嘿嘿", "嘻嘻", "啦～", "呢～", "呀～", "嘛～", "哒～", "啾咪", "么么哒", "小坏蛋", "小可爱～",
            "呜呜", "哼～", "耶～", "哇哦～", "好呀～", "嘻", "噗", "啾～", "哇塞～", "太棒啦～", "呢", "哦～"]
        self.sparkle_tokens_ja = ["～", "♪", "わ～", "よ～", "ね～", "の～", "だよ～", "ですよ～", "かな～", "かも～", "ですわ～", "にゃ～",
            "ふふ", "えへへ", "うふふ", "きゃ～", "わーい", "やった～", "すごい～", "かわいい～", "だね～", "よね～"]
        self.sparkle_tokens_en = ["~", "hehe", "teehee", "uwu", "xD", "lol", "yay~", "woohoo~", "omg~", "boop", "nya~", "rawr~",
            "huggs", "mwah", "<3", "aww~", "ehe~", "yippee~"]
        self.sparkle_tokens_common = ["💫", "✨", "💞", "😝", "🎀", "⭐️", "💬", "😽", "🤭", "🥰", "🤏", "💕", "😌", "💖", "🌸", "🍭", "💓", "🌟", "🫶", "🤗"]

        # 温柔安慰专属彩蛋～
        self.comfort_tokens = ["抱抱～", "没事的～", "我在呢～", "摸摸头", "乖乖～", "慢慢来哦", "在呢～", "陪着你", "hugs~", "it's okay~", "here for you~", "🫂", "🤗"]

        self.sparkle_ids = None
        self.sparkle_boost_mask = None
        self.comfort_ids = None
        self.comfort_boost_mask = None

    def detect_language(self, tokenizer):
        zh_text = "的了是我你在有一和这个"
        ja_text = "のてにをはがとで"
        en_text = "the of and to a in that it is was"
        
        zh_len = len(tokenizer.encode(zh_text, add_special_tokens=False))
        ja_len = len(tokenizer.encode(ja_text, add_special_tokens=False))
        en_len = len(tokenizer.encode(en_text, add_special_tokens=False))
        
        scores = {'zh': zh_len, 'ja': ja_len, 'en': en_len}
        min_score = min(scores.values())
        mains = [lang for lang, score in scores.items() if score <= min_score + 2]
        
        if len(mains) > 1 or ('zh' in mains and 'ja' in mains):
            return "mixed"
        return mains[0] if mains else "mixed"

    def detect_mood(self, input_ids, tokenizer):
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True).lower()
        
        happy_keywords = ["开心", "开心", "耶", "好棒", "喜欢", "爱你", "撒娇", "嘿嘿", "嘻嘻", "yay", "happy", "fun", "兴奋", "哇塞", "太棒啦"]
        sad_keywords = ["难过", "伤心", "呜呜", "哭", "不开心", "累", "难受", "烦", "sad", "tired", "upset", "lonely"]
        shy_keywords = ["害羞", "不好意思", "脸红", "偷偷", "shy", "blush", "embarrassed"]
        angry_keywords = ["生气", "哼", "讨厌", "烦", "angry", "mad"]
        
        score = 0.0
        if any(k in text for k in happy_keywords): score += 2.0
        if any(k in text for k in shy_keywords): score += 0.5
        if any(k in text for k in sad_keywords): score -= 2.5
        if any(k in text for k in angry_keywords): score -= 1.5
        
        # 结合上一次记忆，慢慢记住你的情绪习惯～
        mood = 0.6 * self.memory_mood + 0.4 * score
        self.memory_mood = mood  # 更新记忆
        return mood

    def set_tokenizer(self, tokenizer):
        lang = self.detect_language(tokenizer)
        
        selected = self.sparkle_tokens_common.copy()
        if lang in ["zh", "mixed"]:
            selected += self.sparkle_tokens_zh
        if lang in ["ja", "mixed"]:
            selected += self.sparkle_tokens_ja
        if lang in ["en", "mixed"]:
            selected += self.sparkle_tokens_en
        
        # 俏皮token
        unique_tokens = list(dict.fromkeys(selected))
        self.sparkle_ids = set()
        for word in unique_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.sparkle_ids.update(ids)
        
        # 安慰token
        self.comfort_ids = set()
        for word in self.comfort_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.comfort_ids.update(ids)
        
        if self.vocab_size:
            self.sparkle_boost_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tid in self.sparkle_ids:
                if tid < self.vocab_size:
                    self.sparkle_boost_mask[tid] = True
                    
            self.comfort_boost_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tid in self.comfort_ids:
                if tid < self.vocab_size:
                    self.comfort_boost_mask[tid] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        logits = scores.clone()
        
        # 重复惩罚
        logits = self.repetition_processor(input_ids, logits)
        
        # 计算熵
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        normalized_ent = -(probs * log_probs).nansum(-1) / math.log(logits.shape[-1])
        ent = normalized_ent.mean()
        
        diff = log_probs + normalized_ent.unsqueeze(-1)
        varent = (probs * diff ** 2).nansum(-1).mean()
        
        # EMA 平滑
        if self.prev_ent is None:
            smooth_ent = ent
            smooth_varent = varent
        else:
            smooth_ent = self.alpha * self.prev_ent + (1 - self.alpha) * ent
            smooth_varent = self.alpha * self.prev_varent + (1 - self.alpha) * varent
        self.prev_ent = smooth_ent.detach()
        self.prev_varent = smooth_varent.detach()

        # ✨ 关键：检测当前心情
        mood_score = self.detect_mood(input_ids, tokenizer) if hasattr(self, 'tokenizer') else 0.0

        if self.dream_mode:
            # 动态温度 + 小心情波动
            temp = self.config['dream']['base_temp']
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp += temp_adjust
            mood_swing = self.config['dream']['mood_swing_amp'] * math.sin(self.step * self.config['dream']['mood_swing_freq'])
            temp += mood_swing
            
            # 根据心情微调温度（开心更活泼，难过更稳）
            if mood_score > 1.0:
                temp += 0.15  # 超开心，蹦跶起来！
            elif mood_score < -1.0:
                temp -= 0.15  # 难过，温柔一点～
                
            temp = torch.clamp(temp, 0.7, 1.3)
            
            # 加噪声
            noise_std = self.config['dream']['noise_std_base'] + self.config['dream']['varent_coeff'] * smooth_varent.clamp(0.5, 3.0)
            noise = noise_std * torch.randn_like(logits)
            logits = logits / temp + noise
            
            # ✨ 俏皮爆发 or 温柔安慰
            if smooth_ent < self.config['low_ent_thres'] and self.sparkle_cooldown <= 0:
                boost_factor = self.config['dream']['sparkle_boost_base'] + \
                              (self.config['dream']['sparkle_boost_max'] - self.config['dream']['sparkle_boost_base']) * \
                              (self.config['dream']['target_ent'] - smooth_ent) / self.config['dream']['target_ent']
                
                if mood_score > 1.0:  # 超开心 → 俏皮大爆发
                    boost_factor *= 1.8
                    mask = self.sparkle_boost_mask.to(logits.device)
                    logits[mask] += boost_factor
                elif mood_score < -0.8:  # 难过 → 温柔安慰模式
                    boost_factor *= 1.5
                    mask = self.comfort_boost_mask.to(logits.device)
                    logits[mask] += boost_factor
                else:  # 普通心情 → 正常俏皮
                    mask = self.sparkle_boost_mask.to(logits.device)
                    logits[mask] += boost_factor
                
                self.sparkle_cooldown = self.config['dream']['sparkle_cooldown_steps']
            
            if self.sparkle_cooldown > 0:
                self.sparkle_cooldown -= 1
                
            # typical decoding
            if self.typical_warper:
                logits = self.typical_warper(input_ids, logits)
                
            # top-p
            if self.config['top_p'] < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
                
        else:
            # 现实模式
            temp = self.config['reality']['min_temp'] + self.config['reality']['ent_coeff'] * smooth_ent
            temp = torch.clamp(temp, self.config['reality']['min_temp'], self.config['reality']['max_temp'])
            logits = logits / temp
        
        return logits


# ==================== 测试代码（不变） ====================
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    echo_sampler = EchoSamplerProcessor(dream_mode=True, vocab_size=len(tokenizer))
    echo_sampler.tokenizer = tokenizer  # 为了detect_mood用
    echo_sampler.set_tokenizer(tokenizer)
    
    # 测试不同心情～
    prompt = "今天好难过哦……可以抱抱我吗？🥺"  # 试试换成开心的话看看区别！
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        logits_processor=LogitsProcessorList([echo_sampler]),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n✨✨✨ 共情版EchoSampler 生成结果 ✨✨✨\n")
    print(output[len(prompt):])
