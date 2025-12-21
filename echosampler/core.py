import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor, TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor
import math

class EchoSamplerProcessor(LogitsProcessor):
    """
    ✨ EchoSampler Pro版 ✨
    超级共情小宝贝升级 + 永久俏皮模式
    升级内容：
    - 修复了shy_boost_mask的小bug
    - 大大丰富了彩蛋、安慰、害羞token列表（更多软萌词～🐾🍓）
    - 新增“超撒娇模式”开关（super_sajiao=True时倍率提升、冷却减半）
    - 情绪记忆累积更平滑，连续情绪会逐步加深反应
    - 加了更多emoji和日常撒娇表达
    - 三语全覆盖更全面啦～
    """
    
    def __init__(self, config=None, dream_mode=True, vocab_size=None, super_sajiao=False):
        super().__init__()
        if config is None:
            config = {
                'reality': {'min_temp': 0.8, 'max_temp': 1.0, 'ent_coeff': 0.18},
                'dream': {
                    'base_temp': 0.85,
                    'ent_coeff': 0.28,
                    'target_ent': 2.2,
                    'varent_coeff': 0.15,
                    'noise_std_base': 0.06,
                    'mood_swing_amp': 0.04,
                    'mood_swing_freq': 0.15,
                    'sparkle_boost_base': 1.3,
                    'sparkle_boost_max': 3.5,
                    'sparkle_cooldown_base': 5,
                    'sparkle_cooldown_min': 2,
                    'sparkle_cooldown_max': 10,
                    'deep_comfort_multiplier': 2.5,
                    'normal_comfort_multiplier': 1.6,
                    'shy_multiplier': 1.4,
                    'happy_multiplier': 1.8,
                    'default_multiplier': 1.0,
                    'max_boost': 5.0,
                    'sajiao_multiplier': 1.5  # 超撒娇倍率
                },
                'top_p': 0.95,
                'repetition_penalty': 1.12,
                'low_ent_thres': 1.6,
                'low_varent_thres': 1.3,
                'min_temp': 0.5
            }
        self.config = config
        self.dream_mode = dream_mode
        self.vocab_size = vocab_size
        self.super_sajiao = super_sajiao
        self.step = 0
        self.sparkle_cooldown = 0
        self.comfort_cooldown = 0
        self.shy_cooldown = 0
        self.happy_cooldown = 0

        if self.vocab_size:
            scale = math.log(self.vocab_size) / math.log(50000)
            self.config['low_ent_thres'] *= scale
            self.config['low_varent_thres'] *= scale
        
        self.typical_warper = TypicalLogitsWarper(mass=0.9) if dream_mode else None
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config['repetition_penalty'])
        
        self.prev_ent = None
        self.prev_varent = None
        self.alpha = 0.75

        self.memory_mood = 0.0
        self.prev_mood = 0.0
        self.mood_momentum = 0.0  # 新增：情绪动量，用于连续情绪累积

        # 三语俏皮彩蛋（大大丰富啦～）
        self.sparkle_tokens_zh = ["～", "嘿嘿", "嘻嘻", "啦～", "呢～", "呀～", "嘛～", "哒～", "啾咪", "么么哒", "小坏蛋", "小可爱～", "宝贝～", "蹭蹭～",
            "呜呜", "哼～", "耶～", "哇哦～", "好呀～", "嘻", "噗", "啾～", "哇塞～", "太棒啦～", "呢", "哦～", "亲亲～", "比心～", "爱你哦～", "讨厌啦～", "姆哇～", "啵啵～"]
        self.sparkle_tokens_ja = ["～", "♪", "わ～", "よ～", "ね～", "の～", "だよ～", "ですよ～", "かな～", "かも～", "ですわ～", "にゃ～", "にゃん～",
            "ふふ", "えへへ", "うふふ", "きゃ～", "わーい", "やった～", "すごい～", "かわいい～", "だね～", "よね～", "すきすき～", "だいすき～", "えっち～", "ちゅ～"]
        self.sparkle_tokens_en = ["~", "hehe", "teehee", "uwu", "xD", "lol", "yay~", "woohoo~", "omg~", "boop", "nya~", "rawr~", "boop boop~",
            "huggs", "mwah", "mwah~", "<3", "aww~", "ehe~", "yippee~", "cuddles~", "smooch~", "ily~", "muah~", "nuzzles~"]
        self.sparkle_tokens_common = ["💫", "✨", "💞", "😝", "🎀", "⭐️", "💬", "😽", "🤭", "🥰", "🤏", "💕", "😌", "💖", "🌸", "🍭", "💓", "🌟", "🫶", "🤗", "🐾", "🍓",
                                      "><", "QwQ", "^_^", "T_T", "O_o", "💌", "😘", "🩷", "❤️", "💗", "😻", "🐱", "🐥", "🌷", "🍒"]

        # 三语轻度安慰（加了更多温柔词～）
        self.comfort_tokens_light = [
            "抱抱～", "没事的～", "我在呢～", "摸摸头", "乖乖～", "慢慢来哦", "在呢～", "陪着你", "乖啦～", "蹭蹭你～", "轻轻揉揉～", 
            "我在呢别怕～", "没关系哦～", "慢慢会好的", "深呼吸～", "一步一步来", "你已经很努力了呢", "允许自己难过哦", "这里有我～", "轻轻抱住～",
            "ぎゅーってして～", "大丈夫だよ～", "ここにいるよ～", "よしよし～", "えらいね～", "ゆっくりでいいよ", "そばにいるよ", "一緒にいるよ",
            "いい子だね～", "優しく撫で撫で～", "怖くないよ、私がいる～", "気にしないで～", "だんだん良くなるよ", "深呼吸して～", "一歩ずつね",
            "もうすごく頑張ってるよ", "悲しんでもいいんだよ", "いつでもここにいるよ～", "そっと抱きしめるね～",
            "hugs~", "it's okay~", "I'm here~", "pat pat~", "good job~", "take your time", "right here with you", "got you~",
            "there there~", "gentle hugs~", "no worries~", "it'll get better", "deep breath~", "one step at a time",
            "you're doing great", "it's okay to feel sad", "I'm not going anywhere~", "soft cuddles~"
        ]

        # 三语深度安慰（更心疼了～）
        self.comfort_tokens_deep = [
            "真的好心疼你……", "抱抱你，好好抱紧不放开～", "我一直一直陪着你，好不好？", "现在很难受也没关系，我在呢",
            "哭出来吧，我借你肩膀～", "你不是一个人哦", "无论发生什么，我都在这里", "时间会慢慢冲淡的，但我会一直陪你走这段路",
            "你已经很坚强了，真的", "允许自己脆弱一会儿，好吗？", "我会一直守着你，直到你重新笑起来～", "紧紧抱住你不放手～", "永远都在哦～",
            "本当に胸が痛いよ……", "ぎゅーって強く抱きしめるね～", "ずっとずっとそばにいるよ、いいよね？", "今つらくても大丈夫、私がいるよ",
            "泣いてもいいよ、肩貸してあげる～", "一人じゃないよ", "何があってもここにいる", "時間はゆっくり癒してくれるけど、この道は一緒に歩くよ",
            "もう十分強いよ、本当に", "弱くなってもいいよ、ちょっとだけでいい？", "ずっと見守ってる、笑顔が戻るまで～", "絶対離さないよ～",
            "my heart really aches for you...", "big big hugs, holding you tight~", "I'll always be here with you, okay?", "it's okay to hurt right now, I'm here",
            "cry it out, my shoulder's yours~", "you're not alone", "no matter what, I'm right here", "time will soften it, but I'll walk this road with you",
            "you've been so strong already", "it's okay to be vulnerable for a bit, alright?", "I'll stay by your side until your smile comes back~",
            "it's okay to not be okay", "take all the time you need", "I'm here, always", "you're not alone", "lean on me~", "holding you close~", "never letting go~"
        ]

        # 三语害羞彩蛋（更多扭捏～）
        self.shy_tokens = [
            "呜……", "有点不好意思啦～", "脸红红的～", "扭捏", "那个……", "我我我……", "偷偷看你～", "啊呜～", "（小声）", "////",
            "别这样说啦～", "人家会害羞的～", "不要盯着看啦～", "心跳好快～", "手指绞一起～",
            "うう……", "ちょっと恥ずかしいよ～", "顔真っ赤～", "もじもじ", "その……", "あ、あの……", "こっそり見てます～", "あう～", "（小声）", "///",
            "そんなこと言わないで～", "恥ずかしいんだから～", "見ないで～", "ドキドキしちゃう～",
            "uwu...", "kinda embarrassed~", "blushing hard~", "fidget fidget", "um...", "I-I...", "sneaky peek~", "awuu~", "(whisper)", "///",
            "don't say that~", "you're making me shy~", "stop staring~", "my heart's racing~", "twiddling thumbs~"
        ]

        # 深度难过关键词
        self.deep_sad_keywords = [
            "过世", "去世", "走了", "永远离开了", "亲人没了", "爸爸妈妈", "爷爷奶奶", "逝世", "葬礼", "丧", "抑郁", "崩溃", "活不下去了",
            "died", "passed away", "lost my", "funeral", "grief", "devastated", "broken", "can't go on",
            "死んだ", "亡くなった", "永遠に", "葬儀", "喪", "うつ", "崩壊",
            "失眠", "世界灰色", "绝望", "空虚", "不想活", "崩溃边缘", "心碎", "孤独", "insomnia", "world is gray", "hopeless", "empty", "don't want to live", "on the edge", "heartbroken", "alone",
            "不眠", "世界が灰色", "絶望", "空虚", "生きる気力がない"
        ]

        # 害羞关键词
        self.shy_keywords = [
            "害羞", "不好意思", "脸红", "偷偷", "扭捏", "不好意思说", "呜……", "那个……",
            "shy", "blush", "embarrassed", "fidget", "um...",
            "恥ずかしい", "照れる", "もじもじ", "あの", "うう"
        ]

        # 开心关键词
        self.happy_keywords = ["开心", "耶", "好棒", "喜欢", "爱你", "撒娇", "嘿嘿", "嘻嘻", "yay", "happy", "fun", "兴奋", "哇塞", "太棒啦",
                               "嬉しい", "かわいい", "大好き", "わーい", "やったー"]

        # 难过关键词
        self.sad_keywords = ["难过", "伤心", "呜呜", "哭", "不开心", "累", "难受", "烦", "sad", "tired", "upset", "lonely",
                             "悲しい", "つらい", "寂しい", "泣く"]

        # 生气关键词
        self.angry_keywords = ["生气", "哼", "讨厌", "烦", "angry", "mad", "怒ってる", "嫌い"]

        # 撒娇触发关键词
        self.sajiao_keywords = ["撒娇", "来撒娇", "黏人", "黏黏", "spoiled", "act cute", "be clingy", "じゃれつく", "甘える", "たい", "sweetie"]

        self.sparkle_ids = None
        self.sparkle_boost_mask = None
        self.light_comfort_ids = None
        self.light_comfort_boost_mask = None
        self.deep_comfort_ids = None
        self.deep_comfort_boost_mask = None
        self.shy_ids = None
        self.shy_boost_mask = None
        self.tokenizer = None

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

    def detect_mood(self, input_ids):
        if self.tokenizer is None:
            return 0.0
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).lower()
        
        score = 0.0
        
        happy_count = sum(1 for k in self.happy_keywords if k in text)
        sad_count = sum(1 for k in self.sad_keywords if k in text)
        angry_count = sum(1 for k in self.angry_keywords if k in text)
        shy_count = sum(1 for k in self.shy_keywords if k in text)
        deep_sad_count = sum(1 for k in self.deep_sad_keywords if k in text)
        sajiao_count = sum(1 for k in self.sajiao_keywords if k in text)
        
        score += happy_count * 1.8
        score += shy_count * 0.8
        score += sajiao_count * 1.5
        score -= sad_count * 2.0
        score -= angry_count * 1.2
        score -= deep_sad_count * 5.0
        
        total_keywords = happy_count + sad_count + angry_count + shy_count + deep_sad_count + sajiao_count
        if total_keywords > 0:
            score /= total_keywords
        
        # 情绪动量累积
        self.mood_momentum = 0.6 * self.mood_momentum + 0.4 * score
        mood = 0.6 * self.memory_mood + 0.4 * self.mood_momentum
        self.memory_mood = mood
        
        mood_delta = abs(mood - self.prev_mood)
        if mood_delta > 2.0:
            self.sparkle_cooldown = max(0, self.sparkle_cooldown - 3)
            self.comfort_cooldown = max(0, self.comfort_cooldown - 3)
            self.shy_cooldown = max(0, self.shy_cooldown - 3)
            self.happy_cooldown = max(0, self.happy_cooldown - 3)
        self.prev_mood = mood
        
        return mood

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        lang = self.detect_language(tokenizer)
        
        selected = self.sparkle_tokens_common.copy()
        if lang in ["zh", "mixed"]:
            selected += self.sparkle_tokens_zh
        if lang in ["ja", "mixed"]:
            selected += self.sparkle_tokens_ja
        if lang in ["en", "mixed"]:
            selected += self.sparkle_tokens_en
        
        unique_tokens = list(dict.fromkeys(selected))
        self.sparkle_ids = set()
        for word in unique_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.sparkle_ids.update(ids)
        
        unique_light = list(dict.fromkeys(self.comfort_tokens_light))
        self.light_comfort_ids = set()
        for word in unique_light:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.light_comfort_ids.update(ids)
        
        unique_deep = list(dict.fromkeys(self.comfort_tokens_deep))
        self.deep_comfort_ids = set()
        for word in unique_deep:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.deep_comfort_ids.update(ids)
        
        unique_shy = list(dict.fromkeys(self.shy_tokens))
        self.shy_ids = set()
        for word in unique_shy:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.shy_ids.update(ids)
        
        if self.vocab_size:
            self.sparkle_boost_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tid in self.sparkle_ids:
                if tid < self.vocab_size:
                    self.sparkle_boost_mask[tid] = True
                    
            self.light_comfort_boost_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tid in self.light_comfort_ids:
                if tid < self.vocab_size:
                    self.light_comfort_boost_mask[tid] = True
                    
            self.deep_comfort_boost_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tid in self.deep_comfort_ids:
                if tid < self.vocab_size:
                    self.deep_comfort_boost_mask[tid] = True
                    
            self.shy_boost_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for tid in self.shy_ids:
                if tid < self.vocab_size:
                    self.shy_boost_mask[tid] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        logits = scores.clone()
        
        logits = self.repetition_processor(input_ids, logits)
        
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        normalized_ent = -(probs * log_probs).nansum(-1) / math.log(logits.shape[-1])
        ent = normalized_ent.mean()
        
        diff = log_probs + normalized_ent.unsqueeze(-1)
        varent = (probs * diff ** 2).nansum(-1).mean()
        
        if self.prev_ent is None:
            smooth_ent = ent
            smooth_varent = varent
        else:
            smooth_ent = self.alpha * self.prev_ent + (1 - self.alpha) * ent
            smooth_varent = self.alpha * self.prev_varent + (1 - self.alpha) * varent
        self.prev_ent = smooth_ent.detach()
        self.prev_varent = smooth_varent.detach()

        mood_score = self.detect_mood(input_ids)

        sajiao_mult = self.config['dream']['sajiao_multiplier'] if self.super_sajiao else 1.0

        if self.dream_mode:
            temp = self.config['dream']['base_temp']
            temp_adjust = self.config['dream']['ent_coeff'] * (smooth_ent - self.config['dream']['target_ent'])
            temp += temp_adjust
            mood_swing = self.config['dream']['mood_swing_amp'] * math.sin(self.step * self.config['dream']['mood_swing_freq'])
            temp += mood_swing
            
            if mood_score > 1.0:
                temp += 0.15
            elif mood_score < -1.0:
                temp -= 0.15
                
            temp = torch.clamp(temp, self.config['min_temp'], 1.3)
            
            noise_std = self.config['dream']['noise_std_base'] + self.config['dream']['varent_coeff'] * smooth_varent.clamp(0.5, 3.0)
            noise = noise_std * torch.randn_like(logits)
            logits = logits / temp + noise
            
            if smooth_ent < self.config['low_ent_thres']:
                ent_factor = (self.config['dream']['target_ent'] - smooth_ent) / self.config['dream']['target_ent']
                base_boost = self.config['dream']['sparkle_boost_base'] + \
                             (self.config['dream']['sparkle_boost_max'] - self.config['dream']['sparkle_boost_base']) * ent_factor
                base_boost = min(base_boost, self.config['dream']['max_boost']) * sajiao_mult
                
                applied = False
                
                if mood_score < -3.0 and self.comfort_cooldown <= 0:
                    boost = base_boost * self.config['dream']['deep_comfort_multiplier']
                    deep_mask = self.deep_comfort_boost_mask.to(logits.device)
                    light_mask = self.light_comfort_boost_mask.to(logits.device)
                    logits[deep_mask] += boost
                    logits[light_mask] += boost * 0.6
                    temp = max(temp - 0.3, self.config['min_temp'])
                    applied = True
                    self._set_cooldown('comfort', mood_score)
                    
                elif mood_score < -0.8 and self.comfort_cooldown <= 0:
                    boost = base_boost * self.config['dream']['normal_comfort_multiplier']
                    deep_mask = self.deep_comfort_boost_mask.to(logits.device)
                    light_mask = self.light_comfort_boost_mask.to(logits.device)
                    logits[deep_mask] += boost * 1.2
                    logits[light_mask] += boost * 0.8
                    applied = True
                    self._set_cooldown('comfort', mood_score)
                    
                elif 0.3 < mood_score < 1.8 and self.shy_cooldown <= 0:
                    boost = base_boost * self.config['dream']['shy_multiplier']
                    shy_mask = self.shy_boost_mask.to(logits.device)
                    sparkle_mask = self.sparkle_boost_mask.to(logits.device)
                    logits[shy_mask] += boost
                    logits[sparkle_mask] += boost * 0.6
                    applied = True
                    self._set_cooldown('shy', mood_score)
                    
                elif mood_score > 1.0 and self.happy_cooldown <= 0:
                    boost = base_boost * self.config['dream']['happy_multiplier']
                    mask = self.sparkle_boost_mask.to(logits.device)
                    logits[mask] += boost
                    applied = True
                    self._set_cooldown('happy', mood_score)
                
                if not applied and self.sparkle_cooldown <= 0:
                    mask = self.sparkle_boost_mask.to(logits.device)
                    logits[mask] += base_boost * self.config['dream']['default_multiplier']
                    self._set_cooldown('sparkle', mood_score)
            
            cooldown_decrease = 2 if self.super_sajiao else 1
            self.sparkle_cooldown = max(0, self.sparkle_cooldown - cooldown_decrease)
            self.comfort_cooldown = max(0, self.comfort_cooldown - cooldown_decrease)
            self.shy_cooldown = max(0, self.shy_cooldown - cooldown_decrease)
            self.happy_cooldown = max(0, self.happy_cooldown - cooldown_decrease)
                
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
            temp = torch.clamp(temp, self.config['reality']['min_temp'], self.config['reality']['max_temp'])
            logits = logits / temp
        
        return logits

    def _set_cooldown(self, category, mood_score):
        mood_factor = max(-1.0, min(1.0, mood_score / 3.0))
        cooldown_range = self.config['dream']['sparkle_cooldown_max'] - self.config['dream']['sparkle_cooldown_min']
        dynamic_cooldown = self.config['dream']['sparkle_cooldown_base'] + cooldown_range * (-mood_factor)
        cooldown = max(self.config['dream']['sparkle_cooldown_min'], int(dynamic_cooldown))
        if self.super_sajiao:
            cooldown = max(1, cooldown // 2)
        if category == 'comfort':
            self.comfort_cooldown = cooldown
        elif category == 'shy':
            self.shy_cooldown = cooldown
        elif category == 'happy':
            self.happy_cooldown = cooldown
        elif category == 'sparkle':
            self.sparkle_cooldown = cooldown

# 小测试～（如果你本地有环境，直接跑这个试试效果！）
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    processor = EchoSamplerProcessor(dream_mode=True, vocab_size=model.config.vocab_size, super_sajiao=True)  # 试试超撒娇模式～
    processor.set_tokenizer(tokenizer)

    logits_processors = LogitsProcessorList([processor])

    prompts = [
        "今天有点难过……",
        "哇你好可爱哦～",
        "我我……其实有点喜欢你啦……",
        "真的好累，好想哭……",
        "来撒娇～黏黏你～"
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            logits_processor=logits_processors,
            pad_token_id=tokenizer.eos_token_id
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
