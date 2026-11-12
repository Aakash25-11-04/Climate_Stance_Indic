"""Template-based synthetic comment generator (no API, no LLM).

Hand-written Hindi and Bengali templates per label x style, slot-filled with climate
vocabulary, negators, intensifiers, emojis, and then optionally Romanized / code-mixed.
Everything is driven by a seeded RNG, so a given config always yields the same data.

Styles: formal, slang, sarcastic, implicit, negated, romanized, codemixed, emoji, short, long.
Each generated row carries source_type == "synthetic" and a `style` column.

Honesty note: these are synthetic. They are capped at a share of the training data
(configs/dataset.yaml: synthetic_max_share) and their effect is ablated in Phase 6.
"""
from __future__ import annotations

import random
import re

import pandas as pd

from stance.data.codemix import codemix, romanize_full

# ----------------------------------------------------------------------------- slots
SLOTS = {
    "hi": {
        "TOPIC": ["जलवायु परिवर्तन", "ग्लोबल वार्मिंग", "climate change", "बढ़ता तापमान",
                  "मौसम में बदलाव", "धरती का गर्म होना", "global warming"],
        "INT": ["बहुत", "सच में", "वाकई", "एकदम", "बेहद", "काफी", ""],
        "FAKE": ["झूठ", "साजिश", "स्कैम", "प्रोपेगैंडा", "पश्चिम का एजेंडा", "ड्रामा", "अफवाह",
                 "नाटक", "धोखा"],
        "WHO": ["वैज्ञानिक", "नेता", "मीडिया", "NGO वाले", "अमीर देश", "विदेशी कंपनियां",
                "सरकार", "एक्सपर्ट"],
        "CITY": ["दिल्ली", "मुंबई", "पटना", "लखनऊ", "जयपुर", "भोपाल", "कोलकाता", "रांची"],
        "MONTH": ["मार्च", "अप्रैल", "अक्टूबर", "नवंबर", "फरवरी"],
        "DISASTER": ["बाढ़", "सूखा", "हीटवेव", "चक्रवात", "बेमौसम बारिश", "ग्लेशियर पिघलना"],
        "ACTION": ["पेड़ लगाने", "प्लास्टिक छोड़ने", "कोयला बंद करने", "सोलर अपनाने",
                   "पेट्रोल कम जलाने", "सरकार पर दबाव डालने"],
        "WEATHER": ["हल्की बारिश होगी", "तापमान 32 डिग्री रहेगा", "आसमान साफ रहेगा",
                    "कोहरा छाया रहेगा", "तेज हवाएं चलेंगी"],
        "OFFTOPIC": ["आज का मैच कौन जीतेगा?", "नई फिल्म का ट्रेलर देखा क्या?",
                     "पेट्रोल के दाम फिर बढ़ गए", "भाई ये वीडियो किस कैमरे से शूट किया?",
                     "सब्सक्राइब कर दिया, अच्छा चैनल है", "चुनाव में कौन जीतेगा इस बार?",
                     "कल ऑफिस की छुट्टी है क्या?", "ये गाना बहुत अच्छा है यार"],
        "YEAR": ["2030", "2040", "2050", "2070"],
    },
    "bn": {
        "TOPIC": ["জলবায়ু পরিবর্তন", "বৈশ্বিক উষ্ণায়ন", "climate change", "তাপমাত্রা বৃদ্ধি",
                  "আবহাওয়ার পরিবর্তন", "পৃথিবীর উষ্ণ হওয়া", "global warming"],
        "INT": ["খুব", "সত্যিই", "আসলেই", "একদম", "ভীষণ", "বেশ", ""],
        "FAKE": ["মিথ্যা", "ষড়যন্ত্র", "স্ক্যাম", "প্রোপাগান্ডা", "পশ্চিমাদের এজেন্ডা", "নাটক",
                 "গুজব", "ধোঁকা", "ভুয়া গল্প"],
        "WHO": ["বিজ্ঞানীরা", "নেতারা", "মিডিয়া", "NGO-র লোকেরা", "ধনী দেশগুলো",
                "বিদেশি কোম্পানিগুলো", "সরকার", "এক্সপার্টরা"],
        "CITY": ["ঢাকা", "চট্টগ্রাম", "কলকাতা", "সিলেট", "খুলনা", "রাজশাহী", "বরিশাল", "রংপুর"],
        "MONTH": ["মার্চ", "এপ্রিল", "অক্টোবর", "নভেম্বর", "ফেব্রুয়ারি"],
        "DISASTER": ["বন্যা", "খরা", "তাপপ্রবাহ", "ঘূর্ণিঝড়", "অসময়ের বৃষ্টি", "নদীভাঙন"],
        "ACTION": ["গাছ লাগানো", "প্লাস্টিক ছাড়া", "কয়লা বন্ধ করা", "সোলার ব্যবহার",
                   "কম গাড়ি চালানো", "সরকারকে চাপ দেওয়া"],
        "WEATHER": ["হালকা বৃষ্টি হবে", "তাপমাত্রা ৩২ ডিগ্রি থাকবে", "আকাশ পরিষ্কার থাকবে",
                    "কুয়াশা থাকবে", "ঝড়ো হাওয়া বইবে"],
        "OFFTOPIC": ["আজকের ম্যাচে কে জিতবে?", "নতুন সিনেমার ট্রেলার দেখেছো?",
                     "পেট্রোলের দাম আবার বাড়ল", "ভাই এই ভিডিও কোন ক্যামেরায় শুট করা?",
                     "সাবস্ক্রাইব করলাম, ভালো চ্যানেল", "এবার নির্বাচনে কে জিতবে?",
                     "কাল কি অফিস ছুটি?", "এই গানটা দারুণ লাগল"],
        "YEAR": ["২০৩০", "২০৪০", "২০৫০", "2050"],
    },
}

EMOJI = {
    "Favour": ["🌍", "🌱", "🙏", "😢", "🔥", "💯", "⚠️", "🌡️", "😰", "🌳"],
    "Against": ["😂", "🤣", "🙄", "🤡", "👎", "😏", "💰", "🐑", "🤷", "😴"],
    "None": ["🤔", "📰", "📺", "🙂", "👍", "❓", "🏏", "🎬", "☁️", "📅"],
}

# ----------------------------------------------------------------------------- templates
# Each entry: (style, template). Slots in {BRACES}. "{NOT}" is filled by the negated style.
TEMPLATES = {
    "hi": {
        "Favour": [
            ("formal", "{TOPIC} {INT} गंभीर समस्या है, हमें अभी से {ACTION} पर ध्यान देना चाहिए।"),
            ("formal", "{DISASTER} की बढ़ती घटनाएं साफ दिखाती हैं कि {TOPIC} असली है और खतरनाक है।"),
            ("formal", "{WHO} सालों से चेतावनी दे रहे हैं, {TOPIC} को नज़रअंदाज़ करना आत्मघाती है।"),
            ("formal", "अगर {TOPIC} पर अभी कदम नहीं उठाए तो {YEAR} तक हालात काबू से बाहर हो जाएंगे।"),
            ("slang", "भाई {TOPIC} कोई मज़ाक नहीं है, {CITY} में {MONTH} में ही 42 डिग्री हो रहा है"),
            ("slang", "यार सच में {TOPIC} से डर लगता है, हमारे बच्चों का क्या होगा"),
            ("slang", "जो लोग {TOPIC} को {FAKE} बोलते हैं वो खुद AC में बैठे हैं, बाहर निकल कर देखो"),
            ("sarcastic", "वाह, {TOPIC} {FAKE} है और {CITY} में {MONTH} की गर्मी भी {FAKE} है, है ना 🙄"),
            ("sarcastic", "हाँ भाई, {DISASTER} तो बस संयोग है, {TOPIC} से कोई लेना देना नहीं... बिलकुल"),
            ("implicit", "इस साल {CITY} में {MONTH} में ही पंखे चलने लगे, पहले ऐसा कभी नहीं होता था"),
            ("implicit", "हर साल {DISASTER} बढ़ रही है, कुछ तो गड़बड़ है इस धरती के साथ"),
            ("implicit", "हमारे गांव की नदी सूख गई है और {MONTH} में लू चल रही है, अब तो जागो"),
            ("negated", "ये कहना {INT} गलत है कि {TOPIC} {FAKE} है, सबूत सामने हैं।"),
            ("negated", "{TOPIC} को नकारने का कोई मतलब नहीं, {DISASTER} हमारे सामने है।"),
            ("negated", "मैं नहीं मानता कि {TOPIC} {FAKE} है, {WHO} सही कह रहे हैं"),
            ("short", "{TOPIC} असली है, जागो!"),
            ("short", "{ACTION} ही एकमात्र रास्ता है"),
            ("short", "{TOPIC} {INT} गंभीर है"),
            ("long", "मैंने पिछले दस साल में {CITY} का मौसम बदलते देखा है, {MONTH} में {DISASTER} और गर्मी असहनीय हो गई है, {TOPIC} को {FAKE} कहने वाले {WHO} से पूछो कि उनके बच्चे किस दुनिया में रहेंगे, हमें अभी {ACTION} की ज़रूरत है।"),
        ],
        "Against": [
            ("formal", "{TOPIC} {WHO} द्वारा फैलाया गया {FAKE} है, इसका कोई ठोस सबूत नहीं है।"),
            ("formal", "मौसम हमेशा बदलता रहा है, {TOPIC} को गंभीर समस्या बताना अतिशयोक्ति है।"),
            ("formal", "{TOPIC} के नाम पर {WHO} सिर्फ पैसा कमाना चाहते हैं, यह कोई असली खतरा नहीं है।"),
            ("formal", "{YEAR} तक दुनिया खत्म होने की बातें पहले भी हुई हैं, {TOPIC} भी वैसा ही डर है।"),
            ("slang", "भाई {TOPIC} सब {FAKE} है, {WHO} का धंधा है बस"),
            ("slang", "यार हर साल गर्मी सर्दी आती है, इसमें {TOPIC} कहां से आ गया 😂"),
            ("slang", "ये {TOPIC} वाले बस पैसा बटोरने आए हैं, कुछ नहीं होने वाला"),
            ("sarcastic", "हाँ हाँ, {TOPIC} से {YEAR} में दुनिया खत्म हो जाएगी, जैसे 2012 में हुई थी 🙄"),
            ("sarcastic", "वाह, थोड़ी बारिश हुई और {WHO} ने {TOPIC} बोल दिया, क्या साइंस है"),
            ("sarcastic", "{TOPIC} की चिंता करो, और {WHO} प्राइवेट जेट में घूमें, बढ़िया लॉजिक"),
            ("implicit", "पहले भी {DISASTER} आती थी, तब कोई {FAKE} नहीं फैलाता था"),
            ("implicit", "{CITY} में इस बार {MONTH} में ठंड पड़ रही है, कहां है गर्मी?"),
            ("implicit", "सूरज के चक्र से मौसम बदलता है, इंसान का इसमें कोई हाथ नहीं"),
            ("negated", "मुझे {INT} नहीं लगता कि {TOPIC} कोई असली खतरा है, सब {FAKE} है।"),
            ("negated", "{TOPIC} से कुछ नहीं होने वाला, {WHO} बेवजह डरा रहे हैं"),
            ("negated", "कोई {DISASTER} {TOPIC} की वजह से नहीं आती, ये सब {FAKE} है"),
            ("short", "{TOPIC} = {FAKE}"),
            ("short", "सब {FAKE} है भाई"),
            ("short", "{TOPIC} कोई खतरा नहीं"),
            ("long", "मैं {CITY} में पैदा हुआ हूं और यहां हमेशा {MONTH} में गर्मी पड़ती थी, {DISASTER} भी हर दस साल में आती थी, अब {WHO} इसे {TOPIC} बोलकर {FAKE} फैला रहे हैं ताकि हमसे टैक्स वसूल सकें, मैं इस डर में नहीं आने वाला।"),
        ],
        "None": [
            ("formal", "{TOPIC} पर {CITY} में {MONTH} को एक सेमिनार आयोजित किया जाएगा।"),
            ("formal", "मौसम विभाग के अनुसार कल {CITY} में {WEATHER}।"),
            ("formal", "{WHO} ने {TOPIC} पर नई रिपोर्ट जारी की है, लिंक नीचे है।"),
            ("formal", "{TOPIC} का मतलब क्या होता है? कोई समझा सकता है?"),
            ("slang", "भाई {TOPIC} पर कोई अच्छा वीडियो हो तो बताओ"),
            ("slang", "{OFFTOPIC}"),
            ("slang", "{OFFTOPIC} वैसे वीडियो अच्छा था"),
            ("implicit", "{CITY} में आज {WEATHER}, छाता लेकर निकलना"),
            ("implicit", "{TOPIC} पर स्कूल में निबंध लिखना है, कोई पॉइंट्स बताओ"),
            ("negated", "मुझे {TOPIC} के बारे में ज्यादा नहीं पता, कोई समझाए"),
            ("negated", "इस वीडियो में {TOPIC} के बारे में कुछ नहीं बताया, बस {OFFTOPIC}"),
            ("short", "{OFFTOPIC}"),
            ("short", "{TOPIC} पर वीडियो कब आएगा?"),
            ("short", "कल {CITY} में {WEATHER}"),
            ("long", "मैं {CITY} से हूं और यहां {MONTH} में {WEATHER}, वैसे इस चैनल के वीडियो अच्छे होते हैं, {TOPIC} वाला वीडियो भी देखूंगा, {OFFTOPIC}"),
        ],
    },
    "bn": {
        "Favour": [
            ("formal", "{TOPIC} {INT} গুরুতর সমস্যা, আমাদের এখন থেকেই {ACTION}র দিকে মনোযোগ দিতে হবে।"),
            ("formal", "{DISASTER}র ক্রমবর্ধমান ঘটনা স্পষ্ট দেখায় যে {TOPIC} আসল এবং বিপজ্জনক।"),
            ("formal", "{WHO} বছরের পর বছর সতর্ক করছেন, {TOPIC}কে উপেক্ষা করা আত্মঘাতী।"),
            ("formal", "{TOPIC} নিয়ে এখনই পদক্ষেপ না নিলে {YEAR} সালের মধ্যে পরিস্থিতি নিয়ন্ত্রণের বাইরে চলে যাবে।"),
            ("slang", "ভাই {TOPIC} কোনো মজা না, {CITY}তে {MONTH} মাসেই ৪২ ডিগ্রি হচ্ছে"),
            ("slang", "সত্যি বলতে {TOPIC} নিয়ে ভয় লাগে, আমাদের বাচ্চাদের কী হবে"),
            ("slang", "যারা {TOPIC}কে {FAKE} বলে তারা নিজেরা AC-তে বসে আছে, বাইরে বের হয়ে দেখ"),
            ("sarcastic", "বাহ, {TOPIC} {FAKE} আর {CITY}র {MONTH} মাসের গরমও {FAKE}, তাই না 🙄"),
            ("sarcastic", "হ্যাঁ ভাই, {DISASTER} তো শুধু কাকতালীয়, {TOPIC}র সাথে কোনো সম্পর্ক নেই... একদম"),
            ("implicit", "এই বছর {CITY}তে {MONTH} মাসেই ফ্যান চালাতে হচ্ছে, আগে কখনো এমন হতো না"),
            ("implicit", "প্রতি বছর {DISASTER} বাড়ছে, এই পৃথিবীর সাথে কিছু একটা গড়বড় হচ্ছে"),
            ("implicit", "আমাদের গ্রামের নদী শুকিয়ে গেছে আর {MONTH} মাসে লু বইছে, এবার জাগো"),
            ("negated", "এটা বলা {INT} ভুল যে {TOPIC} {FAKE}, প্রমাণ সামনেই আছে।"),
            ("negated", "{TOPIC}কে অস্বীকার করার কোনো মানে নেই, {DISASTER} আমাদের চোখের সামনে।"),
            ("negated", "আমি মানি না যে {TOPIC} {FAKE}, {WHO} ঠিকই বলছেন"),
            ("short", "{TOPIC} আসল, জাগো!"),
            ("short", "{ACTION}ই একমাত্র পথ"),
            ("short", "{TOPIC} {INT} গুরুতর"),
            ("long", "আমি গত দশ বছরে {CITY}র আবহাওয়া বদলাতে দেখেছি, {MONTH} মাসে {DISASTER} আর গরম অসহ্য হয়ে গেছে, যারা {TOPIC}কে {FAKE} বলে সেই {WHO}কে জিজ্ঞেস করো তাদের বাচ্চারা কোন পৃথিবীতে থাকবে, আমাদের এখনই {ACTION} দরকার।"),
        ],
        "Against": [
            ("formal", "{TOPIC} হলো {WHO}র ছড়ানো {FAKE}, এর কোনো শক্ত প্রমাণ নেই।"),
            ("formal", "আবহাওয়া সবসময়ই বদলেছে, {TOPIC}কে গুরুতর সমস্যা বলা অতিরঞ্জন।"),
            ("formal", "{TOPIC}র নামে {WHO} শুধু টাকা কামাতে চায়, এটা কোনো আসল বিপদ না।"),
            ("formal", "{YEAR} সালে দুনিয়া শেষ হওয়ার কথা আগেও বলা হয়েছে, {TOPIC}ও সেরকম ভয়।"),
            ("slang", "ভাই {TOPIC} পুরোটাই {FAKE}, {WHO}র ব্যবসা"),
            ("slang", "প্রতি বছর গরম শীত আসে, এর মধ্যে {TOPIC} কোথা থেকে এলো 😂"),
            ("slang", "এই {TOPIC}ওয়ালারা শুধু টাকা তুলতে এসেছে, কিছুই হবে না"),
            ("sarcastic", "হ্যাঁ হ্যাঁ, {TOPIC}র জন্য {YEAR} সালে দুনিয়া শেষ হয়ে যাবে, যেমন ২০১২ সালে হয়েছিল 🙄"),
            ("sarcastic", "বাহ, একটু বৃষ্টি হলো আর {WHO} বলে দিল {TOPIC}, কী সায়েন্স"),
            ("sarcastic", "{TOPIC} নিয়ে চিন্তা করো, আর {WHO} প্রাইভেট জেটে ঘুরুক, দারুণ লজিক"),
            ("implicit", "আগেও {DISASTER} হতো, তখন কেউ {FAKE} ছড়াতো না"),
            ("implicit", "{CITY}তে এবার {MONTH} মাসে ঠান্ডা পড়ছে, কোথায় গরম?"),
            ("implicit", "সূর্যের চক্রে আবহাওয়া বদলায়, মানুষের এতে কোনো হাত নেই"),
            ("negated", "আমার {INT} মনে হয় না {TOPIC} কোনো আসল বিপদ, সবই {FAKE}।"),
            ("negated", "{TOPIC}তে কিছুই হবে না, {WHO} অযথা ভয় দেখাচ্ছে"),
            ("negated", "কোনো {DISASTER} {TOPIC}র কারণে হয় না, এসব {FAKE}"),
            ("short", "{TOPIC} = {FAKE}"),
            ("short", "সব {FAKE} ভাই"),
            ("short", "{TOPIC} কোনো বিপদ না"),
            ("long", "আমি {CITY}তে জন্মেছি আর এখানে সবসময় {MONTH} মাসে গরম পড়ত, {DISASTER}ও প্রতি দশ বছরে হতো, এখন {WHO} এটাকে {TOPIC} বলে {FAKE} ছড়াচ্ছে যাতে আমাদের থেকে ট্যাক্স নিতে পারে, আমি এই ভয়ে পড়ছি না।"),
        ],
        "None": [
            ("formal", "{TOPIC} নিয়ে {CITY}তে {MONTH} মাসে একটি সেমিনার আয়োজন করা হবে।"),
            ("formal", "আবহাওয়া অধিদপ্তরের মতে কাল {CITY}তে {WEATHER}।"),
            ("formal", "{WHO} {TOPIC} নিয়ে নতুন রিপোর্ট প্রকাশ করেছে, লিংক নিচে।"),
            ("formal", "{TOPIC} মানে কী? কেউ বুঝিয়ে বলতে পারবেন?"),
            ("slang", "ভাই {TOPIC} নিয়ে ভালো কোনো ভিডিও থাকলে বলো"),
            ("slang", "{OFFTOPIC}"),
            ("slang", "{OFFTOPIC} যাই হোক ভিডিওটা ভালো ছিল"),
            ("implicit", "{CITY}তে আজ {WEATHER}, ছাতা নিয়ে বের হবেন"),
            ("implicit", "{TOPIC} নিয়ে স্কুলে রচনা লিখতে হবে, কেউ পয়েন্ট দাও"),
            ("negated", "আমি {TOPIC} সম্পর্কে বেশি জানি না, কেউ বুঝিয়ে দিন"),
            ("negated", "এই ভিডিওতে {TOPIC} নিয়ে কিছুই বলা হয়নি, শুধু {OFFTOPIC}"),
            ("short", "{OFFTOPIC}"),
            ("short", "{TOPIC} নিয়ে ভিডিও কবে আসবে?"),
            ("short", "কাল {CITY}তে {WEATHER}"),
            ("long", "আমি {CITY} থেকে, এখানে {MONTH} মাসে {WEATHER}, যাই হোক এই চ্যানেলের ভিডিওগুলো ভালো হয়, {TOPIC}র ভিডিওটাও দেখব, {OFFTOPIC}"),
        ],
    },
}

SURFACE_STYLES = ("romanized", "codemixed", "emoji")
_SLOT_RE = re.compile(r"\{([A-Z_]+)\}")


def _fill(template: str, lang: str, rng: random.Random) -> str:
    def rep(m):
        return rng.choice(SLOTS[lang][m.group(1)])

    s = _SLOT_RE.sub(rep, template)
    return re.sub(r"\s{2,}", " ", s).strip()


def generate(lang: str, label: str, n: int, rng: random.Random,
             styles: tuple[str, ...] | None = None) -> list[dict]:
    """Generate `n` synthetic comments. Surface styles (romanized / codemixed / emoji) are
    applied on top of a randomly chosen base template."""
    base = TEMPLATES[lang][label]
    all_styles = sorted({s for s, _ in base}) + list(SURFACE_STYLES)
    styles = tuple(styles or all_styles)
    rows = []
    for i in range(n):
        style = styles[i % len(styles)]
        if style in SURFACE_STYLES:
            bstyle, tpl = rng.choice(base)
            text = _fill(tpl, lang, rng)
            if style == "romanized":
                text = romanize_full(text, lang, rng)
            elif style == "codemixed":
                text = codemix(text, lang, rng)
            else:
                emo = " ".join(rng.choice(EMOJI[label]) for _ in range(rng.randint(1, 3)))
                text = f"{text} {emo}" if rng.random() < 0.7 else f"{emo} {text}"
            style_tag = f"{style}+{bstyle}"
        else:
            cands = [t for s, t in base if s == style]
            text = _fill(rng.choice(cands), lang, rng)
            style_tag = style
        rows.append({"text": text, "label": label, "language": lang, "source": "template",
                     "source_type": "synthetic", "style": style_tag, "orig_label": label,
                     "split_hint": "train"})
    return rows


def generate_dataset(n_per_label_lang: int, seed: int = 42, langs=("hi", "bn"),
                     labels=("Favour", "Against", "None")) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for lang in langs:
        for label in labels:
            rows.extend(generate(lang, label, n_per_label_lang, rng))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(12, seed=1)
    for _, r in df.sample(12, random_state=0).iterrows():
        print(r["language"], r["label"], r["style"], "|", r["text"])
