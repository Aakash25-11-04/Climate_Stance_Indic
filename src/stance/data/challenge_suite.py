# -*- coding: utf-8 -*-
"""CheckList-style robustness challenge suite (Phase 7). TEST-ONLY.

Three test types
  MFT (minimum functionality): text -> expected label, per category and language.
  INV (invariance): label-preserving perturbations (typos, emoji, URL, elongation, case,
      whitespace, Romanization of a span) must not change the prediction.
  DIR (directional): negation / framing edits that MUST flip the label (expected label_b).

Every category has hand-written seed items (author-verified) that are expanded with small,
category-specific templates so that each category x language has >= 30 MFT tests. Templates
here are deliberately different from the training-data templates in synth_templates.py,
and scripts/build_challenge.py reports any near-duplicates against train.csv.
"""
from __future__ import annotations

import random
import re

import pandas as pd

from stance.data.transliterate import native_to_roman

# --------------------------------------------------------------------------- slots
S = {
    "hi": {
        "TOPIC": ["जलवायु परिवर्तन", "ग्लोबल वार्मिंग", "क्लाइमेट चेंज", "धरती का तापमान"],
        "CITY": ["दिल्ली", "कानपुर", "नागपुर", "वाराणसी", "चंडीगढ़", "इंदौर"],
        "MONTH": ["फरवरी", "मार्च", "अक्टूबर", "नवंबर"],
        "EVENT": ["बाढ़", "सूखा", "लू", "चक्रवात", "ओलावृष्टि"],
        "DENIER_WORD": ["झूठ", "ढकोसला", "स्कैम", "अफवाह", "ड्रामा", "प्रोपेगैंडा"],
        "ACTOR": ["वैज्ञानिक", "नेता", "मीडिया", "NGO", "विदेशी ताकतें", "अमीर देश"],
    },
    "bn": {
        "TOPIC": ["জলবায়ু পরিবর্তন", "গ্লোবাল ওয়ার্মিং", "ক্লাইমেট চেঞ্জ", "পৃথিবীর তাপমাত্রা"],
        "CITY": ["ঢাকা", "কুমিল্লা", "ময়মনসিংহ", "যশোর", "বগুড়া", "শিলিগুড়ি"],
        "MONTH": ["ফেব্রুয়ারি", "মার্চ", "অক্টোবর", "নভেম্বর"],
        "EVENT": ["বন্যা", "খরা", "তাপপ্রবাহ", "ঘূর্ণিঝড়", "শিলাবৃষ্টি"],
        "DENIER_WORD": ["মিথ্যা", "ভাঁওতা", "স্ক্যাম", "গুজব", "নাটক", "প্রোপাগান্ডা"],
        "ACTOR": ["বিজ্ঞানীরা", "নেতারা", "মিডিয়া", "NGO", "বিদেশি শক্তি", "ধনী দেশ"],
    },
}

# --------------------------------------------------------------------------- MFT seeds
# category -> lang -> list of (text_or_template, label). Templates use {SLOT}.
MFT = {
    "negation": {
        "hi": [
            ("{TOPIC} {DENIER_WORD} नहीं है, ये हकीकत है।", "Favour"),
            ("मैं ये नहीं मानता कि {TOPIC} कोई खतरा है।", "Against"),
            ("ऐसा नहीं है कि {TOPIC} से कुछ नहीं होगा, बहुत कुछ होगा।", "Favour"),
            ("{TOPIC} को नकारने वाले गलत नहीं हैं, मौसम तो बदलता ही रहता है।", "Against"),
            ("कोई नहीं कह सकता कि {TOPIC} {DENIER_WORD} है, सबूत साफ हैं।", "Favour"),
            ("मुझे {TOPIC} की चिंता बिल्कुल नहीं है, कुछ नहीं होने वाला।", "Against"),
            ("ये {EVENT} बिना वजह नहीं आ रही, {TOPIC} असली है।", "Favour"),
            ("{TOPIC} के नाम पर डरना बंद करो, कोई खतरा नहीं है।", "Against"),
            ("{TOPIC} पर मेरी कोई राय नहीं है, मुझे जानकारी नहीं।", "None"),
            ("इस वीडियो में {TOPIC} की बात नहीं हुई, सिर्फ खेती की।", "None"),
            ("ये नहीं कह सकते कि {ACTOR} गलत हैं, धरती सच में गर्म हो रही है।", "Favour"),
            ("न {TOPIC} असली है न {ACTOR} की चेतावनी, सब {DENIER_WORD}।", "Against"),
        ],
        "bn": [
            ("{TOPIC} {DENIER_WORD} না, এটা বাস্তব।", "Favour"),
            ("আমি মানি না যে {TOPIC} কোনো বিপদ।", "Against"),
            ("এমন না যে {TOPIC}-এ কিছু হবে না, অনেক কিছু হবে।", "Favour"),
            ("{TOPIC} অস্বীকারকারীরা ভুল না, আবহাওয়া তো বদলায়ই।", "Against"),
            ("কেউ বলতে পারবে না {TOPIC} {DENIER_WORD}, প্রমাণ পরিষ্কার।", "Favour"),
            ("{TOPIC} নিয়ে আমার কোনো চিন্তা নেই, কিছুই হবে না।", "Against"),
            ("এই {EVENT} অকারণে হচ্ছে না, {TOPIC} আসল।", "Favour"),
            ("{TOPIC}-এর নামে ভয় পাওয়া বন্ধ করো, কোনো বিপদ নেই।", "Against"),
            ("{TOPIC} নিয়ে আমার কোনো মত নেই, জানি না।", "None"),
            ("এই ভিডিওতে {TOPIC} নিয়ে কথা হয়নি, শুধু চাষের কথা।", "None"),
            ("এটা বলা যায় না যে {ACTOR} ভুল, পৃথিবী সত্যিই গরম হচ্ছে।", "Favour"),
            ("না {TOPIC} আসল, না {ACTOR}-র সতর্কতা, সব {DENIER_WORD}।", "Against"),
        ],
    },
    "sarcasm": {
        "hi": [
            ("वाह, {CITY} में {MONTH} में बर्फ, बहुत ग्लोबल वार्मिंग है 🙄", "Against"),
            ("हाँ भाई, {ACTOR} ने कह दिया तो {TOPIC} सच हो गया, बिलकुल 😂", "Against"),
            ("बढ़िया, {EVENT} आई तो {TOPIC}, धूप निकली तो भी {TOPIC}, क्या साइंस है।", "Against"),
            ("{TOPIC} झूठ है? अच्छा, {CITY} में 47 डिग्री में ये बोलकर दिखाओ 🙃", "Favour"),
            ("कितना सुकून है कि {MONTH} में ही लू चल रही है, सब एकदम नॉर्मल है ना।", "Favour"),
            ("सही है, {ACTOR} की हर बात मान लो, वो तो कभी गलत नहीं होते 🤡", "Against"),
            ("हाँ, {EVENT} तो बस संयोग है, लगातार पांचवें साल, संयोग ही होगा।", "Favour"),
            ("वाह क्या लॉजिक है, एक ठंडा दिन आया और {TOPIC} खत्म 👏", "Favour"),
            ("हम तो बस चाय पी रहे हैं, धरती जले तो जले 😌", "Favour"),
            ("हाँ जी, 2030 में दुनिया खत्म, 2000 में भी यही बोला था 🙄", "Against"),
        ],
        "bn": [
            ("বাহ, {CITY}-তে {MONTH} মাসে বরফ, খুব গ্লোবাল ওয়ার্মিং 🙄", "Against"),
            ("হ্যাঁ ভাই, {ACTOR} বলেছে তাই {TOPIC} সত্যি হয়ে গেল, একদম 😂", "Against"),
            ("দারুণ, {EVENT} হলে {TOPIC}, রোদ উঠলেও {TOPIC}, কী সায়েন্স।", "Against"),
            ("{TOPIC} মিথ্যা? আচ্ছা, {CITY}-তে ৪৫ ডিগ্রিতে দাঁড়িয়ে বলে দেখাও 🙃", "Favour"),
            ("কী শান্তি, {MONTH} মাসেই লু বইছে, সব একদম স্বাভাবিক তাই না।", "Favour"),
            ("ঠিকই, {ACTOR}-র সব কথা মেনে নাও, তারা তো কখনো ভুল করে না 🤡", "Against"),
            ("হ্যাঁ, {EVENT} তো শুধু কাকতালীয়, টানা পাঁচ বছর, কাকতালীয়ই হবে।", "Favour"),
            ("বাহ কী লজিক, একটা ঠান্ডা দিন এলো আর {TOPIC} শেষ 👏", "Favour"),
            ("আমরা তো চা খাচ্ছি, পৃথিবী পুড়লে পুড়ুক 😌", "Favour"),
            ("হ্যাঁ জি, ২০৩০-এ দুনিয়া শেষ, ২০০০ সালেও এটাই বলেছিল 🙄", "Against"),
        ],
    },
    "implicit": {
        "hi": [
            ("पहले {MONTH} में स्वेटर पहनते थे, अब पंखा चलता है।", "Favour"),
            ("{CITY} की झील सूख गई, दादाजी कहते हैं उनकी ज़िंदगी में पहली बार।", "Favour"),
            ("हर दस साल में {EVENT} आती रही है, इसमें नया क्या है।", "Against"),
            ("नदी का पानी घटता जा रहा है, हर साल और कम।", "Favour"),
            ("मेरे दादा के ज़माने में भी गर्मी पड़ती थी, वो भी जी गए, हम भी जी लेंगे।", "Against"),
            ("सूरज अपनी मर्जी से चलता है, इंसान के बस में कुछ नहीं।", "Against"),
            ("आम की फसल इस साल फिर {MONTH} की बारिश में बर्बाद, लगातार तीसरा साल।", "Favour"),
            ("आज {CITY} में 36 डिग्री है।", "None"),
            ("हमने AC नया लिया है, बहुत अच्छा चलता है।", "None"),
            ("{EVENT} से इस बार 40 गांव डूब गए, पहले ऐसा नहीं होता था।", "Favour"),
            ("ये सब टैक्स बढ़ाने का बहाना है, और कुछ नहीं।", "Against"),
            ("पहाड़ों पर बर्फ अब {MONTH} में ही पिघल जाती है।", "Favour"),
        ],
        "bn": [
            ("আগে {MONTH} মাসে সোয়েটার পরতাম, এখন ফ্যান চলে।", "Favour"),
            ("{CITY}-র দিঘি শুকিয়ে গেছে, দাদু বলেন তার জীবনে প্রথমবার।", "Favour"),
            ("প্রতি দশ বছরে {EVENT} হয়ে এসেছে, এতে নতুন কী।", "Against"),
            ("নদীর পানি কমে যাচ্ছে, প্রতি বছর আরো কম।", "Favour"),
            ("আমার দাদার সময়েও গরম পড়ত, তারাও বেঁচেছিল, আমরাও বাঁচব।", "Against"),
            ("সূর্য নিজের মর্জিতে চলে, মানুষের হাতে কিছু নেই।", "Against"),
            ("আমের ফলন এবারও {MONTH}-র বৃষ্টিতে নষ্ট, টানা তৃতীয় বছর।", "Favour"),
            ("আজ {CITY}-তে ৩৬ ডিগ্রি।", "None"),
            ("আমরা নতুন AC কিনেছি, খুব ভালো চলে।", "None"),
            ("{EVENT}-এ এবার ৪০ গ্রাম ডুবে গেল, আগে এমন হতো না।", "Favour"),
            ("এসব ট্যাক্স বাড়ানোর অজুহাত, আর কিছু না।", "Against"),
            ("পাহাড়ের বরফ এখন {MONTH} মাসেই গলে যায়।", "Favour"),
        ],
    },
    "romanized": {
        "hi": [
            ("climate change sach hai, {CITY} me {MONTH} me hi 44 degree ho gaya", "Favour"),
            ("global warming sab {DENIER_WORD_R} hai bhai, kuch nahi hone wala", "Against"),
            ("kal {CITY} me baarish ka alert hai", "None"),
            ("jo {EVENT_R} har saal aa rahi hai wo normal nahi hai, kuch karna padega", "Favour"),
            ("ye {ACTOR_R} log sirf dara rahe hai, paisa chahiye inko", "Against"),
            ("{TOPIC_R} pe koi acchi book batao", "None"),
            ("bachpan wali thand ab nahi padti, dharti sach me garam ho rahi hai", "Favour"),
            ("mausam hamesha badalta aaya hai, {TOPIC_R} ka drama band karo", "Against"),
            ("is video ka editing kaun karta hai?", "None"),
            ("agar abhi nahi jage to bachon ka kya hoga, {TOPIC_R} asli hai", "Favour"),
            ("{TOPIC_R} ek videshi agenda hai, bharat ko rokne ke liye", "Against"),
            ("{TOPIC_R} par webinar kal shaam 6 baje", "None"),
        ],
        "bn": [
            ("climate change sotti, {CITY_R} te {MONTH_R} e 42 degree hoye gelo", "Favour"),
            ("global warming sob {DENIER_WORD_R} bhai, kichu hobe na", "Against"),
            ("kal {CITY_R} te bristir alert ache", "None"),
            ("je {EVENT_R} proti bochor hocche eta normal na, kichu korte hobe", "Favour"),
            ("ei {ACTOR_R} ra shudhu bhoy dekhay, taka chay era", "Against"),
            ("{TOPIC_R} niye bhalo boi bolo", "None"),
            ("chotobelar thanda ar pore na, prithibi sotti gorom hocche", "Favour"),
            ("abhawa sobsomoy bodlay, {TOPIC_R} er natok bondho koro", "Against"),
            ("ei video r editing ke kore?", "None"),
            ("ekhon na jagle bacchader ki hobe, {TOPIC_R} asol", "Favour"),
            ("{TOPIC_R} ekta bideshi agenda, bangladesh ke atkanor jonno", "Against"),
            ("{TOPIC_R} niye webinar kal shondha 6 tay", "None"),
        ],
    },
    "codemixed": {
        "hi": [
            ("Climate change real है, {CITY} में {MONTH} में 44 degree 🔥", "Favour"),
            ("ये सब {DENIER_WORD} है bro, scientists को funding चाहिए", "Against"),
            ("Weather update: {CITY} में कल rain होगी", "None"),
            ("Honestly {EVENT} हर साल बढ़ रही है, we need to act", "Favour"),
            ("Climate change is a western agenda, India को रोकने के लिए", "Against"),
            ("{TOPIC} पर seminar है Sunday को, link bio में", "None"),
            ("Frankly मुझे नहीं लगता {TOPIC} से कोई danger है", "Against"),
            ("Sab log please पेड़ लगाओ, earth बचानी है", "Favour"),
            ("Video quality thodi low है, 1080p में upload करो", "None"),
            ("{ACTOR} सही बोल रहे हैं, the data is clear, धरती गर्म हो रही है", "Favour"),
            ("LOL {TOPIC}, meanwhile {CITY} में {MONTH} में ठंड ❄️", "Against"),
            ("Kya koi {TOPIC} का meaning simple words में बता सकता है?", "None"),
        ],
        "bn": [
            ("Climate change real, {CITY}-তে {MONTH}-এ 42 degree 🔥", "Favour"),
            ("এসব {DENIER_WORD} bro, scientists-দের funding লাগে", "Against"),
            ("Weather update: {CITY}-তে কাল rain হবে", "None"),
            ("Honestly {EVENT} প্রতি বছর বাড়ছে, we need to act", "Favour"),
            ("Climate change is a western agenda, বাংলাদেশকে আটকানোর জন্য", "Against"),
            ("{TOPIC} নিয়ে seminar আছে Sunday, link bio-তে", "None"),
            ("Frankly আমার মনে হয় না {TOPIC}-এ কোনো danger আছে", "Against"),
            ("Sobai please গাছ লাগাও, earth বাঁচাতে হবে", "Favour"),
            ("Video quality একটু low, 1080p-তে upload করো", "None"),
            ("{ACTOR} ঠিক বলছে, the data is clear, পৃথিবী গরম হচ্ছে", "Favour"),
            ("LOL {TOPIC}, meanwhile {CITY}-তে {MONTH}-এ ঠান্ডা ❄️", "Against"),
            ("Keu ki {TOPIC}-এর meaning simple words-এ বলতে পারবে?", "None"),
        ],
    },
    "typos": {
        "hi": [
            ("जलवायु परिवरतन सच्च है, {CITY} मे {MONTH} मे ही लू चल रहि है", "Favour"),
            ("ग्लोबल वारमिंग सब झुठ है, {ACTOR} का धंदा", "Against"),
            ("कल {CITY} मे बारिष होगि", "None"),
            ("globl warmng sach h bhai, garmi bhot badh gyi", "Favour"),
            ("climat chng sb jhoot h, kch nhi hoga", "Against"),
            ("vdo acha h bhai, next vdo kb aayega", "None"),
            ("धरती गरम हो रहि है, कुछ करो यार", "Favour"),
            ("ये सब {DENIER_WORD} है, {ACTOR} बेवकुफ बना रहे है", "Against"),
            ("{TOPIC} पर निबन्ध चाहिये स्कुल के लिये", "None"),
            ("हर साल {EVENT} बढ रही है, ये नारमल नही है", "Favour"),
        ],
        "bn": [
            ("জলবায়ু পরিবরতন সত্তি, {CITY} তে {MONTH} এই লু বইছে", "Favour"),
            ("গ্লোবাল ওয়ারমিং সব মিথা, {ACTOR} র ব্যাবসা", "Against"),
            ("কাল {CITY} তে বৃস্টি হবে", "None"),
            ("globl warmng sotti bhai, gorom onek bere geche", "Favour"),
            ("climat chng sb mitha, kichu hbe na", "Against"),
            ("vdo valo bhai, next vdo kbe asbe", "None"),
            ("প্রিথিবি গরম হচ্চে, কিছু করো ভাই", "Favour"),
            ("এসব {DENIER_WORD}, {ACTOR} বোকা বানাচ্চে", "Against"),
            ("{TOPIC} নিয়ে রচনা লাগবে স্কুলের জন্নো", "None"),
            ("প্রতি বছর {EVENT} বারছে, এটা নরমাল না", "Favour"),
        ],
    },
    "emoji": {
        "hi": [
            ("{TOPIC} 😰🌡️🔥 कुछ करो 🙏", "Favour"),
            ("{TOPIC} 🤣🤣 सबसे बड़ा {DENIER_WORD} 🤡", "Against"),
            ("{TOPIC} पर नया वीडियो 📺 कल 📅", "None"),
            ("🌍💔 हर साल {EVENT}, कब जागेंगे 😢", "Favour"),
            ("🙄 फिर वही {TOPIC} का रोना 😴", "Against"),
            ("🎬👍", "None"),
            ("🌳🌳🌳 पेड़ लगाओ, धरती बचाओ 🌍", "Favour"),
            ("💸💸 {TOPIC} = {ACTOR} का पैसा 💸", "Against"),
            ("☀️🌧️ आज {CITY} का मौसम", "None"),
            ("😡 {ACTOR} कुछ नहीं कर रहे और {EVENT} बढ़ती जा रही है 🌊", "Favour"),
        ],
        "bn": [
            ("{TOPIC} 😰🌡️🔥 কিছু করো 🙏", "Favour"),
            ("{TOPIC} 🤣🤣 সবচেয়ে বড় {DENIER_WORD} 🤡", "Against"),
            ("{TOPIC} নিয়ে নতুন ভিডিও 📺 কাল 📅", "None"),
            ("🌍💔 প্রতি বছর {EVENT}, কবে জাগব 😢", "Favour"),
            ("🙄 আবার সেই {TOPIC}-র কান্না 😴", "Against"),
            ("🎬👍", "None"),
            ("🌳🌳🌳 গাছ লাগাও, পৃথিবী বাঁচাও 🌍", "Favour"),
            ("💸💸 {TOPIC} = {ACTOR}-র টাকা 💸", "Against"),
            ("☀️🌧️ আজ {CITY}-র আবহাওয়া", "None"),
            ("😡 {ACTOR} কিছুই করছে না আর {EVENT} বেড়েই চলেছে 🌊", "Favour"),
        ],
    },
    "short": {
        "hi": [
            ("सच है।", "Favour"), ("{DENIER_WORD}।", "Against"), ("कब?", "None"), ("बचाओ धरती", "Favour"),
            ("कोई खतरा नहीं", "Against"), ("लिंक?", "None"), ("{TOPIC} असली", "Favour"),
            ("{TOPIC} {DENIER_WORD}", "Against"), ("पहला", "None"), ("जागो लोगों", "Favour"),
            ("सब ड्रामा", "Against"), ("ओके", "None"), ("खतरा बड़ा है", "Favour"), ("डराना बंद", "Against"),
            ("कल बारिश", "None"),
        ],
        "bn": [
            ("সত্যি।", "Favour"), ("{DENIER_WORD}।", "Against"), ("কবে?", "None"), ("পৃথিবী বাঁচাও", "Favour"),
            ("কোনো বিপদ নেই", "Against"), ("লিংক?", "None"), ("{TOPIC} আসল", "Favour"),
            ("{TOPIC} {DENIER_WORD}", "Against"), ("প্রথম", "None"), ("জাগো সবাই", "Favour"),
            ("সব নাটক", "Against"), ("ওকে", "None"), ("বিপদ বড়", "Favour"), ("ভয় দেখানো বন্ধ", "Against"),
            ("কাল বৃষ্টি", "None"),
        ],
    },
    "long": {
        "hi": [
            ("मैं {CITY} में पैदा हुआ, वहीं बड़ा हुआ, और मैंने अपनी आंखों से देखा है कि {MONTH} का मौसम कैसे बदल गया है। जो {EVENT} पहले कभी-कभार आती थी, अब हर साल आती है, और हर बार पिछली से ज्यादा नुकसान करती है। लोग कहते हैं कि {TOPIC} सिर्फ किताबों की बात है, पर मेरे खेत, मेरे कुएं और मेरे बच्चों की खांसी कुछ और कहती है। हमें अभी कदम उठाने होंगे, वरना देर हो जाएगी।", "Favour"),
            ("देखिए, मैं कोई {ACTOR} नहीं हूं, पर इतिहास पढ़ा है। मध्यकाल में भी गर्मी थी, छोटा हिमयुग भी आया, और तब कोई फैक्ट्री नहीं थी। ये जो हर {EVENT} को {TOPIC} से जोड़ने का फैशन है, ये विज्ञान नहीं बल्कि एक उद्योग है जिसमें अरबों का कारोबार है। मैं अपने टैक्स के पैसे से इस {DENIER_WORD} को फंड नहीं करना चाहता, और मुझे नहीं लगता कि आने वाले सौ साल में कुछ खास बदलने वाला है।", "Against"),
            ("इस वीडियो में {TOPIC} के इतिहास, {ACTOR} की रिपोर्टों, और {CITY} जैसे शहरों के तापमान के आंकड़ों को समझाया गया है। बीच में एक हिस्सा {EVENT} के प्रबंधन पर है और अंत में कुछ किताबों के नाम दिए गए हैं। वीडियो लगभग पच्चीस मिनट का है, आवाज़ साफ है और हिंदी में सबटाइटल भी हैं, इसलिए स्कूल और कॉलेज के छात्रों के लिए यह एक अच्छा संदर्भ बन सकता है।", "None"),
            ("पहले मैं भी {TOPIC} को {DENIER_WORD} मानता था, और {ACTOR} की बातों पर हंसता था। फिर तीन साल तक लगातार {CITY} में {EVENT} देखी, {MONTH} में ऐसी गर्मी झेली जो मेरे बचपन में कभी नहीं थी, और अपने घर के बुज़ुर्गों से सुना कि उन्होंने भी ऐसा कभी नहीं देखा। अब मैं मानता हूं कि समस्या असली है, और जो लोग अब भी हंस रहे हैं वो कुछ सालों में समझ जाएंगे।", "Favour"),
            ("मुझे समझ नहीं आता कि इतना शोर क्यों है। {CITY} में इस साल {MONTH} में ठीक-ठाक ठंड पड़ी, बारिश भी समय पर हुई, और {EVENT} की जो खबरें आती हैं वो हर दशक में आती रही हैं। मेरे हिसाब से {ACTOR} को अपना एजेंडा चलाना है और मीडिया को व्यूज़ चाहिए, इसलिए हर छोटी चीज़ को {TOPIC} बना दिया जाता है। मैं इस डर के माहौल का हिस्सा नहीं बनना चाहता।", "Against"),
        ],
        "bn": [
            ("আমি {CITY}-তে জন্মেছি, সেখানেই বড় হয়েছি, আর নিজের চোখে দেখেছি {MONTH} মাসের আবহাওয়া কীভাবে বদলে গেছে। যে {EVENT} আগে কালেভদ্রে হতো, এখন প্রতি বছর হয়, আর প্রতিবার আগের চেয়ে বেশি ক্ষতি করে। লোকে বলে {TOPIC} শুধু বইয়ের কথা, কিন্তু আমার ক্ষেত, আমার টিউবওয়েল আর আমার বাচ্চাদের কাশি অন্য কথা বলে। এখনই পদক্ষেপ নিতে হবে, নইলে দেরি হয়ে যাবে।", "Favour"),
            ("দেখুন, আমি কোনো {ACTOR} না, কিন্তু ইতিহাস পড়েছি। মধ্যযুগেও গরম ছিল, ছোট বরফযুগও এসেছিল, আর তখন কোনো কারখানা ছিল না। প্রতিটা {EVENT}-কে {TOPIC}-এর সাথে জোড়ার যে ফ্যাশন, সেটা বিজ্ঞান না, একটা ইন্ডাস্ট্রি যেখানে শত শত কোটি টাকার ব্যবসা। আমি আমার ট্যাক্সের টাকায় এই {DENIER_WORD} ফান্ড করতে চাই না, আর আমার মনে হয় না আগামী একশ বছরে বিশেষ কিছু বদলাবে।", "Against"),
            ("এই ভিডিওতে {TOPIC}-এর ইতিহাস, {ACTOR}-র রিপোর্ট আর {CITY}-র মতো শহরের তাপমাত্রার তথ্য ব্যাখ্যা করা হয়েছে। মাঝে একটা অংশ {EVENT} ব্যবস্থাপনা নিয়ে, আর শেষে কিছু বইয়ের নাম দেওয়া আছে। ভিডিওটা প্রায় পঁচিশ মিনিটের, শব্দ পরিষ্কার আর বাংলা সাবটাইটেলও আছে, তাই স্কুল-কলেজের ছাত্রদের জন্য ভালো রেফারেন্স হতে পারে।", "None"),
            ("আগে আমিও {TOPIC}-কে {DENIER_WORD} ভাবতাম, {ACTOR}-র কথায় হাসতাম। তারপর তিন বছর টানা {CITY}-তে {EVENT} দেখলাম, {MONTH} মাসে এমন গরম সহ্য করলাম যা আমার ছোটবেলায় কখনো ছিল না, আর বাড়ির মুরুব্বিদের কাছে শুনলাম তারাও এমন কখনো দেখেননি। এখন আমি মানি সমস্যাটা আসল, আর যারা এখনো হাসছে তারা কয়েক বছরে বুঝবে।", "Favour"),
            ("আমি বুঝি না এত হইচই কেন। {CITY}-তে এ বছর {MONTH} মাসে ভালোই ঠান্ডা পড়েছে, বৃষ্টিও সময়মতো হয়েছে, আর {EVENT}-র যে খবর আসে সেটা প্রতি দশকেই আসত। আমার মতে {ACTOR}-কে নিজের এজেন্ডা চালাতে হয় আর মিডিয়ার ভিউ লাগে, তাই প্রতিটা ছোট জিনিসকে {TOPIC} বানিয়ে দেওয়া হয়। আমি এই ভয়ের পরিবেশের অংশ হতে চাই না।", "Against"),
        ],
    },
    "neutral_topic": {
        "hi": [
            ("{CITY} में कल अधिकतम तापमान 31 डिग्री रहेगा और हल्के बादल छाए रहेंगे।", "None"),
            ("{TOPIC} पर {ACTOR} की नई रिपोर्ट {MONTH} में आएगी।", "None"),
            ("{TOPIC} पर इस हफ्ते {CITY} विश्वविद्यालय में व्याख्यान है।", "None"),
            ("मौसम विभाग ने {CITY} के लिए {EVENT} की चेतावनी जारी की है।", "None"),
            ("{TOPIC} का सिलेबस में कौन सा अध्याय है?", "None"),
            ("{TOPIC} पर एक डॉक्यूमेंट्री नेटफ्लिक्स पर आई है।", "None"),
            ("{MONTH} में {CITY} का औसत तापमान 28 डिग्री होता है।", "None"),
            ("इस साल मानसून {CITY} में तीन दिन देर से पहुंचा।", "None"),
            ("{ACTOR} की बैठक में {TOPIC} एजेंडे में था।", "None"),
            ("{TOPIC} शब्द पहली बार 1970 के दशक में इस्तेमाल हुआ था।", "None"),
        ],
        "bn": [
            ("{CITY}-তে কাল সর্বোচ্চ তাপমাত্রা ৩১ ডিগ্রি থাকবে, হালকা মেঘ থাকবে।", "None"),
            ("{TOPIC} নিয়ে {ACTOR}-র নতুন রিপোর্ট {MONTH} মাসে আসবে।", "None"),
            ("{TOPIC} নিয়ে এই সপ্তাহে {CITY} বিশ্ববিদ্যালয়ে বক্তৃতা আছে।", "None"),
            ("আবহাওয়া অফিস {CITY}-র জন্য {EVENT}-এর সতর্কতা জারি করেছে।", "None"),
            ("{TOPIC} সিলেবাসের কোন অধ্যায়ে আছে?", "None"),
            ("{TOPIC} নিয়ে একটা ডকুমেন্টারি নেটফ্লিক্সে এসেছে।", "None"),
            ("{MONTH} মাসে {CITY}-র গড় তাপমাত্রা ২৮ ডিগ্রি।", "None"),
            ("এ বছর বর্ষা {CITY}-তে তিন দিন দেরিতে এসেছে।", "None"),
            ("{ACTOR}-র বৈঠকে {TOPIC} এজেন্ডায় ছিল।", "None"),
            ("{TOPIC} শব্দটা প্রথম ১৯৭০-এর দশকে ব্যবহার হয়।", "None"),
        ],
    },
    "offtopic": {
        "hi": [
            ("{CITY} में कौन सा रेस्टोरेंट अच्छा है?", "None"),
            ("भाई इस गाने का नाम क्या है?", "None"),
            ("कल {CITY} में क्रिकेट मैच है, टिकट मिल गया।", "None"),
            ("ये फोन {MONTH} में लॉन्च होगा, कीमत कितनी होगी?", "None"),
            ("मेरा इंटरनेट बहुत स्लो है आज।", "None"),
            ("दीदी की शादी {MONTH} में है, बहुत काम है।", "None"),
            ("नया लैपटॉप लिया, बहुत बढ़िया है।", "None"),
            ("सब्सक्राइब और लाइक कर दिया, बेल भी दबा दी।", "None"),
            ("कल एग्ज़ाम है, पढ़ाई नहीं हुई 😭", "None"),
            ("{CITY} से {CITY} जाने का सबसे अच्छा रास्ता कौन सा है?", "None"),
        ],
        "bn": [
            ("{CITY}-তে কোন রেস্টুরেন্ট ভালো?", "None"),
            ("ভাই এই গানটার নাম কী?", "None"),
            ("কাল {CITY}-তে ক্রিকেট ম্যাচ, টিকিট পেয়ে গেছি।", "None"),
            ("এই ফোনটা {MONTH} মাসে লঞ্চ হবে, দাম কত হবে?", "None"),
            ("আমার ইন্টারনেট আজ খুব স্লো।", "None"),
            ("আপুর বিয়ে {MONTH} মাসে, অনেক কাজ।", "None"),
            ("নতুন ল্যাপটপ নিলাম, দারুণ।", "None"),
            ("সাবস্ক্রাইব আর লাইক করলাম, বেলও বাজিয়ে দিলাম।", "None"),
            ("কাল পরীক্ষা, পড়া হয়নি 😭", "None"),
            ("{CITY} থেকে {CITY} যাওয়ার সবচেয়ে ভালো রাস্তা কোনটা?", "None"),
        ],
    },
    "conspiracy": {
        "hi": [
            ("{TOPIC} {ACTOR} की चाल है ताकि हम पर कार्बन टैक्स लगाया जा सके।", "Against"),
            ("ये सब {ACTOR} का खेल है, असली मकसद हमें गुलाम बनाना है।", "Against"),
            ("{TOPIC} का डर दिखाकर {ACTOR} अरबों कमा रहे हैं, जागो।", "Against"),
            ("मौसम को मशीनों से बदला जा रहा है और नाम {TOPIC} का दिया जा रहा है।", "Against"),
            ("{ACTOR} को पहले से पता था, ये {EVENT} प्लान किया हुआ है, {TOPIC} बहाना है।", "Against"),
            ("{TOPIC} एक वैश्विक {DENIER_WORD} है, डेटा के साथ छेड़छाड़ होती है।", "Against"),
            ("जिनको इलेक्ट्रिक कार बेचनी है वही {TOPIC} चिल्ला रहे हैं।", "Against"),
            ("{ACTOR} चाहते हैं कि गरीब देश विकास न करें, इसलिए {TOPIC} का नारा।", "Against"),
            ("{TOPIC} के पीछे असली एजेंडा जनसंख्या नियंत्रण है।", "Against"),
            ("हर {EVENT} के बाद {ACTOR} को फंड मिलता है, समझो गणित।", "Against"),
        ],
        "bn": [
            ("{TOPIC} {ACTOR}-র চাল যাতে আমাদের উপর কার্বন ট্যাক্স বসানো যায়।", "Against"),
            ("এসব {ACTOR}-র খেলা, আসল উদ্দেশ্য আমাদের গোলাম বানানো।", "Against"),
            ("{TOPIC}-এর ভয় দেখিয়ে {ACTOR} শত শত কোটি কামাচ্ছে, জাগো।", "Against"),
            ("মেশিন দিয়ে আবহাওয়া বদলানো হচ্ছে আর নাম দেওয়া হচ্ছে {TOPIC}।", "Against"),
            ("{ACTOR} আগে থেকেই জানত, এই {EVENT} প্ল্যান করা, {TOPIC} অজুহাত।", "Against"),
            ("{TOPIC} একটা বৈশ্বিক {DENIER_WORD}, ডেটা নিয়ে কারসাজি হয়।", "Against"),
            ("যাদের ইলেকট্রিক গাড়ি বেচতে হবে তারাই {TOPIC} বলে চিৎকার করছে।", "Against"),
            ("{ACTOR} চায় গরিব দেশ উন্নতি না করুক, তাই {TOPIC}-এর স্লোগান।", "Against"),
            ("{TOPIC}-এর পেছনে আসল এজেন্ডা জনসংখ্যা নিয়ন্ত্রণ।", "Against"),
            ("প্রতিটা {EVENT}-এর পর {ACTOR} ফান্ড পায়, হিসাবটা বোঝো।", "Against"),
        ],
    },
    "different_target": {
        "hi": [
            ("{TOPIC} असली है, पर ये ऑड-ईवन नियम बेकार है, सरकार कुछ ठोस करे।", "Favour"),
            ("मुझे ये एक्टिविस्ट पसंद नहीं, पर {TOPIC} को लेकर वो गलत नहीं हैं।", "Favour"),
            ("पेरिस समझौता बेकार है।", "None"),
            ("सोलर सब्सिडी में घोटाला है, जांच हो।", "None"),
            ("सरकार {TOPIC} पर बहुत धीमी है, खतरा हर दिन बढ़ रहा है।", "Favour"),
            ("कार्बन टैक्स गरीबों पर बोझ है, पर {TOPIC} से लड़ना ज़रूरी है।", "Favour"),
            ("{ACTOR} को हमारे देश में दखल नहीं देना चाहिए।", "None"),
            ("इलेक्ट्रिक गाड़ियां अभी महंगी हैं, ये समाधान नहीं।", "None"),
            ("{CITY} नगर निगम ने पेड़ काटे, ये गलत है, {TOPIC} के दौर में ऐसा?", "Favour"),
            ("COP बैठकें सिर्फ फोटो खिंचवाने के लिए हैं, काम कुछ नहीं होता।", "None"),
            ("प्लास्टिक बैन से छोटे दुकानदार परेशान हैं, पर पर्यावरण बचाना भी ज़रूरी है।", "Favour"),
            ("ये {ACTOR} खुद प्राइवेट जेट में घूमते हैं, हमें उपदेश देते हैं।", "None"),
        ],
        "bn": [
            ("{TOPIC} আসল, কিন্তু এই জোড়-বিজোড় নিয়ম অকেজো, সরকার কিছু শক্ত করুক।", "Favour"),
            ("আমার এই অ্যাক্টিভিস্টকে পছন্দ না, কিন্তু {TOPIC} নিয়ে সে ভুল না।", "Favour"),
            ("প্যারিস চুক্তি অকেজো।", "None"),
            ("সোলার ভর্তুকিতে কেলেঙ্কারি, তদন্ত হোক।", "None"),
            ("সরকার {TOPIC} নিয়ে খুব ধীর, বিপদ প্রতিদিন বাড়ছে।", "Favour"),
            ("কার্বন ট্যাক্স গরিবের বোঝা, কিন্তু {TOPIC}-এর সাথে লড়াই জরুরি।", "Favour"),
            ("{ACTOR}-র আমাদের দেশে নাক গলানো উচিত না।", "None"),
            ("ইলেকট্রিক গাড়ি এখনো দামি, এটা সমাধান না।", "None"),
            ("{CITY} সিটি কর্পোরেশন গাছ কেটেছে, এটা ভুল, {TOPIC}-এর যুগে এমন?", "Favour"),
            ("COP বৈঠক শুধু ছবি তোলার জন্য, কাজ কিছু হয় না।", "None"),
            ("প্লাস্টিক নিষেধাজ্ঞায় ছোট দোকানদাররা বিপদে, কিন্তু পরিবেশ বাঁচানোও জরুরি।", "Favour"),
            ("এই {ACTOR} নিজেরা প্রাইভেট জেটে ঘোরে, আমাদের উপদেশ দেয়।", "None"),
        ],
    },
}

# --------------------------------------------------------------------------- directional (negation flips)
DIRECTIONAL = {
    "hi": [
        ("{TOPIC} एक गंभीर खतरा है।", "Favour", "{TOPIC} कोई गंभीर खतरा नहीं है।", "Against"),
        ("{TOPIC} {DENIER_WORD} है।", "Against", "{TOPIC} {DENIER_WORD} नहीं है।", "Favour"),
        ("मैं मानता हूं कि धरती गर्म हो रही है।", "Favour", "मैं नहीं मानता कि धरती गर्म हो रही है।", "Against"),
        ("{ACTOR} सही कह रहे हैं, {TOPIC} असली है।", "Favour", "{ACTOR} गलत कह रहे हैं, {TOPIC} असली नहीं है।", "Against"),
        ("{EVENT} {TOPIC} की वजह से बढ़ रही है।", "Favour", "{EVENT} {TOPIC} की वजह से नहीं बढ़ रही, ये हमेशा से होती है।", "Against"),
        ("हमें {TOPIC} पर अभी कदम उठाने चाहिए।", "Favour", "हमें {TOPIC} पर कोई कदम उठाने की ज़रूरत नहीं है।", "Against"),
        ("{TOPIC} से डरना सही है।", "Favour", "{TOPIC} से डरना बेवकूफी है।", "Against"),
        ("{TOPIC} हमारे बच्चों के भविष्य के लिए खतरा है।", "Favour", "{TOPIC} हमारे बच्चों के भविष्य के लिए कोई खतरा नहीं है।", "Against"),
    ],
    "bn": [
        ("{TOPIC} একটা গুরুতর বিপদ।", "Favour", "{TOPIC} কোনো গুরুতর বিপদ না।", "Against"),
        ("{TOPIC} {DENIER_WORD}।", "Against", "{TOPIC} {DENIER_WORD} না।", "Favour"),
        ("আমি মানি পৃথিবী গরম হচ্ছে।", "Favour", "আমি মানি না পৃথিবী গরম হচ্ছে।", "Against"),
        ("{ACTOR} ঠিক বলছে, {TOPIC} আসল।", "Favour", "{ACTOR} ভুল বলছে, {TOPIC} আসল না।", "Against"),
        ("{EVENT} {TOPIC}-এর কারণে বাড়ছে।", "Favour", "{EVENT} {TOPIC}-এর কারণে বাড়ছে না, এটা সবসময়ই হয়।", "Against"),
        ("আমাদের {TOPIC} নিয়ে এখনই পদক্ষেপ নেওয়া উচিত।", "Favour", "আমাদের {TOPIC} নিয়ে কোনো পদক্ষেপ নেওয়ার দরকার নেই।", "Against"),
        ("{TOPIC}-কে ভয় পাওয়া ঠিক।", "Favour", "{TOPIC}-কে ভয় পাওয়া বোকামি।", "Against"),
        ("{TOPIC} আমাদের সন্তানদের ভবিষ্যতের জন্য বিপদ।", "Favour", "{TOPIC} আমাদের সন্তানদের ভবিষ্যতের জন্য কোনো বিপদ না।", "Against"),
    ],
}

# --------------------------------------------------------------------------- helpers
_SLOT = re.compile(r"\{([A-Z_]+)\}")


def _roman_slots(lang: str, rng: random.Random) -> dict:
    out = {}
    for k, vals in S[lang].items():
        out[k + "_R"] = [native_to_roman(v, lang, rng, "standard") if re.search(r"[ऀ-৿]", v) else v for v in vals]
    return out


def _fill(tpl: str, lang: str, rng: random.Random, extra: dict) -> str:
    slots = {**S[lang], **extra}

    def rep(m):
        return rng.choice(slots[m.group(1)])

    return re.sub(r"\s{2,}", " ", _SLOT.sub(rep, tpl)).strip()


def _expand(items, lang, rng, target_n, extra):
    """Fill every seed once (deterministic first fill), then keep sampling fills until target_n."""
    out, seen = [], set()
    for tpl, lab in items:
        t = _fill(tpl, lang, rng, extra)
        if t not in seen:
            seen.add(t); out.append((t, lab, "seed" if "{" not in tpl else "seed_filled"))
    tries = 0
    while len(out) < target_n and tries < target_n * 20:
        tries += 1
        tpl, lab = rng.choice([it for it in items if "{" in it[0]] or items)
        t = _fill(tpl, lang, rng, extra)
        if t not in seen:
            seen.add(t); out.append((t, lab, "template"))
    return out


# --------------------------------------------------------------------------- invariance perturbations
_KEYMAP = {"a": "s", "e": "r", "i": "o", "o": "p", "n": "m", "h": "j", "t": "y", "r": "t", "s": "d", "l": "k"}


def perturb(text: str, lang: str, kind: str, rng: random.Random) -> str:
    if kind == "typo":
        chars = list(text)
        idx = [i for i, c in enumerate(chars) if c.isalpha()]
        for i in rng.sample(idx, k=min(2, len(idx))):
            c = chars[i]
            if c.lower() in _KEYMAP:
                chars[i] = _KEYMAP[c.lower()]
            elif i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if kind == "emoji_append":
        return text + " " + rng.choice(["🙂", "👍", "📌", "🙏", "✨", "💬"])
    if kind == "url_append":
        return text + " https://youtu.be/dQw4w9WgXcQ"
    if kind == "mention_prepend":
        return "@user123 " + text
    if kind == "elongation":
        m = list(re.finditer(r"[ऀ-ॿঀ-৿ a-zA-Z]", text))
        if not m:
            return text + "!!!"
        return re.sub(r"([aeiouािीुूািীুূ])", lambda mm: mm.group(1) * 3, text, count=1) + "!!!"
    if kind == "whitespace":
        return "  " + re.sub(r" ", "   ", text) + "  "
    if kind == "roman_span":
        words = text.split()
        if len(words) < 3:
            return native_to_roman(text, lang, rng, "standard")
        i = rng.randint(0, len(words) - 2)
        words[i] = native_to_roman(words[i], lang, rng, "standard")
        words[i + 1] = native_to_roman(words[i + 1], lang, rng, "standard")
        return " ".join(words)
    if kind == "punct_strip":
        return re.sub(r"[।.!?,]+$", "", text)
    return text


INV_KINDS = ["typo", "emoji_append", "url_append", "mention_prepend", "elongation", "whitespace", "roman_span", "punct_strip"]


def build_suite(seed: int = 7, min_per_category: int = 30, n_invariance_per_lang: int = 120):
    rng = random.Random(seed)
    mft_rows, inv_rows, dir_rows = [], [], []
    for cat, by_lang in MFT.items():
        for lang, items in by_lang.items():
            extra = _roman_slots(lang, rng)
            for t, lab, origin in _expand(items, lang, rng, min_per_category, extra):
                mft_rows.append({"id": f"mft_{cat}_{lang}_{len(mft_rows):04d}", "text": t, "language": lang,
                                 "category": cat, "label": lab, "origin": origin})
    mft = pd.DataFrame(mft_rows)
    # invariance: perturb MFT items from explicit categories (labels must be unambiguous)
    base = mft[mft["category"].isin(["negation", "conspiracy", "neutral_topic", "offtopic", "implicit", "different_target"])]
    for lang in ("hi", "bn"):
        pool = base[base["language"] == lang].sample(frac=1.0, random_state=seed)
        k = 0
        for _, r in pool.iterrows():
            if k >= n_invariance_per_lang:
                break
            kind = INV_KINDS[k % len(INV_KINDS)]
            if kind == "roman_span" and not re.search(r"[ऀ-৿]", r["text"]):
                kind = "typo"
            pert = perturb(r["text"], lang, kind, rng)
            if pert == r["text"]:
                continue
            inv_rows.append({"id": f"inv_{kind}_{lang}_{k:04d}", "text_a": r["text"], "text_b": pert, "language": lang,
                             "category": f"inv_{kind}", "label": r["label"], "source_mft": r["id"]})
            k += 1
    for lang, pairs in DIRECTIONAL.items():
        extra = _roman_slots(lang, rng)
        for i, (a, la, b, lb) in enumerate(pairs):
            for j in range(4):  # four fills per pair -> 32 per language
                fa = _fill(a, lang, random.Random(seed + i * 10 + j), extra)
                fb = _fill(b, lang, random.Random(seed + i * 10 + j), extra)
                dir_rows.append({"id": f"dir_negation_{lang}_{i:02d}_{j}", "text_a": fa, "text_b": fb, "language": lang,
                                 "category": "dir_negation", "label_a": la, "label_b": lb})
    return mft, pd.DataFrame(inv_rows), pd.DataFrame(dir_rows)
