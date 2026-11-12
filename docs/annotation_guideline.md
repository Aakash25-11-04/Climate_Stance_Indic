# Annotation guideline: stance toward "Climate change and global warming is a serious concern."

Use this to label real YouTube comments (Hindi, Bengali, Romanized, code-mixed) for
`data/gold/test_real.csv`. Read the whole comment before deciding. Label the **stance of the
author toward the claim**, not the sentiment, not the quality of the video.

## Labels

| Label | Give it when the author ... | Examples |
|---|---|---|
| **Favour** | agrees that climate change / global warming is real and a serious problem; calls for action; reports alarming change as real; mocks deniers; worries about consequences; supports the claim even while criticising a specific policy or person | "अब तो जागो, गर्मी हर साल बढ़ रही है"; "ग्रेटा पसंद नहीं पर समस्या असली है"; "jolbayu poriborton niye kichu korte hobe" |
| **Against** | denies, minimises or ridicules the concern: it is a hoax / natural cycle / money scheme / western agenda; says there is no danger; uses a cold day to "disprove" warming; conspiracy framing; mocks people who worry | "सब झूठ है, पैसा कमाने का धंधा"; "প্রতি বছর গরম আসে, এর মধ্যে ক্লাইমেট চেঞ্জ কোথায়"; "hoax hai bhai" |
| **None** | expresses no stance toward the claim: weather reports, event news, questions asking for information, off-topic chat, praise of the video, emoji-only, policy criticism **without** revealing a view on the claim, deflection to another topic | "कल बारिश होगी"; "video acha tha"; "पेरिस समझौता बेकार है" (no view on the claim itself) |

## Decision rules for hard cases

1. **Negation**: resolve it. "ये कहना गलत है कि क्लाइमेट चेंज झूठ है" = Favour. "मुझे नहीं लगता कोई खतरा है" = Against.
2. **Sarcasm**: label the *intended* meaning. "हाँ हाँ, दिसंबर में बर्फ, बहुत ग्लोबल वार्मिंग है 🙄" = Against. "48 डिग्री में भी झूठ है? AC चालू है ना?" = Favour.
3. **Implicit**: observations of unusual change offered as evidence ("बचपन में मई में ठंडी हवा चलती थी") = Favour; "हर युग में मौसम बदला है" = Against. A bare measurement ("आज 34 डिग्री") = None.
4. **Different target**: criticising a policy, politician or activist while accepting the problem = Favour. Criticising them without any signal about the problem = None.
5. **Doubt about human cause only** ("गर्म तो हो रही है पर इंसान की वजह से नहीं") = Against (the claim implies it is a concern we must act on; denying the cause minimises it). Note it in `annotator_note`.
6. **Mixed / unclear**: if you cannot decide after two readings, label None and write why in `annotator_note`.
7. **Replies**: label only the text you see; do not guess the parent comment.
8. Ignore spelling, script and language mix; label Romanized text exactly like native script.

## File format

`data/gold/test_real.csv`, UTF-8, columns: `id,text,language,label,annotator_note`
(`language` = `hi` or `bn` as you perceive it; `annotator_note` optional).
Do **not** add usernames or any personal information.
