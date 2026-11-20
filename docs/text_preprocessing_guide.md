Indic Text Preprocessing Guide

Introduction

Text preprocessing is an important step in Natural Language Processing (NLP). It prepares raw text for machine learning models.

In Indic Climate Stance Detection, preprocessing must preserve the original meaning of Hindi and Bengali comments.

1. Unicode Normalization

Unicode normalization helps represent equivalent characters consistently.

NFC normalization is used to reduce inconsistencies in Indic text.

2. Zero-Width Character Removal

Some text contains invisible Unicode characters that can affect text processing.

Removing unwanted zero-width characters helps maintain consistency.

3. URL and Mention Handling

URLs and social media mentions can introduce irrelevant variations.

They can be represented using special tokens instead of being treated as ordinary words.

4. Hashtag Processing

Hashtags may contain meaningful information.

The text inside hashtags should be preserved whenever possible.

5. Emoji Handling

Emojis can communicate opinions and emotions.

The project considers emoji preservation and alternative handling strategies.

6. Elongated Word Normalization

Social media users often repeat characters for emphasis.

Reducing excessive character repetition can help normalize informal text.

7. Negation Preservation

Words such as “not” and “नहीं” are important for stance detection.

Removing negation can reverse the intended meaning of a sentence.

8. Romanized Text

Hindi and Bengali comments may be written using English alphabets.

Preprocessing should account for spelling variations and Romanized expressions.

Conclusion

Indic text preprocessing requires a balance between normalization and meaning preservation.

The objective is to improve text consistency without removing information necessary for accurate stance classification.
