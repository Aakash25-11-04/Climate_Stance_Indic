import pytest

from stance.preprocess import dedup_key, dominant_script, script_shares


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("जलवायु परिवर्तन!!", "जलवायु परिवर्तन"),
        ("জলবায়ু পরিবর্তন।", "জলবায়ু পরিবর্তন"),
        ("ग्लोबल   वार्मिंग?", "ग्लोबल वार्मिंग"),
        ("Global warming is a HOAX!!! http://t.co/x @bob", "global warming is a hoax"),
        ("see www.example.com now", "see now"),
        ("global_warming", "global_warming"),
    ],
)
def test_dedup_key(text: str, expected: str) -> None:
    assert dedup_key(text) == expected


def test_dedup_key_keeps_words_that_differ_only_in_vowel_sign() -> None:
    assert dedup_key("काला") != dedup_key("कोला")


def test_dedup_key_unifies_composed_and_decomposed_bengali_yya() -> None:
    assert dedup_key("য়") == dedup_key("য়")


def test_dedup_key_drops_zero_width_joiner_inside_word() -> None:
    assert dedup_key("क्‍ष") == "क्ष"


def test_dedup_key_ignores_emoji_variation_selector() -> None:
    assert dedup_key("save the planet \U0001f5e3️\U0001f30d") == "save the planet"


def test_script_shares_ignores_danda_in_bengali() -> None:
    assert script_shares("জলবায়ু পরিবর্তন। গরম।") == {
        "bengali": 1.0,
        "devanagari": 0.0,
        "latin": 0.0,
    }


def test_script_shares_romanized() -> None:
    assert script_shares("garmi bahut hai")["latin"] == 1.0


def test_script_shares_code_mixed() -> None:
    mixed = script_shares("गर्मी ab")
    assert 0 < mixed["devanagari"] < 1
    assert mixed["devanagari"] + mixed["latin"] == pytest.approx(1.0)


def test_script_shares_without_letters_is_all_zero() -> None:
    assert script_shares("123 !!") == {"bengali": 0.0, "devanagari": 0.0, "latin": 0.0}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("जलवायु परिवर्तन एक गंभीर समस्या है", "devanagari"),
        ("জলবায়ু পরিবর্তন একটি গুরুতর সমস্যা।", "bengali"),
        ("global warming sab jhooth hai", "latin"),
        ("ये climate change वाली बात सही है yaar", "mixed"),
        ("🔥🔥 100%", "none"),
    ],
)
def test_dominant_script(text: str, expected: str) -> None:
    assert dominant_script(text) == expected
