#!/usr/bin/env python3
"""Golden tokenization corpus generator for the spacy-rs tokenizer.

Runs the pinned spaCy 3.8.15 (editable install in /tmp/opencode/spacy-venv)
over a curated set of English texts and emits the hermetic fixture
``tests/data/en_tokenization.json``: for every token, its orth, char idx,
spacy flag, and the deterministic lexeme surface attributes (lower/shape/
prefix/suffix/norm) plus the category flags. The Rust golden test replays the
same texts through the native tokenizer and must match byte-for-byte.

Run:
    /tmp/opencode/spacy-venv/bin/python3 src/spacy-rs/tools/gen_golden_corpus.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/opt/src/nlp/spaCy")

import spacy  # noqa: E402

OUT = "src/spacy-rs/tests/data/en_tokenization.json"

CORPUS = [
    # plain sentences
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "This is a test of the tokenizer.",
    # contractions / special cases
    "I can't go.",
    "don't", "won't", "shan't", "y'all", "let's", "c'mon", "gimme", "gonna", "gotta",
    "cannot", "isn't it?", "we've been there.", "they'll come.", "he's here.",
    "I'm going.", "you're right.", "she'd've left.", "wouldn't've",
    "o'clock", "ma'am", "and/or", "w/o",
    # pronouns with apostrophe
    "It's cold.", "The dog's bone.", "James' book.",
    # multi-space / whitespace
    "a  b   c", "  leading", "trailing  ", "tab\there", "line\nbreak", "cr\rreturn",
    "mixed \n whitespace \t runs",
    # punctuation
    "What?!", "Hello, world!", "Are you (really) sure?", "It is—as you see—good.",
    "He said, \"Hello.\"", "Don't 'quote' me.", "['a', 'b']", "{1, 2, 3}",
    "…ellipsis", "3.14 and 42...", "x + y = z", "50% off", "10.5 km/h",
    # emoticons
    ":-) :-( :D 8) ^_^ </3 o.O <3",
    # numbers / times / units
    "It's 5a.m. now.", "at 12p.m.", "3p.m. meeting", "We ran 10km in 2h.",
    "The temperature is 68°F.", "30°C.", "2+2=4", "x^2 + y^2", "a-b", "5-year-old",
    # abbreviations
    "Dr. Smith is here.", "e.g. apples", "i.e. oranges", "Mr. Jones", "St. Louis",
    "Ph.D. students", "etc.", "vs. them", "U.S. citizens", "N.Y. Times", "Apr. 5",
    "a.m.", "p.m.", "Mt. Everest",
    # URLs / emails (lexeme attrs)
    "Visit https://example.com now.", "Go to www.example.org.",
    "mail me at test@example.com.", "check example.co.uk/path?q=1",
    # unicode
    "café éclair naïve", "中文测试", "русский язык", "Ελληνικά", "日本語のテキスト",
    # hyphenation and special shapes
    "dyn-o-mite", "foo---bar", "e-mail", "pre-existing", "well-known",
    "U.S.A.", "R&D", "C++", "C#", "3-D", "T-shirt",
    # possessives / clitics
    "the dogs' park", "children's toys", "rock 'n' roll",
    # long + shapes
    "This sentence has oneHundredAndTwentyThree characters to test word shapes.",
    "We studied the effects of long-range neural network tokenization patterns.",
    "bananas", "hello world hello world hello world hello world",
    # degree + base exceptions
    "It's 100°F.", "°C conversion", "50°F.", "a 2° incline",
    "Mr. Smith and Dr. Jones went to St. Louis.",
    # longer passages
    "The quick brown fox jumps over the lazy dog, and the dog doesn't even notice.",
    "In the U.S., the President's team worked on e.g. tax reform, i.e. a 5-year plan, ",
    "It's been said that 'you can't have your cake and eat it too'—but you can, really.",
    "She bought 2 apples, 3 bananas, and 4 oranges at the farmer's market on 5th Avenue.",
    "The temperature dropped to -10°C overnight, and the roads were covered in ice.",
    "Please send your application to hr@example.com or visit https://jobs.example.org/careers.",
    "The report, published on Dec. 15, covers Q4 performance across N.Y., L.A., and Chicago.",
    "Hmm, that's an interesting point—but I'd have to check with Dr. Smith, Ph.D., first.",
    "The e-commerce company's R&D division developed a state-of-the-art T-shirt printing system.",
    "Could've, should've, would've—all three are acceptable contractions in informal writing.",
    "She'd've gone to the party if she'd known you were coming, but now it's too late.",
    "The meeting is at 2p.m., followed by a 3:30 p.m. workshop and a 5p.m. reception.",
    "Testing 1,000.5 meters, 42km/h, and 98.6°F against the base unit system now.",
    "He said \"It's fine\", then added: 'Really, it's fine'—but we didn't believe him.",
    "Whether it's the 3-D printer, the C++ compiler, or the C# IDE, the tools all work.",
    "A table of contents lists: chapter 1, section 1.1, and appendix A—plus references.",
    "The company's website (www.example.com) and email (info@example.net) are both live.",
    "We traveled from Aix-en-Provence to Saint-Tropez, passing through l'Arbresle.",
    "This sentence, although long and winding, does not contain any particularly unusual tokens.",
    "日本語の文章をトークナイズするテストです。これは長い文です。",
    "Он сказал, что придёт завтра, но не сказал во сколько.",
    "¿Cómo estás? Muy bien, gracias. ¡Hasta luego!",
    "The film—starring Sir Ian McKellen—premiered at 7:30p.m. in N.Y.C.",
    "Whatever you do, don't panic; just stay calm and carry on.",
    "He couldn't've been more wrong, honestly—not even by a little bit.",
    "I'd've said 'maybe', but now I'm not so sure it's worth the effort.",
    "The 64-bit hashes of 'hello', 'world', and 'spaCy' are all different values.",
    "Emoji: 😀 👍 🚀 and the table-flip (╯°□°）╯︵┻━┻ all tokenize fine.",
    "Let's meet at the café on 5th and Main—it's across from the bank, e.g. near St. Mark's.",
]

# Deterministically-random passages: token pools covering affixes, special
# cases, numbers, punctuation, and unicode, assembled with a fixed seed so the
# committed fixture is stable.
import random

_SEED = 20260825
PIECES = [
    "can't", "won't", "shan't", "y'all", "let's", "gimme", "gonna", "gotta",
    "cannot", "isn't", "we've", "they'll", "he's", "I'm", "you're", "she'd",
    "o'clock", "ma'am", "and/or", "w/o", "e.g.", "i.e.", "Dr.", "Mr.",
    "St. Louis", "N.Y.C.", "Ph.D.", "etc.", "vs.", "a.m.", "p.m.",
    "hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "dog",
    "5-year-old", "e-mail", "T-shirt", "3-D", "C++", "C#", "R&D",
    "dyn-o-mite", "foo---bar", "well-known", "pre-existing",
    "123", "3.14", "42", "10km/h", "50%", "2+2", "x^2",
    "-10", "+42", "1,000", "1/2", "100°F", "5a.m.",
    "(", ")", "[", "]", "{", "}", ":", ";", "!", "?", "—", "--", "…",
    "café", "éclair", "naïve", "中文", "日本語", "русский", "Ελληνικά",
    "don't", "it's", "that's", "there's", "here's", "what's",
    "😀", "👍", "<3", ":-)", "o.O",
]

_rand = random.Random(_SEED)
for _ in range(220):
    n = _rand.randint(1, 9)
    pieces = [_rand.choice(PIECES) for _ in range(n)]
    sep = _rand.choice([" ", " ", " ", "  ", "\t", "\n", "  "])
    text = sep.join(pieces)
    if _rand.random() < 0.3:
        text = text + _rand.choice([" ", "", "  ", "!"])
    CORPUS.append(text)

# The lexeme surface attributes to capture per token.
SURFACE = ["lower", "shape", "prefix", "suffix", "norm"]
FLAGS = [
    "is_alpha", "is_ascii", "is_digit", "is_lower", "is_punct", "is_space",
    "is_title", "is_upper", "like_url", "like_num", "like_email", "is_stop",
    "is_bracket", "is_quote", "is_left_punct", "is_right_punct", "is_currency",
]


def main() -> None:
    nlp = spacy.blank("en")
    cases = []
    for text in CORPUS:
        doc = nlp(text)
        tokens = []
        for t in doc:
            tok = {
                "orth": t.text,
                "idx": t.idx,
                "spacy": bool(t.whitespace_),
            }
            for attr in SURFACE:
                tok[attr] = getattr(t, attr + "_")
            for flag in FLAGS:
                tok[flag] = bool(getattr(t, flag))
            tokens.append(tok)
        cases.append({"text": text, "tokens": tokens})
    with open(OUT, "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: {len(cases)} cases")


if __name__ == "__main__":
    main()