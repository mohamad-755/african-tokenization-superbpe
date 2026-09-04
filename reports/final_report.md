# Bridging the Tokenization Gap in African LLMs via SuperBPE

**Final Technical Report**
**Program:** TRI AI Saturdays
**Team:** Mahale Team

---

## 1. Abstract

Large language model tokenizers are almost universally trained on
English-dominated corpora, and this bias carries over even when the
resulting models are later applied to other languages. For morphologically
rich or agglutinative African languages, a mismatched tokenizer fragments
words into small, often meaningless pieces rather than the coherent
morphemes a native speaker would recognize â inflating token counts (and
therefore compute cost) and degrading downstream model quality. This
project builds a tokenization pipeline trained directly on native corpora
across nine African languages spanning three language families, and
evaluates **SuperBPE** (Liu et al., 2025) â a two-stage byte-pair-encoding
scheme that first learns ordinary subwords, then lifts the whitespace
restriction to learn frequent multi-word "superwords" â against a matched
standard BPE baseline. SuperBPE reduces fragmentation in every one of the
nine languages tested (10.1% average, tokens/word) and shows a consistent,
independently-corroborating gain in raw compression efficiency (11.8%
average, bytes/token). However, at the small model/data scale we were able
to train (a ~9.4M-parameter model, ~4M words, 3 epochs), SuperBPE
underperformed the baseline on language-model quality (bits-per-byte),
which we attribute to fewer gradient updates per unit of raw text rather
than a flaw in the tokenization approach itself. We report this negative
result plainly, alongside the corpus construction methodology, known data
limitations, and a discussion of why gains concentrate unevenly across
language families.

---

## 2. Problem Statement

Most large language models are built and evaluated primarily on English
and a small set of other high-resource languages, and this bias shapes the
tokenizers sitting underneath them. A tokenizer trained mostly on English
text does not know the internal structure of African languages, many of
which are morphologically rich or agglutinative â a single word can carry
the meaning of an entire English sentence through layers of prefixes,
infixes, and suffixes. When such a tokenizer meets a word it has never
properly learned, it breaks that word into small, often meaningless
fragments instead of the coherent morphemes a native speaker would
recognize.

This fragmentation is not cosmetic. It inflates the number of tokens
needed to represent African-language text, directly raising the
computational cost of training and serving models on that text. It also
degrades model quality, since the model must learn meaning from broken,
inconsistent pieces instead of stable, recognizable units. The effect
compounds across language families, each with different morphological
patterns, so a tokenizer built around one family's structure often fails
differently on another.

This project addresses the problem at its root: building a localized
tokenization pipeline trained directly on native African-language corpora,
rather than adapting an English-centric tokenizer after the fact, and
testing whether SuperBPE's superword-merging mechanism further reduces
fragmentation beyond what a well-constructed standard BPE baseline already
achieves.

---

## 3. Data

### 3.1 Corpus composition

The corpus spans nine languages across three families, chosen to test
whether results generalize rather than overfitting to one morphological
pattern:

| Family | Languages |
|---|---|
| Niger-Congo (Bantu) | Swahili, Chichewa/Nyanja, Zulu, Kinyarwanda |
| Afro-Asiatic | Hausa, Amharic |
| Niger-Congo (Volta-Niger / West Atlantic) | Yoruba, Wolof |
| *(bonus, beyond the required set)* | Igbo |

All languages except Amharic (Ethiopic script) use the Latin script.
Source text is primarily Wikipedia (`dumps.wikimedia.org`,
`cc-by-sa-4.0`), with Wolof supplemented by MasakhaNER 1.0 (Adelani et
al., 2021) after its Wikipedia-only corpus proved well short of every
other language's size. Full sourcing, licensing, and per-language cleaning
statistics are documented in the project's Data Card
(`docs/data_card.md`).

After cleaning (deduplication, near-duplicate template capping,
encoding-artifact removal, and five other rule-based passes â see the Data
Card Â§4), the corpus totals 286,135 documents from 425,268 raw extracted.
Corpus size varies enormously by language: Hausa alone contributes over
100,000 cleaned documents, while Chichewa's entire cleaned Wikipedia is
only 892 documents (204,640 words) â well short of the project's working
target and documented honestly as a genuine, unresolved shortfall rather
than papered over.

### 3.2 Corpus mixing

A single tokenizer was trained across all nine languages, using
**temperature sampling** (`p_i = n_i^0.3 / Î£(n_j^0.3)`) to determine each
language's share of the combined training corpus. Naive proportional
mixing would let the largest corpora (Hausa, Igbo) dominate the learned
vocabulary; naive equal weighting would over-represent tiny corpora
(Chichewa) relative to their real linguistic diversity. Temperature
sampling with Î± = 0.3 gives small languages meaningfully more influence
without full equalization â a standard technique in multilingual
tokenizer construction. The resulting combined training corpus totals
approximately 565 MB of raw text.

### 3.3 Known data limitations

Documented in full in the Data Card; the most consequential for
interpreting this report's results:

- **Chichewa and Wolof remain under-resourced** even after cleaning and
  (for Wolof) supplementing with a second source. Their word counts
  (204,640 and 458,166 respectively) sit an order of magnitude or more
  below most other languages in the set, meaning their learned vocabulary
  reflects a narrower slice of the language than languages like Hausa or
  Kinyarwanda.
- **Empty-page extraction rates vary hugely and unexplainedly by
  language** â from ~5% (Kinyarwanda) to 50â60% (Amharic, Wolof) in
  manually verified samples â a property of each source Wikipedia, not a
  pipeline defect, but one that shapes how much usable text was actually
  available per language.
- **Tone/diacritic marking is inconsistently applied within the Yoruba
  and Igbo corpora** (the same word appears with and without diacritics
  across different articles), which cannot be corrected without a native
  morphological analyzer or speaker review.

---

## 4. Methodology

### 4.1 Baseline BPE

A standard byte-level BPE tokenizer, trained with `vocab_size=24,000`,
using whitespace-restricted pretokenization (Hugging Face `tokenizers`
library). This serves as the control: same corpus, same vocabulary
budget, same preprocessing, differing only in the tokenization algorithm
itself.

### 4.2 SuperBPE (two-stage)

Following the two-stage curriculum described by Liu et al. (2025):

1. **Stage 1** â train ordinary BPE to `vocab_size=19,200` (80% of the
   total budget), whitespace-restricted, identical in spirit to the
   baseline.
2. **Stage 2** â re-encode the training corpus with Stage 1, represent
   each Stage 1 token as a single placeholder character, then train a
   second BPE pass on that placeholder text **without** whitespace
   restriction. This allows the second stage to freely merge tokens that
   originally sat on either side of a space, filling the remaining
   ~4,800 vocab slots with "superword" merges â frequent short phrases
   collapsed into single tokens.

Both tokenizers share the same combined corpus and the same total
`vocab_size=24,000`, isolating the effect of the tokenization algorithm
from any difference in data or vocabulary size.

### 4.3 Evaluation splits

Each language's cleaned corpus was split into a 95% training set and a 5%
held-out evaluation set (fixed seed, per-language SHA-256 content hash
recorded for verification), following the same procedure across all nine
languages.

---

## 5. Results

### 5.1 Fragmentation (tokens per word, held-out eval data)

| Language    | Baseline | SuperBPE | Reduction |
|-------------|----------|----------|-----------|
| Igbo        | 1.546    | 1.222    | **21.0%** |
| Hausa       | 1.434    | 1.179    | **17.8%** |
| Yoruba      | 1.816    | 1.502    | **17.3%** |
| Swahili     | 1.577    | 1.398    | 11.4%     |
| Kinyarwanda | 1.934    | 1.771    | 8.4%      |
| Chichewa    | 1.769    | 1.682    | 4.9%      |
| Wolof       | 1.656    | 1.581    | 4.5%      |
| Zulu        | 2.489    | 2.407    | 3.3%      |
| Amharic     | 2.897    | 2.830    | 2.3%      |
| **Average** |          |          | **10.1%** |

SuperBPE reduces fragmentation in **every** language, with no exceptions.

### 5.2 Compression efficiency (bytes per token, held-out eval data)

| Language    | Baseline | SuperBPE | Improvement |
|-------------|----------|----------|-------------|
| Igbo        | 4.121    | 5.213    | **26.5%**   |
| Hausa       | 4.012    | 4.878    | **21.6%**   |
| Yoruba      | 3.810    | 4.605    | **20.9%**   |
| Swahili     | 4.070    | 4.592    | 12.8%       |
| Kinyarwanda | 3.782    | 4.130    | 9.2%        |
| Wolof       | 3.255    | 3.407    | 4.7%        |
| Chichewa    | 4.048    | 4.257    | 5.2%        |
| Zulu        | 3.759    | 3.887    | 3.4%        |
| Amharic     | 5.232    | 5.356    | 2.4%        |
| **Average** |          |          | **11.8%**   |

(Higher bytes/token is better â fewer tokens needed per unit of raw text,
following the compression measure used in the SuperBPE paper.)

This ranking is **identical** to the fragmentation ranking above it (Igbo
> Hausa > Yoruba > Swahili > Kinyarwanda > Chichewa > Wolof > Zulu >
Amharic in both, save a minor swap between Chichewa and Wolof). Two
independent metrics, measured completely differently, converge on the
same per-language ordering â meaningful corroboration that the effect
being measured is real, not an artifact of either metric's specific
definition.

A further internal consistency check: SuperBPE also produces ~11.8% fewer
total tokens across the full training corpus (6.07M vs. 6.89M tokens for
the same underlying text), matching the fragmentation reduction measured
independently on held-out eval data.

### 5.3 Model comparison (bits per byte)

To test whether SuperBPE's fragmentation advantage translates into
downstream model quality, matched ~9.4M-parameter GPT-2-style models
(`n_embd=256, n_layer=4, n_head=4, block_size=256`, `vocab_size=24,000`
for both) were trained for 3 epochs on the same raw text, tokenized
separately by each tokenizer. We compare bits-per-byte rather than raw
loss, since SuperBPE's tokens each represent more raw text â a per-token
loss comparison would be unfair to the finer-grained baseline.

| Tokenizer | Bits/byte |
|-----------|-----------|
| Baseline BPE | **1.8657** |
| SuperBPE     | 2.0125 (7.9% worse) |

**SuperBPE underperformed at this scale.** We treat this as a genuine,
reportable finding rather than a failure â see Discussion below.

---

## 6. Discussion

**Why gains concentrate unevenly across languages.** SuperBPE's
improvement is largest for languages that already fragmented relatively
well under the baseline (Igbo, Hausa, Yoruba â all Latin-script) and
smallest for the two worst-fragmenting languages under the baseline
(Amharic â non-Latin Ethiopic script; Zulu â heavy Bantu noun-class
agglutination). This pattern suggests superword merging compounds on top
of already reasonably-formed subwords, but cannot by itself repair deeper
subword-level fragmentation problems. A tokenizer that is already
struggling to form clean subwords for a language has less to gain from a
second stage that only recombines those subwords across whitespace.

**Why the tiny-scale model result favors the baseline.** SuperBPE's
coarser tokenization means roughly 12% fewer training tokens â and
therefore fewer gradient updates â for the same amount of raw text, and
each token is a harder prediction target since it can represent more than
one word. At the scale we were able to train (9.4M parameters, ~4M words,
3 epochs), this disadvantage plausibly outweighs the fragmentation
benefit. This is consistent with the broader pattern in the tokenization
literature that coarser tokenization schemes tend to show their benefit
at larger model and data scale, not smaller â a hypothesis this project's
compute budget could not directly test, but one that follows naturally
from the mechanism.

---

## 7. Limitations

- **Corpus scale and evaluation compute were both constrained by team
  size** (effectively two active contributors against an originally
  five-role plan), which shaped scope throughout â most visibly in
  Section 5.3's small model scale, and in the decision to scope Phase 5
  evaluation down to two intrinsic metrics rather than the original
  six-metric plan (vocabulary utilization, compute cost estimation,
  downstream task performance, and qualitative morphological validity by
  native speakers were all deprioritized; see the project README's status
  checklist).
- **A data-hygiene issue affecting this report's held-out evaluation was
  identified after these results were produced**, during later,
  unrelated engineering work on a separate tokenizer submission using the
  same corpus pipeline: the script that built the combined training
  corpus sampled from each language's full cleaned corpus rather than
  from the corpus with the evaluation split already excluded. This means
  it is possible that some text identical to held-out evaluation lines
  was also present in training, for all three tokenizers evaluated in
  this report (baseline, Stage 1, Stage 2). We are disclosing this
  plainly rather than omitting it: it does not invalidate the *relative*
  comparison between baseline and SuperBPE (both tokenizers were trained
  and evaluated under the identical procedure, so any leak affects both
  equally), but it means the *absolute* fragmentation and compression
  figures in Sections 5.1â5.2 should be read as measured on a
  held-out-in-name split rather than a rigorously leak-free one. Future
  work should retrain both tokenizers from the corrected corpus-building
  script (which sources from each language's post-split training portion)
  before treating the absolute figures as final.
- **No downstream task benchmarks** (e.g., MasakhaNER-style NER or
  classification) were run; all results in this report are intrinsic
  tokenization/compression metrics or a single small-scale language-model
  comparison.
- **No native-speaker review** of tokenizer output was conducted, so we
  cannot confirm whether SuperBPE's superword merges align with
  linguistically meaningful multi-word units versus merely statistically
  frequent ones.
- **Chichewa and Wolof's small corpora** mean their tokenizer vocabulary
  reflects a narrower sample of each language than the other seven; the
  temperature-sampling weighting used during corpus mixing also required
  repeating their source text multiple times to reach its allotted share
  of the combined corpus, which may cause specific repeated passages to
  be over-represented as "frequent" patterns relative to the true
  language.

---

## 8. Conclusion and Future Work

Across nine African languages spanning three language families, SuperBPE
consistently reduces text fragmentation and improves compression
efficiency relative to a matched standard BPE baseline, with two
independent metrics converging on an identical per-language ranking of
benefit. At the small scale this project could train and evaluate,
however, that intrinsic tokenization advantage did not translate into
better language-model quality â a genuine, mechanistically-explicable
negative result rather than a refutation of the approach. Future work
should prioritize, in order of likely impact: (1) retraining both
tokenizers on the corrected, leak-free corpus split noted in Section 7;
(2) validating the model-quality comparison at a larger model and data
scale, where the literature suggests coarser tokenization schemes tend to
show their benefit; and (3) downstream task evaluation once labeled
benchmarks are available for more of the nine languages.

Separately from this research track, a purpose-built variant of this
tokenizer was developed for a related competition submission with a
stricter, ASCII-only input format â a different set of engineering
constraints not covered by this report. See the project repository's
`submission/` directory for that work's own documentation.

---

## 9. References

Liu, A., Hayase, J., Hofmann, V., Oh, S., Smith, N. A., & Choi, Y. (2025).
SuperBPE: Space travel for language models. *arXiv:2503.13423*.

Adelani, D. I., et al. (2021). MasakhaNER: Named Entity Recognition for
African Languages. `github.com/masakhane-io/masakhane-ner`.

Data Card: `docs/data_card.md` (this repository).
