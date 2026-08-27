# Bridging the Tokenization Gap in African LLMs via SuperBPE

Comparing **SuperBPE** (Liu et al., 2025) against a standard BPE baseline across
9 African languages — measuring whether "superword" tokenization meaningfully
reduces text fragmentation, and whether that improvement translates into
better model performance.

**Program:** TRI AI Saturdays
**Status:** Phases 1–4 complete. Phase 5 (evaluation) in progress.

---

## Why this matters

Most tokenizers used in large language models are trained mostly on English
text. When they encounter morphologically rich or agglutinative African
languages, they fragment words into small, often meaningless pieces instead
of the coherent morphemes a native speaker would recognize. This inflates
token counts (raising compute cost) and degrades model quality (the model
learns from broken pieces instead of stable units).

This project builds a tokenization pipeline trained directly on native
African-language text, and tests whether SuperBPE's approach — learning
ordinary subwords first, then lifting the whitespace restriction to learn
frequent multi-word "superwords" — reduces fragmentation compared to a
matched standard BPE baseline.

## Languages

Amharic (am) · Hausa (ha) · Igbo (ig) · Chichewa/Nyanja (ny) ·
Kinyarwanda (rw) · Swahili (sw) · Wolof (wo) · Yoruba (yo) · Zulu (zu)

Spanning three language families (Niger-Congo/Bantu, Afro-Asiatic,
Niger-Congo/Volta-Niger) to test whether results generalize rather than
overfitting to one morphological pattern. See
[`docs/data_card.md`](./docs/data_card.md) for corpus documentation,
cleaning methodology, and known limitations (currently a stub — the full
Data Card lives in Google Drive, see `docs/drive_structure.md`).

## Methodology

### Combined training corpus (temperature sampling, α = 0.3)

Corpus sizes across the 9 languages vary by over 200x (Hausa: 46M words vs.
Chichewa: 205K words). Naive proportional mixing would let large corpora
dominate the learned vocabulary; naive equal weighting would over-represent
tiny corpora. We use temperature sampling — `p_i = n_i^0.3 / Σ(n_j^0.3)` — to
give small languages meaningfully more influence without full equalization.

### Baseline BPE

Standard byte-level BPE, `vocab_size=24,000`, whitespace-restricted
pretokenization (Hugging Face `tokenizers` library).

### SuperBPE (two-stage)

1. **Stage 1** — train ordinary BPE to `vocab_size=19,200` (80% of budget),
   whitespace-restricted, same as the baseline approach.
2. **Stage 2** — re-encode the corpus with Stage 1, represent each Stage 1
   token as a single placeholder character, then train a second BPE pass
   on that placeholder text **without** whitespace restriction — free to
   merge tokens that originally sat on either side of a space. This fills
   the remaining ~4,800 vocab slots with "superword" merges.

Both tokenizers share the same combined corpus and the same total
`vocab_size=24,000`, isolating the effect of the tokenization algorithm
itself.

## Results

### Fragmentation (tokens per word, held-out eval data)

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
Gains are largest for languages that already fragmented relatively well
under the baseline (Igbo, Hausa, Yoruba — all Latin-script) and smallest for
the two worst-fragmenting languages (Amharic — non-Latin Ethiopic script;
Zulu — heavy Bantu noun-class agglutination). This suggests superword
merging compounds on top of reasonably-formed subwords, but can't by itself
fix deeper subword-level fragmentation problems.

A consistency check: SuperBPE also produces ~11.8% fewer total tokens across
the training corpus (6.07M vs. 6.89M tokens for the same text), matching the
fragmentation reduction measured independently on eval data.

### Model comparison (bits per byte, tiny GPT-2-style model)

To test whether SuperBPE's fragmentation advantage translates into better
model performance, we trained matched ~9.4M-parameter GPT-2-style models
(`n_embd=256, n_layer=4, n_head=4, block_size=256`, `vocab_size=24,000` for
both) for 3 epochs on the same raw text, tokenized separately by each
tokenizer. We compare **bits-per-byte** rather than raw loss, since
SuperBPE's tokens each represent more raw text — a raw per-token loss
comparison would be unfair.

| Tokenizer | Bits/byte |
|-----------|-----------|
| Baseline BPE | **1.8657** |
| SuperBPE     | 2.0125 (7.9% worse) |

**SuperBPE underperformed at this tiny scale.** Its coarser tokenization
means ~12% fewer training tokens/gradient updates for the same raw text,
and each token is a harder prediction target — a disadvantage that likely
outweighs the fragmentation benefit when the model and dataset are this
small and undertrained (9.4M params, ~4M words, 3 epochs). This is
consistent with the broader pattern that coarse tokenization schemes tend
to show benefits at larger model/data scale, not smaller. We treat this as
a genuine, reportable finding rather than a failure — full details in
[`results/model_comparison_results.json`](./results/model_comparison_results.json).

## Repository structure

```
.
├── README.md
├── LICENSE
├── docs/
│   ├── data_card.md              # corpus sourcing, cleaning, per-language stats (stub)
│   └── drive_structure.md        # expected Google Drive layout for large artifacts
├── scripts/                      # Phase 1 data pipeline (01_download_and_extract.sh -> 05_convert_masakhaner_to_corpus.py)
├── notebooks/
│   └── phase2_3_tokenizer_training.ipynb   # Phase 2 (baseline BPE) + Phase 3 (SuperBPE)
├── results/
│   ├── baseline_fragmentation_results.json
│   ├── superbpe_fragmentation_results.json
│   └── model_comparison_results.json
└── .gitignore                    # excludes large corpus/tokenizer files (kept in Drive)
```

Large data files (combined corpus, tokenizer `.json` files, trained model
weights) are intentionally excluded from git and kept in Google Drive —
see `docs/drive_structure.md` for the exact expected paths.

## Status / next steps

- [x] Phase 1 — Data engineering (corpus collection, cleaning, eval splits)
- [x] Phase 2 — Baseline BPE tokenizer, trained and validated
- [x] Phase 3 — SuperBPE tokenizer, trained and validated
- [x] Phase 4 — Model integration: matched tiny language models trained on
      each tokenizer's output, compared via bits-per-byte (SuperBPE 7.9%
      worse at this scale — see results above)
- [ ] Phase 5 — Evaluation (scoped down from the original 6-metric plan to
      1–2 lightweight metrics, prioritizing a strong final report over thin
      coverage):
  - [ ] Compression efficiency (bytes/token) per language — in progress
  - [ ] Vocabulary utilization — optional/lower priority
  - [ ] Compute cost estimate — optional/lower priority
  - [ ] Downstream task performance — likely out of scope (would need
        labeled benchmarks, e.g. MasakhaNER-style tasks)
  - [ ] Qualitative morphological validity — likely out of scope (no
        native-speaker reviewer currently available)
- [ ] Phase 6 — Final report and codebase cleanup

## References

Liu, A., Hayase, J., Hofmann, V., Oh, S., Smith, N. A., & Choi, Y. (2025).
SuperBPE: Space travel for language models. *arXiv:2503.13423*.
