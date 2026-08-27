# Data Card — TRI-Project: African Tokenization (SuperBPE vs. BPE)

> **Status: stub.** The full Data Card (per-language raw/cleaned document
> counts, detailed cleaning reports, licensing notes) was produced during
> Phase 1 but currently lives only as `DATA_CARD.md` in the project's
> Google Drive folder (see `docs/drive_structure.md`) — it was never copied
> into this repo. This file captures the parts referenced elsewhere in the
> repo so those links aren't dangling; replace/expand it with the full
> Drive version when convenient.

## Languages and sources

9 African languages, sourced primarily from Wikipedia dumps, cleaned via
`scripts/01_download_and_extract.sh` → `scripts/04_holdout_split.py`:

Amharic (am) · Hausa (ha) · Igbo (ig) · Chichewa/Nyanja (ny) ·
Kinyarwanda (rw) · Swahili (sw) · Wolof (wo) · Yoruba (yo) · Zulu (zu)

### Section 2.1 — Wolof: Wikipedia + MasakhaNER

Wolof Wikipedia alone was too small to be usable, so it was supplemented
with the MasakhaNER dataset (NER-tagged text, tags stripped and rejoined
into plain sentences via `scripts/05_convert_masakhaner_to_corpus.py`).
This is the only language with a combined source, which is why its cleaned/
training folder is named `wo_combined/` rather than `wo_wikipedia/` — see
`docs/drive_structure.md` for the resulting path inconsistency.

## Cleaning pipeline

Applied uniformly per language (`scripts/03_clean_corpus.py`):

1. **Minimum word count filter** — a 20-document manual spot check
   (`scripts/02_spot_check_sample.py`) found 6/20 (30%) extracted
   "articles" were empty stubs (date pages, infobox-only pages); these are
   dropped.
2. **Exact deduplication** — Wikipedia dumps can contain redirect artifacts
   or duplicate snapshots; articles are hashed and exact repeats dropped.
3. (See `scripts/03_clean_corpus.py` docstring for the remaining rules —
   not yet transcribed here.)

## Held-out evaluation split

5% of each language's cleaned corpus, seed=42, held out **before** any
tokenizer/model training touches the data (`scripts/04_holdout_split.py`),
stored under a top-level `eval_holdout/` folder specifically so it can't be
accidentally swept into a later "grab everything in cleaned/" step. See
`docs/drive_structure.md` for the exact expected paths, including the
Wolof naming inconsistency (`wo_combined/` for training data vs.
`eval_holdout/wo_wikipedia/` for the eval split).

## Known limitations

- Document counts (raw extracted / final cleaned) are **not** a proxy for
  corpus size in words or bytes — corpus sizes vary by over 200x across
  languages (Hausa ~46M words vs. Chichewa ~205K words). Always measure
  actual words/bytes for anything size-sensitive (e.g. temperature
  sampling), not document counts.
- No native-speaker morphological review has been done on the cleaned
  corpora or tokenizer outputs (see README Phase 5 status — likely scoped
  out for this iteration, noted as a limitation in the final report).
