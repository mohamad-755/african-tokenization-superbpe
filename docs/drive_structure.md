# Expected Google Drive structure

The scripts in `scripts/` and notebooks in `notebooks/` expect the
following layout under `TRI-Project/` in Google Drive. This data is not
included in the git repo (see `.gitignore`) — see `data_card.md` for full
sourcing and cleaning documentation.

```
TRI-Project/
├── am_wikipedia/
│   ├── cleaned/
│   │   ├── corpus.txt              # cleaned training text for this language
│   │   └── cleaning_report.md      # output of 03_clean_corpus.py
│   ├── train/
│   │   └── corpus.txt              # train split (post eval-holdout)
│   └── spot_check_report_am.md     # manual spot-check notes (02_spot_check_sample.py)
│
├── ha_wikipedia/            (same structure as am_wikipedia)
├── ig_wikipedia/            (same structure)
├── ny_wikipedia/            (same structure)
├── rw_wikipedia/            (same structure)
├── sw_wikipedia/            (same structure)
├── yo_wikipedia/            (same structure)
├── zu_wikipedia/            (same structure)
│
├── wo_combined/             # Wolof: note this is wo_combined, not wo_wikipedia --
│   ├── cleaned/             # combines Wikipedia + MasakhaNER (see data_card.md, Section 2.1)
│   └── train/
│
├── eval_holdout/            # top-level folder, NOT nested inside each language folder
│   ├── am_wikipedia/
│   │   └── eval.txt
│   ├── ha_wikipedia/
│   │   └── eval.txt
│   ├── ig_wikipedia/
│   │   └── eval.txt
│   ├── ny_wikipedia/
│   │   └── eval.txt
│   ├── rw_wikipedia/
│   │   └── eval.txt
│   ├── sw_wikipedia/
│   │   └── eval.txt
│   ├── wo_wikipedia/        # note: eval folder uses wo_wikipedia, unlike the training data above
│   │   └── eval.txt
│   ├── yo_wikipedia/
│   │   └── eval.txt
│   └── zu_wikipedia/
│       └── eval.txt
│
├── combined_corpus.txt              # Phase 2: temperature-sampled (α=0.3) combined training text
├── stage2_meta_corpus.txt           # Phase 3: placeholder-character corpus for Stage 2 training
│
├── baseline_bpe_tokenizer.json      # Phase 2 output, vocab_size=24000
├── superbpe_stage1_tokenizer.json   # Phase 3 Stage 1 output, vocab_size=19200
├── superbpe_stage2_raw.json         # Phase 3 Stage 2 output, vocab_size=24000
│
├── model_train_text.txt             # Phase 4: training subset (19,000 lines)
├── model_eval_text.txt              # Phase 4: held-out subset (1,000 lines)
│
├── DATA_CARD.md
└── Data_Card_African_T...           # (original uploaded copy)
```

## Known naming inconsistency (documented, not a bug)

Wolof's folder name differs between the training-data side and the
eval-holdout side:
- Cleaned/training data: `wo_combined/` (reflects the combined
  Wikipedia + MasakhaNER source — see `data_card.md` Section 2.1)
- Eval holdout: `eval_holdout/wo_wikipedia/`

This was flagged during Phase 2/3 as a path-handling gotcha worth keeping
in mind when writing code that loops over all 9 languages by a single
naming pattern.

## Reproducing from scratch

1. Run `scripts/01_download_and_extract.sh` through `05_convert_masakhaner_to_corpus.py`
   in order to rebuild the raw/cleaned corpora for each language (Phase 1).
2. Run the Phase 2/3 notebook (`notebooks/phase2_3_tokenizer_training.ipynb`)
   to rebuild `combined_corpus.txt` and both tokenizers.
3. Run the Phase 4 notebook to rebuild `model_train_text.txt` /
   `model_eval_text.txt` and train the comparison models.