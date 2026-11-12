# Ask What Forgetting Would Cost — reproducibility package

Companion files for the preprint *Ask What Forgetting Would Cost: A Label-Free Judge Gates Nightly Replay of Free-Text Facts* (S. Goin del Rio, 2026), Zenodo, doi:10.5281/zenodo.24164503.

## Contents

| Folder | File | What it is |
|---|---|---|
| `data/` | `facts_150.txt` | The 150 facts: `exam_idx|label|category|language|sentence|question|key(;variants)`. Labels are used only for evaluation. |
| `scripts/` | `e21b_qwen_seed0.py` | Qwen2.5-0.5B, seed 0: gate + Random / Oracle, then Forgotten / Judge. |
| | `e21b_qwen_seeds1-4.py` | Qwen2.5-0.5B, seeds 1–4: Random, Forgotten, Judge (lr 2e-4 fixed by the seed-0 gate). |
| | `e21c_llama_seeds0-2.py` | Llama-3.2-1B replication: Random vs Judge, seeds 0–2, same pre-fixed lr ladder. |
| | `e20_judge_selection.py` | Earlier comparison that chose the judge (direct P(Yes) vs "dreaming"). |
| `logs/` | `*.log` | Complete run logs. Each ends with a CSV whose `aciertos` column is the per-fact outcome (150 characters, 1 = retained). The Qwen seeds 1–4 log and the Llama log also contain the start of an interrupted earlier session; only complete runs appear in the final CSV. |
| `analysis/` | `verify_results.py` | Recomputes per-seed counts, means and the exact McNemar tests from the logs. |
| | `make_figure.py` | Draws Figure 1. |
| | `e22_keyword_baseline.py`, `e22_keyword_preregistration.md` | Keyword selector (selection stage only) and its word list and rule, fixed before computing. |
| | `e22b_judge_salient_split.py` | Re-scores the facts with the judge (GPU, no training) and prints the explicit/implicit split of its salient facts. |
| `paper/` | `amygdala2.tex`, `amygdala2.pdf` | The manuscript. |

## Running

The scripts are self-contained (facts embedded) and were run on Google Colab with one L4 GPU.
Cell 1: `!pip install -q peft` (the scripts uninstall an incompatible `torchao` themselves); cell 2: the script.
Runs resume from per-run JSON files if interrupted. Qwen runs take about 34 min each, Llama runs about 22 min.
The Llama weights are loaded from the public mirror `unsloth/Llama-3.2-1B-Instruct`.

Script comments and log messages are in Spanish. Key terms: *azar* = Random, *olvido* = Forgotten-first,
*juez* = Judge, *oráculo* = Oracle, *críticos* = critical facts, *triviales* = trivial, *trampas* = traps,
*cultura* = general-knowledge check, *puerta* = gate, *semilla* = seed, *noche* = night.

To check the numbers: `cd analysis && python verify_results.py` (needs numpy and scipy).

## License

The Zenodo record releases everything under CC BY 4.0. In this repository the code is also available under the repository's MIT license; data, logs and manuscript remain CC BY 4.0.
