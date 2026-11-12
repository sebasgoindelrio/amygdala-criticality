# Amygdala: criticality-gated replay for nightly LoRA consolidation

Two preprints on deciding what a continually learning language model should replay, so that the facts that would hurt to forget are the ones it keeps.

**Author:** [Sebastian Goin del Rio](https://orcid.org/0009-0006-1147-9995) · Independent researcher, Switzerland · [sebastiangoindelrio@gmail.com](mailto:sebastiangoindelrio@gmail.com)

| | Paper | DOI |
|---|---|---|
| 1 | Surprise Is Not Importance: Criticality Read from Hidden States Gates Sleep-Like Consolidation | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24164569.svg)](https://doi.org/10.5281/zenodo.24164569) |
| 2 | Ask What Forgetting Would Cost: A Label-Free Judge Gates Nightly Replay of Free-Text Facts | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24164503.svg)](https://doi.org/10.5281/zenodo.24164503) |

## Paper 2 — Ask What Forgetting Would Cost

Can a model decide what is worth remembering without anyone telling it what matters? On 150 free-text facts in Spanish and English, a 1.5B judge asked once per fact whether forgetting it could cause harm (no human labels) gates fifteen nights of LoRA replay. In `Qwen2.5-0.5B-Instruct` it retains 19.6 of 38 critical facts against 8.8 for random replay and 11.0 for forgotten-first replay (five seeds, exact McNemar p = 1.2e-8 and 3.3e-6), with the largest gain on implicit critical facts; a replication with `Llama-3.2-1B-Instruct` gives 22.0 against 11.7. A keyword selector marks 12 traps as salient where the judge marks none.

Everything needed to check or rerun it is in [`paper2/`](paper2/): the facts, the Colab scripts, the complete logs with per-fact outcomes, and `analysis/verify_results.py`, which recomputes every number in the paper. See [`paper2/README.md`](paper2/README.md).

## Paper 1 — Surprise Is Not Importance

A model that writes new facts into a LoRA adapter every night forgets what it does not rehearse, so a fixed replay budget is a choice of what matters. The usual guides are surprise and detected forgetting. This work tests a third signal: **criticality**—whether losing a fact would bear on health, safety, money, or legal standing—read by a small probe from the frozen model's own residual stream. In `Qwen2.5-0.5B-Instruct` the probe ranks critical facts on a synthetic 100-fact benchmark at AUROC 0.99; answer-loss surprise scores 0.53. Used as the gate of ten nights of replay, the **Amygdala** probe retains substantially more critical facts than random replay, surprise, or replaying forgotten facts, while staying near a label oracle on three seeds. Controls address lexical shortcuts, streaming quotas, layer depth, ablations, and out-of-template generalization. Full methods, limits, and figures are in the [Zenodo preprint](https://doi.org/10.5281/zenodo.24164569) and in [`paper/`](paper/).

## What is in this repository

| Path | Description |
|------|-------------|
| `paper/amygdala-manuscript.pdf` | Author's PDF of paper 1 (CC-BY-4.0 via Zenodo) |
| `paper/amygdala-manuscript.tex` | LaTeX source (figures not bundled here; see Zenodo for full submission archive) |
| `paper2/` | Paper 2: manuscript, data, scripts, logs and verification code |
| `CITATION.cff` | Machine-readable citation metadata |
| `LICENSE` | MIT license for repository scaffolding and future code |

**Coming soon (paper 1):** cleaned experiment scripts, Kalandor-style synthetic data generators, and Colab/Kaggle notebooks used in the paper. Until then, cite the Zenodo record for the scientific artifact.

## How to cite

Please cite the Zenodo preprints. Paper 1 (version 1: doi:10.5281/zenodo.24131189; complete version 2 with data: doi:10.5281/zenodo.24164569):

```bibtex
@misc{goin_del_rio_2026_surprise,
  author       = {Goin del Rio, Sebastian},
  title        = {Surprise Is Not Importance: Criticality Read from Hidden States Gates Sleep-Like Consolidation},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24164569},
  url          = {https://doi.org/10.5281/zenodo.24164569}
}
```

Paper 2:

```bibtex
@misc{goin_del_rio_2026_forgetting,
  author       = {Goin del Rio, Sebastian},
  title        = {Ask What Forgetting Would Cost: A Label-Free Judge Gates Nightly Replay of Free-Text Facts},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24164503},
  url          = {https://doi.org/10.5281/zenodo.24164503}
}
```

If you use this GitHub repository as a software entry point, also cite the repository URL and note the Zenodo DOIs above. `CITATION.cff` is provided for tools that read [Citation File Format](https://citation-file-format.github.io/).

## License

- **Manuscripts, figures and data:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) on Zenodo ([paper 1](https://doi.org/10.5281/zenodo.24164569), [paper 2](https://doi.org/10.5281/zenodo.24164503)).
- **Code in this repository:** [MIT](LICENSE); see `LICENSE` for the full text and the CC-BY note for the paper.

## Contact

Questions and collaboration: **sebastiangoindelrio@gmail.com** · ORCID [0009-0006-1147-9995](https://orcid.org/0009-0006-1147-9995)
