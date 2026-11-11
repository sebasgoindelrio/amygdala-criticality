# Surprise Is Not Importance: Criticality Read from Hidden States Gates Sleep-Like Consolidation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24131189.svg)](https://doi.org/10.5281/zenodo.24131189)

**Author:** [Sebastian Goin del Rio](https://orcid.org/0009-0006-1147-9995) · Independent researcher, Switzerland · [sebastiangoindelrio@gmail.com](mailto:sebastiangoindelrio@gmail.com)

## Abstract

A model that writes new facts into a LoRA adapter every night forgets what it does not rehearse, so a fixed replay budget is a choice of what matters. The usual guides are surprise and detected forgetting. This work tests a third signal: **criticality**—whether losing a fact would bear on health, safety, money, or legal standing—read by a small probe from the frozen model's own residual stream. In `Qwen2.5-0.5B-Instruct` the probe ranks critical facts on a synthetic 100-fact benchmark at AUROC 0.99; answer-loss surprise scores 0.53. Used as the gate of ten nights of replay, the **Amygdala** probe retains substantially more critical facts than random replay, surprise, or replaying forgotten facts, while staying near a label oracle on three seeds. Controls address lexical shortcuts, streaming quotas, layer depth, ablations, and out-of-template generalization. Full methods, limits, and figures are in the [Zenodo preprint](https://doi.org/10.5281/zenodo.24131189) and in `paper/`.

## What is in this repository

| Path | Description |
|------|-------------|
| `paper/amygdala-manuscript.pdf` | Author's PDF of the preprint (CC-BY-4.0 via Zenodo) |
| `paper/amygdala-manuscript.tex` | LaTeX source (figures not bundled here; see Zenodo for full submission archive) |
| `CITATION.cff` | Machine-readable citation metadata |
| `LICENSE` | MIT license for repository scaffolding and future code |

**Coming soon:** cleaned experiment scripts, Kalandor-style synthetic data generators, and Colab/Kaggle notebooks used in the paper. Until then, cite the Zenodo record for the scientific artifact.

## How to cite

Please cite the Zenodo preprint (canonical version):

```bibtex
@misc{goin_del_rio_2026_surprise,
  author       = {Goin del Rio, Sebastian},
  title        = {Surprise Is Not Importance: Criticality Read from Hidden States Gates Sleep-Like Consolidation},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24131189},
  url          = {https://doi.org/10.5281/zenodo.24131189}
}
```

If you use this GitHub repository as a software entry point, also cite the repository URL and note the Zenodo DOI above. `CITATION.cff` is provided for tools that read [Citation File Format](https://citation-file-format.github.io/).

## License

- **Manuscript and figures:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) on [Zenodo](https://doi.org/10.5281/zenodo.24131189).
- **This repository (metadata and future code):** [MIT](LICENSE); see `LICENSE` for the full text and the CC-BY note for the paper.

## Contact

Questions and collaboration: **sebastiangoindelrio@gmail.com** · ORCID [0009-0006-1147-9995](https://orcid.org/0009-0006-1147-9995)
