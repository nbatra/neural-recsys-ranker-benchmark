# DLRM vs DCN-v2 vs XGBoost: Three-Way Ranker Comparison

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

> A lightweight comparison project that loads pre-computed results from [DLRM_Recommendation](https://github.com/nbatra/dlrm-recsys) and [DCN-v2_Recommendation](https://github.com/nbatra/dcn-v2-recsys), performing a unified three-way statistical comparison of all re-ranking models.

## Key Results

### Accuracy (Mean NDCG@10, Two-Tower Retrieval Pipeline)

| Rank | Model | NDCG@10 | vs XGBoost | FDR-Significant Metrics |
|------|-------|---------|------------|------------------------|
| 1 | DLRM | 0.6664 | +0.74% | 7/8 |
| 2 | DCN-v2 | 0.6631 | +0.23% | 1/8 |
| 3 | XGBoost | 0.6615 | baseline | -- |

### Inference Latency (200 candidates, CPU, P99)

| Model | P50 | P99 | Budget (< 5ms) |
|-------|-----|-----|----------------|
| XGBoost | 0.042 ms | 0.054 ms | PASS |
| DCN-v2 | 0.295 ms | 0.389 ms | PASS |
| DLRM | 0.665 ms | 0.832 ms | PASS |

### Recommendation

Deploy **DLRM** as the production re-ranker. It achieves the highest accuracy with 7/8 FDR-significant improvements over XGBoost, while DCN-v2 only achieves 1/8. DLRM's field-aware dot-product architecture better exploits the natural feature grouping (retrieval, user, item, cross features). All models meet latency budgets.

## Methodology

- **Paired evaluation**: All three models score identical FAISS-retrieved candidates per user
- **Significance testing**: Paired t-tests with Benjamini-Hochberg FDR correction (alpha=0.05)
- **Bootstrap CIs**: 10,000 resamples for 95% confidence intervals
- **Effect sizes**: Cohen's d for practical significance assessment
- **Latency**: Isolated sub-process benchmarks (500 warmup + 1000 timed trials)

## Project Structure

```
DLRM_DCN_XGBoost_Ranker_Comparison/
  notebooks/
    01_three_way_comparison.ipynb   -- Main comparison notebook
  plots/
    three_way_comparison_ci.png     -- Pairwise CI comparison plot
    latency_comparison.png          -- Latency bar chart
  per_user_metrics.npz              -- Pre-computed per-user metrics (236 KB)
  latency_results.json              -- Pre-computed latency benchmarks
  predictions.npz                   -- Raw model predictions (22 MB)
  .venv/                            -- Python 3.13 virtual environment
```

## Setup

```bash
cd DLRM_DCN_XGBoost_Ranker_Comparison
# Ensure sibling projects exist with trained models:
#   ../DLRM_Recommendation/models/dlrm_ranker.pt
#   ../DCN-v2_Recommendation/models/dcn_v2_ranker.pt
#   ../DLRM_Recommendation/models/xgboost_ranker.json

# Create venv and install dependencies
uv venv --python 3.13 .venv
uv pip install numpy scipy matplotlib ipykernel nbconvert

# Run notebook (uses pre-computed data, no GPU needed)
.venv/bin/jupyter lab notebooks/
```

## Dependencies

The notebook itself only requires numpy, scipy, and matplotlib (no torch or xgboost needed at runtime since metrics are pre-computed). The pre-computation scripts require torch and xgboost from the sibling project venvs.

## Dataset

MovieLens 25M -- 2.16M test samples across 4,006 users, evaluated on FAISS-retrieved candidate sets of ~200 items per user.

## Key Takeaways

1. **DLRM is the clear winner for this feature set.** Its field-aware dot-product interactions explicitly model cross-field relationships that the tree-based approach only approximates through recursive splits.

2. **DCN-v2's polynomial cross layers offer marginal gains.** The Cross Network's learned weight matrices provide little advantage over XGBoost when the feature space is relatively small (105 dims) and well-engineered.

3. **All models meet production latency budgets with room to spare.** The ranking stage is not the bottleneck -- retrieval (FAISS search) and feature assembly dominate end-to-end latency.

4. **Statistical rigor separates real gains from noise.** FDR correction is essential when testing multiple metrics simultaneously -- it prevents promoting a model based on cherry-picked significant results.

---

## Author

Built by **Nipun Batra**

[![GitHub](https://img.shields.io/badge/GitHub-nbatra-181717?logo=github)](https://github.com/nbatra)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nipunbatra-0A66C2?logo=linkedin)](https://www.linkedin.com/in/nipunbatra/)

---

## License

This project is released for educational and portfolio purposes. The MovieLens 25M dataset is provided by [GroupLens Research](https://grouplens.org/) under their own terms of use.

<details>
<summary>Keywords</summary>

DLRM, DCN-v2, Deep & Cross Network, Deep Learning Recommendation Model, XGBoost LambdaMART, Learning to Rank, Model Comparison, Recommendation Ranking, Neural Ranking, Re-Ranking, Feature Interactions, Dot-Product Architecture, Cross Network, Benjamini-Hochberg, FDR Correction, Bootstrap Confidence Intervals, Cohen's d, Effect Size, Statistical Testing, A/B Testing, Latency Benchmarking, MovieLens 25M, RecSys, Production ML, Inference Latency, NDCG, Ranking Metrics

</details>
