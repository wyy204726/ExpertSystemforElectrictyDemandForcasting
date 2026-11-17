# An Efficient Expert System for Short-Term Electricity Consumption Forecasting

Official code and experimental data for the paper:

> **An Efficient Expert System for Short-Term Electricity Consumption Forecasting**
> (manuscript under review; introduces the **ECHOES** model)

## Overview

Short-term electricity consumption forecasting is difficult because real-world
data mixes stable patterns with sudden load changes and measurement noise.
**ECHOES** addresses this with two complementary ideas:

1. **Layered (hierarchical) hyperparameter optimization** — instead of tuning
   all hyperparameters of a deep network jointly, ECHOES first identifies the
   hidden layer with the greatest influence on prediction accuracy, then
   optimizes the learning rate and Dropout rates layer by layer with classical
   and evolutionary optimizers (ED, RBMO, SA, SMAC3, BO, L-BFGS-B).
2. **An expert system with time-decay weighting and clustering gating** —
   historical samples are weighted by a time-decay function (recent samples
   matter more), and K-means partitions the data into regions. A gating rule
   then selects among specialized expert predictions:
   - **Peak period** (Jan, Feb, Jul, Aug, Sep, Nov, Dec): take the **highest**
     prediction (overheating-warning oriented)
   - **Valley period** (Mar, Apr, Oct): take the **lowest** prediction
   - **Normal period** (May, Jun): take the prediction **closest to the
     cluster-center oil temperature**

Tested on a real transformer dataset (features: OLL, LULL, OT), ECHOES
outperforms eight baseline models with different mechanisms (Informer,
DLinear, PatchTST, iTransformer, TimeMixer, ModernTCN, PULSE, PhaseFormer):

| Metric | IA | R² | MSE | MAE | RMSE |
|---|---|---|---|---|---|
| ECHOES (avg of 10 runs) | **0.968** | **0.947** | **1.801** | **0.897** | **1.262** |

Robustness is further verified under Gaussian noise and simulated sudden
high/low load variations.

## Repository Structure

```
ExpertSystemforElectrictyDemandForcasting/
├── ModelHierarchicalOptimization/          # Paper Sec. 2.2 — Layered Optimization
│   ├── try_HP0_default.py                  #   Baseline: default hyperparameters (no tuning)
│   ├── try_HP1.py                          #   Layer-wise optimization, stage 1
│   ├── try_HP2.py                          #   Layer-wise optimization, stage 2
│   ├── try_HP3_BO_BFGS-B.py                #   HP3 tuned by Bayesian Optimization + L-BFGS-B
│   ├── try_HP3_RBMO_SA.py                  #   HP3 tuned by RBMO + Simulated Annealing
│   └── try_HP3_SMAC.py                     #   HP3 tuned by SMAC3
├── JointOptimizationOfResults/
│   ├── Time-DecayWeightedLongitudinalOptimization/   # Paper Sec. 2.3.2 — TDW
│   │   ├── RelevantParameterOptimization/
│   │   │   ├── Alpha_chose.py              #   Decay factor α selection (K-means based)
│   │   │   └── WindowSize_chose.py         #   Sliding window size selection
│   │   ├── DecayWeightedVisualization/
│   │   │   └── Alpha_view.py               #   Visualization of the decay weighting
│   │   └── SpecificWeightingMethod/
│   │       └── ensemble_weighted.py        #   Time-decay weighted ensemble (α = 0.9, window = 5)
│   └── ClusteringHistoricalDataforHorizontalOptimization/
│       └── LateralSelectionAbstractGatingModel.py    # Paper Sec. 2.3.3 — gating model
└── PreExperimentalDataAndOtherImportantData/
    ├── FindingAndDeterminingTheLayerWithTheGreatestInfluence/
    │   └── ANA_dropout_1/2/3_test1.xlsx    # Dropout-based layer-influence analysis
    ├── Lossfunctioncurvesofthebaselinemodels/
    │   └── baseline_1.csv ... baseline_9.csv  # Training loss curves of baseline models
    └── RepeatedExperiment.xlsx             # Results of the 10 repeated experiments
```

## Requirements

- Python ≥ 3.8
- tensorflow 2.x, scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, joblib

```bash
pip install tensorflow scikit-learn pandas numpy scipy matplotlib seaborn joblib
```

## Usage

1. **Data** — the scripts expect the ETT-format transformer dataset
    and a `model_predictions.csv`
   file containing the true values and the three expert predictions.
   Update the hard-coded `file_path` variables at the top of each script
   to point to your local data.
2. **Hierarchical optimization** — run the scripts under
   `ModelHierarchicalOptimization/` in order (`try_HP0_default` →
   `try_HP1` → `try_HP2` → one of the `try_HP3_*` variants) to reproduce
   the layered optimization comparison.
3. **Time-decay weighting** — run `Alpha_chose.py` / `WindowSize_chose.py`
   to select the decay factor and window size, then `ensemble_weighted.py`
   to produce the weighted ensemble predictions.
4. **Gating model** — `LateralSelectionAbstractGatingModel.py` loads the
   precomputed clustering results (`kmeans_model.pkl`,
   `standard_scaler.pkl`, `cluster_centers.csv`) and applies the
   peak/valley/normal gating rules to the three expert predictions.

## Citation

If you use this code or data, please cite:

```bibtex
@article{echoes2026,
  title   = {An Efficient Expert System for Short-Term Electricity Consumption Forecasting},
  journal = {Annals of the New York Academy of Sciences},
  note    = {under review},
  year    = {2026}
}
```

## License

For academic research use. Please contact the authors for other usage.
