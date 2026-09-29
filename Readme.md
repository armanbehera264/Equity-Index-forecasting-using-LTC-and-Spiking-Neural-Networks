# Directionally Consistent Equity Index Forecasting with Liquid Time-Constant and Spiking Neural Networks

Code and research paper for **"Development and Performance Evaluation of Liquid Time-Constant and Spiking Neural Network-Based Intelligent Models for Directionally Consistent Equity Index Forecasting."**

Co-authored with Shayam Ahmad, Haroon Iqbal, Adyasha Rath, and Ganapati Panda (C. V. Raman Global University / University of Edinburgh).

> **Core finding:** for daily equity index forecasting, standard point-accuracy metrics (MAE, MAPE, R²) are almost uninformative — even a zero-parameter "predict today's close" rule scores R² > 0.96. **Directional accuracy and trading outcome are what actually separate models**, and under that lens a Liquid Time-Constant (LTC) network is the most consistent performer across three indices and four forecast horizons.

---

## 1. What this project does

Eight forecasting models are built and evaluated **under one shared protocol** to predict the next close of a major equity index:

| # | Model | Family |
|---|---|---|
| 1 | Naive persistence | Zero-parameter baseline |
| 2 | Naive drift | Zero-parameter baseline |
| 3 | ARIMA(5,1,0), walk-forward | Classical statistical |
| 4 | Random Forest (200 trees) | Tree ensemble |
| 5 | LSTM (64 hidden units) | Recurrent neural network |
| 6 | **Liquid Time-Constant network (LTC)** | Continuous-time recurrent, learned time constants |
| 7 | Reservoir-style random Spiking NN (RanSNN) | Spiking, reservoir computing (only readout trained) |
| 8 | Eventprop SNN | Spiking, trained with an exact-gradient rule |

Each model is tested on **3 indices** (BSE Sensex, DJIA, S&P 500) × **4 horizons** (1, 3, 5, 7 trading days) × **10 random seeds** = a shared, harmonised protocol of 12 experiments per model, reported with mean ± standard deviation across seeds.

## 2. Why this matters

Classical forecasting literature (e.g. Majhi, Panda & Sahoo's FLANN/bacterial-foraging papers) almost always reports MAPE alone. But for a daily closing-price series, the day-to-day change is tiny relative to the price level — so **any point-accuracy metric is dominated by the level, not the move**. This paper quantifies that problem directly with an entropy-weighted TOPSIS ranking: across the twelve index/horizon combinations, R² receives an entropy-derived weight below 1% in eleven of them, while directional accuracy typically absorbs more than half the discriminating weight. The paper's contribution is therefore as much methodological (how to evaluate these models honestly) as architectural (which model wins).

## 3. Data & features

- **Source:** Yahoo Finance daily OHLCV, Jan 1 2000 – Dec 31 2025 (6,409 trading days for BSE Sensex; 6,538 for DJIA/S&P 500).
- **12 technical indicators** per day, built from close/high/low/volume:
  - EMA10, EMA20, EMA30 (exponential moving averages)
  - Accumulation/Distribution Oscillator (ADO)
  - 14-day stochastic oscillator (%K) and its 3-day average (%D)
  - RSI(9), RSI(14)
  - Price rate of change PROC(12), PROC(27)
  - 12-day close and high price acceleration
- **Windowing:** 30-day lookback window of all 12 indicators → forecast the close `h` days ahead, for `h ∈ {1, 3, 5, 7}`.
- **Target:** models are trained on the **standardized simple return** over the horizon (not raw price level), which is then converted back to a price prediction — this keeps model capacity focused on the smaller, stationary component instead of the large, non-stationary price level.
- **Split:** a single, fixed **chronological** 70/15/15 train/val/test split (no shuffling, no walk-forward re-splitting) — explicitly *not* claimed as walk-forward validation. Test window: ~Feb 2022 – Dec 2025.
- Feature and target scalers are fit **only** on the training partition to avoid leakage.

## 4. Model notes

- **LSTM** — standard single-layer gated recurrent unit (64 hidden units), linear readout on the final hidden state.
- **LTC** — continuous-time recurrent network (`ncps` library, `AutoNCP` wiring, 32 units) whose time constant is itself a learned function of input and hidden state, integrated with a closed-form ODE solver over the 30-step window.
- **RanSNN** — a frozen, randomly-initialised input projection feeding leaky integrate-and-fire (LIF) neurons; only the linear readout is trained (reservoir computing). Cheapest true sequence model, O(TH) per step.
- **Eventprop SNN** — shares LIF dynamics with RanSNN, but every parameter is trained via a custom `torch.autograd.Function` implementing an exact adjoint-sensitivity backward pass through spike-threshold crossings, rather than a smoothed surrogate gradient. Documented in the code/paper as a **discrete-time adaptation**, not a verified reproduction of the continuous-time original.
- **A ninth model was removed during development:** a FLANN-style network with a `tanh` output activation. Its (−1, 1) output range couldn't represent the standardized return targets, which routinely fall outside that range — a genuine capacity bottleneck. It was dropped rather than patched, and this negative result is reported transparently in the paper (Section 2.9 / Section 4).

All trainable models use Adam (lr = 1e-3, batch size 64, MSE loss on standardized returns, up to 60 epochs, early stopping with patience 10), and every stochastic model is run under the same 10 seeds `{42, 100, 2026, 7, 13, 99, 314, 271, 8, 21}`.

## 5. Evaluation methodology

Four layers of evaluation are applied identically to all 12 experiments:

1. **Point accuracy** — MAE, MAPE, R² (shown to be near-uninformative here, but reported for comparability with classical literature).
2. **Directional accuracy (MDA)** — fraction of days the model correctly calls the sign of the move; the metric shown to actually discriminate models.
3. **Significance tests against the Naive Drift baseline** — paired Wilcoxon signed-rank test and Diebold–Mariano test (Harvey–Leybourne–Newbold small-sample correction), run day-by-day over the single test window.
4. **Entropy-weighted TOPSIS** — multi-criteria ranking across MAE, MAPE, R², MDA, with weights derived from each criterion's Shannon entropy across models (criteria all models agree on get ~0 weight; criteria that differentiate models get most of the weight), reported under both equal and entropy-derived weighting.
5. **Trading simulation** — a simple long/short strategy (`position = sign(predicted return)`) with a 5 bps transaction cost per position change, run once per seed. Reports cumulative return, annualised Sharpe ratio, and max drawdown, benchmarked against buy-and-hold.

## 6. Headline results (1-day horizon, all three indices)

| Index | Metric that separates models | Winner |
|---|---|---|
| BSE Sensex | MDA / Sharpe | LTC (near-top, though Naive Drift's MDA is unusually high on this index) |
| DJIA | MDA (52.5%) / Sharpe (0.57) | **LTC** |
| S&P 500 | MDA (53.1%) / Sharpe | **LTC** (RanSNN close behind) |

Key patterns:
- **Point accuracy is nearly flat** — every model (including Naive persistence) scores R² > 0.96 on every index/horizon; the spread between the best and worst *trainable* model is a few hundredths of a percentage point.
- **Random Forest is the standout underperformer** economically — it regresses on a flattened, order-blind representation of the 30-day window, which appears to overfit on DJIA/S&P 500 (though it is more competitive on the thinner-liquidity BSE Sensex).
- **The LTC network's directional advantage widens, not narrows, with horizon** — e.g. on the BSE Sensex it goes from 51.2% (1-day) to 57.9% (7-day) MDA, consistent with its indicator features being built from multi-day rolling windows.
- **LTC has the lowest seed-to-seed variance** among the four neural models on every index/horizon — the most reproducible model, at the cost of being the most computationally expensive per step (same asymptotic order as LSTM, plus ODE-solver overhead).
- **No model beats simple buy-and-hold** over the 2022–2025 test window, which was a strongly appreciating regime for all three indices — a reminder that statistically significant directional skill in a daily long/short scheme is not automatically more profitable than just holding the index.

Full per-horizon, per-index tables (point accuracy, significance tests, trading simulation) are in the paper's Section 4 (1-day) and Appendix A (3/5/7-day).

## 7. Repository structure

```
├── README.md
├── paper.pdf                                    # Full research paper
├── bse_forecasting_pipeline_{1,3,5,7}day.ipynb   # BSE Sensex pipelines, per horizon
├── djia_forecasting_pipeline_{1,3,5,7}day.ipynb  # DJIA pipelines, per horizon
├── sp500_forecasting_pipeline_{1,3,5,7}day.ipynb # S&P 500 pipelines, per horizon
└── figures/                                      # Actual-vs-predicted and equity-curve plots referenced in the paper
```

Each per-index, per-horizon notebook is a self-contained pipeline (12 in total across the study) and follows the same structure — using `sp500_forecasting_pipeline_1day.ipynb` as the reference:

1. **Setup** — installs (`yfinance`, `snntorch`, `ncps`, `statsmodels`), seeding, device selection, global config (`SEEDS`, `HORIZON`, `TRANSACTION_COST_BPS = 5`, `EPOCHS = 60`).
2. **Data collection** — pulls OHLCV via `yfinance` for the relevant ticker (`^GSPC` for S&P 500) over 2000–2025.
3. **Feature extraction** — builds the 12 technical indicators (`ema`, `ado`, `stochastic`, `rsi`, `proc`, `price_acceleration` → `build_features`).
4. **Windowing** — `make_windows()` constructs 30-day lookback windows and the return/price targets for the given horizon.
5. **Chronological split** — fixed 70/15/15 split, with feature and return scalers fit only on train.
6. **Metrics** — `mape`, `directional_accuracy`, `compute_metrics`.
7. **Training harness** — `train_model()` (Adam, early stopping, records loss history) and `predict_price_deep()`.
8. **Model classes** — `LSTMRegressor`, `LNNRegressor` (wraps `ncps.torch.LTC` with `AutoNCP` wiring), `RanSNNRegressor` (frozen input layer + `snntorch.Leaky`), `EventpropSNNRegressor` with a custom `EventpropLIFFunction` (`torch.autograd.Function`) implementing the exact-gradient backward pass.
9. **Classical baselines** — `naive_drift_predict`, `arima_walk_forward_predict`, plus a scikit-learn `RandomForestRegressor`.
10. **Significance tests** — `wilcoxon_vs_baseline`, `diebold_mariano`.
11. **Trading simulation** — `trading_simulation()` (per-seed Sharpe, cumulative return, max drawdown vs. buy-and-hold).
12. **Train + evaluate everything** — runs all 8 models across the 10 seeds.
13. **Results table, training-curve plots, actual-vs-predicted plots, MAPE/MDA bar charts.**
14. **TOPSIS ranking** — `topsis()` and `entropy_weights()`.
15. **Notes** — in-notebook caveats (fixed split, seed-averaged plotting, Eventprop honesty note, FLANN removal rationale) that map directly onto the paper's stated limitations.

> Per the notebook's own versioning notes: this S&P 500 notebook is a direct clone of the confirmed-working DJIA v5 pipeline with only the ticker changed to `^GSPC` — the BSE Sensex and DJIA notebooks (and the other horizons) follow the identical structure, changing only the ticker and `HORIZON`.

## 8. Limitations (stated explicitly in the paper)

- **Single fixed chronological holdout**, not a rolling/walk-forward evaluation — all results are conditional on one ~4-year test window (Feb 2022 – Dec 2025), which happened to be a sustained bull market for all three indices.
- **Simplified trading cost model** — flat 5 bps per position change; no market impact or financing costs.
- **Eventprop SNN** is a discrete-time adaptation of the algorithm at the resolution of the 30-day window, not an independently verified reproduction of the continuous-time original.
- Confined to **3 indices** and a **technical-indicator-only** feature set — no macroeconomic or intraday signals.

## 9. Suggested future work (from the paper's conclusion)

- Rolling/expanding walk-forward evaluation across bull, bear, and range-bound regimes.
- Widen the feature set with macroeconomic and cross-index signals.
- A controlled comparison of Eventprop (exact gradient) vs. surrogate-gradient training for the spiking network, to isolate the effect of the gradient rule from the effect of the spiking representation itself.
- Reintroduce the removed bounded-activation model as a deliberate ablation study rather than an incidental development note.

## 10. Key references

- Hasani, R. et al. — *Liquid time constant networks*, AAAI 2021
- Nowotny, T., Turner, J.P., Knight, J.C. — *Loss shaping enhances exact gradient learning with Eventprop in spiking neural networks*, 2025
- Maass, W. et al. — *Real time computing without stable states*, Neural Computation, 2002
- Hwang, C.L., Yoon, K. — *Multiple Attribute Decision Making* (TOPSIS), 1981
- Diebold, F.X., Mariano, R.S. — *Comparing predictive accuracy*, 1995
- Majhi, R., Panda, G., Sahoo, G. — *FLANN-based model for forecasting of stock markets*, 2009

---

*Code availability (per the paper): the full Jupyter notebook implementation is publicly available at [github.com/shayamahmad/stock-forecasting](https://github.com/shayamahmad/stock-forecasting.git).*
