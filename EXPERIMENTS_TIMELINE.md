# Experiment timeline (chronological)

Each row is one Kubeflow run, in the order it was launched. Committed commits marked ⭐.

| # | val_loss | name | summary |
|---|---|---|---|
| 1 | **0.09042** ⭐ | lstm_ema | EMA(decay=0.999) of weights, evaluated as best — first commit |
| 2 | 0.09069 | lstm_ema_smooth | plain CosineAnnealingLR (no warm restart) |
| 3 | **0.09027** ⭐ | lstm_ema_t20 | T_0=20, T_mult=2 (vs T_0=10) |
| 4 | 0.09034 | lstm_ema_d9995 | EMA decay=0.9995 |
| 5 | collapsed | lstm_ema_wide | hidden_dim=256 |
| 6 | 0.09365 | lstm_ema_wd | Adam weight_decay=1e-4 |
| 7 | 0.09160 | lstm_ema_clip | grad clip max_norm=1.0 |
| 8 | collapsed | lstm_ema_drop05 | dropout=0.5 |
| 9 | 0.09119 | lstm_ema_drop03 | dropout=0.3 |
| 10 | **0.08994** ⭐ | lstm_ema_lr003 | lr=0.003 (vs 0.005); first sub-0.09 result |
| 11 | 0.09085 | lstm_ema_lr002 | lr=0.002 |
| 12 | 0.09030 | lstm_ema_t30 | T_0=30 |
| 13 | 0.09192 | lstm_ema_inoise | input Gaussian noise (sigma=0.05) on numericals |
| 14 | 0.09138 | lstm_ema_bz256 | batch_size=256 |
| 15 | 0.09126 | lstm_ema_multi3 | train on last 3 timesteps' MAE |
| 16 | 0.09025 | lstm_ema_postnorm | LayerNorm after LSTM, before linear |
| 17 | 0.09052 | lstm_ema_adamw | AdamW, weight_decay=0.01 |
| 18 | collapsed | lstm_ema_nodrop | dropout=0 |
| 19 | 0.09153 | lstm_ema_3layer | num_layers=3 |
| 20 | **0.08934** ⭐ | lstm_ema_embnorm | Embedding(max_norm=1.0) on top of EMA + T_0=20 + lr=0.003 |
| 21 | 0.08976 | lstm_ema_emb05 | embedding max_norm=0.5 (tighter) |
| 22 | 0.09002 | lstm_ema_emb2 | embedding max_norm=2.0 |
| 23 | 0.09098 | lstm_ema_emb15 | embedding max_norm=1.5 |
| 24 | 0.09027 | lstm_ema_init | orthogonal LSTM init + forget bias=1 |
| 25 | 0.09044 | lstm_ema_d998 | EMA decay=0.998 |
| 26 | 0.09121 | lstm_ema_embp1 | larger embedding dim (+1 to log formula) |
| 27 | 0.08979 | lstm_ema_ar | activation regularization on LSTM output |
| 28 | collapsed | lstm_ema_diff | in-model first-difference features |
| 29 | 0.09081 | lstm_ema_fe | sequence-level lag/diff/rolling/EWMA (7 features) |
| 30 | 0.09091 | lstm_ema_fe3 | simpler FE (lag1+diff1+EWMA only) |
| 31 | 0.09049 | lstm_ema_agg | aggregate FE (terminal/channel/product mean) |
| 32 | 0.09068 | lstm_ema_cyclic | sin/cos cyclical month-of-year features |
| 33 | 0.09039 | lstm_ema_2lstm | 2 parallel LSTMs, hidden states averaged |
| 34 | 0.09296 | lstm_ema_huber | smooth_l1_loss (Huber, beta=0.1) for training |
| 35 | 0.09003 | lstm_ema_ens10 | 10 parallel LSTMs, predictions averaged |
| 36 | 0.09089 | lstm_ema_h64 | hidden_dim=64 |
| 37 | 0.09123 | gru_ema | GRU instead of LSTM |
| 38 | failed | lstm_ema_attn | softmax attention pool over LSTM outputs |
| 39 | 0.09091 | lstm_ema_lr0035 | lr=0.0035 |
| 40 | 0.09046 | lstm_ema_la | Lookahead optimizer (k=6, alpha=0.5) |
| 41 | failed | lstm_ema_2state | per-epoch ensemble of EMA + current weights |
| 42 | collapsed | lstm_ema_mc | MC Dropout eval (5 forward passes with dropout on) |
| 43 | 0.09054 | lstm_ema_residual | softplus(linear(y) + num) — input residual |
| 44 | 0.09028 | lstm_ema_drop035 | dropout=0.35 |
| 45 | collapsed | tcn_ema | dilated TCN replacing LSTM |
| 46 | collapsed | lstm_ema_ffn | residual FFN block on LSTM output |
| 47 | 0.09110 | lstm_ema_1layer | num_layers=1 |
| 48 | 0.09017 | lstm_ema_long | patience=25, max_epochs=80 |
| 49 | collapsed | lstm_ema_elu | ELU+1 activation instead of softplus |
| 50 | collapsed | lstm_ema_seed42 | seed=42 (revealed init dead-zone fragility) |
| 51 | 0.08948 | lstm_ema_warmup | + LinearLR warmup over 2 epochs (start_factor=0.1) |
| 52 | collapsed | lstm_ema_bz128 | batch_size=128 |
| 53 | 0.09018 | lstm_ema_bz1024 | batch_size=1024 |
| 54 | collapsed | lstm_ema_lr0028 | lr=0.0028 |
| 55 | collapsed | lstm_ema_clip10 | grad clip max_norm=10.0 |
| 56 | collapsed | lstm_ema_embsel | selective max_norm (only for cardinality > 50) |
| 57 | 0.09123 | lstm_ema_bias1 | linear bias init = 1.0 |
| 58 | 0.08971 | lstm_ema_nosoftplus | remove softplus on output, raw linear |
| 59 | 0.10122 | lstm_ema_volw | volatility-weighted training loss |
| 60 | 0.08973 | lstm_ema_scale | learnable per-target output scale parameter |
| 61 | collapsed | lstm_ema_h160 | hidden_dim=160 |
| 62 | 0.09074 | lstm_ema_init01 | xavier_uniform_(gain=0.1) on linear |
| 63 | 0.08948 | lstm_ema_warm1 | 1-epoch warmup (start_factor=0.3) |
| 64 | 0.09046 | lstm_ema_embinit | embedding init Normal(0, 0.1) |
| 65 | 0.08992 | lstm_ema_lr001 | lr=0.001 |
| 66 | 0.09108 | lstm_ema_t15 | T_0=15 |
| 67 | collapsed | lstm_ema_rdrop | R-Drop (consistency loss between two dropout passes) |
| 68 | 0.08985 | lstm_ema_t25 | T_0=25 |
| 69 | 0.09090 | lstm_ema_clip5 | grad clip max_norm=5.0 |
| 70 | 0.09148 | lstm_ema_meanpool | mix of last-step and mean-pooled LSTM output |
| 71 | 0.09002 | lstm_ema_mixup | mixup on numericals+targets, alpha=0.4 |
| 72 | 0.09061 | lstm_ema_mixup02 | mixup alpha=0.2, max_epochs=80 |
| 73 | 0.09864 | lstm_ema_manimix | manifold mixup on LSTM hidden states |
| 74 | 0.08977 | lstm_ema_onecycle | OneCycleLR scheduler, pct_start=0.2 |
| 75 | 0.08988 | lstm_ema_d9985 | EMA decay=0.9985 |
| 76 | collapsed | lstm_ema_freqgrad | Embedding(scale_grad_by_freq=True) |
| 77 | 0.08967 | lstm_ema_quantile | predict 3 quantiles (.25/.5/.75) with pinball loss, output median |
| 78 | collapsed | lstm_ema_2head | 2-layer MLP output head |
| 79 | failed | lstm_ema_bestfinal | end-of-training ensemble (best-EMA + final-EMA) v1 |
| 80 | collapsed | lstm_ema_bestfinal2 | end-of-training ensemble v2 (training itself collapsed) |
| 81 | 0.09030 | lstm_ema_embnorm_v2 | sanity rerun of HEAD config — exposed ±0.001 run-to-run noise |

**Pattern of progression:**
- Runs 1–10: rapid early wins from EMA, scheduler tuning, and lr — three commits inside the first 10 experiments.
- Runs 11–20: explored regularization, optimizer, and architecture variations; only `lstm_ema_embnorm` (run 20) committed.
- Runs 21–50: increasingly creative architectural and loss experiments, plus first feature-engineering attempts; nothing improved over 0.08934. Collapse failures start appearing as configs perturb the seed-fragile init.
- Runs 51–80: hyperparameter fine-tuning, ensembling, and exotic losses (mixup, quantile, R-Drop). Best near-misses around 0.0895–0.0897 but no further commit.
- Run 81: re-running the HEAD config produced 0.09030, revealing that the headline 0.08934 was within run-to-run variance.
