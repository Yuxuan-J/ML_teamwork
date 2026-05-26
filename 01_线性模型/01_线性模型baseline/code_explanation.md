# 线性模型 Baseline 代码解读

## 总览

本 notebook 实现**逐日截面 OLS + Walk-Forward 预测**基线，用于量化多因子选股。核心思路：每天用当日因子值做 OLS 回归得到系数 β_t，β 序列经 EWM 平滑后，用 **t-1 日的 β** 预测 **t 日的收益**，严格杜绝未来信息泄漏。

```
230 因子 → [IC 筛选 76] → [共线去重 38] → 逐日 OLS β → EWM 平滑 → shift(1) 预测 → IC 评估
```

---

## Step 0: 参数配置

```python
FOLD_ID         = 1        # fold_1: train 2016-2019, test 2020 H1
LABEL_COL       = "FWD_RET_10D"  # 10日前向收益
WINSORIZE_MAD   = 3.0      # 3 倍 MAD 去极值
IC_THRESHOLD    = 0.02     # |IC| > 0.02
T_STAT_THRESHOLD= 2.0      # |t| > 2
CORR_THRESHOLD  = 0.7      # |ρ| > 0.7 视为共线
EWM_HALFLIFE    = 20       # β 平滑半衰期 (交易日)
```

所有阈值集中在顶部，方便调参对比。

---

## Step 1: 函数定义 (14 个)

全部函数含 `输入/输出` docstring。分四组：

| 组 | 函数 | 作用 |
|----|------|------|
| 数据 | `load_fold` | 读 CSV，按 trade_date+ts_code 排序 |
| 数据 | `identify_columns` | 自动识别 `ALPHA_*` 特征 / `FWD_RET_*` 标签 |
| 预处理 | `cross_section_winsorize` | 逐日 MAD 截尾 |
| 预处理 | `cross_section_demean` | 逐日去均值 (中性化降级) |
| 预处理 | `cross_section_zscore` | 逐日标准化 |
| 筛选 | `compute_ic_panel` | 逐日 Spearman IC 面板 |
| 筛选 | `summarize_ic` | IC mean/std/t/pos_ratio |
| 筛选 | `filter_by_ic` | 阈值过滤 |
| 筛选 | `spearman_union_find` | 共线性并查集分组 |
| 筛选 | `select_representatives` | 每组选 \|IC\| 最大 |
| 建模 | `daily_ols_beta` | 逐日 `np.linalg.lstsq` |
| 建模 | `smooth_beta_ewm` | `ewm(halflife=20).mean()` |
| 预测 | `predict_walkforward` | `X[t] @ β_ewm[t-1]` |
| 评估 | `evaluate_ic` | 日度 Spearman + 汇总 |

---

## Step 2: 数据加载

```
fold_1_train.csv → train_df: 282,719 行, 976 日 (2015-12-31 ~ 2019-12-31)
fold_1_test.csv  → test_df:   32,331 行, 108 日 (2020-01-16 ~ 2020-07-01)
```

230 个因子 (`ALPHA_1` ~ `ALPHA_230`)，标签 `FWD_RET_10D`。训练集和测试集标签缺失率均为 0%（fold 切分时已 dropna）。

---

## Step 3: 截面预处理 (三步流水线)

**train 和 test 各自独立处理**，不跨集混合——这是防泄漏第一关。

### 3a. MAD Winsorize (去极值)

```python
median = g[cols].median()
mad    = (g[cols] - median).abs().median() * 1.4826   # 1.4826 = 正态修正系数
lo, hi = median ± 3 * mad
g[cols] = g[cols].clip(lo, hi)
```

逐日分组 (`groupby("trade_date")`)，每日截面对每个因子独立计算 MAD 上下界并截断。3 倍 MAD 对非正态分布比 Z-score (3σ) 更稳健。

### 3b. Demean (中性化降级)

数据无 `industry`/`size`/`mkt_cap` 列，无法做标准行业+市值中性化。退而求其次：每列减当日截面均值，消除截面水平偏移。后续 Lasso/Ridge 版本可补充行业映射表做真正中性化。

### 3c. Z-score (标准化)

`(x - mean) / std`，`ddof=0`（总体标准差）。标准差为 0 的列替换为 1，避免除零。

---

## Step 4: IC 初筛

```python
ic_panel  = compute_ic_panel(train_df, feature_cols, label_col)   # 976日 × 230因子
ic_summary = summarize_ic(ic_panel)   # mean/std/t/pos_ratio
kept_ic    = filter_by_ic(ic_summary, ic_th=0.02, t_th=2.0)
```

**只用训练集**计算。每日对每个因子与 `FWD_RET_10D` 算 Spearman 秩相关，得 IC 时间序列。汇总后：

- `ic_mean`: 日均 IC
- `t_stat`: `ic_mean / ic_std * sqrt(n)` — 衡量 IC 稳定性
- 过滤: `|ic_mean| > 0.02` **且** `|t_stat| > 2.0`

**结果**：230 → 76 个因子通过。

---

## Step 5: 共线性去重 (Union-Find)

```python
groups      = spearman_union_find(train_df, kept_ic, corr_th=0.7)
kept_final  = select_representatives(groups, ic_summary)
```

### 算法

1. 计算 76 个因子间 **Spearman 相关矩阵** (`|ρ|`)
2. 对角置 0，`|ρ| > 0.7` 的因子对连边
3. **并查集** (Union-Find) 找连通分量，每组即共线因子簇
4. 每组选 `|ic_mean|` 最大者为代表

### 为什么用 Spearman 而非 Pearson

因子截面分布常非正态，Spearman 基于秩，对厚尾/偏态稳健。O(n²) 复杂度的并查集 + 路径压缩确保即使 230 因子也能秒出。

**结果**：76 → 38 个因子。最大一组 13 个因子共线，代表为 `ALPHA_94`。

---

## Step 6: 逐日截面 OLS

```python
beta_train = daily_ols_beta(train_df, x_cols, label_col)  # 976日 × (38+1)
beta_test  = daily_ols_beta(test_df,  x_cols, label_col)  # 108日 × (38+1)
```

每日切一片截面数据 `X_t (N_t × K)`, `y_t (N_t,)`，`np.linalg.lstsq` 求解 `β_t = (XᵀX)⁻¹Xᵀy`。自动加截距列 (`add_const=True`)。

**为什么 train 和 test 都跑 OLS？** 训练 β 是 EWM 的主体，测试 β 为 EWM 提供末尾若干天的增量——但预测时用的是 `shift(1)` 滞后值，所以测试日的 β 不会泄漏到当天预测。

**安全过滤**：`mask.sum() < X.shape[1] + 5` 时跳过该日（样本数不足，OLS 不稳定）。

**结果**：976 训练日 + 108 测试日 = 1084 天 β 序列。

---

## Step 7: EWM 平滑 + Shift (Walk-Forward 核心)

```python
beta_all      = pd.concat([beta_train, beta_test]).sort_index()
beta_ewm      = beta_all.ewm(halflife=20, adjust=False).mean()
beta_for_pred = beta_ewm.shift(1)   # ← 关键!!
```

### 为什么需要 EWM

逐日 OLS 的 β_t 噪声极大（单日截面估计不稳定）。EWM（指数加权移动平均）用历史 β 的加权平均替代当日 β：

```
β_ewm[t] = α * β[t] + (1-α) * β_ewm[t-1]
α = 1 - exp(-ln(2) / halflife) ≈ 0.034  (halflife=20)
```

半衰期 20 天意味着 ~20 天前的 β 权重衰减到一半。

### shift(1) 的作用

`shift(1)` 让 t 日能用的 β 是 **t-1 日及之前** 的 EWM 值。这是 walk-forward 的灵魂——预测时点 t 看不到 t 日本身的任何信息（包括 t 日的 OLS β）。

### 为什么 EWM 而非 SMA

EWM 无需固定窗口长度，不会在窗口边界产生跳变，且计算更轻量。

---

## Step 8: Walk-Forward 预测

```python
pred_df = predict_walkforward(test_df, x_cols, beta_for_pred)
# 对每个测试日 d:
#   y_pred[d] = X[d] @ β_for_pred[d]
# β_for_pred[d] = β_ewm[d-1]  ← t-1 日信息
```

### NaN 处理

部分股票在截面预处理后特征仍含 NaN（如整列 constant 导致 zscore 为 NaN），`predict_walkforward` 中 `np.isfinite` 未显式过滤——`X @ b` 会自动产生 NaN 预测值。评估阶段 `dropna()` 处理这些行。

**结果**：32,331 行预测，108 个测试日，y_pred 范围 [-0.080, 0.104]。

---

## Step 9: IC 评估

```python
ic_series, summary = evaluate_ic(pred_df)
# 每日: spearmanr(y_pred, y_true)
# 汇总: ic_mean / ic_std / ic_ir / pos_ratio / n_days
```

### IC IR 公式

```
IC_IR = (ic_mean / ic_std) * sqrt(252 / 10)
```

- `252/10`: 年化因子。10 日调仓周期，一年约 25.2 个调仓周期
- IR > 0.5 通常认为因子有选股能力

**结果**：

| 指标 | 值 |
|------|-----|
| ic_mean | 0.1357 |
| ic_std | 0.1204 |
| ic_ir | **5.66** |
| pos_ratio | 87.96% |
| n_days | 108 |

IC 均值 0.136、正 IC 占比 88%、IR 5.66——基线模型选股能力显著。

---

## Step 10: 持久化输出

5 个文件写入 `output/20260526/`：

| 文件 | 内容 | 形状 |
|------|------|------|
| `fold_1_y_pred.parquet` | trade_date, ts_code, y_pred, y_true | 32,331 × 4 |
| `fold_1_ic_series.csv` | 日度 IC | 108 行 |
| `fold_1_summary.csv` | 6 项汇总指标 | 1 行 |
| `fold_1_kept_factors.txt` | 保留因子名 | 38 行 |
| `fold_1_beta_history.parquet` | 全区间 β (平滑前) | 1084 × 39 |

---

## 防泄漏设计总结

| 环节 | 防泄漏措施 |
|------|-----------|
| 预处理 | train/test **各自独立** winsorize/demean/zscore，不跨集 |
| IC 筛选 | **只用 train** 计算 IC，保留列表传到 test |
| 共线去重 | **只用 train** 计算相关矩阵+分组 |
| OLS 拟合 | train/test 分别逐日拟合，但只用 `β_ewm[t-1]` 预测 |
| EWM + shift | `shift(1)` 确保 t 日看不到 t 日信息 |
| OLS 残差 | β 拟合用全量 train，残差仅用于 EWM，不参与预测 |
