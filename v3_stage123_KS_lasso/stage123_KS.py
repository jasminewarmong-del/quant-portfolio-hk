"""
快手 (1024.HK) 量化策略 - 阶段 1+2+3
因子: PB Ratio, Momentum 60d, 20d Momentum, 20d Volatility,
      Volatility 60d (ann.), Relative Turnover, PE Ratio
（Paying User Growth 89%缺失、Amihud 全0 已剔除）

1) 数据合并 + dropna + 时间序列 8/2 划分
2) winsorize + PolynomialFeatures(degree=2) + Standardize
   + LassoCV(5折) + 1-SE rule + 选中频率筛选
3) 用选中变量做 OLS 重拟合，输出多项式系数与各指标
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# === 配置 ===
SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_KS"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = [
    "PB Ratio",
    "Momentum 60d",
    "20d Momentum",
    "20d Volatility",
    "Volatility 60d (ann.)",
    "Relative Turnover",
    "PE Ratio",
]
SHORT = {
    "y (5d Log Return)": "y",
    "PB Ratio": "PB",
    "Momentum 60d": "M60",
    "20d Momentum": "M20",
    "20d Volatility": "V20",
    "Volatility 60d (ann.)": "V60",
    "Relative Turnover": "RT",
    "PE Ratio": "PE",
}
FREQ_THRESH = 3 / 5
RANDOM_STATE = 42

# === 1. 数据合并 + dropna + 时间序列划分 ===
sheet1 = pd.read_excel(SRC, sheet_name="Sheet1")
sheet2 = pd.read_excel(SRC, sheet_name="Sheet2")
df = pd.merge(
    sheet1[["Trade Date", "y (5d Log Return)"]],
    sheet2[["Trade Date"] + FACTORS], on="Trade Date",
).rename(columns=SHORT)
factor_cols = [SHORT[c] for c in FACTORS]

df = df.dropna().reset_index(drop=True)
print(f"[Stage 1] 样本数（dropna后）: {len(df)}")
print(f"[Stage 1] 时间范围: {df['Trade Date'].min().date()} -> {df['Trade Date'].max().date()}")

n = len(df)
n_train = int(n * 0.8)
train = df.iloc[:n_train].copy()
test = df.iloc[n_train:].copy()
print(f"[Stage 1] Train: {len(train)}  ({train['Trade Date'].min().date()} -> {train['Trade Date'].max().date()})")
print(f"[Stage 1] Test : {len(test)}  ({test['Trade Date'].min().date()} -> {test['Trade Date'].max().date()})")

X_train_raw = train[factor_cols].values
X_test_raw = test[factor_cols].values
y_train = train["y"].values
y_test = test["y"].values

# Winsorize 1%/99%（仅在 train 上拟合分位）
lows = np.quantile(X_train_raw, 0.01, axis=0)
highs = np.quantile(X_train_raw, 0.99, axis=0)
X_train_w = np.clip(X_train_raw, lows[None, :], highs[None, :])
X_test_w = np.clip(X_test_raw, lows[None, :], highs[None, :])
print(f"[Stage 1] Winsorize 下界: {dict(zip(factor_cols, np.round(lows, 4)))}")
print(f"[Stage 1] Winsorize 上界: {dict(zip(factor_cols, np.round(highs, 4)))}")

# === 2. 多项式 + 标准化 + LassoCV 5折 ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w)
X_test_poly = poly.transform(X_test_w)
feature_names = poly.get_feature_names_out(factor_cols)
n_poly = len(feature_names)
print(f"\n[Stage 2] 多项式特征数: {n_poly} (7主效应 + 7平方 + 21交互 = 35)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_test_s = scaler.transform(X_test_poly)

# LassoCV 5折选 alpha（TimeSeriesSplit，避免时间序列泄露）
kf = TimeSeriesSplit(n_splits=5)
lasso_cv = LassoCV(alphas=np.logspace(-5, 1, 200), cv=kf, random_state=RANDOM_STATE, max_iter=200000)
lasso_cv.fit(X_train_s, y_train)
min_alpha = lasso_cv.alpha_
print(f"[Stage 2] LassoCV 最优 alpha (CV MSE最小): {min_alpha:.6f}")
print(f"[Stage 2] CV 方式: TimeSeriesSplit(n_splits=5) — 时间序列顺序验证，无 shuffle")

# 1-SE rule
mean_per_alpha = np.mean(lasso_cv.mse_path_, axis=1)
std_per_alpha = np.std(lasso_cv.mse_path_, axis=1, ddof=1)
best_idx = int(np.argmin(mean_per_alpha))
cv_mse_min = float(mean_per_alpha[best_idx])
std_at_min = float(std_per_alpha[best_idx])
threshold = cv_mse_min + std_at_min
print(f"[Stage 2] 最优alpha下: CV MSE={cv_mse_min:.6f}, std={std_at_min:.6f}, 阈值={threshold:.6f}")

alphas_sorted = lasso_cv.alphas_
feasible = mean_per_alpha <= threshold
candidate_alphas = alphas_sorted[feasible]
best_alpha = float(np.max(candidate_alphas))
print(f"[Stage 2] 1-SE rule 选定 alpha: {best_alpha:.6f}")

# 整训练集 Lasso
lasso_full = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
lasso_full.fit(X_train_s, y_train)
cv_mse = cv_mse_min
n_nonzero = int((np.abs(lasso_full.coef_) > 1e-10).sum())
print(f"[Stage 2] 整训练集 Lasso 非零变量数: {n_nonzero} / {n_poly}")

# 5折选中频率
sel_count = np.zeros(n_poly, dtype=int)
for tr_idx, _ in kf.split(X_train_s):
    m = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    sel_count += (np.abs(m.coef_) > 1e-10).astype(int)

freq_table = pd.DataFrame(
    {
        "feature": feature_names,
        "selection_count": sel_count,
        "frequency": sel_count / 5,
        "full_lasso_coef": lasso_full.coef_,
    }
).sort_values(["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)
print("\n[Stage 2] 5折选中频率表（按频率降序）:")
print(freq_table.to_string(index=False))

# 选中变量：整训练集非零 AND 5折>=3/5
full_nonzero_mask = np.abs(lasso_full.coef_) > 1e-10
freq_mask = (sel_count / 5) >= FREQ_THRESH
combined_mask = full_nonzero_mask & freq_mask
selected = [feature_names[i] for i in range(n_poly) if combined_mask[i]]

only_full = int((full_nonzero_mask & ~freq_mask).sum())
only_freq = int((~full_nonzero_mask & freq_mask).sum())
both = int(combined_mask.sum())
neither = int((~full_nonzero_mask & ~freq_mask).sum())
print(f"\n[Stage 2] 判据统计（整训练集非零 AND 5折>=3/5）:")
print(f"  两个条件都满足 (保留): {both}")
print(f"  仅整训练集非零 (剔除): {only_full}")
print(f"  仅5折>=3/5 (剔除): {only_freq}")
print(f"  两个条件都不满足 (剔除): {neither}")
print(f"[Stage 2] 最终选中变量 ({len(selected)}): {selected}")

# 如果 Lasso 选中太多（>15），切换到 Ridge top-k
use_ridge = False
if len(selected) > 15 or len(selected) == 0:
    use_ridge = True
    print(f"\n[Stage 2] Lasso 选中 {len(selected)} 个变量，切换到 Ridge top-10 方案")
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 5, 200), cv=kf)
    ridge_cv.fit(X_train_s, y_train)
    ridge_alpha = ridge_cv.alpha_
    print(f"[Stage 2] RidgeCV 最优 alpha: {ridge_alpha:.6f}")
    ridge_full = Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)
    ridge_full.fit(X_train_s, y_train)
    ridge_coef = ridge_full.coef_
    abs_coef = np.abs(ridge_coef)
    top10_idx = set(np.argsort(abs_coef)[::-1][:10].tolist())
    cv_mse_scores = -cross_val_score(Ridge(alpha=ridge_alpha), X_train_s, y_train,
                                     cv=kf, scoring="neg_mean_squared_error")
    cv_mse = float(np.mean(cv_mse_scores))
    print(f"[Stage 2] Ridge CV MSE: {cv_mse:.6f}")
    # 5折稳定性
    sel_count_r = np.zeros(n_poly, dtype=int)
    for tr_idx, _ in kf.split(X_train_s):
        m = Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)
        m.fit(X_train_s[tr_idx], y_train[tr_idx])
        fold_top = set(np.argsort(np.abs(m.coef_))[::-1][:10].tolist())
        for i in fold_top:
            sel_count_r[i] += 1
    full_top_mask = np.array([i in top10_idx for i in range(n_poly)])
    freq_mask_r = (sel_count_r / 5) >= FREQ_THRESH
    combined_mask = full_top_mask & freq_mask_r
    selected = [feature_names[i] for i in range(n_poly) if combined_mask[i]]
    freq_table = pd.DataFrame(
        {"feature": feature_names, "ridge_coef": ridge_coef, "abs_coef": abs_coef,
         "topk_count": sel_count_r, "frequency": sel_count_r / 5}
    ).sort_values(["frequency", "abs_coef"], ascending=[False, False]).reset_index(drop=True)
    print("\n[Stage 2] Ridge 5折 top-10 稳定性表:")
    print(freq_table.to_string(index=False))
    print(f"\n[Stage 2] Ridge 最终选中变量 ({len(selected)}): {selected}")

selected_idx = [list(feature_names).index(f) for f in selected]

# === 3. OLS 重拟合 ===
X_train_sel = X_train_s[:, selected_idx]
X_test_sel = X_test_s[:, selected_idx]
X_train_sel_int = sm.add_constant(X_train_sel)
X_test_sel_int = sm.add_constant(X_test_sel)

ols = sm.OLS(y_train, X_train_sel_int).fit()
print("\n[Stage 3] OLS summary:")
print(ols.summary())

y_train_pred = ols.predict(X_train_sel_int)
y_test_pred = ols.predict(X_test_sel_int)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n[Stage 3] 指标:")
print(f"  Train MSE : {train_mse:.6f}")
print(f"  Test  MSE : {test_mse:.6f}")
print(f"  CV    MSE : {cv_mse:.6f}")
print(f"  R²        : {ols.rsquared:.4f}")
print(f"  Adj R²    : {ols.rsquared_adj:.4f}")
print(f"  AIC       : {ols.aic:.2f}")
print(f"  BIC       : {ols.bic:.2f}")
print(f"  n_train   : {len(y_train)}, n_test: {len(y_test)}, n_features: {len(selected)}")

coef_table = pd.DataFrame(
    {"feature": ["intercept"] + selected, "coef": ols.params,
     "std_err": ols.bse, "t": ols.tvalues, "p": ols.pvalues}
)
print("\n[Stage 3] OLS 系数表:")
print(coef_table.to_string(index=False))

print("\n[Stage 3] 多项式（标准化空间）:")
terms = [f"{ols.params[0]:.6f}"]
for i, name in enumerate(selected):
    terms.append(f"({ols.params[i+1]:.6f}) * {name}")
print("y ≈ " + "\n    + ".join(terms[1:]) + f"\n    + {terms[0]}")

# === 保存 ===
freq_table.to_csv(f"{OUT_DIR}/stage2_selection_frequency.csv", index=False)
coef_table.to_csv(f"{OUT_DIR}/stage3_ols_coefficients.csv", index=False)
pd.DataFrame([{
    "method": "ridge" if use_ridge else "lasso_1se",
    "best_alpha": best_alpha if not use_ridge else ridge_alpha,
    "cv_mse": cv_mse,
    "train_mse_ols": train_mse,
    "test_mse_ols": test_mse,
    "r2": ols.rsquared, "r2_adj": ols.rsquared_adj,
    "aic": ols.aic, "bic": ols.bic,
    "n_features_selected": len(selected),
    "n_train": len(y_train), "n_test": len(y_test),
    "freq_threshold": FREQ_THRESH,
}]).to_csv(f"{OUT_DIR}/stage3_metrics.csv", index=False)
print(f"\n[Saved] 结果文件位于 {OUT_DIR}/")
