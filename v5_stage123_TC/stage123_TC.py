"""
携程 (9961.HK) 阶段 1+2+3 - TimeSeriesSplit 版
因子: PB Ratio, PE Ratio, ROE (%), 20d Momentum, MAU Growth (YoY %)

1) 数据合并 + dropna + 时间序列 8/2 划分
2) winsorize + PolynomialFeatures(degree=2) + Standardize
   + RidgeCV(TimeSeriesSplit) + top-10 稳定性筛选
3) OLS 重拟合
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# === 配置 ===
SRC = "/workspace/.uploads/35657faa-847c-4c1d-b404-5b5165eb83ec_TC_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_TC"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = ["PB Ratio", "PE Ratio", "ROE (%)", "20d Momentum", "MAU Growth (YoY %)"]
SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "PE Ratio": "PE",
         "ROE (%)": "ROE", "20d Momentum": "M20", "MAU Growth (YoY %)": "MAUg"}
FREQ_THRESH = 3 / 5

# === 1. 数据准备 ===
print("=" * 70)
print("携程 (TC) Stage 1-3 (TimeSeriesSplit)")
print("=" * 70)

s1 = pd.read_excel(SRC, sheet_name="Sheet1")
s2 = pd.read_excel(SRC, sheet_name="Sheet2")

df = pd.merge(
    s1[["Trade Date", "y (5d Log Return)", "Close (HKD)"]],
    s2[["Trade Date"] + FACTORS],
    on="Trade Date"
).rename(columns=SHORT)

factor_cols = [SHORT[c] for c in FACTORS]
print(f"\n[Stage 1] 原始样本: {len(df)}")
print(f"各因子缺失:\n{df[factor_cols + ['y']].isnull().sum().to_string()}")

df = df.dropna().reset_index(drop=True)
n = len(df)
n_train = int(n * 0.8)
train = df.iloc[:n_train].copy()
test = df.iloc[n_train:].copy()

X_train_raw = train[factor_cols].values
X_test_raw = test[factor_cols].values
y_train = train["y"].values
y_test = test["y"].values

print(f"\n[Stage 1] dropna 后样本: {n}")
print(f"  Train: {len(train)} ({train['Trade Date'].iloc[0].date()} -> {train['Trade Date'].iloc[-1].date()})")
print(f"  Test:  {len(test)} ({test['Trade Date'].iloc[0].date()} -> {test['Trade Date'].iloc[-1].date()})")

# winsorize (1%/99% 分位裁剪，参数只用 train)
lows = np.quantile(X_train_raw, 0.01, axis=0)
highs = np.quantile(X_train_raw, 0.99, axis=0)
X_train_w = np.clip(X_train_raw, lows[None, :], highs[None, :])
X_test_w = np.clip(X_test_raw, lows[None, :], highs[None, :])

# PolynomialFeatures(degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w)
X_test_poly = poly.transform(X_test_w)
feat_names = poly.get_feature_names_out(factor_cols)
n_poly = len(feat_names)
print(f"\n[Stage 1] 多项式特征数: {n_poly}")

# Standardize (多项式特征)
scaler_p = StandardScaler()
X_train_s = scaler_p.fit_transform(X_train_poly)
X_test_s = scaler_p.transform(X_test_poly)

# 另存一份原始因子标准化 (给泰勒替换用)
scaler_raw = StandardScaler()
X_train_raw_s = scaler_raw.fit_transform(X_train_w)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_idx = {f: i for i, f in enumerate(factor_cols)}

tss = TimeSeriesSplit(n_splits=5)

# === 2. Ridge top-10 变量选择（固定 alpha） ===
print("\n" + "=" * 70)
print("[Stage 2] Ridge top-10 稳定性筛选 (固定 alpha=1.0, TimeSeriesSplit)")
print("=" * 70)

# 固定 alpha（不再用 RidgeCV 自动选）
ridge_alpha = 1.0
print(f"Ridge alpha = {ridge_alpha:.6f} (固定)")

# 整训练集 Ridge 拟合
ridge_full = Ridge(alpha=ridge_alpha).fit(X_train_s, y_train)
top10_full = set(np.argsort(np.abs(ridge_full.coef_))[-10:])

# 5折稳定性
sel_count = np.zeros(n_poly, dtype=int)
for tr_idx, _ in tss.split(X_train_s):
    m = Ridge(alpha=ridge_alpha)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    for i in np.argsort(np.abs(m.coef_))[-10:]:
        sel_count[i] += 1

# 联合判据
combined = [(i in top10_full) and (sel_count[i] / 5 >= FREQ_THRESH)
           for i in range(n_poly)]
selected = [feat_names[i] for i in range(n_poly) if combined[i]]
print(f"Ridge top-10 选出 {len(selected)} 个变量:")
for v in selected:
    print(f"  {v}")

# 保存频率表
freq_table = pd.DataFrame({
    "feature": feat_names,
    "ridge_coef": ridge_full.coef_,
    "abs_coef": np.abs(ridge_full.coef_),
    "topk_count": sel_count,
    "frequency": sel_count / 5,
    "in_top10_full": [i in top10_full for i in range(n_poly)],
    "selected": combined,
}).sort_values(["frequency", "abs_coef"], ascending=[False, False]).reset_index(drop=True)
freq_table.to_csv(f"{OUT_DIR}/stage2_selection_frequency.csv", index=False)

# === 3. OLS 重拟合 ===
print("\n" + "=" * 70)
print("[Stage 3] OLS 重拟合")
print("=" * 70)

X_train_sel = X_train_s[:, [list(feat_names).index(v) for v in selected]]
X_test_sel = X_test_s[:, [list(feat_names).index(v) for v in selected]]

X_train_int = sm.add_constant(X_train_sel, has_constant='add')
X_test_int = sm.add_constant(X_test_sel, has_constant='add')
ols = sm.OLS(y_train, X_train_int).fit()

# CV MSE
cv_scores = -cross_val_score(LinearRegression(), X_train_sel, y_train,
                             cv=tss, scoring="neg_mean_squared_error")
cv_mse = float(np.mean(cv_scores))
cv_se = float(np.std(cv_scores, ddof=1) / np.sqrt(5))

# Test
y_test_pred = ols.predict(X_test_int)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Train
y_train_pred = ols.predict(X_train_int)
train_mse = mean_squared_error(y_train, y_train_pred)

# AIC/BIC
n_obs = len(y_train)
rss = np.sum((y_train - y_train_pred) ** 2)
k = len(selected) + 1
aic = float(ols.aic)
bic = float(ols.bic)

print(f"\n变量数: {len(selected)}")
print(f"Train R²  = {ols.rsquared:.4f}")
print(f"Adj R²    = {ols.rsquared_adj:.4f}")
print(f"Train MSE = {train_mse:.6f}")
print(f"Test MSE  = {test_mse:.6f}")
print(f"Test R²   = {test_r2:.4f}")
print(f"CV MSE    = {cv_mse:.6f} ± {cv_se:.6f}")
print(f"AIC       = {aic:.2f}")
print(f"BIC       = {bic:.2f}")

# 系数表
coef_table = pd.DataFrame({
    "feature": ["intercept"] + selected,
    "coef": ols.params,
    "std_err": ols.bse,
    "t": ols.tvalues,
    "p": ols.pvalues,
}).sort_values("p").reset_index(drop=True)
coef_table.to_csv(f"{OUT_DIR}/stage3_ols_coefficients.csv", index=False)

print(f"\n系数表 (按 p 值升序):")
print(coef_table.to_string(index=False))

# 指标汇总
metrics = pd.DataFrame({
    "metric": ["n_features", "n_train", "n_test", "train_mse", "test_mse",
               "test_r2", "cv_mse", "cv_se", "r2", "adj_r2", "aic", "bic"],
    "value": [len(selected), len(train), len(test), train_mse, test_mse,
              test_r2, cv_mse, cv_se, ols.rsquared, ols.rsquared_adj, aic, bic]
})
metrics.to_csv(f"{OUT_DIR}/stage3_metrics.csv", index=False)

print(f"\n[Saved] {OUT_DIR}/")
