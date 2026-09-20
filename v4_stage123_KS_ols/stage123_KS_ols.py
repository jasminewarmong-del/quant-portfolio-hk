"""
快手 (1024.HK) 量化策略 - 阶段 1+2+3 (无 Lasso 版)
因子: PB Ratio, Momentum 60d, 20d Momentum, 20d Volatility,
      Volatility 60d (ann.), Relative Turnover, PE Ratio

1) 数据合并 + dropna + 时间序列 8/2 划分
2) winsorize + PolynomialFeatures(degree=2) + Standardize
3) 直接 OLS 拟合全部 35 项，输出多项式系数与各指标
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# === 配置 ===
SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_KS_ols"
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

# === 2. 多项式展开 + 标准化 ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w)
X_test_poly = poly.transform(X_test_w)
feature_names = poly.get_feature_names_out(factor_cols)
n_poly = len(feature_names)
print(f"\n[Stage 2] 多项式特征数: {n_poly} (7主效应 + 7平方 + 21交互 = 35)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_test_s = scaler.transform(X_test_poly)

# === 3. OLS 拟合全部 35 项 ===
X_train_int = sm.add_constant(X_train_s)
X_test_int = sm.add_constant(X_test_s)

ols = sm.OLS(y_train, X_train_int).fit()
print("\n[Stage 3] OLS summary:")
print(ols.summary())

y_train_pred = ols.predict(X_train_int)
y_test_pred = ols.predict(X_test_int)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 5折 CV MSE（用 sklearn 的 LinearRegression 等价于无截距 OLS，但 add_constant 后等价）
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = -cross_val_score(LinearRegression(), X_train_int, y_train,
                             cv=kf, scoring="neg_mean_squared_error")
cv_mse = float(np.mean(cv_scores))
cv_std = float(np.std(cv_scores, ddof=1))

print("\n[Stage 3] 指标:")
print(f"  Train MSE : {train_mse:.6f}")
print(f"  Test  MSE : {test_mse:.6f}")
print(f"  CV    MSE : {cv_mse:.6f}  (std={cv_std:.6f})")
print(f"  R²        : {ols.rsquared:.4f}")
print(f"  Adj R²    : {ols.rsquared_adj:.4f}")
print(f"  AIC       : {ols.aic:.2f}")
print(f"  BIC       : {ols.bic:.2f}")
print(f"  F-stat    : {ols.fvalue:.4f}  (p={ols.f_pvalue:.4e})")
print(f"  n_train   : {len(y_train)}, n_test: {len(y_test)}, n_features: {n_poly}")

# 系数表
coef_table = pd.DataFrame(
    {"feature": ["intercept"] + list(feature_names),
     "coef": ols.params, "std_err": ols.bse,
     "t": ols.tvalues, "p": ols.pvalues}
)
# 标记显著性
def sig_star(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""
coef_table["sig"] = coef_table["p"].apply(sig_star)
print("\n[Stage 3] OLS 系数表（按 |coef| 降序）:")
coef_sorted = coef_table.iloc[1:].reindex(coef_table.iloc[1:].coef.abs().sort_values(ascending=False).index)
print(coef_sorted.to_string(index=False))

# 多项式表达式
print("\n[Stage 3] 多项式（标准化空间）:")
terms = []
for i, name in enumerate(feature_names):
    terms.append(f"({ols.params[i+1]:.6f}) * {name}")
print("y ≈ " + "\n    + ".join(terms) + f"\n    + ({ols.params[0]:.6f})")

# 显著变量
sig_vars = coef_table[coef_table["p"] < 0.05].copy()
print(f"\n[Stage 3] 显著变量 (p<0.05): {len(sig_vars)} / {n_poly+1}")
if len(sig_vars) > 0:
    print(sig_vars.to_string(index=False))

# === 保存 ===
coef_table.to_csv(f"{OUT_DIR}/stage3_ols_coefficients.csv", index=False)
pd.DataFrame([{
    "method": "ols_full",
    "train_mse": train_mse, "test_mse": test_mse, "cv_mse": cv_mse, "cv_std": cv_std,
    "r2": ols.rsquared, "r2_adj": ols.rsquared_adj,
    "aic": ols.aic, "bic": ols.bic,
    "f_stat": ols.fvalue, "f_pvalue": ols.f_pvalue,
    "n_features": n_poly, "n_train": len(y_train), "n_test": len(y_test),
}]).to_csv(f"{OUT_DIR}/stage3_metrics.csv", index=False)
print(f"\n[Saved] 结果文件位于 {OUT_DIR}/")
