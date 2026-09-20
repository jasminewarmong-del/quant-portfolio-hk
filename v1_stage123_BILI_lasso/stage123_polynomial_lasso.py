"""
Bilibili (9626.HK) 量化策略 - 阶段 1+2+3
1) 数据合并 + dropna + 时间序列 8/2 划分
2) winsorize + PolynomialFeatures(degree=2) + Standardize + LassoCV(5折) + 选中频率筛选
3) 用选中变量做 OLS 重拟合，输出多项式系数与各指标
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# === 配置 ===
SRC = "/workspace/.uploads/d4d06402-aa6f-40b0-b7c8-c4d694a048eb_BILI_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = [
    "PB Ratio",
    "Momentum 120d",
    "DAU Growth (YoY %)",
    "MAU Growth (YoY %)",
    "Momentum 60d",
    "Rel. Strength vs HSTECH",
]
SHORT = {
    "y (5d Log Return)": "y",
    "PB Ratio": "PB",
    "Momentum 120d": "M120",
    "DAU Growth (YoY %)": "DAUg",
    "MAU Growth (YoY %)": "MAUg",
    "Momentum 60d": "M60",
    "Rel. Strength vs HSTECH": "RS",
}
FREQ_THRESH = 3 / 5  # 5折中至少3折被选中
RANDOM_STATE = 42

# === 1. 数据合并 + dropna + 时间序列划分 ===
sheet1 = pd.read_excel(SRC, sheet_name="Sheet1")
sheet2 = pd.read_excel(SRC, sheet_name="Sheet2")
df = pd.merge(
    sheet1[["Trade Date", "y (5d Log Return)"]],
    sheet2[["Trade Date"] + FACTORS],
    on="Trade Date",
).rename(columns=SHORT)
factor_cols = [SHORT[c] for c in FACTORS]

df = df.dropna().reset_index(drop=True)
print(f"[Stage 1] 样本数（dropna后）: {len(df)}")
print(f"[Stage 1] 时间范围: {df['Trade Date'].min().date()} -> {df['Trade Date'].max().date()}")

# 时间序列 8/2 划分（不 shuffle）
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

# Winsorize 1%/99%（仅在 train 上拟合分位，应用到两边）
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
print(f"\n[Stage 2] 多项式特征数: {len(feature_names)} (6主效应 + 6平方 + 15交互 = 27)")

# 标准化多项式特征（仅在 train 上 fit）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_test_s = scaler.transform(X_test_poly)

# LassoCV 5折选 alpha
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lasso_cv = LassoCV(alphas=np.logspace(-5, 1, 200), cv=kf, random_state=RANDOM_STATE, max_iter=200000)
lasso_cv.fit(X_train_s, y_train)
min_alpha = lasso_cv.alpha_
print(f"[Stage 2] LassoCV 最优 alpha (CV MSE最小): {min_alpha:.6f}")

# 1-SE rule: alpha 上调一个标准差（更稀疏的解）
# mse_path_ shape = (n_alphas, n_folds)，按折数求平均/标准差
mean_per_alpha = np.mean(lasso_cv.mse_path_, axis=1)  # 每个alpha的5折平均MSE, 长度=n_alphas
std_per_alpha = np.std(lasso_cv.mse_path_, axis=1, ddof=1)  # 每个alpha的5折标准差
best_idx = int(np.argmin(mean_per_alpha))
cv_mse_min = float(mean_per_alpha[best_idx])
std_at_min = float(std_per_alpha[best_idx])
threshold = cv_mse_min + std_at_min  # 上调一个标准差的阈值
print(f"[Stage 2] 最优alpha下: CV MSE = {cv_mse_min:.6f}, std = {std_at_min:.6f}, 阈值 = {threshold:.6f}")

# 在所有 CV MSE <= 阈值 的 alpha 中，选最大的 alpha（最稀疏解）
alphas_sorted = lasso_cv.alphas_  # sklearn 默认降序（从大到小）
feasible = mean_per_alpha <= threshold
candidate_alphas = alphas_sorted[feasible]
best_alpha = float(np.max(candidate_alphas))
print(f"[Stage 2] 1-SE rule 选定 alpha (更稀疏): {best_alpha:.6f}")

# 用 1-SE rule 的 alpha 在整训练集重新拟合 Lasso
lasso_full = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
lasso_full.fit(X_train_s, y_train)
cv_mse = cv_mse_min  # 仍报告最优alpha的CV MSE作为参考
print(f"[Stage 2] 整训练集 Lasso 非零变量数: {int((np.abs(lasso_full.coef_) > 1e-10).sum())} / {len(feature_names)}")

full_nonzero = np.abs(lasso_full.coef_) > 1e-10
print(f"[Stage 2] 全训练集 Lasso 非零变量数: {int(full_nonzero.sum())} / {len(feature_names)}")

# 5折选中频率（用 1-SE rule 的 alpha 在每折上单独 fit）
sel_count = np.zeros(len(feature_names), dtype=int)
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

# 选中变量：整训练集 Lasso 非零 AND 5折中至少3折被选中（交集）
full_nonzero_mask = np.abs(lasso_full.coef_) > 1e-10
freq_mask = (sel_count / 5) >= FREQ_THRESH
combined_mask = full_nonzero_mask & freq_mask
selected = [feature_names[i] for i in range(len(feature_names)) if combined_mask[i]]

# 剔除情况统计
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

# 列出被剔除的变量
excluded = [feature_names[i] for i in range(len(feature_names)) if not combined_mask[i]]
if excluded:
    print(f"[Stage 2] 被剔除的变量: {excluded}")

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
print(f"  CV    MSE (Lasso 5折, 最优alpha): {cv_mse:.6f}")
print(f"  R²        : {ols.rsquared:.4f}")
print(f"  Adj R²    : {ols.rsquared_adj:.4f}")
print(f"  AIC       : {ols.aic:.2f}")
print(f"  BIC       : {ols.bic:.2f}")
print(f"  n_train   : {len(y_train)}, n_test: {len(y_test)}, n_features: {len(selected)}")

# 系数表
coef_table = pd.DataFrame(
    {
        "feature": ["intercept"] + selected,
        "coef": ols.params,
        "std_err": ols.bse,
        "t": ols.tvalues,
        "p": ols.pvalues,
    }
)
print("\n[Stage 3] OLS 系数表:")
print(coef_table.to_string(index=False))

# 多项式表达式（标准化空间下）
print("\n[Stage 3] 多项式（标准化多项式特征空间）:")
terms = [f"{ols.params[0]:.6f}"]
for i, name in enumerate(selected):
    terms.append(f"({ols.params[i+1]:.6f}) * {name}")
print("y ≈ " + "\n    + " + "\n    + ".join(terms[1:]) + f"\n    + {terms[0]}")

# === 保存 ===
freq_table.to_csv(f"{OUT_DIR}/stage2_selection_frequency.csv", index=False)
coef_table.to_csv(f"{OUT_DIR}/stage3_ols_coefficients.csv", index=False)
pd.DataFrame(
    [
        {
            "best_alpha": best_alpha,
            "cv_mse_lasso": cv_mse,
            "train_mse_ols": train_mse,
            "test_mse_ols": test_mse,
            "r2": ols.rsquared,
            "r2_adj": ols.rsquared_adj,
            "aic": ols.aic,
            "bic": ols.bic,
            "n_features_selected": len(selected),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "freq_threshold": FREQ_THRESH,
        }
    ]
).to_csv(f"{OUT_DIR}/stage3_metrics.csv", index=False)
print(f"\n[Saved] 结果文件位于 {OUT_DIR}/")
