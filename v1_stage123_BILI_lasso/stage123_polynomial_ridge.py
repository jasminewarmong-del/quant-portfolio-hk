"""
Bilibili (9626.HK) 量化策略 - 阶段 1+2+3 (Ridge 版)
1) 数据合并 + dropna + 时间序列 8/2 划分
2) winsorize + PolynomialFeatures(degree=2) + Standardize + RidgeCV(5折) + top-10稳定性选择
3) 用选中变量做 OLS 重拟合，输出多项式系数与各指标
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
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
K_TOP = 10            # 保留 top-k 变量
FREQ_THRESH = 3 / 5   # 5折中至少3折进入top-k
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

# === 2. 多项式 + 标准化 + RidgeCV 5折 ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w)
X_test_poly = poly.transform(X_test_w)
feature_names = poly.get_feature_names_out(factor_cols)
print(f"\n[Stage 2] 多项式特征数: {len(feature_names)} (6主效应 + 6平方 + 15交互 = 27)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_test_s = scaler.transform(X_test_poly)

# RidgeCV 5折选 alpha
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ridge_cv = RidgeCV(alphas=np.logspace(-3, 5, 200), cv=kf)
ridge_cv.fit(X_train_s, y_train)
best_alpha = ridge_cv.alpha_
print(f"[Stage 2] RidgeCV 最优 alpha: {best_alpha:.6f}")

# 整训练集 Ridge 系数（RidgeCV本身就在全train上拟合，但显式重跑一次以确认）
ridge_full = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
ridge_full.fit(X_train_s, y_train)
full_coef = ridge_full.coef_.copy()

# 整训练集 CV MSE（用5折平均）
from sklearn.model_selection import cross_val_score
cv_mse_scores = -cross_val_score(Ridge(alpha=best_alpha), X_train_s, y_train,
                                 cv=kf, scoring="neg_mean_squared_error")
cv_mse = float(np.mean(cv_mse_scores))
print(f"[Stage 2] Ridge 5折 CV MSE (最优alpha下): {cv_mse:.6f} (std={np.std(cv_mse_scores, ddof=1):.6f})")

# 整训练集 top-k by |coef|
abs_coef = np.abs(full_coef)
topk_idx_full = set(np.argsort(abs_coef)[::-1][:K_TOP].tolist())
print(f"[Stage 2] 整训练集 Ridge top-{K_TOP} (按|coef|): {[feature_names[i] for i in sorted(topk_idx_full)]}")

# 5折稳定性：每折 fit Ridge(alpha=best)，记录每折 top-k by |coef|，统计频率
sel_count = np.zeros(len(feature_names), dtype=int)
for tr_idx, _ in kf.split(X_train_s):
    m = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    fold_top = set(np.argsort(np.abs(m.coef_))[::-1][:K_TOP].tolist())
    for i in fold_top:
        sel_count[i] += 1

freq_table = pd.DataFrame(
    {
        "feature": feature_names,
        "ridge_coef": full_coef,
        "abs_coef": abs_coef,
        "topk_count": sel_count,
        "frequency": sel_count / 5,
    }
).sort_values(["frequency", "abs_coef"], ascending=[False, False]).reset_index(drop=True)
print("\n[Stage 2] 5折 top-k 稳定性表（按频率降序，频率相同按|coef|降序）:")
print(freq_table.to_string(index=False))

# 选中变量：整训练集 top-k AND 5折中至少3折进入top-k（交集）
full_top_mask = np.array([i in topk_idx_full for i in range(len(feature_names))])
freq_mask = (sel_count / 5) >= FREQ_THRESH
combined_mask = full_top_mask & freq_mask
selected = [feature_names[i] for i in range(len(feature_names)) if combined_mask[i]]

# 若交集为空或太少，则回退到"整训练集top-k OR 5折>=3/5"，仍不够则用整训练集top-k
if len(selected) < 3:
    print(f"[Stage 2] 警告：交集仅 {len(selected)} 项，回退到 OR 规则")
    combined_mask = full_top_mask | freq_mask
    selected = [feature_names[i] for i in range(len(feature_names)) if combined_mask[i]]
    if len(selected) < 3:
        print(f"[Stage 2] 警告：OR规则仍只 {len(selected)} 项，回退到整训练集top-{K_TOP}")
        selected = [feature_names[i] for i in sorted(topk_idx_full)]

# 统计
only_full = int((full_top_mask & ~freq_mask).sum())
only_freq = int((~full_top_mask & freq_mask).sum())
both = int((full_top_mask & freq_mask).sum())
neither = int((~full_top_mask & ~freq_mask).sum())
print(f"\n[Stage 2] 判据统计（整训练集top-{K_TOP} AND 5折>=3/5）:")
print(f"  两个条件都满足 (保留): {both}")
print(f"  仅整训练集top-{K_TOP} (剔除): {only_full}")
print(f"  仅5折>=3/5 (剔除): {only_freq}")
print(f"  两个条件都不满足 (剔除): {neither}")
print(f"[Stage 2] 最终选中变量 ({len(selected)}): {selected}")

excluded = [feature_names[i] for i in range(len(feature_names)) if not combined_mask[i]]
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
print(f"  CV    MSE (Ridge 5折, 最优alpha): {cv_mse:.6f}")
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

# 多项式表达式（标准化空间）
print("\n[Stage 3] 多项式（标准化多项式特征空间）:")
terms = [f"{ols.params[0]:.6f}"]
for i, name in enumerate(selected):
    terms.append(f"({ols.params[i+1]:.6f}) * {name}")
print("y ≈ " + "\n    + ".join(terms[1:]) + f"\n    + {terms[0]}")

# === 保存 ===
freq_table.to_csv(f"{OUT_DIR}/stage2_ridge_topk_stability.csv", index=False)
coef_table.to_csv(f"{OUT_DIR}/stage3_ridge_ols_coefficients.csv", index=False)
pd.DataFrame(
    [
        {
            "best_alpha_ridge": best_alpha,
            "cv_mse_ridge": cv_mse,
            "cv_mse_std": np.std(cv_mse_scores, ddof=1),
            "train_mse_ols": train_mse,
            "test_mse_ols": test_mse,
            "r2": ols.rsquared,
            "r2_adj": ols.rsquared_adj,
            "aic": ols.aic,
            "bic": ols.bic,
            "n_features_selected": len(selected),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "k_top": K_TOP,
            "freq_threshold": FREQ_THRESH,
        }
    ]
).to_csv(f"{OUT_DIR}/stage3_ridge_metrics.csv", index=False)
print(f"\n[Saved] 结果文件位于 {OUT_DIR}/")
