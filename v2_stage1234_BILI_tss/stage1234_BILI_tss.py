"""
Bilibili (9626.HK) 阶段 1-4 - TimeSeriesSplit 版
因子: PB Ratio, Momentum 120d, DAU Growth (YoY %),
      MAU Growth (YoY %), Momentum 60d, Rel. Strength vs HSTECH

1) 数据合并 + dropna + 时间序列 8/2 划分
2) winsorize + PolynomialFeatures(degree=2) + Standardize
   + RidgeCV(TimeSeriesSplit) + top-10 稳定性筛选
3) OLS 重拟合
4) 泰勒基函数替换（单项 ablation, TimeSeriesSplit CV）
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
SRC = "/workspace/.uploads/d4d06402-aa6f-40b0-b7c8-c4d694a048eb_BILI_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_BILI_tss"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = ["PB Ratio", "Momentum 120d", "DAU Growth (YoY %)",
           "MAU Growth (YoY %)", "Momentum 60d", "Rel. Strength vs HSTECH"]
SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 120d": "M120",
         "DAU Growth (YoY %)": "DAUg", "MAU Growth (YoY %)": "MAUg",
         "Momentum 60d": "M60", "Rel. Strength vs HSTECH": "RS"}
FREQ_THRESH = 3 / 5
RANDOM_STATE = 42

# === 基函数 ===
def safe_clip(z, lo, hi): return np.clip(z, lo, hi)
def g_exp(z): return np.expm1(safe_clip(z, -3, 3))
def g_ln1pz(z): return np.log1p(safe_clip(z, -0.999, None))
def g_sin(z): return np.sin(safe_clip(z, -3*np.pi, 3*np.pi))
def g_tanh(z): return np.tanh(safe_clip(z, -3, 3))
def g_cos(z): return 2*(1 - np.cos(safe_clip(z, -1.55, 1.55)))
def g_sec(z):
    x = safe_clip(z, -1.55, 1.55)
    return 2*(1/np.cos(x) - 1)
def g_pow05(z): return (1 + safe_clip(z, -0.999, None))**0.5 - 1
def g_pow2(z): return (1 + safe_clip(z, -0.999, None))**2 - 1
def g_powm1(z): return (1 + safe_clip(z, -0.999, None))**(-1) - 1

LINEAR_BASES = {"exp": g_exp, "pow_0.5": g_pow05, "pow_2": g_pow2, "pow_-1": g_powm1}
SQUARE_BASES = {"cos": g_cos, "sec": g_sec}
INTERACTION_BASES = {"ln1pz": g_ln1pz, "sin": g_sin, "tanh": g_tanh, "exp": g_exp}

def classify_var(var):
    if " " in var: return "interaction"
    elif "^2" in var: return "square"
    else: return "linear"

def build_design_matrix(X_raw_s, X_poly_s, selected, replacements, raw_idx, feat_names):
    cols = []
    for var in selected:
        if var in replacements:
            basis_name = replacements[var]
            vtype = classify_var(var)
            if vtype == "square":
                base = var.replace("^2", "")
                x = X_raw_s[:, raw_idx[base]]
                cols.append(SQUARE_BASES[basis_name](x))
            elif vtype == "linear":
                x = X_raw_s[:, raw_idx[var]]
                cols.append(LINEAR_BASES[basis_name](x))
            else:
                idx = list(feat_names).index(var)
                z = X_poly_s[:, idx]
                cols.append(INTERACTION_BASES[basis_name](z))
        else:
            idx = list(feat_names).index(var)
            cols.append(X_poly_s[:, idx])
    return np.column_stack(cols)

# === 1. 数据准备 ===
print("=" * 70)
print("BILI Stage 1-4 (TimeSeriesSplit)")
print("=" * 70)

sheet1 = pd.read_excel(SRC, sheet_name="Sheet1")
sheet2 = pd.read_excel(SRC, sheet_name="Sheet2")
df = pd.merge(sheet1[["Trade Date", "y (5d Log Return)"]],
              sheet2[["Trade Date"] + FACTORS], on="Trade Date").rename(columns=SHORT)
factor_cols = [SHORT[c] for c in FACTORS]
df = df.dropna().reset_index(drop=True)

n = len(df); n_train = int(n * 0.8)
train = df.iloc[:n_train].copy(); test = df.iloc[n_train:].copy()
X_train_raw = train[factor_cols].values; X_test_raw = test[factor_cols].values
y_train = train["y"].values; y_test = test["y"].values

lows = np.quantile(X_train_raw, 0.01, axis=0); highs = np.quantile(X_train_raw, 0.99, axis=0)
X_train_w = np.clip(X_train_raw, lows[None, :], highs[None, :])
X_test_w = np.clip(X_test_raw, lows[None, :], highs[None, :])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w); X_test_poly = poly.transform(X_test_w)
feat_names = poly.get_feature_names_out(factor_cols)
n_poly = len(feat_names)
print(f"[Stage 1] 样本数: {n}, Train: {len(train)}, Test: {len(test)}, 多项式特征: {n_poly}")

scaler_p = StandardScaler(); X_train_s = scaler_p.fit_transform(X_train_poly)
X_test_s = scaler_p.transform(X_test_poly)
scaler_raw = StandardScaler(); X_train_raw_s = scaler_raw.fit_transform(X_train_w)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_idx = {f: i for i, f in enumerate(factor_cols)}

tss = TimeSeriesSplit(n_splits=5)

# === 2. Ridge top-10 变量选择 ===
ridge_cv = RidgeCV(alphas=np.logspace(-3, 5, 200), cv=tss)
ridge_cv.fit(X_train_s, y_train)
ridge_alpha = float(ridge_cv.alpha_)
print(f"[Stage 2] RidgeCV alpha={ridge_alpha:.6f}")

ridge_full = Ridge(alpha=ridge_alpha).fit(X_train_s, y_train)
top10_full = set(np.argsort(np.abs(ridge_full.coef_))[-10:])

# 5折稳定性
sel_count = np.zeros(n_poly, dtype=int)
for tr_idx, _ in tss.split(X_train_s):
    m = Ridge(alpha=ridge_alpha)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    for i in np.argsort(np.abs(m.coef_))[-10:]:
        sel_count[i] += 1

combined = [(i in top10_full) and (sel_count[i]/5 >= FREQ_THRESH) for i in range(n_poly)]
selected = [feat_names[i] for i in range(n_poly) if combined[i]]
print(f"[Stage 2] Ridge top-10 选出 {len(selected)} 个变量: {selected}")

freq_table = pd.DataFrame({
    "feature": feat_names, "ridge_coef": ridge_full.coef_,
    "abs_coef": np.abs(ridge_full.coef_), "topk_count": sel_count, "frequency": sel_count/5
}).sort_values(["frequency", "abs_coef"], ascending=[False, False]).reset_index(drop=True)
freq_table.to_csv(f"{OUT_DIR}/stage2_selection_frequency.csv", index=False)

# === 3. OLS 基线 ===
X_train_design = build_design_matrix(X_train_raw_s, X_train_s, selected, {}, raw_idx, feat_names)
X_test_design = build_design_matrix(X_test_raw_s, X_test_s, selected, {}, raw_idx, feat_names)
X_train_int = sm.add_constant(X_train_design, has_constant='add')
X_test_int = sm.add_constant(X_test_design, has_constant='add')
ols_baseline = sm.OLS(y_train, X_train_int).fit()

cv_scores = -cross_val_score(LinearRegression(), X_train_design, y_train,
                              cv=tss, scoring="neg_mean_squared_error")
baseline_cv_mse = float(np.mean(cv_scores))
baseline_cv_se = float(np.std(cv_scores, ddof=1) / np.sqrt(5))
baseline_test_mse = mean_squared_error(y_test, ols_baseline.predict(X_test_int))
baseline_test_r2 = r2_score(y_test, ols_baseline.predict(X_test_int))

print(f"\n[Stage 3] OLS 基线:")
print(f"  Train R² = {ols_baseline.rsquared:.4f}")
print(f"  CV MSE = {baseline_cv_mse:.6f} ± {baseline_cv_se:.6f}")
print(f"  Test MSE = {baseline_test_mse:.6f}, Test R² = {baseline_test_r2:.4f}")

coef_table = pd.DataFrame({
    "feature": ["intercept"] + selected, "coef": ols_baseline.params,
    "std_err": ols_baseline.bse, "t": ols_baseline.tvalues, "p": ols_baseline.pvalues
})
coef_table.to_csv(f"{OUT_DIR}/stage3_ols_coefficients.csv", index=False)

# === 4. 泰勒替换（单项 ablation）===
candidates = []
for var in selected:
    vtype = classify_var(var)
    if vtype == "linear": bases = LINEAR_BASES
    elif vtype == "square": bases = SQUARE_BASES
    else: bases = INTERACTION_BASES
    for bname in bases:
        candidates.append((var, vtype, bname))

print(f"\n[Stage 4] 候选模型数: {len(candidates)} + 1 基线 = {len(candidates)+1}")

results = []
for i, (var, vtype, bname) in enumerate(candidates):
    replacements = {var: bname}
    X_train_d = build_design_matrix(X_train_raw_s, X_train_s, selected, replacements, raw_idx, feat_names)
    X_test_d = build_design_matrix(X_test_raw_s, X_test_s, selected, replacements, raw_idx, feat_names)
    X_train_i = sm.add_constant(X_train_d, has_constant='add')
    X_test_i = sm.add_constant(X_test_d, has_constant='add')
    
    try:
        ols = sm.OLS(y_train, X_train_i).fit()
    except Exception as e:
        print(f"[{i+1}/{len(candidates)}] {var} -> {bname}: FAILED")
        continue
    
    cv_scores = -cross_val_score(LinearRegression(), X_train_d, y_train,
                                  cv=tss, scoring="neg_mean_squared_error")
    cv_mse = float(np.mean(cv_scores))
    cv_se = float(np.std(cv_scores, ddof=1) / np.sqrt(5))
    test_mse = mean_squared_error(y_test, ols.predict(X_test_i))
    test_r2 = r2_score(y_test, ols.predict(X_test_i))
    improvement = (cv_mse - baseline_cv_mse) / baseline_cv_mse * 100
    
    results.append({
        "replaced_var": var, "var_type": vtype, "basis": bname,
        "cv_mse": cv_mse, "cv_se": cv_se, "test_mse": test_mse, "test_r2": test_r2,
        "train_r2": ols.rsquared, "aic": float(ols.aic), "bic": float(ols.bic),
        "cv_mse_improvement_pct": improvement,
    })
    
    marker = "✓" if cv_mse < baseline_cv_mse else " "
    print(f"[{i+1:2d}/{len(candidates)}] {var:10s} -> {bname:8s} | CV={cv_mse:.6f}±{cv_se:.6f} | Test={test_mse:.6f} | {marker} {improvement:+.2f}%")

# 加基线
results.append({
    "replaced_var": "None (baseline)", "var_type": "-", "basis": "poly",
    "cv_mse": baseline_cv_mse, "cv_se": baseline_cv_se,
    "test_mse": baseline_test_mse, "test_r2": baseline_test_r2,
    "train_r2": ols_baseline.rsquared, "aic": float(ols_baseline.aic),
    "bic": float(ols_baseline.bic), "cv_mse_improvement_pct": 0.0,
})

df_results = pd.DataFrame(results).sort_values("cv_mse").reset_index(drop=True)
df_results.to_csv(f"{OUT_DIR}/stage4_candidate_models.csv", index=False)

# === 汇总 ===
print("\n" + "=" * 70)
print("Stage 4 Top 10 by CV MSE")
print("=" * 70)
print(df_results.head(10).to_string(index=False))

print("\n=== 各类型最优 ===")
for vt in ["interaction", "square", "linear"]:
    sub = df_results[df_results["var_type"] == vt]
    if len(sub) > 0:
        b = sub.iloc[0]
        print(f"{vt:12s}: {b['replaced_var']:10s} -> {b['basis']:8s} | CV={b['cv_mse']:.6f} ({b['cv_mse_improvement_pct']:+.2f}%)")

best = df_results.iloc[0]
print(f"\n=== 最终推荐（CV MSE 最优）===")
print(f"  替换: {best['replaced_var']} -> {best['basis']}")
print(f"  CV MSE = {best['cv_mse']:.6f} ± {best['cv_se']:.6f}")
print(f"  Test MSE = {best['test_mse']:.6f}, Test R² = {best['test_r2']:.4f}")
print(f"  vs baseline: {best['cv_mse_improvement_pct']:+.2f}%")

print(f"\n[Saved] {OUT_DIR}/")
