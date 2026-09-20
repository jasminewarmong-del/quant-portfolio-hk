"""
快手 (1024.HK) 阶段 4 - 泰勒基函数替换（TimeSeriesSplit 版）
基于 stage123_KS.py 的 TimeSeriesSplit Ridge top-10 选出的 8 个变量：
  ['PB', 'PE', 'PB^2', 'PB V60', 'PB PE', 'M60 M20', 'V60 PE', 'PE^2']

方法：单项 ablation（每次只替换一个项，固定其余 7 个为原多项式形式）
CV：TimeSeriesSplit(n_splits=5)（与 stage2 一致，无时间泄露）
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# === 配置 ===
SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_KS_stage4_tss"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = ["PB Ratio", "Momentum 60d", "20d Momentum", "20d Volatility",
           "Volatility 60d (ann.)", "Relative Turnover", "PE Ratio"]
SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 60d": "M60",
         "20d Momentum": "M20", "20d Volatility": "V20",
         "Volatility 60d (ann.)": "V60", "Relative Turnover": "RT", "PE Ratio": "PE"}
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

# 各类型候选
LINEAR_BASES = {"exp": g_exp, "pow_0.5": g_pow05, "pow_2": g_pow2, "pow_-1": g_powm1}
SQUARE_BASES = {"cos": g_cos, "sec": g_sec}
INTERACTION_BASES = {"ln1pz": g_ln1pz, "sin": g_sin, "tanh": g_tanh, "exp": g_exp}

def classify_var(var):
    if " " in var: return "interaction"
    elif "^2" in var: return "square"
    else: return "linear"

def build_design_matrix(X_raw_s, X_poly_s, selected, replacements, raw_idx, feat_names):
    """构造设计矩阵：replacements 里的变量用基函数替换，其余用原多项式"""
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
            else:  # interaction
                idx = list(feat_names).index(var)
                z = X_poly_s[:, idx]
                cols.append(INTERACTION_BASES[basis_name](z))
        else:
            idx = list(feat_names).index(var)
            cols.append(X_poly_s[:, idx])
    return np.column_stack(cols)

# === 1. 数据准备（复用 stage123_KS 逻辑，TimeSeriesSplit）===
print("=" * 70)
print("Stage 4 - Taylor Replacement (TimeSeriesSplit, KS)")
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

scaler_p = StandardScaler(); X_train_s = scaler_p.fit_transform(X_train_poly)
X_test_s = scaler_p.transform(X_test_poly)
scaler_raw = StandardScaler(); X_train_raw_s = scaler_raw.fit_transform(X_train_w)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_idx = {f: i for i, f in enumerate(factor_cols)}

# TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5)

# 变量选择（复用 stage123_KS 的 Ridge top-10 逻辑）
lasso_cv = LassoCV(alphas=np.logspace(-5, 1, 200), cv=tss, random_state=RANDOM_STATE, max_iter=200000)
lasso_cv.fit(X_train_s, y_train)
mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
std_mse = np.std(lasso_cv.mse_path_, axis=1, ddof=1)
best_idx = int(np.argmin(mean_mse))
thresh = float(mean_mse[best_idx]) + float(std_mse[best_idx])
alpha_1se = float(np.max(lasso_cv.alphas_[mean_mse <= thresh]))

lasso_full = Lasso(alpha=alpha_1se, max_iter=200000, random_state=RANDOM_STATE).fit(X_train_s, y_train)
sel_count = np.zeros(len(feat_names), dtype=int)
for tr_idx, _ in tss.split(X_train_s):
    m = Lasso(alpha=alpha_1se, max_iter=200000, random_state=RANDOM_STATE)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    sel_count += (np.abs(m.coef_) > 1e-10).astype(int)
combined = (np.abs(lasso_full.coef_) > 1e-10) & ((sel_count / 5) >= FREQ_THRESH)
lasso_selected = [feat_names[i] for i in range(len(feat_names)) if combined[i]]

# Lasso 选中为 0，切换 Ridge top-10
print(f"[Lasso 1-SE] 选中 {len(lasso_selected)} 个变量（TimeSeriesSplit 下全归零），切换 Ridge top-10")
ridge_cv = RidgeCV(alphas=np.logspace(-3, 5, 200), cv=tss)
ridge_cv.fit(X_train_s, y_train)
ridge_alpha = float(ridge_cv.alpha_)
ridge_full = Ridge(alpha=ridge_alpha).fit(X_train_s, y_train)
top10_full = set(np.argsort(np.abs(ridge_full.coef_))[-10:])
sel_count_r = np.zeros(len(feat_names), dtype=int)
for tr_idx, _ in tss.split(X_train_s):
    m = Ridge(alpha=ridge_alpha)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    for i in np.argsort(np.abs(m.coef_))[-10:]:
        sel_count_r[i] += 1
combined_r = [(i in top10_full) and (sel_count_r[i]/5 >= FREQ_THRESH) for i in range(len(feat_names))]
selected = [feat_names[i] for i in range(len(feat_names)) if combined_r[i]]
print(f"[Ridge top-10] alpha={ridge_alpha:.6f}, 选中 {len(selected)} 个: {selected}")

# === 2. 阶段 3 基线：OLS 全 poly ===
X_train_design = build_design_matrix(X_train_raw_s, X_train_s, selected, {}, raw_idx, feat_names)
X_test_design = build_design_matrix(X_test_raw_s, X_test_s, selected, {}, raw_idx, feat_names)
X_train_int = sm.add_constant(X_train_design, has_constant='add')
X_test_int = sm.add_constant(X_test_design, has_constant='add')
ols_baseline = sm.OLS(y_train, X_train_int).fit()

# CV MSE for baseline
cv_scores = -cross_val_score(LinearRegression(), X_train_design, y_train,
                              cv=tss, scoring="neg_mean_squared_error")
baseline_cv_mse = float(np.mean(cv_scores))
baseline_cv_se = float(np.std(cv_scores, ddof=1) / np.sqrt(5))
baseline_test_mse = mean_squared_error(y_test, ols_baseline.predict(X_test_int))
baseline_test_r2 = r2_score(y_test, ols_baseline.predict(X_test_int))

print(f"\n[Baseline] CV MSE = {baseline_cv_mse:.6f} ± {baseline_cv_se:.6f}")
print(f"[Baseline] Test MSE = {baseline_test_mse:.6f}, Test R² = {baseline_test_r2:.4f}")

# === 3. 单项 ablation: 生成候选模型 ===
candidates = []
for var in selected:
    vtype = classify_var(var)
    if vtype == "linear":
        bases = LINEAR_BASES
    elif vtype == "square":
        bases = SQUARE_BASES
    else:
        bases = INTERACTION_BASES
    for bname, bfunc in bases.items():
        candidates.append((var, vtype, bname))

print(f"\n[Stage 4] 候选模型数: {len(candidates)} + 1 基线 = {len(candidates)+1}")

# === 4. 跑所有候选模型 ===
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
        print(f"[{i+1}/{len(candidates)}] {var} -> {bname}: FAILED ({e})")
        continue
    
    cv_scores = -cross_val_score(LinearRegression(), X_train_d, y_train,
                                  cv=tss, scoring="neg_mean_squared_error")
    cv_mse = float(np.mean(cv_scores))
    cv_se = float(np.std(cv_scores, ddof=1) / np.sqrt(5))
    cv_std = float(np.std(cv_scores, ddof=1))
    test_mse = mean_squared_error(y_test, ols.predict(X_test_i))
    test_r2 = r2_score(y_test, ols.predict(X_test_i))
    
    improvement = (cv_mse - baseline_cv_mse) / baseline_cv_mse * 100
    
    results.append({
        "replaced_var": var, "var_type": vtype, "basis": bname,
        "cv_mse": cv_mse, "cv_se": cv_se, "cv_std": cv_std,
        "test_mse": test_mse, "test_r2": test_r2,
        "train_r2": ols.rsquared, "train_adj_r2": ols.rsquared_adj,
        "aic": float(ols.aic), "bic": float(ols.bic),
        "cv_mse_improvement_pct": improvement,
    })
    
    marker = "  "
    if cv_mse < baseline_cv_mse: marker = "✓"
    print(f"[{i+1:2d}/{len(candidates)}] {var:8s} -> {bname:8s} | CV MSE={cv_mse:.6f} ± {cv_se:.6f} | Test={test_mse:.6f} | {marker} {improvement:+.2f}%")

# 加基线
results_with_baseline = results + [{
    "replaced_var": "None (baseline)", "var_type": "-", "basis": "poly",
    "cv_mse": baseline_cv_mse, "cv_se": baseline_cv_se, "cv_std": float(np.std(cv_scores, ddof=1)),
    "test_mse": baseline_test_mse, "test_r2": baseline_test_r2,
    "train_r2": ols_baseline.rsquared, "train_adj_r2": ols_baseline.rsquared_adj,
    "aic": float(ols_baseline.aic), "bic": float(ols_baseline.bic),
    "cv_mse_improvement_pct": 0.0,
}]

df_results = pd.DataFrame(results_with_baseline).sort_values("cv_mse").reset_index(drop=True)
df_results.to_csv(f"{OUT_DIR}/stage4_candidate_models.csv", index=False)

# === 5. 汇总输出 ===
print("\n" + "=" * 70)
print("Stage 4 Results - Top 10 by CV MSE")
print("=" * 70)
print(df_results.head(10).to_string(index=False))

print("\n=== 按类型看最优替换 ===")
for vtype in ["interaction", "square", "linear"]:
    sub = df_results[df_results["var_type"] == vtype]
    if len(sub) > 0:
        best = sub.iloc[0]
        print(f"{vtype:12s}: {best['replaced_var']:8s} -> {best['basis']:8s} | CV MSE={best['cv_mse']:.6f} ({best['cv_mse_improvement_pct']:+.2f}%)")

# 最终推荐
best_model = df_results.iloc[0]
print(f"\n=== 最终推荐（CV MSE 最优）===")
print(f"  替换: {best_model['replaced_var']} -> {best_model['basis']}")
print(f"  CV MSE = {best_model['cv_mse']:.6f} ± {best_model['cv_se']:.6f}")
print(f"  Test MSE = {best_model['test_mse']:.6f}")
print(f"  Test R² = {best_model['test_r2']:.4f}")
print(f"  vs baseline CV MSE: {best_model['cv_mse_improvement_pct']:+.2f}%")

# 也看 Test MSE 最优（仅参考，不参与 selection）
best_test = df_results.sort_values("test_mse").iloc[0]
print(f"\n=== 参考：Test MSE 最优（不参与 selection）===")
print(f"  替换: {best_test['replaced_var']} -> {best_test['basis']}")
print(f"  CV MSE = {best_test['cv_mse']:.6f}, Test MSE = {best_test['test_mse']:.6f}")

print(f"\n[Saved] {OUT_DIR}/stage4_candidate_models.csv")
