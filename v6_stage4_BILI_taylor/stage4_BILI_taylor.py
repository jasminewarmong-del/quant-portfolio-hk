"""
Bilibili (9626.HK) 量化策略 - 阶段 4：泰勒基函数替换（单项 ablation）
与快手 stage4 思路完全一致：
- 每次只替换一个项，固定其余 9 项为原多项式形式
- 主效应 x_i → e^x, (1+x)^a  (a ∈ {0.5, 2, -1})
- 平方项 x_i² → cos(x_i), sec(x_i)
- 交互项 x_i·x_j → ln(1+z), sin(z), tanh(z), e^z-1  (z = x_i·x_j)

Bilibili Ridge top-10 选出的 10 个变量：
['M120', 'MAUg', 'PB M120', 'PB DAUg', 'PB MAUg',
 'M120 DAUg', 'M120 MAUg', 'DAUg^2', 'DAUg RS', 'MAUg RS']

可替换项分类：
- 主效应 (linear): M120, MAUg → 4 基函数 each
- 平方项 (square): DAUg^2 → 2 基函数
- 交互项 (interaction): 7 个 → 4 基函数 each
合计 2*4 + 1*2 + 7*4 = 38 单项替换 + 1 基线 = 39 个模型
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# === 配置 ===
SRC = "/workspace/.uploads/d4d06402-aa6f-40b0-b7c8-c4d694a048eb_BILI_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_BILI_stage4"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = [
    "PB Ratio", "Momentum 120d", "DAU Growth (YoY %)",
    "MAU Growth (YoY %)", "Momentum 60d", "Rel. Strength vs HSTECH",
]
SHORT = {
    "y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 120d": "M120",
    "DAU Growth (YoY %)": "DAUg", "MAU Growth (YoY %)": "MAUg",
    "Momentum 60d": "M60", "Rel. Strength vs HSTECH": "RS",
}
TOPK = 10  # Ridge top-10
FREQ_THRESH = 3 / 5
RANDOM_STATE = 42

# === 1. 数据准备 + Ridge top-10 选变量（复现阶段2-3 逻辑）===
sheet1 = pd.read_excel(SRC, sheet_name="Sheet1")
sheet2 = pd.read_excel(SRC, sheet_name="Sheet2")
df = pd.merge(
    sheet1[["Trade Date", "y (5d Log Return)"]],
    sheet2[["Trade Date"] + FACTORS], on="Trade Date",
).rename(columns=SHORT)
factor_cols = [SHORT[c] for c in FACTORS]
df = df.dropna().reset_index(drop=True)

n = len(df); n_train = int(n * 0.8)
train = df.iloc[:n_train].copy(); test = df.iloc[n_train:].copy()
X_train_raw = train[factor_cols].values; X_test_raw = test[factor_cols].values
y_train = train["y"].values; y_test = test["y"].values

lows = np.quantile(X_train_raw, 0.01, axis=0)
highs = np.quantile(X_train_raw, 0.99, axis=0)
X_train_w = np.clip(X_train_raw, lows[None, :], highs[None, :])
X_test_w = np.clip(X_test_raw, lows[None, :], highs[None, :])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w)
X_test_poly = poly.transform(X_test_w)
feature_names = poly.get_feature_names_out(factor_cols)
n_poly = len(feature_names)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_poly)
X_test_s = scaler.transform(X_test_poly)

# 原始单因子标准化（用于构造 cos(x_i), sec(x_i)）
scaler_raw = StandardScaler()
X_train_raw_s = scaler_raw.fit_transform(X_train_w)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_factor_idx = {f: i for i, f in enumerate(factor_cols)}

# Ridge top-10 + 5折稳定性
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ridge_cv = RidgeCV(alphas=np.logspace(-5, 2, 200), cv=kf)
ridge_cv.fit(X_train_s, y_train)
best_alpha = float(ridge_cv.alpha_)
print(f"[Stage 2] Ridge 最优 alpha = {best_alpha:.6f}")

ridge_full = Ridge(alpha=best_alpha)
ridge_full.fit(X_train_s, y_train)
coef_abs = np.abs(ridge_full.coef_)
topk_idx_full = set(np.argsort(coef_abs)[-TOPK:])

# 5折稳定性
sel_count = np.zeros(n_poly, dtype=int)
for tr_idx, _ in kf.split(X_train_s):
    m = Ridge(alpha=best_alpha)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    topk_fold = set(np.argsort(np.abs(m.coef_))[-TOPK:])
    for i in topk_fold:
        sel_count[i] += 1

freq_mask = (sel_count / 5) >= FREQ_THRESH
combined_mask = np.array([
    (i in topk_idx_full) and freq_mask[i] for i in range(n_poly)
])
selected = [feature_names[i] for i in range(n_poly) if combined_mask[i]]
selected_idx = [list(feature_names).index(f) for f in selected]
print(f"[Stage 2] Ridge top-{TOPK} + 5折稳定性选出: {selected}")

# === 2. 基函数定义（与快手完全一致）===
def safe_clip(z, lo, hi): return np.clip(z, lo, hi)
def g_exp(z): return np.expm1(safe_clip(z, -3, 3))
def g_pow_a(z, a): return np.power(1.0 + safe_clip(z, -0.999, None), a) - 1.0
def g_cos(x): return np.cos(x)
def g_sec(x): return 1.0 / np.cos(safe_clip(x, -1.55, 1.55))
def g_ln1pz(z): return np.log1p(safe_clip(z, -0.999, None))
def g_sin(z): return np.sin(safe_clip(z, -3*np.pi, 3*np.pi))
def g_tanh(z): return np.tanh(safe_clip(z, -3, 3))

LINEAR_BASES = [
    ("exp", g_exp),
    ("pow_0.5", lambda z: g_pow_a(z, 0.5)),
    ("pow_2", lambda z: g_pow_a(z, 2)),
    ("pow_-1", lambda z: g_pow_a(z, -1)),
]
SQUARE_BASES = [
    ("cos", g_cos),
    ("sec", g_sec),
]
INTERACTION_BASES = [
    ("ln1pz", g_ln1pz),
    ("sin", g_sin),
    ("tanh", g_tanh),
    ("exp", g_exp),
]

def classify_var(var):
    if " " in var: return "interaction"
    elif "^2" in var: return "square"
    else: return "linear"

def build_cols_with_replacement(replace_var, basis_func, var_type):
    """构造 10 列：replace_var 用 basis_func 替换，其余原 poly"""
    train_cols = []; test_cols = []
    for var in selected:
        if var == replace_var:
            if var_type == "square":
                base = var.replace("^2", "")
                idx = raw_factor_idx[base]
                tr_x = X_train_raw_s[:, idx]; te_x = X_test_raw_s[:, idx]
                train_cols.append(basis_func(tr_x))
                test_cols.append(basis_func(te_x))
            elif var_type == "linear":
                idx = raw_factor_idx[var]
                tr_x = X_train_raw_s[:, idx]; te_x = X_test_raw_s[:, idx]
                train_cols.append(basis_func(tr_x))
                test_cols.append(basis_func(te_x))
            elif var_type == "interaction":
                idx = list(feature_names).index(var)
                tr_z = X_train_s[:, idx]; te_z = X_test_s[:, idx]
                train_cols.append(basis_func(tr_z))
                test_cols.append(basis_func(te_z))
        else:
            idx = list(feature_names).index(var)
            train_cols.append(X_train_s[:, idx])
            test_cols.append(X_test_s[:, idx])
    return np.column_stack(train_cols), np.column_stack(test_cols)

def run_model(X_tr, X_te, label):
    X_tr_int = sm.add_constant(X_tr, has_constant='add')
    X_te_int = sm.add_constant(X_te, has_constant='add')
    ols = sm.OLS(y_train, X_tr_int).fit()
    y_tr_pred = ols.predict(X_tr_int)
    y_te_pred = ols.predict(X_te_int)
    cv_scores = -cross_val_score(LinearRegression(), X_tr, y_train,
                                 cv=kf, scoring="neg_mean_squared_error")
    return {
        "replace_var": label["replace_var"], "var_type": label["var_type"],
        "basis": label["basis"],
        "train_mse": mean_squared_error(y_train, y_tr_pred),
        "test_mse": mean_squared_error(y_test, y_te_pred),
        "cv_mse": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores, ddof=1)),
        "r2": ols.rsquared, "r2_adj": ols.rsquared_adj,
        "aic": ols.aic, "bic": ols.bic, "n_features": X_tr.shape[1],
    }

# === 3. 基线模型（全 poly）===
results = []
X_train_base = X_train_s[:, selected_idx]
X_test_base = X_test_s[:, selected_idx]
res_base = run_model(X_train_base, X_test_base,
                     {"replace_var": "(baseline)", "var_type": "-", "basis": "poly"})
results.append(res_base)
print(f"\n[Stage 4] 基线 (全 poly): CV MSE={res_base['cv_mse']:.6f}, "
      f"Test MSE={res_base['test_mse']:.6f}, AIC={res_base['aic']:.2f}")

# === 4. 单项替换 ===
total = 0
for var in selected:
    vtype = classify_var(var)
    bases = (LINEAR_BASES if vtype == "linear"
             else SQUARE_BASES if vtype == "square"
             else INTERACTION_BASES)
    for basis_name, basis_func in bases:
        try:
            X_tr, X_te = build_cols_with_replacement(var, basis_func, vtype)
            r = run_model(X_tr, X_te,
                          {"replace_var": var, "var_type": vtype, "basis": basis_name})
            results.append(r)
            total += 1
            print(f"[Stage 4] {var} -> {basis_name}: CV MSE={r['cv_mse']:.6f}, "
                  f"Test MSE={r['test_mse']:.6f}")
        except Exception as e:
            print(f"[Stage 4] {var} -> {basis_name} FAILED: {e}")
            results.append({"replace_var": var, "var_type": vtype, "basis": basis_name,
                            "train_mse": np.nan, "test_mse": np.nan, "cv_mse": np.nan,
                            "cv_std": np.nan, "r2": np.nan, "r2_adj": np.nan,
                            "aic": np.nan, "bic": np.nan, "n_features": len(selected)})

print(f"\n[Stage 4] 共跑 {total} 个单项替换 + 1 基线 = {total+1} 个模型")

# === 5. 汇总 ===
df_res = pd.DataFrame(results)
baseline_cv = df_res.loc[0, "cv_mse"]
df_res["cv_mse_diff_vs_baseline"] = df_res["cv_mse"] - baseline_cv
df_res["cv_mse_pct_change"] = (df_res["cv_mse"] - baseline_cv) / baseline_cv * 100
df_res = df_res.sort_values("cv_mse", ascending=True).reset_index(drop=True)

print("\n[Stage 4] Top-10 最优 (CV MSE 升序):")
print(df_res.head(10).to_string(index=False))

print("\n[Stage 4] 各类型最优:")
for vtype in ["linear", "square", "interaction"]:
    sub = df_res[df_res["var_type"] == vtype].head(3)
    print(f"\n--- {vtype} Top-3 ---")
    print(sub.to_string(index=False))

df_res.to_csv(f"{OUT_DIR}/stage4_candidate_models.csv", index=False)
print(f"\n[Saved] {OUT_DIR}/stage4_candidate_models.csv")
