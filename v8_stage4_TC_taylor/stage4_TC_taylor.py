"""
腾讯 (TC) 阶段 4 - 泰勒基函数替换（单项 ablation）
基于 stage123_TC.py 选出的 7 个变量:
  PB, ROE, M20, MAUg^2, PE ROE, PE MAUg, ROE MAUg
"""

import os, numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

SRC = "/workspace/.uploads/35657faa-847c-4c1d-b404-5b5165eb83ec_TC_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_TC_stage4"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = ["PB Ratio", "PE Ratio", "ROE (%)", "20d Momentum", "MAU Growth (YoY %)"]
SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "PE Ratio": "PE",
         "ROE (%)": "ROE", "20d Momentum": "M20", "MAU Growth (YoY %)": "MAUg"}
SELECTED = ['PB', 'ROE', 'M20', 'MAUg^2', 'PE ROE', 'PE MAUg', 'ROE MAUg']
RIDGE_ALPHA = 1.0

# 基函数
def sc(z, lo, hi): return np.clip(z, lo, hi)
def g_exp(z): return np.expm1(sc(z, -3, 3))
def g_ln1pz(z): return np.log1p(sc(z, -0.999, None))
def g_sin(z): return np.sin(sc(z, -3*np.pi, 3*np.pi))
def g_tanh(z): return np.tanh(sc(z, -3, 3))
def g_cos(z): return 2*(1 - np.cos(sc(z, -1.55, 1.55)))
def g_sec(z):
    x = sc(z, -1.55, 1.55); return 2*(1/np.cos(x) - 1)
def g_pow05(z): return (1 + sc(z, -0.999, None))**0.5 - 1
def g_pow2(z): return (1 + sc(z, -0.999, None))**2 - 1
def g_powm1(z): return (1 + sc(z, -0.999, None))**(-1) - 1

LINEAR_BASES = {"exp": g_exp, "pow_0.5": g_pow05, "pow_2": g_pow2, "pow_-1": g_powm1}
SQUARE_BASES = {"cos": g_cos, "sec": g_sec}
INTERACTION_BASES = {"ln1pz": g_ln1pz, "sin": g_sin, "tanh": g_tanh, "exp": g_exp}

def classify_var(var):
    if " " in var: return "interaction"
    elif "^2" in var: return "square"
    else: return "linear"

def build_design(X_raw_s, X_poly_s, selected, replacements, raw_idx, feat_names):
    cols = []
    for var in selected:
        if var in replacements:
            bname = replacements[var]; vtype = classify_var(var)
            if vtype == "square":
                base = var.replace("^2", "")
                x = X_raw_s[:, raw_idx[base]]
                cols.append(SQUARE_BASES[bname](x))
            elif vtype == "linear":
                x = X_raw_s[:, raw_idx[var]]
                cols.append(LINEAR_BASES[bname](x))
            else:
                idx = list(feat_names).index(var)
                z = X_poly_s[:, idx]
                cols.append(INTERACTION_BASES[bname](z))
        else:
            idx = list(feat_names).index(var)
            cols.append(X_poly_s[:, idx])
    return np.column_stack(cols)

# === 数据准备 ===
print("="*70); print("腾讯 Stage 4 泰勒替换"); print("="*70)
s1 = pd.read_excel(SRC, sheet_name="Sheet1")
s2 = pd.read_excel(SRC, sheet_name="Sheet2")
df = pd.merge(s1[["Trade Date","y (5d Log Return)","Close (HKD)"]],
              s2[["Trade Date"]+FACTORS], on="Trade Date").rename(columns=SHORT)
factor_cols = [SHORT[c] for c in FACTORS]
df = df.dropna().reset_index(drop=True)
n = len(df); n_train = int(n*0.8)
train = df.iloc[:n_train].copy(); test = df.iloc[n_train:].copy()
X_train_raw = train[factor_cols].values; X_test_raw = test[factor_cols].values
y_train = train["y"].values; y_test = test["y"].values
lows = np.quantile(X_train_raw, 0.01, axis=0); highs = np.quantile(X_train_raw, 0.99, axis=0)
X_train_w = np.clip(X_train_raw, lows[None,:], highs[None,:])
X_test_w = np.clip(X_test_raw, lows[None,:], highs[None,:])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_w); X_test_poly = poly.transform(X_test_w)
feat_names = poly.get_feature_names_out(factor_cols)
scaler_p = StandardScaler(); X_train_s = scaler_p.fit_transform(X_train_poly)
X_test_s = scaler_p.transform(X_test_poly)
scaler_raw = StandardScaler(); X_train_raw_s = scaler_raw.fit_transform(X_train_w)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_idx = {f: i for i, f in enumerate(factor_cols)}
tss = TimeSeriesSplit(n_splits=5)

# === 基线 OLS ===
X_train_d = build_design(X_train_raw_s, X_train_s, SELECTED, {}, raw_idx, feat_names)
X_test_d = build_design(X_test_raw_s, X_test_s, SELECTED, {}, raw_idx, feat_names)
X_train_i = sm.add_constant(X_train_d, has_constant='add')
X_test_i = sm.add_constant(X_test_d, has_constant='add')
ols_base = sm.OLS(y_train, X_train_i).fit()
cv_scores = -cross_val_score(LinearRegression(), X_train_d, y_train, cv=tss, scoring="neg_mean_squared_error")
base_cv = float(np.mean(cv_scores)); base_se = float(np.std(cv_scores, ddof=1)/np.sqrt(5))
base_test = mean_squared_error(y_test, ols_base.predict(X_test_i))
base_test_r2 = r2_score(y_test, ols_base.predict(X_test_i))
print(f"\n基线: CV={base_cv:.6f}±{base_se:.6f}, Test={base_test:.6f}, R²={base_test_r2:.4f}")

# === 候选模型 ===
candidates = []
for var in SELECTED:
    vt = classify_var(var)
    bases = LINEAR_BASES if vt=="linear" else (SQUARE_BASES if vt=="square" else INTERACTION_BASES)
    for bn in bases: candidates.append((var, vt, bn))
print(f"候选模型数: {len(candidates)} + 1 基线 = {len(candidates)+1}")

results = []
for i, (var, vt, bn) in enumerate(candidates):
    reps = {var: bn}
    Xtr = build_design(X_train_raw_s, X_train_s, SELECTED, reps, raw_idx, feat_names)
    Xte = build_design(X_test_raw_s, X_test_s, SELECTED, reps, raw_idx, feat_names)
    Xi = sm.add_constant(Xtr, has_constant='add'); Xti = sm.add_constant(Xte, has_constant='add')
    try:
        ols = sm.OLS(y_train, Xi).fit()
    except Exception as e:
        print(f"[{i+1}/{len(candidates)}] {var} -> {bn}: FAILED"); continue
    cvs = -cross_val_score(LinearRegression(), Xtr, y_train, cv=tss, scoring="neg_mean_squared_error")
    cvm = float(np.mean(cvs)); cvs_se = float(np.std(cvs, ddof=1)/np.sqrt(5))
    tm = mean_squared_error(y_test, ols.predict(Xti)); tr2 = r2_score(y_test, ols.predict(Xti))
    imp = (cvm - base_cv)/base_cv*100
    results.append({"replaced_var":var,"var_type":vt,"basis":bn,"cv_mse":cvm,"cv_se":cvs_se,
                    "test_mse":tm,"test_r2":tr2,"train_r2":ols.rsquared,
                    "aic":float(ols.aic),"bic":float(ols.bic),"cv_mse_improvement_pct":imp})
    mk = "✓" if cvm < base_cv else " "
    print(f"[{i+1:2d}/{len(candidates)}] {var:10s} -> {bn:8s} | CV={cvm:.6f}±{cvs_se:.6f} | Test={tm:.6f} | {mk} {imp:+.2f}%")

results.append({"replaced_var":"None (baseline)","var_type":"-","basis":"poly",
                "cv_mse":base_cv,"cv_se":base_se,"test_mse":base_test,"test_r2":base_test_r2,
                "train_r2":ols_base.rsquared,"aic":float(ols_base.aic),"bic":float(ols_base.bic),
                "cv_mse_improvement_pct":0.0})
df_r = pd.DataFrame(results).sort_values("cv_mse").reset_index(drop=True)
df_r.to_csv(f"{OUT_DIR}/stage4_candidate_models.csv", index=False)

print("\n"+"="*70); print("Top 10 by CV MSE"); print("="*70)
print(df_r.head(10).to_string(index=False))
print("\n=== 各类型最优 ===")
for vt in ["interaction","square","linear"]:
    sub = df_r[df_r["var_type"]==vt]
    if len(sub)>0:
        b = sub.iloc[0]
        print(f"{vt:12s}: {b['replaced_var']:10s} -> {b['basis']:8s} | CV={b['cv_mse']:.6f} ({b['cv_mse_improvement_pct']:+.2f}%)")
best = df_r.iloc[0]
print(f"\n=== 最终推荐（CV MSE 最优）===")
print(f"  替换: {best['replaced_var']} -> {best['basis']}")
print(f"  CV MSE = {best['cv_mse']:.6f} ± {best['cv_se']:.6f}")
print(f"  Test MSE = {best['test_mse']:.6f}, Test R² = {best['test_r2']:.4f}")
print(f"  vs baseline: {best['cv_mse_improvement_pct']:+.2f}%")
print(f"\n[Saved] {OUT_DIR}/")
