"""
修复版：补跑 stage4 中失败的 7 个模型
- PE -> pow_0.5, pow_2, pow_-1 (3个)
- M20^2 -> cos, sec (2个)
- V20^2 -> cos, sec (2个)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
FACTORS = ["PB Ratio", "Momentum 60d", "20d Momentum", "20d Volatility",
           "Volatility 60d (ann.)", "Relative Turnover", "PE Ratio"]
SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 60d": "M60",
         "20d Momentum": "M20", "20d Volatility": "V20",
         "Volatility 60d (ann.)": "V60", "Relative Turnover": "RT", "PE Ratio": "PE"}
FREQ_THRESH = 3 / 5
RANDOM_STATE = 42

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

scaler_raw = StandardScaler()
X_train_raw_s = scaler_raw.fit_transform(X_train_w)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_factor_idx = {f: i for i, f in enumerate(factor_cols)}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lasso_cv = LassoCV(alphas=np.logspace(-5, 1, 200), cv=kf, random_state=RANDOM_STATE, max_iter=200000)
lasso_cv.fit(X_train_s, y_train)
mean_per_alpha = np.mean(lasso_cv.mse_path_, axis=1)
std_per_alpha = np.std(lasso_cv.mse_path_, axis=1, ddof=1)
best_idx = int(np.argmin(mean_per_alpha))
threshold = float(mean_per_alpha[best_idx]) + float(std_per_alpha[best_idx])
best_alpha = float(np.max(lasso_cv.alphas_[mean_per_alpha <= threshold]))
lasso_full = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
lasso_full.fit(X_train_s, y_train)
sel_count = np.zeros(n_poly, dtype=int)
for tr_idx, _ in kf.split(X_train_s):
    m = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    sel_count += (np.abs(m.coef_) > 1e-10).astype(int)
combined_mask = (np.abs(lasso_full.coef_) > 1e-10) & ((sel_count / 5) >= FREQ_THRESH)
selected = [feature_names[i] for i in range(n_poly) if combined_mask[i]]
selected_idx = [list(feature_names).index(f) for f in selected]
print(f"选中 {len(selected)} 个变量: {selected}")

# === 基函数 ===
def safe_clip(z, lo, hi): return np.clip(z, lo, hi)
def g_pow_a(z, a): return np.power(1.0 + safe_clip(z, -0.999, None), a) - 1.0
def g_cos(x): return np.cos(x)
def g_sec(x): return 1.0 / np.cos(safe_clip(x, -1.55, 1.55))

# 修复：对 PE 这种一次项，pow_a 的输入应该是标准化后的 PE 值（一维）
# 对 M20^2 这种平方项，cos/sec 的输入是标准化后的 M20 原始值（一维）

kf_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def run_model(X_tr, X_te, label):
    """跑一个模型，返回指标"""
    X_tr_int = sm.add_constant(X_tr, has_constant='add')
    X_te_int = sm.add_constant(X_te, has_constant='add')
    ols = sm.OLS(y_train, X_tr_int).fit()
    y_tr_pred = ols.predict(X_tr_int)
    y_te_pred = ols.predict(X_te_int)
    # 用 LinearRegression 做 CV（避免 Lasso alpha=0 警告）
    from sklearn.linear_model import LinearRegression
    cv_scores = -cross_val_score(LinearRegression(), X_tr, y_train,
                                 cv=kf_cv, scoring="neg_mean_squared_error")
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

def build_cols_with_replacement(replace_var, basis_func, var_type):
    """构造 11 列：replace_var 用 basis_func 替换，其余原 poly"""
    train_cols = []; test_cols = []
    for var in selected:
        if var == replace_var:
            if var_type == "square":
                # M20^2 -> 取原始 M20 标准化值，过 cos/sec
                base = var.replace("^2", "")
                idx = raw_factor_idx[base]
                tr_x = X_train_raw_s[:, idx]
                te_x = X_test_raw_s[:, idx]
                train_cols.append(basis_func(tr_x))
                test_cols.append(basis_func(te_x))
            elif var_type == "linear":
                # PE -> 取原始 PE 标准化值，过 pow_a
                idx = raw_factor_idx[var]
                tr_x = X_train_raw_s[:, idx]
                te_x = X_test_raw_s[:, idx]
                train_cols.append(basis_func(tr_x))
                test_cols.append(basis_func(te_x))
        else:
            idx = list(feature_names).index(var)
            train_cols.append(X_train_s[:, idx])
            test_cols.append(X_test_s[:, idx])
    return np.column_stack(train_cols), np.column_stack(test_cols)

# 补跑 7 个失败的
fix_models = [
    ("PE", "linear", "pow_0.5", lambda z: g_pow_a(z, 0.5)),
    ("PE", "linear", "pow_2", lambda z: g_pow_a(z, 2)),
    ("PE", "linear", "pow_-1", lambda z: g_pow_a(z, -1)),
    ("M20^2", "square", "cos", g_cos),
    ("M20^2", "square", "sec", g_sec),
    ("V20^2", "square", "cos", g_cos),
    ("V20^2", "square", "sec", g_sec),
]

results = []
for rv, vt, bn, bf in fix_models:
    try:
        X_tr, X_te = build_cols_with_replacement(rv, bf, vt)
        r = run_model(X_tr, X_te, {"replace_var": rv, "var_type": vt, "basis": bn})
        results.append(r)
        print(f"[Fix] {rv} -> {bn}: CV MSE={r['cv_mse']:.6f}, AIC={r['aic']:.2f}, "
              f"shape={X_tr.shape}")
    except Exception as e:
        print(f"[Fix] {rv} -> {bn} FAILED: {e}")

# 保存
df_fix = pd.DataFrame(results)
print("\n[Fix] 补跑结果:")
print(df_fix.to_string(index=False))
df_fix.to_csv("/workspace/output_KS_stage4/stage4_fix_models.csv", index=False)
print("[Saved] /workspace/output_KS_stage4/stage4_fix_models.csv")
