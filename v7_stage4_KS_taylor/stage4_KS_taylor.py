"""
快手 (1024.HK) 量化策略 - 阶段 4：泰勒基函数替换（单项 ablation）
基于 stage123_KS.py 的 Lasso 1-SE 选出的 11 个变量

替换规则：
- 一次项 x_i → e^x, (1+x)^a  (a ∈ {0.5, 2, -1})
- 平方项 x_i² → cos(x_i), sec(x_i)
- 交互项 x_i·x_j → ln(1+z), sin(z), tanh(z), e^z-1  (z = x_i·x_j)

每次只替换一个项，固定其余 10 项为原多项式形式。
共 51 个单项替换 + 1 基线 = 52 个模型。
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# === 配置 ===
SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
OUT_DIR = "/workspace/output_KS_stage4"
os.makedirs(OUT_DIR, exist_ok=True)

FACTORS = [
    "PB Ratio", "Momentum 60d", "20d Momentum", "20d Volatility",
    "Volatility 60d (ann.)", "Relative Turnover", "PE Ratio",
]
SHORT = {
    "y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 60d": "M60",
    "20d Momentum": "M20", "20d Volatility": "V20",
    "Volatility 60d (ann.)": "V60", "Relative Turnover": "RT", "PE Ratio": "PE",
}
FREQ_THRESH = 3 / 5
RANDOM_STATE = 42

# === 1. 数据准备 + Lasso 选变量（与 stage123_KS 完全一致）===
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

# Lasso 1-SE
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lasso_cv = LassoCV(alphas=np.logspace(-5, 1, 200), cv=kf, random_state=RANDOM_STATE, max_iter=200000)
lasso_cv.fit(X_train_s, y_train)
mean_per_alpha = np.mean(lasso_cv.mse_path_, axis=1)
std_per_alpha = np.std(lasso_cv.mse_path_, axis=1, ddof=1)
best_idx = int(np.argmin(mean_per_alpha))
cv_mse_min = float(mean_per_alpha[best_idx])
threshold = cv_mse_min + float(std_per_alpha[best_idx])
feasible = mean_per_alpha <= threshold
best_alpha = float(np.max(lasso_cv.alphas_[feasible]))

lasso_full = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
lasso_full.fit(X_train_s, y_train)
sel_count = np.zeros(n_poly, dtype=int)
for tr_idx, _ in kf.split(X_train_s):
    m = Lasso(alpha=best_alpha, max_iter=200000, random_state=RANDOM_STATE)
    m.fit(X_train_s[tr_idx], y_train[tr_idx])
    sel_count += (np.abs(m.coef_) > 1e-10).astype(int)

full_nonzero_mask = np.abs(lasso_full.coef_) > 1e-10
freq_mask = (sel_count / 5) >= FREQ_THRESH
combined_mask = full_nonzero_mask & freq_mask
selected = [feature_names[i] for i in range(n_poly) if combined_mask[i]]
selected_idx = [list(feature_names).index(f) for f in selected]
print(f"[Stage 2] 选中 {len(selected)} 个变量: {selected}")

# 同时拿到原始单因子标准化值（用于构造 cos(x_i), sec(x_i) 等）
# 注意：scaler 是对多项式特征做标准化，单因子标准化需要单独做
scaler_raw = StandardScaler()
X_train_raw_s = scaler_raw.fit_transform(X_train_w)  # (n_train, 7)
X_test_raw_s = scaler_raw.transform(X_test_w)
raw_factor_idx = {f: i for i, f in enumerate(factor_cols)}

# === 2. 基函数定义 ===
def safe_clip(z, lo, hi):
    return np.clip(z, lo, hi)

def g_exp(z):
    """e^z - 1, 泰勒一阶 = z"""
    return np.expm1(safe_clip(z, -3, 3))

def g_pow_a(z, a):
    """(1+z)^a - 1, 泰勒一阶 = a*z"""
    return np.power(1 + safe_clip(z, -0.999, None), a) - 1

def g_cos(x):
    """cos(x), 泰勒 = 1 - x²/2, 替换 x²"""
    return np.cos(x)

def g_sec(x):
    """sec(x), 泰勒 = 1 + x²/2, 替换 x²"""
    return 1.0 / np.cos(safe_clip(x, -1.55, 1.55))

def g_ln1pz(z):
    """ln(1+z), 泰勒一阶 = z"""
    return np.log1p(safe_clip(z, -0.999, None))

def g_sin(z):
    """sin(z), 泰勒一阶 = z"""
    return np.sin(safe_clip(z, -3*np.pi, 3*np.pi))

def g_tanh(z):
    """tanh(z), 泰勒一阶 = z"""
    return np.tanh(safe_clip(z, -3, 3))

def g_exp1pz(z):
    """e^z - 1, 泰勒一阶 = z (交互项用)"""
    return np.expm1(safe_clip(z, -3, 3))

# 候选基函数映射
# (变量名, 变量类型, 候选基函数列表)
# 候选基函数: (basis_name, transform_func)
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
    ("exp", g_exp1pz),
]

# === 3. 构造候选模型 ===
def get_poly_feature(name):
    """从多项式特征矩阵里取一列"""
    idx = list(feature_names).index(name)
    return X_train_s[:, idx], X_test_s[:, idx]

def get_raw_factor(name):
    """从原始单因子标准化矩阵取一列（用于 cos/sec）"""
    idx = raw_factor_idx[name]
    return X_train_raw_s[:, idx], X_test_raw_s[:, idx]

def build_feature_matrix(replace_var, basis_name, basis_func, var_type):
    """构造 11 列特征矩阵：其中 replace_var 用 basis_func 替换，其余保持原多项式"""
    train_cols = []
    test_cols = []
    feature_labels = []
    for var in selected:
        if var == replace_var:
            # 替换
            if var_type == "square":
                # 平方项 x_i² → 基函数(x_i)
                base_factor = var.split()[0]  # 例如 "M20^2" -> "M20"
                tr_x, te_x = get_raw_factor(base_factor)
                tr_new = basis_func(tr_x)
                te_new = basis_func(te_x)
            elif var_type == "linear":
                # 一次项 x_i → 基函数(x_i)
                tr_x, te_x = get_raw_factor(var)
                tr_new = basis_func(tr_x)
                te_new = basis_func(te_x)
            elif var_type == "interaction":
                # 交互项 x_i·x_j → 基函数(z), z = x_i * x_j
                tr_z, te_z = get_poly_feature(var)
                tr_new = basis_func(tr_z)
                te_new = basis_func(te_z)
            train_cols.append(tr_new)
            test_cols.append(te_new)
            feature_labels.append(f"{basis_name}({var})")
        else:
            tr_col, te_col = get_poly_feature(var)
            train_cols.append(tr_col)
            test_cols.append(te_col)
            feature_labels.append(var)
    return np.column_stack(train_cols), np.column_stack(test_cols), feature_labels

def classify_var(var):
    """判断变量类型"""
    if " " in var:  # 交互项
        return "interaction"
    elif "^2" in var or "²" in var:  # 平方项
        return "square"
    else:  # 一次项
        return "linear"

def get_base_factor_of_square(var):
    """M20^2 -> M20"""
    return var.replace("^2", "").replace("²", "")

# === 4. 跑所有候选模型 ===
results = []
kf_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 4.1 基线模型（全部原多项式）
X_train_base = X_train_s[:, selected_idx]
X_test_base = X_test_s[:, selected_idx]
X_train_base_int = sm.add_constant(X_train_base)
X_test_base_int = sm.add_constant(X_test_base)
ols_base = sm.OLS(y_train, X_train_base_int).fit()
y_train_pred = ols_base.predict(X_train_base_int)
y_test_pred = ols_base.predict(X_test_base_int)
cv_scores_base = -cross_val_score(
    Lasso(alpha=0, max_iter=200000, fit_intercept=True),  # 等价 OLS
    X_train_base, y_train, cv=kf_cv, scoring="neg_mean_squared_error"
)
results.append({
    "replace_var": "(baseline)",
    "var_type": "-",
    "basis": "poly",
    "train_mse": mean_squared_error(y_train, y_train_pred),
    "test_mse": mean_squared_error(y_test, y_test_pred),
    "cv_mse": float(np.mean(cv_scores_base)),
    "cv_std": float(np.std(cv_scores_base, ddof=1)),
    "r2": ols_base.rsquared, "r2_adj": ols_base.rsquared_adj,
    "aic": ols_base.aic, "bic": ols_base.bic,
    "n_features": len(selected),
})
print(f"\n[Stage 4] 基线 (全 poly): CV MSE={results[-1]['cv_mse']:.6f}, "
      f"AIC={ols_base.aic:.2f}, BIC={ols_base.bic:.2f}")

# 4.2 单项替换
total_models = 0
for var in selected:
    vtype = classify_var(var)
    if vtype == "linear":
        bases = LINEAR_BASES
    elif vtype == "square":
        bases = SQUARE_BASES
    else:
        bases = INTERACTION_BASES
    
    for basis_name, basis_func in bases:
        try:
            X_tr, X_te, labels = build_feature_matrix(var, basis_name, basis_func, vtype)
            X_tr_int = sm.add_constant(X_tr)
            X_te_int = sm.add_constant(X_te)
            ols = sm.OLS(y_train, X_tr_int).fit()
            y_tr_pred = ols.predict(X_tr_int)
            y_te_pred = ols.predict(X_te_int)
            cv_scores = -cross_val_score(
                Lasso(alpha=0, max_iter=200000, fit_intercept=True),
                X_tr, y_train, cv=kf_cv, scoring="neg_mean_squared_error"
            )
            results.append({
                "replace_var": var,
                "var_type": vtype,
                "basis": basis_name,
                "train_mse": mean_squared_error(y_train, y_tr_pred),
                "test_mse": mean_squared_error(y_test, y_te_pred),
                "cv_mse": float(np.mean(cv_scores)),
                "cv_std": float(np.std(cv_scores, ddof=1)),
                "r2": ols.rsquared, "r2_adj": ols.rsquared_adj,
                "aic": ols.aic, "bic": ols.bic,
                "n_features": len(selected),
            })
            total_models += 1
            print(f"[Stage 4] {var} -> {basis_name}: CV MSE={results[-1]['cv_mse']:.6f}, "
                  f"AIC={ols.aic:.2f}")
        except Exception as e:
            print(f"[Stage 4] {var} -> {basis_name} FAILED: {e}")
            results.append({
                "replace_var": var, "var_type": vtype, "basis": basis_name,
                "train_mse": np.nan, "test_mse": np.nan, "cv_mse": np.nan,
                "cv_std": np.nan, "r2": np.nan, "r2_adj": np.nan,
                "aic": np.nan, "bic": np.nan, "n_features": len(selected),
            })

print(f"\n[Stage 4] 共跑 {total_models} 个单项替换 + 1 基线 = {total_models+1} 个模型")

# === 5. 汇总 + 排序 ===
df_res = pd.DataFrame(results)
df_res["cv_mse_diff_vs_baseline"] = df_res["cv_mse"] - df_res.loc[0, "cv_mse"]
df_res["cv_mse_pct_change"] = (df_res["cv_mse"] - df_res.loc[0, "cv_mse"]) / df_res.loc[0, "cv_mse"] * 100
df_res = df_res.sort_values("cv_mse", ascending=True).reset_index(drop=True)

print("\n[Stage 4] 全部候选模型（按 CV MSE 升序）:")
print(df_res.to_string(index=False))

print("\n[Stage 4] Top-10 最优（CV MSE 最低）:")
print(df_res.head(10).to_string(index=False))

print("\n[Stage 4] 各变量类型的最优替换:")
for vtype in ["linear", "square", "interaction"]:
    sub = df_res[df_res["var_type"] == vtype].head(3)
    print(f"\n--- {vtype} 类型 Top-3 ---")
    print(sub.to_string(index=False))

# 保存
df_res.to_csv(f"{OUT_DIR}/stage4_candidate_models.csv", index=False)
print(f"\n[Saved] {OUT_DIR}/stage4_candidate_models.csv")
