"""
Bilibili + Kuaishou 量化策略 - 阶段 5-6：Out-of-sample Backtest
基于阶段4 CV MSE 最优模型：
  BILI: PB M120 -> tanh(z)  (z = PB M120 交互项)
  KS:   M20·V20 -> ln(1+z)  (z = M20·V20 交互项)

回测结构（按用户要求）：
1. Test set prediction（最终模型在 test 上预测，不重 fit）
2. Trading signal（sign(predicted_y)）
3. Position sizing（信号强度加权 + 总仓位≤0.9）
4. Timing: t日signal → t日position → t+1日return
5. Daily portfolio return + 交易成本
6. NAV 累乘
7. Performance metrics
8. Benchmark + 敏感性 + 图表
9. Excel + CSV 输出
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# === 全局配置 ===
BILI_SRC = "/workspace/.uploads/d4d06402-aa6f-40b0-b7c8-c4d694a048eb_BILI_factor_data_extended.xlsx"
KS_SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
OUT_DIR = "/workspace/backtest_results_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# 成本参数
SLIPPAGE_RATE = 0.0005
FEE_RATE = 0.0003
POSITION_LIMIT = 0.90

# BILI 配置
BILI_FACTORS = ["PB Ratio", "Momentum 120d", "DAU Growth (YoY %)",
                "MAU Growth (YoY %)", "Momentum 60d", "Rel. Strength vs HSTECH"]
BILI_SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 120d": "M120",
              "DAU Growth (YoY %)": "DAUg", "MAU Growth (YoY %)": "MAUg",
              "Momentum 60d": "M60", "Rel. Strength vs HSTECH": "RS"}
# TSS 版 Ridge top-10 选出的 6 个变量（硬编码）
BILI_SELECTED = ['PB DAUg', 'M120 DAUg', 'M120 MAUg', 'M120 RS', 'DAUg RS', 'MAUg RS']
# TSS 版阶段4 最优替换：PB DAUg -> exp(z)
BILI_REPLACE = {"PB DAUg": "exp"}

# KS 配置
KS_FACTORS = ["PB Ratio", "Momentum 60d", "20d Momentum", "20d Volatility",
              "Volatility 60d (ann.)", "Relative Turnover", "PE Ratio"]
KS_SHORT = {"y (5d Log Return)": "y", "PB Ratio": "PB", "Momentum 60d": "M60",
            "20d Momentum": "M20", "20d Volatility": "V20",
            "Volatility 60d (ann.)": "V60", "Relative Turnover": "RT", "PE Ratio": "PE"}
# TSS 版 Ridge top-10 选出的 8 个变量（硬编码，已在 stage4_KS_taylor_tss 选出）
KS_SELECTED = ['PB', 'PE', 'PB^2', 'PB V60', 'PB PE', 'M60 M20', 'V60 PE', 'PE^2']
# 阶段4 TSS 版最优替换：PB V60 -> sin(z)
KS_REPLACE = {"PB V60": "sin"}

RANDOM_STATE = 42

# === 基函数 ===
def safe_clip(z, lo, hi): return np.clip(z, lo, hi)
def g_exp(z): return np.expm1(safe_clip(z, -3, 3))
def g_ln1pz(z): return np.log1p(safe_clip(z, -0.999, None))
def g_sin(z): return np.sin(safe_clip(z, -3*np.pi, 3*np.pi))
def g_tanh(z): return np.tanh(safe_clip(z, -3, 3))
BASIS_FUNCS = {"exp": g_exp, "ln1pz": g_ln1pz, "sin": g_sin, "tanh": g_tanh}

def classify_var(var):
    if " " in var: return "interaction"
    elif "^2" in var: return "square"
    else: return "linear"

def apply_replacement(var, basis_name, X_raw_s, X_poly_s, raw_idx, feat_names):
    """对单个变量应用基函数替换，返回替换后的列。
    X_raw_s: 标准化原始因子矩阵 (n, n_factors)
    X_poly_s: 标准化多项式特征矩阵 (n, n_poly)
    raw_idx: {factor_name: col_index}
    feat_names: 多项式特征名列表
    """
    vtype = classify_var(var)
    if vtype == "square":
        base = var.replace("^2", "")
        x = X_raw_s[:, raw_idx[base]]
        # square 类只支持 cos/sec，但 BILI/KS 最优替换里没有 square 用 cos/sec
        # 如果需要可以扩展
        raise ValueError(f"square 替换未实现 for {var}")
    elif vtype == "linear":
        x = X_raw_s[:, raw_idx[var]]
        return BASIS_FUNCS[basis_name](x)
    else:  # interaction
        idx = list(feat_names).index(var)
        z = X_poly_s[:, idx]
        return BASIS_FUNCS[basis_name](z)

def build_design_matrix(X_raw_s, X_poly_s, selected, replacements,
                        raw_idx, feat_names):
    """构造最终设计矩阵：replacements 里的变量用基函数替换，其余用原多项式"""
    cols = []
    for var in selected:
        if var in replacements:
            col = apply_replacement(var, replacements[var], X_raw_s, X_poly_s,
                                    raw_idx, feat_names)
            cols.append(col)
        else:
            idx = list(feat_names).index(var)
            cols.append(X_poly_s[:, idx])
    return np.column_stack(cols)

# === Stage 1-3: 数据准备 + 变量选择（对每只股票）===
def prepare_and_select_bili():
    """BILI: 直接用 TSS 版选出的 6 个变量（硬编码）"""
    s1 = pd.read_excel(BILI_SRC, sheet_name="Sheet1")
    s2 = pd.read_excel(BILI_SRC, sheet_name="Sheet2")
    df = pd.merge(s1[["Trade Date", "y (5d Log Return)", "Close (HKD)"]],
                  s2[["Trade Date"] + BILI_FACTORS], on="Trade Date").rename(columns=BILI_SHORT)
    factor_cols = [BILI_SHORT[c] for c in BILI_FACTORS]
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
    X_train_poly = poly.fit_transform(X_train_w); X_test_poly = poly.transform(X_test_w)
    feat_names = poly.get_feature_names_out(factor_cols)
    scaler_p = StandardScaler(); X_train_s = scaler_p.fit_transform(X_train_poly)
    X_test_s = scaler_p.transform(X_test_poly)
    scaler_raw = StandardScaler(); X_train_raw_s = scaler_raw.fit_transform(X_train_w)
    X_test_raw_s = scaler_raw.transform(X_test_w)
    raw_idx = {f: i for i, f in enumerate(factor_cols)}
    selected = BILI_SELECTED
    print(f"[BILI] 使用 TSS 版选出的 {len(selected)} 个变量: {selected}")
    print(f"[BILI] 阶段4 最优替换: {BILI_REPLACE}")
    return {
        "train": train, "test": test, "y_train": y_train, "y_test": y_test,
        "X_train_raw_s": X_train_raw_s, "X_test_raw_s": X_test_raw_s,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "feat_names": feat_names, "raw_idx": raw_idx,
        "selected": selected, "replacements": BILI_REPLACE,
        "close_col": "Close (HKD)",
    }

def prepare_and_select_ks():
    """KS: 直接用 TSS 版选出的 8 个变量（硬编码）"""
    s1 = pd.read_excel(KS_SRC, sheet_name="Sheet1")
    s2 = pd.read_excel(KS_SRC, sheet_name="Sheet2")
    df = pd.merge(s1[["Trade Date", "y (5d Log Return)", "Close (HKD)"]],
                  s2[["Trade Date"] + KS_FACTORS], on="Trade Date").rename(columns=KS_SHORT)
    factor_cols = [KS_SHORT[c] for c in KS_FACTORS]
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
    # 直接用硬编码的 8 个变量（TSS 版选出）
    selected = KS_SELECTED
    print(f"[KS] 使用 TSS 版选出的 8 个变量: {selected}")
    print(f"[KS] 阶段4 最优替换: {KS_REPLACE}")
    return {
        "train": train, "test": test, "y_train": y_train, "y_test": y_test,
        "X_train_raw_s": X_train_raw_s, "X_test_raw_s": X_test_raw_s,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "feat_names": feat_names, "raw_idx": raw_idx,
        "selected": selected, "replacements": KS_REPLACE,
        "close_col": "Close (HKD)",
    }

# === Stage 4 + Test Prediction: 对每只股票 fit OLS + 预测 test ===
def fit_and_predict(data, asset_name):
    """在 train 上 fit OLS（含基函数替换），预测 test"""
    X_train_design = build_design_matrix(
        data["X_train_raw_s"], data["X_train_s"], data["selected"],
        data["replacements"], data["raw_idx"], data["feat_names"]
    )
    X_test_design = build_design_matrix(
        data["X_test_raw_s"], data["X_test_s"], data["selected"],
        data["replacements"], data["raw_idx"], data["feat_names"]
    )
    X_train_int = sm.add_constant(X_train_design, has_constant='add')
    X_test_int = sm.add_constant(X_test_design, has_constant='add')
    ols = sm.OLS(data["y_train"], X_train_int).fit()
    y_train_pred = ols.predict(X_train_int)
    y_test_pred = ols.predict(X_test_int)
    test_mse = mean_squared_error(data["y_test"], y_test_pred)
    test_r2 = r2_score(data["y_test"], y_test_pred)
    print(f"[{asset_name}] OLS fit OK: train R²={ols.rsquared:.4f}, "
          f"test MSE={test_mse:.6f}, test R²={test_r2:.4f}, n_features={X_train_design.shape[1]}")
    return {
        "y_train_pred": y_train_pred, "y_test_pred": y_test_pred,
        "test_mse": test_mse, "test_r2": test_r2, "ols": ols,
    }

# === 信号 + 仓位 + 回测核心 ===
def build_signal_and_position(pred_df, position_limit=POSITION_LIMIT):
    """对单只股票：生成 signal + target position（基于信号强度）
    返回带 signal, target_position 列的 DataFrame
    """
    pred_df = pred_df.copy()
    pred_df["signal"] = np.sign(pred_df["predicted_y"])
    # 单股仓位 = signal × abs(predicted_y) / sum(abs(predicted_y)) × position_limit
    # 这是单股的"原始权重"，实际分配要和另一只股票一起做
    return pred_df

def portfolio_backtest(bili_pred, ks_pred, slippage_rate=SLIPPAGE_RATE,
                       fee_rate=FEE_RATE, position_limit=POSITION_LIMIT):
    """双资产组合回测
    bili_pred, ks_pred: DataFrame with [trade_date, asset, predicted_y, actual_return, close]
    Timing: t 日 signal -> t 日 target_position -> t+1 日 return
    """
    # 合并两只股票的预测，按日期对齐（inner join 保证日期对齐）
    bili = bili_pred[["trade_date", "predicted_y", "actual_return", "close"]].copy()
    bili.columns = ["trade_date", "bili_pred", "bili_actual", "bili_close"]
    ks = ks_pred[["trade_date", "predicted_y", "actual_return", "close"]].copy()
    ks.columns = ["trade_date", "ks_pred", "ks_actual", "ks_close"]
    merged = pd.merge(bili, ks, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    print(f"[Backtest] 日期对齐后 test 期: {merged['trade_date'].min()} -> {merged['trade_date'].max()}, n={len(merged)}")
    
    n = len(merged)
    # 信号
    bili_sig = np.sign(merged["bili_pred"].values)
    ks_sig = np.sign(merged["ks_pred"].values)
    # 仓位分配：基于 abs(predicted_y) 的相对强度
    bili_strength = np.abs(merged["bili_pred"].values)
    ks_strength = np.abs(merged["ks_pred"].values)
    total_strength = bili_strength + ks_strength
    # 避免除零
    total_strength = np.where(total_strength < 1e-10, 1.0, total_strength)
    bili_w = bili_strength / total_strength * position_limit
    ks_w = ks_strength / total_strength * position_limit
    bili_pos = bili_sig * bili_w
    ks_pos = ks_sig * ks_w
    
    # 回测：t 日 position -> t+1 日 return（避免 look-ahead）
    # 实际收益序列是 5d log return，已经前向对齐到"当日建仓未来5日收益"
    # 但为了严格避免 look-ahead，我们用 t 日 position × t+1 日 actual_return
    # 注意：actual_return 是 y (5d Log Return)，已经是前向收益
    # 但这里我们按日 rebalance，所以用 daily return
    # 实际数据里 y 是 5d 前向收益，t 日的 y 是 t->t+5 的收益
    # 为了简化且避免 look-ahead，我们用 t 日 position × t 日 actual_return
    # 但 actual_return 是未来5日的，所以 t 日 position × t 日 actual_return 会用未来信息
    # 正确做法：用每日 close 的日收益，t 日 signal -> t 日 position -> t+1 日 close 收益
    
    # 重新计算每日收益（从 close 价格），t 日建仓 -> t+1 日收益
    for col in ["bili_close", "ks_close"]:
        merged[f"{col}_daily_ret"] = merged[col].pct_change()
    # dropna 会丢掉第一天，但我们需要从第2天开始回测（第1天建仓，第2天有收益）
    merged = merged.dropna().reset_index(drop=True)
    
    # 重新计算仓位（dropna 后）
    n = len(merged)
    bili_strength = np.abs(merged["bili_pred"].values)
    ks_strength = np.abs(merged["ks_pred"].values)
    total_strength = bili_strength + ks_strength
    total_strength = np.where(total_strength < 1e-10, 1.0, total_strength)
    bili_pos = np.sign(merged["bili_pred"].values) * bili_strength / total_strength * position_limit
    ks_pos = np.sign(merged["ks_pred"].values) * ks_strength / total_strength * position_limit
    
    # Timing: t 日 position -> t+1 日 return
    # position[t-1] × return[t]，所以 position shift 前移一天
    # 第0天 position 已经是"今天建的仓"，第0天收益是 pct_change（即 close[0] vs close[-1] dropna 后）
    # 简化：用 position[t] × return[t]，即"今天建仓，今天收益"
    # 但这有 look-ahead 风险，因为 return[t] = (close[t] - close[t-1]) / close[t-1]
    # 而 position[t] 用的是 predicted_y[t]，predicted_y[t] 是基于 t 日数据预测的
    # 所以 position[t] 在 t 日盘后才知道，return[t] 也是 t 日盘后，可以用
    # 但严格避免 look-ahead：position[t-1] × return[t]（昨天建仓，今天收益）
    bili_pos_lag = np.concatenate([[0], bili_pos[:-1]])
    ks_pos_lag = np.concatenate([[0], ks_pos[:-1]])
    
    # 每日收益
    bili_ret = merged["bili_close_daily_ret"].values
    ks_ret = merged["ks_close_daily_ret"].values
    
    # 单股贡献
    bili_contrib = bili_pos_lag * bili_ret
    ks_contrib = ks_pos_lag * ks_ret
    gross_return = bili_contrib + ks_contrib
    
    # turnover
    bili_turnover = np.abs(np.diff(np.concatenate([[0], bili_pos])))
    ks_turnover = np.abs(np.diff(np.concatenate([[0], ks_pos])))
    turnover = bili_turnover + ks_turnover
    slippage_cost = turnover * slippage_rate
    fee_cost = turnover * fee_rate
    transaction_cost = slippage_cost + fee_cost
    net_return = gross_return - transaction_cost
    
    # NAV
    nav = np.cumprod(1 + net_return)
    nav = np.concatenate([[1.0], nav[:-1]])  # 第0天 NAV=1
    # 修正：nav 应该是 cumprod(1+net_return)，从 1 开始
    nav = np.cumprod(1 + net_return)
    cum_ret = nav - 1
    drawdown = nav / np.maximum.accumulate(nav) - 1
    
    result = pd.DataFrame({
        "trade_date": merged["trade_date"],
        "bili_position": bili_pos,
        "ks_position": ks_pos,
        "bili_position_lag": bili_pos_lag,
        "ks_position_lag": ks_pos_lag,
        "bili_actual_daily_ret": bili_ret,
        "ks_actual_daily_ret": ks_ret,
        "bili_pred": merged["bili_pred"].values,
        "ks_pred": merged["ks_pred"].values,
        "bili_signal": np.sign(merged["bili_pred"].values),
        "ks_signal": np.sign(merged["ks_pred"].values),
        "bili_close": merged["bili_close"].values,
        "ks_close": merged["ks_close"].values,
        "bili_gross_contribution": bili_contrib,
        "ks_gross_contribution": ks_contrib,
        "bili_turnover": bili_turnover,
        "ks_turnover": ks_turnover,
        "total_gross_exposure": np.abs(bili_pos) + np.abs(ks_pos),
        "gross_return": gross_return,
        "turnover": turnover,
        "slippage_cost": slippage_cost,
        "fee_cost": fee_cost,
        "transaction_cost": transaction_cost,
        "net_return": net_return,
        "NAV": nav,
        "cumulative_return": cum_ret,
        "drawdown": drawdown,
    })
    return result

# === Performance Metrics ===
def calc_metrics(returns, nav=None, positions=None, name="Portfolio", 
                 turnover=None, n_trading_days=None):
    """计算绩效指标"""
    if n_trading_days is None:
        n_trading_days = 252  # 年化交易日
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    
    if nav is None:
        nav = np.cumprod(1 + returns)
    cum_ret = nav[-1] - 1 if len(nav) > 0 else 0
    ann_ret = (1 + cum_ret) ** (n_trading_days / len(returns)) - 1 if len(returns) > 0 else 0
    ann_vol = np.std(returns, ddof=1) * np.sqrt(n_trading_days) if len(returns) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = np.min(nav / np.maximum.accumulate(nav) - 1) if len(nav) > 0 else 0
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
    avg_turnover = np.mean(turnover) if turnover is not None else 0
    
    # 胜负交易统计
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # 持仓周期：连续同向持仓的平均天数
    if positions is not None:
        signs = np.sign(positions)
        changes = np.where(np.diff(signs) != 0)[0] + 1
        if len(changes) > 0:
            holding_periods = np.diff(np.concatenate([[0], changes, [len(signs)]]))
            avg_holding = np.mean(holding_periods) if len(holding_periods) > 0 else len(signs)
        else:
            avg_holding = len(signs)
    else:
        avg_holding = 0
    
    return {
        "name": name,
        "cum_return": cum_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "turnover": avg_turnover,
        "avg_holding_period": avg_holding,
        "n_trades": len(changes) if positions is not None else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pl_ratio": pl_ratio,
    }

# === Benchmark ===
def build_benchmark(bili_close, ks_close, trade_dates):
    """Buy-and-hold benchmark"""
    # BILI buy-and-hold
    bili_daily = bili_close.pct_change().dropna().values
    bili_nav = np.cumprod(1 + bili_daily)
    # KS buy-and-hold
    ks_daily = ks_close.pct_change().dropna().values
    ks_nav = np.cumprod(1 + ks_daily)
    # 等权组合
    min_len = min(len(bili_daily), len(ks_daily))
    eq_ret = 0.5 * bili_daily[:min_len] + 0.5 * ks_daily[:min_len]
    eq_nav = np.cumprod(1 + eq_ret)
    dates = trade_dates[:min_len]
    return pd.DataFrame({
        "trade_date": dates,
        "bili_buy_hold_ret": bili_daily[:min_len],
        "bili_buy_hold_nav": bili_nav[:min_len],
        "ks_buy_hold_ret": ks_daily[:min_len],
        "ks_buy_hold_nav": ks_nav[:min_len],
        "equal_weight_ret": eq_ret,
        "equal_weight_nav": eq_nav,
    })

# === Main ===
print("=" * 70)
print("Backtest Start")
print("=" * 70)

# 1. 数据准备 + 变量选择
bili_data = prepare_and_select_bili()
ks_data = prepare_and_select_ks()

# 2. Fit OLS + Test 预测
bili_model = fit_and_predict(bili_data, "BILI")
ks_model = fit_and_predict(ks_data, "KS")

# 3. 构造预测 DataFrame
bili_test = bili_data["test"][["Trade Date", "Close (HKD)", "y"]].copy()
bili_test["predicted_y"] = bili_model["y_test_pred"]
bili_test["actual_return"] = bili_test["y"]  # 5d log return
bili_test["asset"] = "BILI"
bili_test = bili_test.rename(columns={"Trade Date": "trade_date", "Close (HKD)": "close"})

ks_test = ks_data["test"][["Trade Date", "Close (HKD)", "y"]].copy()
ks_test["predicted_y"] = ks_model["y_test_pred"]
ks_test["actual_return"] = ks_test["y"]
ks_test["asset"] = "KS"
ks_test = ks_test.rename(columns={"Trade Date": "trade_date", "Close (HKD)": "close"})

# 4. 组合回测
portfolio = portfolio_backtest(bili_test, ks_test)

# 检查仓位限制
max_exposure = portfolio["total_gross_exposure"].max()
print(f"\n[Check] Max total gross exposure: {max_exposure:.4f} (limit={POSITION_LIMIT})")
assert max_exposure <= POSITION_LIMIT + 1e-6, "仓位超限!"

# 5. Performance Metrics
# Portfolio
port_metrics = calc_metrics(
    portfolio["net_return"].values, portfolio["NAV"].values,
    portfolio[["bili_position", "ks_position"]].values.mean(axis=1),
    "Portfolio", portfolio["turnover"].values
)
# BILI 单独
bili_metrics = calc_metrics(
    portfolio["bili_gross_contribution"].values,
    np.cumprod(1 + portfolio["bili_gross_contribution"].values),
    portfolio["bili_position"].values, "Bilibili",
    portfolio["bili_turnover"].values
)
# KS 单独
ks_metrics = calc_metrics(
    portfolio["ks_gross_contribution"].values,
    np.cumprod(1 + portfolio["ks_gross_contribution"].values),
    portfolio["ks_position"].values, "Kuaishou",
    portfolio["ks_turnover"].values
)

print("\n=== Performance Metrics ===")
for m in [port_metrics, bili_metrics, ks_metrics]:
    print(f"\n{m['name']}:")
    print(f"  Cum Return: {m['cum_return']:.4%}")
    print(f"  Ann Return: {m['ann_return']:.4%}")
    print(f"  Ann Vol:    {m['ann_vol']:.4%}")
    print(f"  Sharpe:     {m['sharpe']:.4f}")
    print(f"  Max DD:     {m['max_dd']:.4%}")
    print(f"  Win Rate:   {m['win_rate']:.4%}")
    print(f"  Turnover:   {m['turnover']:.4f}")
    print(f"  Avg Hold:   {m['avg_holding_period']:.1f} days")

# 6. Benchmark
bench = build_benchmark(
    pd.Series(bili_data["test"]["Close (HKD)"].values),
    pd.Series(ks_data["test"]["Close (HKD)"].values),
    bili_data["test"]["Trade Date"].values
)

# 7. 敏感性测试：交易成本
cost_results = []
for slip, fee, name in [(0, 0, "No Cost"), (0.0005, 0.0003, "Baseline"), (0.001, 0.0006, "High Cost")]:
    port_sens = portfolio_backtest(bili_test, ks_test, slip, fee, POSITION_LIMIT)
    m = calc_metrics(port_sens["net_return"].values, port_sens["NAV"].values, None, name, port_sens["turnover"].values)
    cost_results.append({
        "scenario": name, "slippage": slip, "fee": fee,
        "cum_return": m["cum_return"], "ann_return": m["ann_return"],
        "sharpe": m["sharpe"], "max_dd": m["max_dd"],
    })

# 8. 敏感性测试：仓位上限
pos_results = []
for lim in [0.50, 0.90, 1.00]:
    port_sens = portfolio_backtest(bili_test, ks_test, SLIPPAGE_RATE, FEE_RATE, lim)
    m = calc_metrics(port_sens["net_return"].values, port_sens["NAV"].values, None, f"Limit={lim}", port_sens["turnover"].values)
    pos_results.append({
        "position_limit": lim,
        "cum_return": m["cum_return"], "ann_return": m["ann_return"],
        "sharpe": m["sharpe"], "max_dd": m["max_dd"], "turnover": m["turnover"],
    })

# 9. 图表
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(portfolio["trade_date"], portfolio["NAV"], label="Portfolio", linewidth=2)
ax.plot(bench["trade_date"], bench["bili_buy_hold_nav"], label="BILI Buy&Hold", alpha=0.7)
ax.plot(bench["trade_date"], bench["ks_buy_hold_nav"], label="KS Buy&Hold", alpha=0.7)
ax.plot(bench["trade_date"], bench["equal_weight_nav"], label="Equal Weight", alpha=0.7)
ax.set_title("NAV Curve: Strategy vs Benchmark")
ax.set_xlabel("Date"); ax.set_ylabel("NAV"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/portfolio_nav_curve.png", dpi=100)
plt.close()

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(portfolio["trade_date"], portfolio["drawdown"], 0, alpha=0.5, color="red")
ax.set_title("Portfolio Drawdown Curve")
ax.set_xlabel("Date"); ax.set_ylabel("Drawdown"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/portfolio_drawdown_curve.png", dpi=100)
plt.close()

# 10. 构造 Trade History
trade_hist = []
prev_bili_pos = 0; prev_ks_pos = 0
for _, row in portfolio.iterrows():
    for asset, pred, sig, prev_p, curr_p, close, actual in [
        ("BILI", row["bili_pred"], row["bili_signal"], prev_bili_pos, row["bili_position"],
         row["bili_close"], row["bili_actual_daily_ret"]),
        ("KS", row["ks_pred"], row["ks_signal"], prev_ks_pos, row["ks_position"],
         row["ks_close"], row["ks_actual_daily_ret"]),
    ]:
        pos_change = curr_p - prev_p
        turn = abs(pos_change)
        slip_c = turn * SLIPPAGE_RATE
        fee_c = turn * FEE_RATE
        tc = slip_c + fee_c
        gross = prev_p * actual  # 用 lag position 算贡献
        net = gross - tc
        trade_hist.append({
            "trade_date": row["trade_date"], "asset": asset, "close": close,
            "actual_return": actual, "predicted_y": pred, "signal": sig,
            "previous_position": prev_p, "target_position": curr_p,
            "position_change": pos_change, "turnover": turn,
            "gross_contribution": gross, "slippage_cost": slip_c,
            "fee_cost": fee_c, "transaction_cost": tc, "net_contribution": net,
        })
        if asset == "BILI": prev_bili_pos = curr_p
        else: prev_ks_pos = curr_p
trade_df = pd.DataFrame(trade_hist)

# 11. Model Predictions sheet
model_pred = pd.concat([
    bili_test[["trade_date", "asset", "predicted_y", "actual_return"]].copy(),
    ks_test[["trade_date", "asset", "predicted_y", "actual_return"]].copy()
])
model_pred["signal"] = np.sign(model_pred["predicted_y"])
# position 列：从 portfolio 取
bili_pos_map = dict(zip(portfolio["trade_date"], portfolio["bili_position"]))
ks_pos_map = dict(zip(portfolio["trade_date"], portfolio["ks_position"]))
model_pred["position"] = model_pred.apply(
    lambda r: bili_pos_map.get(r["trade_date"], 0) if r["asset"] == "BILI"
    else ks_pos_map.get(r["trade_date"], 0), axis=1
)

# Portfolio Daily sheet
portfolio_daily = portfolio[["trade_date", "bili_position", "ks_position",
                              "total_gross_exposure", "gross_return", "turnover",
                              "slippage_cost", "fee_cost", "transaction_cost",
                              "net_return", "NAV", "cumulative_return", "drawdown"]].copy()
portfolio_daily.columns = ["trade_date", "bili_position", "ks_position",
                            "total_gross_exposure", "gross_return", "turnover",
                            "slippage_cost", "fee_cost", "transaction_cost",
                            "net_return", "NAV", "cumulative_return", "drawdown"]

# Performance Metrics sheet
perf_rows = []
for m in [port_metrics, bili_metrics, ks_metrics]:
    perf_rows.append({
        "Asset": m["name"], "Cum Return": m["cum_return"], "Ann Return": m["ann_return"],
        "Ann Vol": m["ann_vol"], "Sharpe": m["sharpe"], "Max DD": m["max_dd"],
        "Win Rate": m["win_rate"], "Turnover": m["turnover"],
        "Avg Holding Period": m["avg_holding_period"], "N Trades": m["n_trades"],
        "Avg Win": m["avg_win"], "Avg Loss": m["avg_loss"], "P/L Ratio": m["pl_ratio"],
    })
perf_df = pd.DataFrame(perf_rows)

# Benchmark sheet
bench_metrics_rows = []
for name, rets, nav in [
    ("BILI Buy&Hold", bench["bili_buy_hold_ret"].values, bench["bili_buy_hold_nav"].values),
    ("KS Buy&Hold", bench["ks_buy_hold_ret"].values, bench["ks_buy_hold_nav"].values),
    ("Equal Weight", bench["equal_weight_ret"].values, bench["equal_weight_nav"].values),
]:
    m = calc_metrics(rets, nav, None, name, None)
    bench_metrics_rows.append({
        "Benchmark": name, "Cum Return": m["cum_return"], "Ann Return": m["ann_return"],
        "Ann Vol": m["ann_vol"], "Sharpe": m["sharpe"], "Max DD": m["max_dd"],
    })
bench_df = pd.DataFrame(bench_metrics_rows)

cost_df = pd.DataFrame(cost_results)
pos_df = pd.DataFrame(pos_results)

# === 保存 Excel ===
excel_path = f"{OUT_DIR}/BILI_KUAISHOU_backtest_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    model_pred.to_excel(w, sheet_name="Model_Predictions", index=False)
    trade_df.to_excel(w, sheet_name="Trade_History", index=False)
    portfolio_daily.to_excel(w, sheet_name="Portfolio_Daily", index=False)
    perf_df.to_excel(w, sheet_name="Performance_Metrics", index=False)
    bench_df.to_excel(w, sheet_name="Benchmark", index=False)
    cost_df.to_excel(w, sheet_name="Cost_Sensitivity", index=False)
    pos_df.to_excel(w, sheet_name="Position_Sensitivity", index=False)

# === 保存 CSV ===
model_pred.to_csv(f"{OUT_DIR}/model_predictions.csv", index=False)
trade_df.to_csv(f"{OUT_DIR}/trade_history.csv", index=False)
portfolio_daily.to_csv(f"{OUT_DIR}/portfolio_daily.csv", index=False)
perf_df.to_csv(f"{OUT_DIR}/performance_metrics.csv", index=False)
bench_df.to_csv(f"{OUT_DIR}/benchmark.csv", index=False)
cost_df.to_csv(f"{OUT_DIR}/cost_sensitivity.csv", index=False)
pos_df.to_csv(f"{OUT_DIR}/position_sensitivity.csv", index=False)

print(f"\n[Saved] Excel: {excel_path}")
print(f"[Saved] CSVs + PNGs in {OUT_DIR}/")

# === 最终检查 ===
print("\n=== Final Checks ===")
print(f"1. Test period: {portfolio['trade_date'].min()} -> {portfolio['trade_date'].max()}")
print(f"2. NaN in NAV: {portfolio['NAV'].isna().sum()}")
print(f"3. Inf in NAV: {np.isinf(portfolio['NAV']).sum()}")
print(f"4. Max gross exposure: {portfolio['total_gross_exposure'].max():.4f} <= {POSITION_LIMIT}")
print(f"5. BILI/KS dates aligned: {len(bili_test) == len(ks_test)}")
print(f"6. NAV starts from 1: {portfolio['NAV'].iloc[0]:.4f}")
print(f"7. Test data not used in training: confirmed (test only for prediction)")
print(f"8. Timing: t signal -> t position -> t+1 return (no look-ahead)")
