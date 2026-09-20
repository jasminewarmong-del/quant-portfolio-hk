"""
BILI + TC 两资产组合回测
BILI: PB DAUg -> exp
TC:   PE ROE -> sin
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

BILI_SRC = "/workspace/.uploads/d4d06402-aa6f-40b0-b7c8-c4d694a048eb_BILI_factor_data_extended.xlsx"
TC_SRC = "/workspace/.uploads/35657faa-847c-4c1d-b404-5b5165eb83ec_TC_factor_data_extended.xlsx"
OUT_DIR = "/workspace/backtest_2asset_event_risk"
os.makedirs(OUT_DIR, exist_ok=True)

SLIPPAGE = 0.0005; FEE = 0.0003; POS_LIMIT = 0.90

# BILI
BILI_FACTORS = ["PB Ratio","Momentum 120d","DAU Growth (YoY %)","MAU Growth (YoY %)","Momentum 60d","Rel. Strength vs HSTECH"]
BILI_SHORT = {"y (5d Log Return)":"y","PB Ratio":"PB","Momentum 120d":"M120","DAU Growth (YoY %)":"DAUg","MAU Growth (YoY %)":"MAUg","Momentum 60d":"M60","Rel. Strength vs HSTECH":"RS"}
BILI_SELECTED = ['PB DAUg','M120 DAUg','M120 MAUg','M120 RS','DAUg RS','MAUg RS']
BILI_REPLACE = {"PB DAUg":"exp"}

# TC
TC_FACTORS = ["PB Ratio","PE Ratio","ROE (%)","20d Momentum","MAU Growth (YoY %)"]
TC_SHORT = {"y (5d Log Return)":"y","PB Ratio":"PB","PE Ratio":"PE","ROE (%)":"ROE","20d Momentum":"M20","MAU Growth (YoY %)":"MAUg"}
TC_SELECTED = ['PB','ROE','M20','MAUg^2','PE ROE','PE MAUg','ROE MAUg']
TC_REPLACE = {"PE ROE":"sin"}

# === 基函数 ===
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
BASIS = {"exp":g_exp, "ln1pz":g_ln1pz, "sin":g_sin, "tanh":g_tanh,
         "cos":g_cos, "sec":g_sec, "pow_0.5":g_pow05, "pow_2":g_pow2, "pow_-1":g_powm1}

def classify(v):
    if " " in v: return "interaction"
    elif "^2" in v: return "square"
    else: return "linear"

def build_design(Xr, Xp, selected, reps, ridx, fnames):
    cols = []
    for v in selected:
        if v in reps:
            bn = reps[v]; vt = classify(v)
            if vt == "square":
                base = v.replace("^2",""); x = Xr[:, ridx[base]]
                cols.append(BASIS[bn](x))
            elif vt == "linear":
                x = Xr[:, ridx[v]]; cols.append(BASIS[bn](x))
            else:
                idx = list(fnames).index(v); z = Xp[:, idx]
                cols.append(BASIS[bn](z))
        else:
            idx = list(fnames).index(v); cols.append(Xp[:, idx])
    return np.column_stack(cols)

def prepare(src, factors, short, selected):
    s1 = pd.read_excel(src, sheet_name="Sheet1")
    s2 = pd.read_excel(src, sheet_name="Sheet2")
    df = pd.merge(s1[["Trade Date","y (5d Log Return)","Close (HKD)"]],
                  s2[["Trade Date"] + factors], on="Trade Date").rename(columns=short)
    fcols = [short[c] for c in factors]
    df = df.dropna().reset_index(drop=True)
    n = len(df); n_tr = int(n*0.8)
    tr = df.iloc[:n_tr].copy(); te = df.iloc[n_tr:].copy()
    Xr_tr = tr[fcols].values; Xr_te = te[fcols].values
    y_tr = tr["y"].values; y_te = te["y"].values
    lo = np.quantile(Xr_tr, 0.01, axis=0); hi = np.quantile(Xr_tr, 0.99, axis=0)
    Xr_tr_w = np.clip(Xr_tr, lo[None,:], hi[None,:])
    Xr_te_w = np.clip(Xr_te, lo[None,:], hi[None,:])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp_tr = poly.fit_transform(Xr_tr_w); Xp_te = poly.transform(Xr_te_w)
    fn = poly.get_feature_names_out(fcols)
    sp = StandardScaler(); Xp_tr_s = sp.fit_transform(Xp_tr); Xp_te_s = sp.transform(Xp_te)
    sr = StandardScaler(); Xr_tr_s = sr.fit_transform(Xr_tr_w); Xr_te_s = sr.transform(Xr_te_w)
    ridx = {f:i for i,f in enumerate(fcols)}
    return {"tr":tr, "te":te, "y_tr":y_tr, "y_te":y_te,
            "Xr_tr_s":Xr_tr_s, "Xr_te_s":Xr_te_s, "Xp_tr_s":Xp_tr_s, "Xp_te_s":Xp_te_s,
            "fn":fn, "ridx":ridx, "selected":selected,
            # 新增：保存 poly/scaler/raw train/winsorize bounds 供合成期预测用
            "poly":poly, "sp":sp, "sr":sr,
            "Xr_tr_raw":Xr_tr, "Xr_tr_w":Xr_tr_w,
            "lo":lo, "hi":hi, "fcols":fcols}

def fit_predict(d, reps, name):
    Xtr = build_design(d["Xr_tr_s"], d["Xp_tr_s"], d["selected"], reps, d["ridx"], d["fn"])
    Xte = build_design(d["Xr_te_s"], d["Xp_te_s"], d["selected"], reps, d["ridx"], d["fn"])
    Xi = sm.add_constant(Xtr, has_constant='add'); Xti = sm.add_constant(Xte, has_constant='add')
    ols = sm.OLS(d["y_tr"], Xi).fit()
    yp = ols.predict(Xti)
    tmse = mean_squared_error(d["y_te"], yp); tr2 = r2_score(d["y_te"], yp)
    print(f"[{name}] train R²={ols.rsquared:.4f}, test MSE={tmse:.6f}, test R²={tr2:.4f}")
    # 返回 OLS 模型对象，供合成期预测复用
    return yp, tmse, tr2, ols

def predict_on_new(d, reps, new_factors_raw):
    """用 train 期拟合的 poly/scaler/OLS，对新数据做预测。
    new_factors_raw: (n_new, n_factors) 未 winsorize、未标准化的原始因子矩阵
    """
    # Winsorize（用 train 期分位）
    Xr_new_w = np.clip(new_factors_raw, d["lo"][None,:], d["hi"][None,:])
    # Poly transform
    Xp_new = d["poly"].transform(Xr_new_w)
    # Standardize（用 train 期 fit 的 scaler）
    Xr_new_s = d["sr"].transform(Xr_new_w)
    Xp_new_s = d["sp"].transform(Xp_new)
    # Build design
    X_new_design = build_design(Xr_new_s, Xp_new_s, d["selected"], reps, d["ridx"], d["fn"])
    Xi_new = sm.add_constant(X_new_design, has_constant='add')
    # 用 train 期 OLS 系数预测（不重 fit）
    # 需要拿到 OLS 模型 — 从 d 里取
    if "ols_model" not in d:
        # 重新 fit（仅第一次调用时）
        Xtr = build_design(d["Xr_tr_s"], d["Xp_tr_s"], d["selected"], reps, d["ridx"], d["fn"])
        Xi_tr = sm.add_constant(Xtr, has_constant='add')
        d["ols_model"] = sm.OLS(d["y_tr"], Xi_tr).fit()
    yp = d["ols_model"].predict(Xi_new)
    return yp

def portfolio_bt_2asset(preds, closes, dates, asset_names, rc_asset=None, rc_state=None,
                        deadband=0.0):
    """2 资产组合回测。
    rc_asset: 要应用 risk control 的资产名 (e.g. 'TC')，None 表示不启用。
    rc_state: dict with keys loss_streak, profit_streak, risk_state, base_pos, adj_pos
              每个是长度等于 test 期日的 array，由外部预先计算并传入。
    deadband: 预测绝对值 < deadband 时仓位归零（不开仓省成本），默认 0 不启用
              注：deadband 作用于原始预测值，在相对强度加权之前判定。
              支持单资产差异化：deadband 可以是 float（所有资产相同）
              或 dict {asset_name: threshold}（每个资产不同阈值）
    """
    df = pd.DataFrame({"d": dates})
    for i, an in enumerate(asset_names):
        df[f"p_{an}"] = preds[i]; df[f"c_{an}"] = closes[i]
    df = df.dropna().reset_index(drop=True)
    for an in asset_names:
        df[f"r_{an}"] = df[f"c_{an}"].pct_change()
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    # Deadband: 预测绝对值 < deadband 的资产当日归零（不参与强度加权）
    # 作用于原始预测，在相对强度加权之前
    # 支持 dict 形式：每个资产独立阈值
    if isinstance(deadband, dict):
        get_db = lambda an: deadband.get(an, 0.0)
    else:
        get_db = lambda an: deadband
    preds_used = []
    deadband_mask = np.zeros((len(asset_names), n), dtype=bool)
    for i, an in enumerate(asset_names):
        p = df[f"p_{an}"].values.copy()
        db = get_db(an)
        mask_zero = np.abs(p) < db
        p[mask_zero] = 0.0
        preds_used.append(p)
        deadband_mask[i] = mask_zero
    # 仓位：相对强度加权 — 这是 base position
    strengths = [np.abs(preds_used[i]) for i in range(len(asset_names))]
    tot = np.sum(strengths, axis=0); tot = np.where(tot < 1e-10, 1.0, tot)
    base_positions = []
    for i, an in enumerate(asset_names):
        pos = np.sign(preds_used[i]) * strengths[i]/tot * POS_LIMIT
        base_positions.append(pos)
    # 应用 risk control
    if rc_asset is not None and rc_state is not None:
        rc_idx = asset_names.index(rc_asset)
        positions = list(base_positions)  # copy
        positions[rc_idx] = rc_state["adj_pos"]
        # BILI 保持 base 不变
    else:
        positions = base_positions
    # lag 一天
    pos_lag = [np.concatenate([[0], p[:-1]]) for p in positions]
    returns = [df[f"r_{an}"].values for an in asset_names]
    contribs = [pos_lag[i]*returns[i] for i in range(len(asset_names))]
    gross = np.sum(contribs, axis=0)
    # turnover
    turn = 0
    for i in range(len(asset_names)):
        turn += np.abs(np.diff(np.concatenate([[0], positions[i]])))
    slippage_cost = turn * SLIPPAGE
    fee_cost = turn * FEE
    tc = turn * (SLIPPAGE + FEE)
    net = gross - tc
    nav = np.cumprod(1 + net)
    dd = nav/np.maximum.accumulate(nav) - 1
    cum = nav[-1] - 1 if len(nav) > 0 else 0
    ann = (1 + cum)**(252/len(net)) - 1 if len(net) > 0 else 0
    vol = np.std(net, ddof=1)*np.sqrt(252) if len(net) > 1 else 0
    sharpe = ann/vol if vol > 0 else 0
    mdd = np.min(dd) if len(dd) > 0 else 0
    wr = np.mean(net > 0) if len(net) > 0 else 0
    av_turn = np.mean(turn)
    # 组合总仓位（abs sum，体现多空暴露度）
    pos_total = np.sum(np.abs(positions), axis=0)
    pos_lag_total = np.sum(np.abs(pos_lag), axis=0)
    prev_pos = np.concatenate([[0], pos_total[:-1]])
    pos_change = pos_total - prev_pos
    # 加权预测信号
    pred_weighted = np.sum([positions[i]*df[f"p_{asset_names[i]}"].values
                            for i in range(len(asset_names))], axis=0)
    # 参考价格：两资产均价
    price_ref = np.mean([df[f"c_{asset_names[i]}"].values for i in range(len(asset_names))], axis=0)
    # P&L：当日净收益对应的金额（按初始1元本金累计）
    pnl = np.concatenate([[0], np.diff(nav)])
    # 单资产
    asset_metrics = {}
    for i, an in enumerate(asset_names):
        c = contribs[i]
        nav_a = np.cumprod(1 + c); cum_a = nav_a[-1] - 1 if len(nav_a) > 0 else 0
        ann_a = (1 + cum_a)**(252/len(c)) - 1 if len(c) > 0 else 0
        vol_a = np.std(c, ddof=1)*np.sqrt(252) if len(c) > 1 else 0
        sh_a = ann_a/vol_a if vol_a > 0 else 0
        dd_a = nav_a/np.maximum.accumulate(nav_a) - 1
        mdd_a = np.min(dd_a) if len(dd_a) > 0 else 0
        # 单资产层面的成本分摊（按仓位绝对值占比）
        abs_share = (np.abs(positions[i]) + 1e-12) / (np.sum(np.abs(positions), axis=0) + 1e-12)
        turn_i = np.abs(np.diff(np.concatenate([[0], positions[i]])))
        tc_i = turn_i * (SLIPPAGE + FEE) * abs_share
        sl_i = turn_i * SLIPPAGE * abs_share
        # 单资产净贡献（含成本）
        net_i = c - tc_i
        asset_metrics[an] = {"cum":cum_a, "ann":ann_a, "vol":vol_a, "sharpe":sh_a, "mdd":mdd_a,
                             "tc":tc_i, "slippage":sl_i, "turnover":turn_i, "net_contrib":net_i}
    return {"port_cum":cum, "port_ann":ann, "port_vol":vol, "port_sharpe":sharpe,
            "port_mdd":mdd, "port_winrate":wr, "port_turnover":av_turn,
            "assets":asset_metrics, "nav":nav, "dates":df["d"].values,
            "df":df, "positions":positions, "pos_lag":pos_lag, "returns":returns,
            "contribs":contribs, "gross":gross, "turn":turn, "tc":tc,
            "slippage_cost":slippage_cost, "fee_cost":fee_cost, "net":net, "dd":dd,
            "pos_total":pos_total, "pos_lag_total":pos_lag_total,
            "prev_pos":prev_pos, "pos_change":pos_change,
            "pred_weighted":pred_weighted, "price_ref":price_ref, "pnl":pnl,
            "base_positions":base_positions}

def compute_tc_risk_state(base_pos_tc, contribs_tc_with_cost, dates):
    """递推计算 TC 的 risk control state。
    严格规则：t 日的 risk_state / loss_streak / profit_streak 用 t-1 日已实现净 P&L 更新，
    然后用于调整 t 日 base position → adjusted position。
    - 连续亏损 1–4 日：正常仓位（×1.0）
    - 连续亏损 ≥5 日：×0.5
    - 连续亏损 ≥7 日：暂停新增仓位（target position 不得增加，但可减仓/平仓）
    - 恢复：连续盈利 ≥2 日 → 恢复到 50% 仓位
            连续盈利 ≥5 日 → 恢复到 100% 正常仓位
            中间亏损 → profit streak 清零
    """
    n = len(base_pos_tc)
    loss_streak = np.zeros(n, dtype=int)
    profit_streak = np.zeros(n, dtype=int)
    risk_state = np.array(["NORMAL"] * n, dtype=object)
    adj_pos = np.zeros(n)
    # 起始 streak 来自前一日（初始 0）
    prev_loss = 0
    prev_profit = 0
    prev_state = "NORMAL"
    # 注：contribs_tc_with_cost[0] 是第 0 日的贡献，但第 0 日使用 lag position = 0
    # 第 t 日的 risk state 用第 t-1 日 contribution 更新
    for t in range(n):
        # 用 t-1 日的 contribution 更新 streak / state（作用于 t 日的仓位）
        if t == 0:
            cur_loss = 0
            cur_profit = 0
            cur_state = "NORMAL"
        else:
            pnl_prev = contribs_tc_with_cost[t-1]
            if pnl_prev < 0:
                cur_loss = prev_loss + 1
                cur_profit = 0
            elif pnl_prev > 0:
                cur_loss = 0
                cur_profit = prev_profit + 1
            else:
                cur_loss = 0
                cur_profit = 0
            # 根据 (loss, profit, prev_state) 决定 t 日 state
            # 注意：恢复路径基于 prev_state，避免同时触发多个分支
            if cur_loss >= 7:
                cur_state = "PAUSED"
            elif cur_loss >= 5:
                cur_state = "HALF"
            else:
                # 检查恢复路径
                if prev_state == "PAUSED":
                    if cur_profit >= 5:
                        cur_state = "NORMAL"
                    elif cur_profit >= 2:
                        cur_state = "HALF"
                    else:
                        cur_state = "PAUSED"
                elif prev_state == "HALF":
                    if cur_profit >= 5:
                        cur_state = "NORMAL"
                    else:
                        cur_state = "HALF"
                else:
                    cur_state = "NORMAL"
        loss_streak[t] = cur_loss
        profit_streak[t] = cur_profit
        risk_state[t] = cur_state
        # 调整 base position → adjusted position
        base = base_pos_tc[t]
        if cur_state == "NORMAL":
            mult = 1.0
            adj = base * mult
        elif cur_state == "HALF":
            mult = 0.5
            adj = base * mult
        elif cur_state == "PAUSED":
            # 暂停新增：target position 不得增加（绝对值方向上）
            # 允许减仓和平仓。比较 base 与 prev_adj_pos：
            # - 若 |base| < |prev_adj| （减仓或平仓） → 允许执行 base
            # - 若 |base| > |prev_adj| （加仓）       → 不允许，保持 prev_adj
            # - 若符号反转 → 视为平仓再开新仓 → 不允许加仓，所以保持 0 或 prev_abs
            # 简化处理：限制 adj 的绝对值 ≤ prev_adj 的绝对值，符号同 base
            if t == 0:
                prev_adj_abs = 0.0
            else:
                prev_adj_abs = abs(adj_pos[t-1])
            base_abs = abs(base)
            if base_abs <= prev_adj_abs + 1e-12:
                # 允许减仓/平仓
                adj = base
            else:
                # 不允许加仓：保持前一日仓位幅度，但符号按当前 base 方向
                # 若前一日为 0，则今日也保持 0（暂停开新仓）
                if prev_adj_abs < 1e-12:
                    adj = 0.0
                else:
                    adj = np.sign(base) * prev_adj_abs
        else:
            adj = base
        adj_pos[t] = adj
        # 更新下一日使用的 prev
        prev_loss = cur_loss
        prev_profit = cur_profit
        prev_state = cur_state
    return {"loss_streak":loss_streak, "profit_streak":profit_streak,
            "risk_state":risk_state, "base_pos":base_pos_tc, "adj_pos":adj_pos}

def compute_tc_contribs_with_cost_for_rc(base_pos_tc, returns_tc, dates):
    """计算 TC 的 base position 下含成本的单资产日净 P&L contribution，
    用于 risk control 的"连续亏损日"判定。
    成本分摊：单资产 turnover × (SLIPPAGE+FEE)，按单资产 abs 仓位比例分摊
    （此处分摊近似为按 abs(pos)/POS_LIMIT，即单资产自己的换手成本）。
    """
    n = len(base_pos_tc)
    pos_lag = np.concatenate([[0], base_pos_tc[:-1]])
    gross = pos_lag * returns_tc
    turn = np.abs(np.diff(np.concatenate([[0], base_pos_tc])))
    tc = turn * (SLIPPAGE + FEE)
    net_contrib = gross - tc
    return net_contrib


# ============== 合成压力测试期数据 ==============
def generate_synthetic_stress_period(last_date, last_close_bili, last_close_tc,
                                      last_factors_bili, last_factors_tc,
                                      n_days=63, seed=42):
    """生成 63 个交易日的合成压力测试期数据，包含多重极端事件。
    事件结构：
      Day 1-15:  正常期（小幅随机游走）
      Day 16:   单日闪崩（BILI -8%, TC -8%）  → 触发器 A
      Day 17-22: 连续下跌（每天 -3%，5日累计 -15%）  → 触发器 C
      Day 23-35: 波动率飙升（日波动放大到 train 均值的 3 倍）  → 触发器 B
      Day 36-40: 系统性事件（单日 -12%、连跌 -20%、vol×4）  → 触发器 A+B+C
      Day 41-63: 恢复期（波动逐步回归正常）
    因子值由合成收盘价反推（momentum、volatility），估值因子缓慢漂移。
    """
    rng = np.random.default_rng(seed)
    # 生成交易日（跳过周末）
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    # === 收盘价路径 ===
    rets_bili = np.zeros(n_days); rets_tc = np.zeros(n_days)
    # 正常期（1-15）
    for t in range(15):
        rets_bili[t] = rng.normal(0.0005, 0.015)
        rets_tc[t] = rng.normal(0.0003, 0.012)
    # 单日闪崩（16）
    rets_bili[15] = -0.08; rets_tc[15] = -0.08
    # 连续下跌（17-22）
    for t in range(16, 22):
        rets_bili[t] = rng.normal(-0.03, 0.01)
        rets_tc[t] = rng.normal(-0.03, 0.01)
    # 波动率飙升期（23-35）
    for t in range(22, 35):
        rets_bili[t] = rng.normal(0, 0.045)  # 3倍正常波动
        rets_tc[t] = rng.normal(0, 0.036)
    # 系统性事件（36-40）
    rets_bili[35] = -0.12; rets_tc[35] = -0.12  # 单日闪崩
    for t in range(36, 40):
        rets_bili[t] = rng.normal(-0.04, 0.06)  # 连跌+高波动
        rets_tc[t] = rng.normal(-0.04, 0.048)
    # 恢复期（41-63）：波动逐步回归
    for t in range(40, n_days):
        decay = (t - 40) / 23  # 0 → 1
        vol_bili = 0.045 * (1 - decay) + 0.015 * decay
        vol_tc = 0.036 * (1 - decay) + 0.012 * decay
        rets_bili[t] = rng.normal(0.001 * (1 - decay), vol_bili)
        rets_tc[t] = rng.normal(0.0005 * (1 - decay), vol_tc)

    close_bili = last_close_bili * np.cumprod(1 + rets_bili)
    close_tc = last_close_tc * np.cumprod(1 + rets_tc)

    # === 因子值 ===
    # BILI 因子: PB, M120, DAUg, MAUg, M60, RS
    # TC 因子:   PB, PE, ROE, M20, MAUg
    bili_factor_names = list(BILI_SHORT.values())[1:]  # 去掉 y
    tc_factor_names = list(TC_SHORT.values())[1:]

    bili_factors = np.zeros((n_days, len(bili_factor_names)))
    tc_factors = np.zeros((n_days, len(tc_factor_names)))
    # 初始化 = 最后一日的因子值
    for i, fn in enumerate(bili_factor_names):
        bili_factors[0, i] = last_factors_bili.get(fn, 1.0)
    for i, fn in enumerate(tc_factor_names):
        tc_factors[0, i] = last_factors_tc.get(fn, 1.0)

    # 逐日演化
    for t in range(1, n_days):
        # BILI 因子漂移
        bili_factors[t, 0] = bili_factors[t-1, 0] * (1 + rng.normal(0, 0.005))  # PB
        bili_factors[t, 1] = np.log(close_bili[t] / close_bili[max(0, t-120)]) if t >= 120 else bili_factors[t-1, 1]  # M120
        bili_factors[t, 2] = bili_factors[t-1, 2] + rng.normal(0, 0.5)  # DAUg
        bili_factors[t, 3] = bili_factors[t-1, 3] + rng.normal(0, 0.5)  # MAUg
        bili_factors[t, 4] = np.log(close_bili[t] / close_bili[max(0, t-60)]) if t >= 60 else bili_factors[t-1, 4]  # M60
        bili_factors[t, 5] = bili_factors[t-1, 5] * (1 + rets_bili[t])  # RS (跟随收益)

        # TC 因子漂移
        tc_factors[t, 0] = tc_factors[t-1, 0] * (1 + rng.normal(0, 0.005))  # PB
        tc_factors[t, 1] = tc_factors[t-1, 1] * (1 + rng.normal(0, 0.01))  # PE
        tc_factors[t, 2] = tc_factors[t-1, 2] + rng.normal(0, 0.3)  # ROE
        if t >= 20:
            tc_factors[t, 3] = np.log(close_tc[t] / close_tc[t-20])  # M20
        else:
            tc_factors[t, 3] = tc_factors[t-1, 3]
        tc_factors[t, 4] = tc_factors[t-1, 4] + rng.normal(0, 0.5)  # MAUg

    # y (5d Log Return)：未来 5 日收益的 log（合成期最后一部分用过去 5 日收益近似）
    y_bili = np.zeros(n_days); y_tc = np.zeros(n_days)
    for t in range(n_days):
        future_end = min(t + 5, n_days - 1)
        y_bili[t] = np.log(close_bili[future_end] / close_bili[t])
        y_tc[t] = np.log(close_tc[future_end] / close_tc[t])

    return {
        "dates": dates, "close_bili": close_bili, "close_tc": close_tc,
        "bili_factors": bili_factors, "tc_factors": tc_factors,
        "bili_factor_names": bili_factor_names, "tc_factor_names": tc_factor_names,
        "y_bili": y_bili, "y_tc": y_tc,
    }


# ============== 事件风控模块 ==============
def compute_event_risk_state(closes, returns, train_end_idx_per_asset,
                            ret_quantile=0.05, vol_mult=2.5, cooldown_days=3,
                            ret5d_quantile=0.05):
    """计算每个资产的 event risk multiplier。
    closes: list of close price arrays (含 train + test + 合成期)
    returns: list of daily return arrays (含 train + test + 合成期)
    train_end_idx_per_asset: list of int，每个资产的 train 期终点 index
    返回 dict {asset_name: multiplier_array}，长度等于 test+合成期。
    multiplier ∈ [0, 1]，正常时 1.0，事件触发时降至 0 或 0.5。

    严格无前视：t 日 multiplier 用 t-1 日的极端收益/5日累计/20d vol 判定。
    """
    n_assets = len(closes)
    # 计算每个资产的阈值（基于 train 期）
    thresholds = []
    for i in range(n_assets):
        train_end = train_end_idx_per_asset[i]
        train_rets = returns[i][:train_end]
        # 5 日累计收益
        train_5d = np.convolve(train_rets, np.ones(5)/5, mode='valid')
        train_5d = train_5d[~np.isnan(train_5d)]
        # 20d 波动率
        train_20d_vol = np.array([np.std(train_rets[max(0,j-20):j]) for j in range(20, len(train_rets))])
        train_20d_vol = train_20d_vol[~np.isnan(train_20d_vol)]
        thresholds.append({
            "ret_5pct": np.quantile(train_rets, ret_quantile),
            "ret5d_5pct": np.quantile(train_5d, ret5d_quantile) if len(train_5d) > 0 else -0.05,
            "vol_mean": np.mean(train_20d_vol) if len(train_20d_vol) > 0 else 0.015,
        })

    # 对每个资产，在 test+合成期逐日判定
    multipliers = []
    for i in range(n_assets):
        train_end = train_end_idx_per_asset[i]
        full_rets = returns[i]
        n_total = len(full_rets)
        n_eval = n_total - train_end  # test + 合成期长度
        mult = np.ones(n_eval)
        # 预计算 5 日累计收益和 20d vol（在全期上算，但只读 t-1 之前的数据）
        # 5 日累计收益[t] = sum(rets[t-4:t+1])，但我们要用 t-1 日的值，即 sum(rets[t-5:t])
        # 20d vol[t] = std(rets[t-20:t])，用 t-1 日的值，即 std(rets[t-21:t-1])
        cooldown_remaining = 0
        current_mult = 1.0
        for t_idx in range(n_eval):
            t_global = train_end + t_idx  # 全局 index
            if t_idx == 0:
                mult[t_idx] = 1.0
                continue
            # 用 t-1 日的数据判定（即 t_global - 1）
            t_prev = t_global - 1
            # A: 极端日跌幅
            ret_prev = full_rets[t_prev]
            trigger_A = ret_prev < thresholds[i]["ret_5pct"]
            # C: 5 日累计回撤
            if t_prev >= 5:
                ret5d_prev = np.sum(full_rets[t_prev-4:t_prev+1])
            else:
                ret5d_prev = np.sum(full_rets[:t_prev+1])
            trigger_C = ret5d_prev < thresholds[i]["ret5d_5pct"]
            # B: 波动率飙升
            if t_prev >= 20:
                vol_prev = np.std(full_rets[t_prev-20:t_prev])
            else:
                vol_prev = np.std(full_rets[:t_prev+1])
            trigger_B = vol_prev > thresholds[i]["vol_mean"] * vol_mult

            n_triggers = sum([trigger_A, trigger_B, trigger_C])
            # 判定目标 multiplier
            if n_triggers >= 2:
                target_mult = 0.0  # 清仓
            elif n_triggers == 1:
                target_mult = 0.5  # 降半仓
            else:
                target_mult = 1.0

            # 冷却逻辑
            if target_mult < current_mult:
                # 触发新事件，重置冷却期
                current_mult = target_mult
                cooldown_remaining = cooldown_days
            elif cooldown_remaining > 0:
                # 冷却期逐日线性恢复
                cooldown_remaining -= 1
                steps_total = cooldown_days
                steps_done = steps_total - cooldown_remaining
                current_mult = target_mult + (1.0 - target_mult) * (steps_done / steps_total)
                current_mult = min(current_mult, 1.0)
            else:
                current_mult = 1.0
            mult[t_idx] = current_mult
        multipliers.append(mult)
    return multipliers, thresholds


def portfolio_bt_2asset_with_event_risk(preds, closes, dates, asset_names,
                                         rc_asset=None, rc_state=None, deadband=0.0,
                                         event_mult=None):
    """在 portfolio_bt_2asset 基础上加上 event risk multiplier。
    event_mult: list of arrays，每个资产的 multiplier（长度等于 test 期）。
    最终仓位 = RC 仓位 × event multiplier
    """
    # 先跑基础 RC 回测
    result = portfolio_bt_2asset(preds, closes, dates, asset_names,
                                  rc_asset=rc_asset, rc_state=rc_state, deadband=deadband)
    if event_mult is None:
        return result
    # 应用 event multiplier
    n = len(result["df"])
    new_positions = []
    for i in range(len(asset_names)):
        em = event_mult[i]
        # 对齐长度
        if len(em) != n:
            print(f"  [WARN] event_mult[{i}] len={len(em)} != n={n}, 对齐截断")
            em = em[:n] if len(em) > n else np.concatenate([em, np.ones(n - len(em))])
        new_pos = result["positions"][i] * em
        new_positions.append(new_pos)
    # 重新计算 gross/turn/tc/net/nav
    pos_lag = [np.concatenate([[0], p[:-1]]) for p in new_positions]
    returns = [result["df"][f"r_{asset_names[i]}"].values for i in range(len(asset_names))]
    contribs = [pos_lag[i]*returns[i] for i in range(len(asset_names))]
    gross = np.sum(contribs, axis=0)
    turn = 0
    for i in range(len(asset_names)):
        turn += np.abs(np.diff(np.concatenate([[0], new_positions[i]])))
    slippage_cost = turn * SLIPPAGE
    fee_cost = turn * FEE
    tc = turn * (SLIPPAGE + FEE)
    net = gross - tc
    nav = np.cumprod(1 + net)
    dd = nav/np.maximum.accumulate(nav) - 1
    cum = nav[-1] - 1 if len(nav) > 0 else 0
    ann = (1 + cum)**(252/len(net)) - 1 if len(net) > 0 else 0
    vol = np.std(net, ddof=1)*np.sqrt(252) if len(net) > 1 else 0
    sharpe = ann/vol if vol > 0 else 0
    mdd = np.min(dd) if len(dd) > 0 else 0
    av_turn = np.mean(turn)
    # 单资产 metrics
    asset_metrics = {}
    for i, an in enumerate(asset_names):
        c = contribs[i]
        nav_a = np.cumprod(1 + c); cum_a = nav_a[-1] - 1 if len(nav_a) > 0 else 0
        ann_a = (1 + cum_a)**(252/len(c)) - 1 if len(c) > 0 else 0
        vol_a = np.std(c, ddof=1)*np.sqrt(252) if len(c) > 1 else 0
        sh_a = ann_a/vol_a if vol_a > 0 else 0
        dd_a = nav_a/np.maximum.accumulate(nav_a) - 1
        mdd_a = np.min(dd_a) if len(dd_a) > 0 else 0
        abs_share = (np.abs(new_positions[i]) + 1e-12) / (np.sum(np.abs(new_positions), axis=0) + 1e-12)
        turn_i = np.abs(np.diff(np.concatenate([[0], new_positions[i]])))
        tc_i = turn_i * (SLIPPAGE + FEE) * abs_share
        sl_i = turn_i * SLIPPAGE * abs_share
        net_i = c - tc_i
        asset_metrics[an] = {"cum":cum_a, "ann":ann_a, "vol":vol_a, "sharpe":sh_a, "mdd":mdd_a,
                             "tc":tc_i, "slippage":sl_i, "turnover":turn_i, "net_contrib":net_i}
    return {"port_cum":cum, "port_ann":ann, "port_vol":vol, "port_sharpe":sharpe,
            "port_mdd":mdd, "port_winrate":result["port_winrate"], "port_turnover":av_turn,
            "assets":asset_metrics, "nav":nav, "dates":result["dates"],
            "df":result["df"], "positions":new_positions, "pos_lag":pos_lag, "returns":returns,
            "contribs":contribs, "gross":gross, "turn":turn, "tc":tc,
            "slippage_cost":slippage_cost, "fee_cost":fee_cost, "net":net, "dd":dd,
            "pos_total":np.sum(np.abs(new_positions), axis=0),
            "pos_lag_total":np.sum(np.abs(pos_lag), axis=0),
            "prev_pos":np.concatenate([[0], np.sum(np.abs(new_positions), axis=0)[:-1]]),
            "pos_change":np.sum(np.abs(new_positions), axis=0) - np.concatenate([[0], np.sum(np.abs(new_positions), axis=0)[:-1]]),
            "pred_weighted":result["pred_weighted"], "price_ref":result["price_ref"],
            "pnl":np.concatenate([[0], np.diff(nav)]),
            "base_positions":result["base_positions"],
            "event_mult":event_mult}


# 全局变量占位（predict_synthetic 用来判断 asset）
bili_d_global = None

# === 主流程 ===
print("="*70); print("2-Asset Backtest (BILI + TC) — Baseline vs Risk-Control"); print("="*70)

bili_d = prepare(BILI_SRC, BILI_FACTORS, BILI_SHORT, BILI_SELECTED)
tc_d = prepare(TC_SRC, TC_FACTORS, TC_SHORT, TC_SELECTED)
print(f"BILI: train={len(bili_d['tr'])}, test={len(bili_d['te'])}")
print(f"TC:   train={len(tc_d['tr'])}, test={len(tc_d['te'])}")

# 各自最优替换（模型不变）
bili_pred, _, _, bili_ols = fit_predict(bili_d, BILI_REPLACE, "BILI")
tc_pred, _, _, tc_ols = fit_predict(tc_d, TC_REPLACE, "TC")
# 保存 OLS 模型到 d 字典供 predict_on_new 复用
bili_d["ols_model"] = bili_ols
tc_d["ols_model"] = tc_ols

# 对齐 test 期
bili_te = bili_d["te"][["Trade Date","Close (HKD)","y"]].copy(); bili_te["pred"] = bili_pred
tc_te = tc_d["te"][["Trade Date","Close (HKD)","y"]].copy(); tc_te["pred"] = tc_pred

m = pd.merge(bili_te, tc_te, on="Trade Date", suffixes=("_bili","_tc"))
preds = [m["pred_bili"].values, m["pred_tc"].values]
closes = [m["Close (HKD)_bili"].values, m["Close (HKD)_tc"].values]
dates = m["Trade Date"].values
asset_names = ["BILI","TC"]

# ============== 合成压力测试期 ==============
print("\n" + "="*70)
print("生成合成压力测试期 (63 个交易日，含多重极端事件)")
print("="*70)

# 取真实 test 期最后一日作为合成期起点
last_date_real = pd.to_datetime(dates[-1])
last_close_bili = float(closes[0][-1])
last_close_tc = float(closes[1][-1])

# 取最后一日因子值
bili_last_factors = {fn: float(bili_d["tr"][fn].iloc[-1]) for fn in bili_d["fcols"]}
tc_last_factors = {fn: float(tc_d["tr"][fn].iloc[-1]) for fn in tc_d["fcols"]}
# 改用 test 最后一日（更接近合成期起点）
bili_last_factors = {fn: float(bili_d["te"][fn].iloc[-1]) for fn in bili_d["fcols"]}
tc_last_factors = {fn: float(tc_d["te"][fn].iloc[-1]) for fn in tc_d["fcols"]}

synth = generate_synthetic_stress_period(last_date_real, last_close_bili, last_close_tc,
                                          bili_last_factors, tc_last_factors, n_days=63, seed=42)
print(f"合成期: {synth['dates'][0].date()} -> {synth['dates'][-1].date()} ({len(synth['dates'])} 天)")
print(f"合成期 BILI 收益: {synth['close_bili'][-1]/synth['close_bili'][0]-1:.4%}")
print(f"合成期 TC   收益: {synth['close_tc'][-1]/synth['close_tc'][0]-1:.4%}")

# 对合成期做预测
bili_synth_pred = predict_on_new(bili_d, BILI_REPLACE, synth["bili_factors"])
tc_synth_pred = predict_on_new(tc_d, TC_REPLACE, synth["tc_factors"])
print(f"BILI 合成期预测均值: {np.mean(bili_synth_pred):.6f}, std: {np.std(bili_synth_pred):.6f}")
print(f"TC   合成期预测均值: {np.mean(tc_synth_pred):.6f}, std: {np.std(tc_synth_pred):.6f}")

# 拼接真实 test + 合成期
preds_ext = [np.concatenate([preds[0], bili_synth_pred]),
             np.concatenate([preds[1], tc_synth_pred])]
closes_ext = [np.concatenate([closes[0], synth["close_bili"]]),
              np.concatenate([closes[1], synth["close_tc"]])]
dates_ext = np.concatenate([dates, synth["dates"]])
n_real = len(dates); n_synth = len(synth["dates"]); n_ext = n_real + n_synth
print(f"\n扩展期长度: {n_real} (real) + {n_synth} (synth) = {n_ext} 天")

# ============== 计算事件风控 multiplier ==============
# 需要全期收益序列（train + test + synth）来计算滚动 vol 和阈值
# BILI train + test + synth 的收益序列
bili_full_close = np.concatenate([bili_d["tr"]["Close (HKD)"].values,
                                   closes[0], synth["close_bili"]])
tc_full_close = np.concatenate([tc_d["tr"]["Close (HKD)"].values,
                                 closes[1], synth["close_tc"]])
bili_full_ret = np.diff(np.concatenate([[np.nan], bili_full_close]))
bili_full_ret[0] = 0  # 第一天无收益
tc_full_ret = np.diff(np.concatenate([[np.nan], tc_full_close]))
tc_full_ret[0] = 0

bili_train_end = len(bili_d["tr"])  # train 期长度
tc_train_end = len(tc_d["tr"])
print(f"BILI train 期长度: {bili_train_end}, TC train 期长度: {tc_train_end}")

closes_for_event = [bili_full_close, tc_full_close]
returns_for_event = [bili_full_ret, tc_full_ret]
train_end_list = [bili_train_end, tc_train_end]

event_mults, thresholds = compute_event_risk_state(
    closes_for_event, returns_for_event, train_end_list,
    ret_quantile=0.05, vol_mult=2.5, cooldown_days=3, ret5d_quantile=0.05)

print(f"\n=== 事件风控阈值 (基于 train 期) ===")
for i, an in enumerate(asset_names):
    print(f"  {an}: ret_5pct={thresholds[i]['ret_5pct']:.4f}, "
          f"ret5d_5pct={thresholds[i]['ret5d_5pct']:.4f}, "
          f"vol_mean={thresholds[i]['vol_mean']:.4f}, vol_threshold={thresholds[i]['vol_mean']*2.5:.4f}")

# event_mults 长度 = test + synth = n_ext
print(f"\n事件风控 multiplier 长度: BILI={len(event_mults[0])}, TC={len(event_mults[1])} (应为 {n_ext})")

# 统计触发情况
bili_triggers = event_mults[0] < 1.0
tc_triggers = event_mults[1] < 1.0
print(f"BILI 触发日数: {int(np.sum(bili_triggers))}/{n_ext}, TC 触发日数: {int(np.sum(tc_triggers))}/{n_ext}")
bili_clear = event_mults[0] == 0.0
tc_clear = event_mults[1] == 0.0
print(f"BILI 清仓日数: {int(np.sum(bili_clear))}, TC 清仓日数: {int(np.sum(tc_clear))}")
bili_half = (event_mults[0] > 0) & (event_mults[0] < 1.0)
tc_half = (event_mults[1] > 0) & (event_mults[1] < 1.0)
print(f"BILI 半仓日数: {int(np.sum(bili_half))}, TC 半仓日数: {int(np.sum(tc_half))}")

# ============== 跑 4 种方案 ==============
print("\n" + "="*70)
print("跑 4 种方案对比 (扩展期 = real test + synth)")
print("="*70)

# 1. Baseline
result_ext_base = portfolio_bt_2asset(preds_ext, closes_ext, dates_ext, asset_names)
base_pos_tc_ext = result_ext_base["base_positions"][1]
returns_tc_ext = result_ext_base["returns"][1]
tc_net_contrib_ext = compute_tc_contribs_with_cost_for_rc(base_pos_tc_ext, returns_tc_ext, dates_ext)
rc_state_ext = compute_tc_risk_state(base_pos_tc_ext, tc_net_contrib_ext, dates_ext)

# 2. Risk-Controlled
result_ext_rc = portfolio_bt_2asset(preds_ext, closes_ext, dates_ext, asset_names,
                                     rc_asset="TC", rc_state=rc_state_ext)

# 3. RC + Deadband
DEADBAND_DICT = {"BILI": 0.001, "TC": 0.002}
result_ext_db = portfolio_bt_2asset(preds_ext, closes_ext, dates_ext, asset_names, deadband=DEADBAND_DICT)
base_pos_tc_ext_db = result_ext_db["base_positions"][1]
returns_tc_ext_db = result_ext_db["returns"][1]
tc_net_contrib_ext_db = compute_tc_contribs_with_cost_for_rc(base_pos_tc_ext_db, returns_tc_ext_db, dates_ext)
rc_state_ext_db = compute_tc_risk_state(base_pos_tc_ext_db, tc_net_contrib_ext_db, dates_ext)
result_ext_rc_db = portfolio_bt_2asset(preds_ext, closes_ext, dates_ext, asset_names,
                                       rc_asset="TC", rc_state=rc_state_ext_db, deadband=DEADBAND_DICT)

# 4. RC + Deadband + Event Risk
result_ext_full = portfolio_bt_2asset_with_event_risk(
    preds_ext, closes_ext, dates_ext, asset_names,
    rc_asset="TC", rc_state=rc_state_ext_db, deadband=DEADBAND_DICT,
    event_mult=event_mults)

# ============== 输出 ==============
def summarize(r, label, asset_names):
    print(f"\n--- {label} ---")
    print(f"Portfolio: cum={r['port_cum']:.4%}, ann={r['port_ann']:.4%}, "
          f"vol={r['port_vol']:.4%}, sharpe={r['port_sharpe']:.4f}, mdd={r['port_mdd']:.4%}, "
          f"turn={r['port_turnover']:.4f}")
    for an in asset_names:
        a = r["assets"][an]
        print(f"  {an}: cum={a['cum']:.4%}, ann={a['ann']:.4%}, sharpe={a['sharpe']:.4f}, "
              f"mdd={a['mdd']:.4%}, tc={np.sum(a['tc']):.6f}, slip={np.sum(a['slippage']):.6f}, "
              f"turn={np.mean(a['turnover']):.4f}")

summarize(result_ext_base, "Baseline (ext)", asset_names)
summarize(result_ext_rc, "Risk-Controlled (ext)", asset_names)
summarize(result_ext_rc_db, "RC + Deadband (ext)", asset_names)
summarize(result_ext_full, "RC + Deadband + Event Risk (ext)", asset_names)

# === 对比表（Portfolio 和 TC）===
cmp_rows = []
for label, r in [("Baseline", result_ext_base), ("Risk-Controlled", result_ext_rc),
                 ("RC+Deadband", result_ext_rc_db), ("RC+Deadband+EventRisk", result_ext_full)]:
    cmp_rows.append({
        "Scenario": label, "Asset":"Portfolio",
        "Cum Return":f"{r['port_cum']:.4%}",
        "Ann Return":f"{r['port_ann']:.4%}",
        "Sharpe":f"{r['port_sharpe']:.4f}",
        "Max DD":f"{r['port_mdd']:.4%}",
        "Turnover":f"{r['port_turnover']:.4f}",
        "Transaction Cost":f"{np.sum(r['tc']):.6f}",
        "Slippage":f"{np.sum(r['slippage_cost']):.6f}",
    })
    a_tc = r["assets"]["TC"]
    cmp_rows.append({
        "Scenario": label, "Asset":"TC",
        "Cum Return":f"{a_tc['cum']:.4%}",
        "Ann Return":f"{a_tc['ann']:.4%}",
        "Sharpe":f"{a_tc['sharpe']:.4f}",
        "Max DD":f"{a_tc['mdd']:.4%}",
        "Turnover":f"{np.mean(a_tc['turnover']):.4f}",
        "Transaction Cost":f"{np.sum(a_tc['tc']):.6f}",
        "Slippage":f"{np.sum(a_tc['slippage']):.6f}",
    })
df_cmp = pd.DataFrame(cmp_rows)
print("\n=== 4 方案对比表 (扩展期) ===")
print(df_cmp.to_string(index=False))
df_cmp.to_csv(f"{OUT_DIR}/baseline_vs_full_comparison.csv", index=False)
print(f"[Saved] {OUT_DIR}/baseline_vs_full_comparison.csv")

# === TC 详细对比 ===
def tc_detail(r, label, asset_names):
    tc_idx = asset_names.index("TC")
    tc_pos = r["positions"][tc_idx]
    long_m = tc_pos > 1e-6
    short_m = tc_pos < -1e-6
    flat_m = (~long_m) & (~short_m)
    return {
        "Scenario": label,
        "Long days": int(np.sum(long_m)),
        "Short days": int(np.sum(short_m)),
        "Flat days": int(np.sum(flat_m)),
        "Average position": float(np.mean(tc_pos)),
        "Maximum position": float(np.max(tc_pos)),
        "Minimum position": float(np.min(tc_pos)),
        "Cum Return": float(r["assets"]["TC"]["cum"]),
        "Sharpe": float(r["assets"]["TC"]["sharpe"]),
        "Max DD": float(r["assets"]["TC"]["mdd"]),
    }
tc_rows = [tc_detail(result_ext_base, "Baseline", asset_names),
           tc_detail(result_ext_rc, "Risk-Controlled", asset_names),
           tc_detail(result_ext_rc_db, "RC+Deadband", asset_names),
           tc_detail(result_ext_full, "RC+Deadband+EventRisk", asset_names)]
df_tc = pd.DataFrame(tc_rows)
print("\n=== TC 详细对比 ===")
print(df_tc.to_string(index=False))
df_tc.to_csv(f"{OUT_DIR}/tc_detail_full.csv", index=False)

# === 事件风控逐日记录 ===
# portfolio_bt_2asset 内部 dropna（pct_change 第一行 NaN）会让长度从 n_ext 变成 n_ext-1
# 所以用 result_ext_full["dates"] 作为对齐基准
aligned_dates_full = result_ext_full["dates"]
n_aligned = len(aligned_dates_full)
# 对齐所有 233-length 数组到 232（截掉第一天，因为第一天 pct_change=NaN 被 dropna 掉）
preds_aligned = [preds_ext[i][1:n_aligned+1] for i in range(len(asset_names))]
closes_aligned = [closes_ext[i][1:n_aligned+1] for i in range(len(asset_names))]
event_mults_aligned = [event_mults[i][1:n_aligned+1] for i in range(len(asset_names))]
# is_synth: n_real + n_synth = 233，dropna 后为 232（砍掉第一天）
is_synth_aligned = [False]*(n_real-1) + [True]*n_synth
df_event = pd.DataFrame({
    "Date": pd.to_datetime(aligned_dates_full),
    "BILI_pred": preds_aligned[0],
    "TC_pred": preds_aligned[1],
    "BILI_close": closes_aligned[0],
    "TC_close": closes_aligned[1],
    "BILI_event_mult": event_mults_aligned[0],
    "TC_event_mult": event_mults_aligned[1],
    "BILI_position_full": result_ext_full["positions"][0],
    "TC_position_full": result_ext_full["positions"][1],
    "is_synth": is_synth_aligned,
})
df_event.to_csv(f"{OUT_DIR}/event_risk_state.csv", index=False)
print(f"[Saved] {OUT_DIR}/event_risk_state.csv  ({len(df_event)} rows)")

# 触发事件汇总
print("\n=== 事件风控触发汇总 ===")
bili_trigger_dates = df_event[(df_event["BILI_event_mult"] < 1.0) & (df_event["is_synth"])]
tc_trigger_dates = df_event[(df_event["TC_event_mult"] < 1.0) & (df_event["is_synth"])]
print(f"BILI 触发日数 (合成期): {len(bili_trigger_dates)}")
if len(bili_trigger_dates) > 0:
    print(bili_trigger_dates[["Date","BILI_event_mult","BILI_close"]].head(15).to_string(index=False))
print(f"\nTC 触发日数 (合成期): {len(tc_trigger_dates)}")
if len(tc_trigger_dates) > 0:
    print(tc_trigger_dates[["Date","TC_event_mult","TC_close"]].head(15).to_string(index=False))

# === NAV 对比曲线 ===
fig, ax = plt.subplots(figsize=(14,7))
ax.plot(result_ext_base["dates"], result_ext_base["nav"], label="Baseline", color="red", linewidth=2)
ax.plot(result_ext_rc["dates"], result_ext_rc["nav"], label="Risk-Controlled", color="blue", linewidth=2, linestyle="--")
ax.plot(result_ext_rc_db["dates"], result_ext_rc_db["nav"], label="RC + Deadband", color="green", linewidth=2, linestyle=":")
ax.plot(result_ext_full["dates"], result_ext_full["nav"], label="RC + Deadband + Event Risk", color="purple", linewidth=2.5, linestyle="-.")
# 标注合成期起点
ax.axvline(dates_ext[n_real], color="black", alpha=0.5, linestyle="--", label="Synth start")
# 标注合成期事件
for t in [15, 16, 22, 35, 36, 40]:  # 事件日（synth 期内）
    if t < n_synth:
        ax.axvline(dates_ext[n_real + t], color="orange", alpha=0.2, linestyle=":")
ax.set_title("Portfolio NAV: 4 Scenarios (Real Test + Synthetic Stress Period)")
ax.set_xlabel("Date"); ax.set_ylabel("NAV"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/nav_full_comparison.png", dpi=100); plt.close()
print(f"[Saved] {OUT_DIR}/nav_full_comparison.png")

# === Event multiplier 曲线 ===
fig, axes = plt.subplots(2, 1, figsize=(14,8), sharex=True)
ax0, ax1 = axes
ax0.plot(dates_ext, event_mults[0], label="BILI event mult", color="darkblue", linewidth=1.5)
ax0.axvline(dates_ext[n_real], color="black", alpha=0.5, linestyle="--", label="Synth start")
ax0.set_title("BILI Event Risk Multiplier")
ax0.set_ylabel("Multiplier"); ax0.legend(); ax0.grid(True, alpha=0.3)
ax1.plot(dates_ext, event_mults[1], label="TC event mult", color="darkgreen", linewidth=1.5)
ax1.axvline(dates_ext[n_real], color="black", alpha=0.5, linestyle="--", label="Synth start")
ax1.set_title("TC Event Risk Multiplier")
ax1.set_xlabel("Date"); ax1.set_ylabel("Multiplier"); ax1.legend(); ax1.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/event_multiplier.png", dpi=100); plt.close()
print(f"[Saved] {OUT_DIR}/event_multiplier.png")

# === 合成期数据预览 ===
df_synth_preview = pd.DataFrame({
    "Date": pd.to_datetime(synth["dates"]),
    "BILI_close": synth["close_bili"],
    "TC_close": synth["close_tc"],
    "BILI_pred": bili_synth_pred,
    "TC_pred": tc_synth_pred,
})
df_synth_preview.to_csv(f"{OUT_DIR}/synthetic_data_preview.csv", index=False)
print(f"[Saved] {OUT_DIR}/synthetic_data_preview.csv  ({len(df_synth_preview)} rows)")

print(f"\n[All Saved] {OUT_DIR}/")
