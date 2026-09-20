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
OUT_DIR = "/workspace/backtest_2asset"
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
            "fn":fn, "ridx":ridx, "selected":selected}

def fit_predict(d, reps, name):
    Xtr = build_design(d["Xr_tr_s"], d["Xp_tr_s"], d["selected"], reps, d["ridx"], d["fn"])
    Xte = build_design(d["Xr_te_s"], d["Xp_te_s"], d["selected"], reps, d["ridx"], d["fn"])
    Xi = sm.add_constant(Xtr, has_constant='add'); Xti = sm.add_constant(Xte, has_constant='add')
    ols = sm.OLS(d["y_tr"], Xi).fit()
    yp = ols.predict(Xti)
    tmse = mean_squared_error(d["y_te"], yp); tr2 = r2_score(d["y_te"], yp)
    print(f"[{name}] train R²={ols.rsquared:.4f}, test MSE={tmse:.6f}, test R²={tr2:.4f}")
    return yp, tmse, tr2

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

# === 主流程 ===
print("="*70); print("2-Asset Backtest (BILI + TC) — Baseline vs Risk-Control"); print("="*70)

bili_d = prepare(BILI_SRC, BILI_FACTORS, BILI_SHORT, BILI_SELECTED)
tc_d = prepare(TC_SRC, TC_FACTORS, TC_SHORT, TC_SELECTED)
print(f"BILI: train={len(bili_d['tr'])}, test={len(bili_d['te'])}")
print(f"TC:   train={len(tc_d['tr'])}, test={len(tc_d['te'])}")

# 各自最优替换（模型不变）
bili_pred, _, _ = fit_predict(bili_d, BILI_REPLACE, "BILI")
tc_pred, _, _ = fit_predict(tc_d, TC_REPLACE, "TC")

# 对齐 test 期
bili_te = bili_d["te"][["Trade Date","Close (HKD)","y"]].copy(); bili_te["pred"] = bili_pred
tc_te = tc_d["te"][["Trade Date","Close (HKD)","y"]].copy(); tc_te["pred"] = tc_pred

m = pd.merge(bili_te, tc_te, on="Trade Date", suffixes=("_bili","_tc"))
preds = [m["pred_bili"].values, m["pred_tc"].values]
closes = [m["Close (HKD)_bili"].values, m["Close (HKD)_tc"].values]
dates = m["Trade Date"].values
asset_names = ["BILI","TC"]

# === Baseline ===
result_base = portfolio_bt_2asset(preds, closes, dates, asset_names)
# 用 baseline 的 TC base position 来计算含成本 P&L contribution（用于 risk control streak 判定）
# 注：RC 的"亏损日"基于 baseline position 下的 TC 净 P&L，避免与 RC 自身循环依赖
base_pos_tc = result_base["base_positions"][1]  # TC index = 1
returns_tc = result_base["returns"][1]
tc_net_contrib_baseline = compute_tc_contribs_with_cost_for_rc(base_pos_tc, returns_tc, dates)

# === 计算 TC 的 risk state ===
rc_state = compute_tc_risk_state(base_pos_tc, tc_net_contrib_baseline, dates)

# === Risk-Controlled run ===
result_rc = portfolio_bt_2asset(preds, closes, dates, asset_names, rc_asset="TC", rc_state=rc_state)

# === Risk-Controlled + Deadband (差异化) run ===
# Deadband 在 risk control 之前生效：
#   1. BILI 预测 |p| < 0.001 归零
#   2. TC   预测 |p| < 0.002 归零（TC 预测弱，更激进过滤）
#   3. 用归零后的预测重算 base position 和 TC 含成本 P&L
#   4. 再算 risk state 和 adjusted position
DEADBAND_DICT = {"BILI": 0.001, "TC": 0.002}
print(f"\n[Deadband 差异化] BILI={DEADBAND_DICT['BILI']}, TC={DEADBAND_DICT['TC']}")
result_db = portfolio_bt_2asset(preds, closes, dates, asset_names, deadband=DEADBAND_DICT)
base_pos_tc_db = result_db["base_positions"][1]
returns_tc_db = result_db["returns"][1]
tc_net_contrib_db = compute_tc_contribs_with_cost_for_rc(base_pos_tc_db, returns_tc_db, dates)
rc_state_db = compute_tc_risk_state(base_pos_tc_db, tc_net_contrib_db, dates)
result_rc_db = portfolio_bt_2asset(preds, closes, dates, asset_names, rc_asset="TC",
                                   rc_state=rc_state_db, deadband=DEADBAND_DICT)

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

summarize(result_base, "Baseline", asset_names)
summarize(result_rc, "Risk-Controlled (TC only)", asset_names)
summarize(result_rc_db, "Risk-Controlled + Deadband (TC only)", asset_names)

# === 对比表（Portfolio 和 TC）===
cmp_rows = []
for label, r in [("Baseline", result_base), ("Risk-Controlled", result_rc),
                 ("RC+Deadband", result_rc_db)]:
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
print("\n=== 对比表 ===")
print(df_cmp.to_string(index=False))
df_cmp.to_csv(f"{OUT_DIR}/baseline_vs_rc_comparison.csv", index=False)

# === TC 详细对比（含 long/short/flat、streak）===
def tc_detail(r, label, asset_names):
    # 用 TC 的"自身实际仓位"判定多/空/平（基准位置 = adj_pos for RC, base_pos for baseline）
    # 注意：positions 在 RC 中是 adj_pos；在 baseline 中等于 base_pos
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
tc_rows = [tc_detail(result_base, "Baseline", asset_names),
           tc_detail(result_rc, "Risk-Controlled", asset_names),
           tc_detail(result_rc_db, "RC+Deadband", asset_names)]
df_tc = pd.DataFrame(tc_rows)
print("\n=== TC 详细对比 ===")
print(df_tc.to_string(index=False))
df_tc.to_csv(f"{OUT_DIR}/tc_detail_baseline_vs_rc.csv", index=False)

# === Risk state 逐日记录 ===
# 注意：streak 是更新到 t 日的状态，作用于 t 日仓位
# 为了让用户能核对"减仓/暂停是否按规则触发"，同时输出 t-1 日 TC 净 P&L（用于触发判断）
# 注：result_base["dates"] 已经过 portfolio_bt_2asset 内部 dropna 对齐，长度可能 < len(dates)
aligned_dates = result_base["dates"]
prev_pnl = np.concatenate([[np.nan], tc_net_contrib_baseline[:-1]]) if len(tc_net_contrib_baseline) > 0 else np.array([np.nan])
# 对齐长度
n_dates = len(aligned_dates)
prev_pnl = prev_pnl[:n_dates]
df_rc_state = pd.DataFrame({
    "Date": pd.to_datetime(aligned_dates),
    "TC base_position": rc_state["base_pos"],
    "TC adjusted_position": rc_state["adj_pos"],
    "risk_state": rc_state["risk_state"],
    "loss_streak": rc_state["loss_streak"],
    "profit_streak": rc_state["profit_streak"],
    "prev_day_TC_net_pnl": prev_pnl,           # 用于判定 streak 的 t-1 日 P&L
    "TC return (baseline)": returns_tc,
    "TC net_contrib (baseline)": tc_net_contrib_baseline,
})
df_rc_state.to_csv(f"{OUT_DIR}/tc_risk_state.csv", index=False)
print(f"[Saved] {OUT_DIR}/tc_risk_state.csv  ({len(df_rc_state)} rows)")

# 触发事件汇总
print("\n=== Risk state 触发事件 ===")
state_changes = df_rc_state[df_rc_state["risk_state"] != df_rc_state["risk_state"].shift(1).fillna("NORMAL")]
print(state_changes[["Date","risk_state","loss_streak","profit_streak","TC base_position","TC adjusted_position"]].to_string(index=False))

# === NAV 对比曲线 ===
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(result_base["dates"], result_base["nav"], label="Baseline", color="red", linewidth=2)
ax.plot(result_rc["dates"], result_rc["nav"], label="Risk-Controlled (TC)", color="blue", linewidth=2, linestyle="--")
ax.plot(result_rc_db["dates"], result_rc_db["nav"], label="RC + Deadband (TC)", color="green", linewidth=2, linestyle=":")
ax.set_title("Portfolio NAV: Baseline vs Risk-Controlled vs RC+Deadband")
ax.set_xlabel("Date"); ax.set_ylabel("NAV"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/nav_baseline_vs_rc.png", dpi=100); plt.close()

# === TC 仓位对比曲线 ===
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(result_base["dates"], result_base["positions"][1], label="TC base position", color="gray", linewidth=1, alpha=0.7)
ax.plot(result_rc["dates"], result_rc["positions"][1], label="TC adjusted position", color="blue", linewidth=1.5)
# 标注 risk state 切换点
for _, row in state_changes.iterrows():
    ax.axvline(row["Date"], color="red", alpha=0.3, linestyle=":")
ax.set_title("TC Position: Base vs Risk-Controlled")
ax.set_xlabel("Date"); ax.set_ylabel("Position"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/tc_position_baseline_vs_rc.png", dpi=100); plt.close()

print(f"\n[Saved] {OUT_DIR}/")
