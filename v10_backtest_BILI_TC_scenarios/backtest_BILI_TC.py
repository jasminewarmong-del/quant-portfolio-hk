"""
BILI + TC (腾讯) 双资产组合回测 - 4 场景对比
场景1: 双基线 (BILI 无替换 + TC 无替换)
场景2: BILI 替换 + TC 基线 (PB DAUg→exp)
场景3: BILI 基线 + TC 替换 (PB→pow_0.5)
场景4: 双替换 (BILI: PB DAUg→exp, TC: PB→pow_0.5)
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
OUT_DIR = "/workspace/backtest_BILI_TC"
os.makedirs(OUT_DIR, exist_ok=True)

SLIPPAGE = 0.0005; FEE = 0.0003; POS_LIMIT = 0.90

# BILI 配置
BILI_FACTORS = ["PB Ratio","Momentum 120d","DAU Growth (YoY %)","MAU Growth (YoY %)","Momentum 60d","Rel. Strength vs HSTECH"]
BILI_SHORT = {"y (5d Log Return)":"y","PB Ratio":"PB","Momentum 120d":"M120","DAU Growth (YoY %)":"DAUg","MAU Growth (YoY %)":"MAUg","Momentum 60d":"M60","Rel. Strength vs HSTECH":"RS"}
BILI_SELECTED = ['PB DAUg','M120 DAUg','M120 MAUg','M120 RS','DAUg RS','MAUg RS']
BILI_REPLACE = {"PB DAUg": "exp"}

# TC 配置
TC_FACTORS = ["PB Ratio","PE Ratio","ROE (%)","20d Momentum","MAU Growth (YoY %)"]
TC_SHORT = {"y (5d Log Return)":"y","PB Ratio":"PB","PE Ratio":"PE","ROE (%)":"ROE","20d Momentum":"M20","MAU Growth (YoY %)":"MAUg"}
TC_SELECTED = ['PB','ROE','M20','MAUg^2','PE ROE','PE MAUg','ROE MAUg']
TC_REPLACE = {"PB": "pow_0.5"}

# 基函数
def sc(z,lo,hi): return np.clip(z,lo,hi)
def g_exp(z): return np.expm1(sc(z,-3,3))
def g_ln1pz(z): return np.log1p(sc(z,-0.999,None))
def g_sin(z): return np.sin(sc(z,-3*np.pi,3*np.pi))
def g_tanh(z): return np.tanh(sc(z,-3,3))
def g_cos(z): return 2*(1-np.cos(sc(z,-1.55,1.55)))
def g_sec(z):
    x=sc(z,-1.55,1.55); return 2*(1/np.cos(x)-1)
def g_pow05(z): return (1+sc(z,-0.999,None))**0.5-1
def g_pow2(z): return (1+sc(z,-0.999,None))**2-1
def g_powm1(z): return (1+sc(z,-0.999,None))**(-1)-1
BASIS = {"exp":g_exp,"ln1pz":g_ln1pz,"sin":g_sin,"tanh":g_tanh,
         "cos":g_cos,"sec":g_sec,"pow_0.5":g_pow05,"pow_2":g_pow2,"pow_-1":g_powm1}

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
                  s2[["Trade Date"]+factors], on="Trade Date").rename(columns=short)
    fcols = [short[c] for c in factors]
    df = df.dropna().reset_index(drop=True)
    n = len(df); n_tr = int(n*0.8)
    tr = df.iloc[:n_tr].copy(); te = df.iloc[n_tr:].copy()
    Xr_tr = tr[fcols].values; Xr_te = te[fcols].values
    y_tr = tr["y"].values; y_te = te["y"].values
    lo = np.quantile(Xr_tr,0.01,axis=0); hi = np.quantile(Xr_tr,0.99,axis=0)
    Xr_tr_w = np.clip(Xr_tr,lo[None,:],hi[None,:]); Xr_te_w = np.clip(Xr_te,lo[None,:],hi[None,:])
    poly = PolynomialFeatures(degree=2,include_bias=False)
    Xp_tr = poly.fit_transform(Xr_tr_w); Xp_te = poly.transform(Xr_te_w)
    fn = poly.get_feature_names_out(fcols)
    sp = StandardScaler(); Xp_tr_s = sp.fit_transform(Xp_tr); Xp_te_s = sp.transform(Xp_te)
    sr = StandardScaler(); Xr_tr_s = sr.fit_transform(Xr_tr_w); Xr_te_s = sr.transform(Xr_te_w)
    ridx = {f:i for i,f in enumerate(fcols)}
    return {"tr":tr,"te":te,"y_tr":y_tr,"y_te":y_te,
            "Xr_tr_s":Xr_tr_s,"Xr_te_s":Xr_te_s,"Xp_tr_s":Xp_tr_s,"Xp_te_s":Xp_te_s,
            "fn":fn,"ridx":ridx,"selected":selected}

def fit_predict(d, reps, name):
    Xtr = build_design(d["Xr_tr_s"],d["Xp_tr_s"],d["selected"],reps,d["ridx"],d["fn"])
    Xte = build_design(d["Xr_te_s"],d["Xp_te_s"],d["selected"],reps,d["ridx"],d["fn"])
    Xi = sm.add_constant(Xtr,has_constant='add'); Xti = sm.add_constant(Xte,has_constant='add')
    ols = sm.OLS(d["y_tr"],Xi).fit()
    yp = ols.predict(Xti)
    tmse = mean_squared_error(d["y_te"],yp); tr2 = r2_score(d["y_te"],yp)
    print(f"[{name}] train R²={ols.rsquared:.4f}, test MSE={tmse:.6f}, test R²={tr2:.4f}")
    return yp, tmse, tr2

def portfolio_bt(pred1, close1, pred2, close2, dates):
    """双资产组合回测，返回 metrics dict"""
    df = pd.DataFrame({"d":dates,"p1":pred1,"c1":close1,"p2":pred2,"c2":close2})
    df = df.dropna().reset_index(drop=True)
    df["r1"] = df["c1"].pct_change(); df["r2"] = df["c2"].pct_change()
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    s1 = np.abs(df["p1"].values); s2 = np.abs(df["p2"].values)
    tot = s1+s2; tot = np.where(tot<1e-10,1.0,tot)
    pos1 = np.sign(df["p1"].values)*s1/tot*POS_LIMIT
    pos2 = np.sign(df["p2"].values)*s2/tot*POS_LIMIT
    pos1_lag = np.concatenate([[0],pos1[:-1]]); pos2_lag = np.concatenate([[0],pos2[:-1]])
    r1 = df["r1"].values; r2 = df["r2"].values
    contrib1 = pos1_lag*r1; contrib2 = pos2_lag*r2
    gross = contrib1+contrib2
    turn = np.abs(np.diff(np.concatenate([[0],pos1])))+np.abs(np.diff(np.concatenate([[0],pos2])))
    tc = turn*(SLIPPAGE+FEE)
    net = gross-tc
    nav = np.cumprod(1+net)
    dd = nav/np.maximum.accumulate(nav)-1
    cum = nav[-1]-1 if len(nav)>0 else 0
    ann = (1+cum)**(252/len(net))-1 if len(net)>0 else 0
    vol = np.std(net,ddof=1)*np.sqrt(252) if len(net)>1 else 0
    sharpe = ann/vol if vol>0 else 0
    mdd = np.min(dd) if len(dd)>0 else 0
    wr = np.mean(net>0) if len(net)>0 else 0
    av_turn = np.mean(turn)
    # 单资产
    nav1 = np.cumprod(1+contrib1); cum1 = nav1[-1]-1 if len(nav1)>0 else 0
    ann1 = (1+cum1)**(252/len(contrib1))-1 if len(contrib1)>0 else 0
    vol1 = np.std(contrib1,ddof=1)*np.sqrt(252) if len(contrib1)>1 else 0
    sh1 = ann1/vol1 if vol1>0 else 0
    dd1 = nav1/np.maximum.accumulate(nav1)-1; mdd1 = np.min(dd1) if len(dd1)>0 else 0
    nav2 = np.cumprod(1+contrib2); cum2 = nav2[-1]-1 if len(nav2)>0 else 0
    ann2 = (1+cum2)**(252/len(contrib2))-1 if len(contrib2)>0 else 0
    vol2 = np.std(contrib2,ddof=1)*np.sqrt(252) if len(contrib2)>1 else 0
    sh2 = ann2/vol2 if vol2>0 else 0
    dd2 = nav2/np.maximum.accumulate(nav2)-1; mdd2 = np.min(dd2) if len(dd2)>0 else 0
    return {"port_cum":cum,"port_ann":ann,"port_vol":vol,"port_sharpe":sharpe,"port_mdd":mdd,
            "port_winrate":wr,"port_turnover":av_turn,
            "a1_cum":cum1,"a1_ann":ann1,"a1_vol":vol1,"a1_sharpe":sh1,"a1_mdd":mdd1,
            "a2_cum":cum2,"a2_ann":ann2,"a2_vol":vol2,"a2_sharpe":sh2,"a2_mdd":mdd2,
            "nav":nav,"dates":df["d"].values}

# === Main ===
print("="*70); print("BILI + TC Backtest (4 scenarios)"); print("="*70)

bili_d = prepare(BILI_SRC, BILI_FACTORS, BILI_SHORT, BILI_SELECTED)
tc_d = prepare(TC_SRC, TC_FACTORS, TC_SHORT, TC_SELECTED)
print(f"BILI: train={len(bili_d['tr'])}, test={len(bili_d['te'])}")
print(f"TC:   train={len(tc_d['tr'])}, test={len(tc_d['te'])}")

# 3 场景：BILI 保持替换，TC 对比 PB→exp 和 PE ROE→exp
scenarios = [
    ("S1: BILI替换 + TC基线", BILI_REPLACE, {}),
    ("S2: BILI替换 + TC: PB→exp", BILI_REPLACE, {"PB": "exp"}),
    ("S3: BILI替换 + TC: PE ROE→exp", BILI_REPLACE, {"PE ROE": "exp"}),
]

all_results = []
for name, bili_rep, tc_rep in scenarios:
    print(f"\n--- {name} ---")
    bili_pred, bili_mse, bili_r2 = fit_predict(bili_d, bili_rep, "BILI")
    tc_pred, tc_mse, tc_r2 = fit_predict(tc_d, tc_rep, "TC")
    # 取 test 期对齐
    bili_te = bili_d["te"][["Trade Date","Close (HKD)","y"]].copy()
    bili_te["pred"] = bili_pred
    tc_te = tc_d["te"][["Trade Date","Close (HKD)","y"]].copy()
    tc_te["pred"] = tc_pred
    merged = pd.merge(bili_te, tc_te, on="Trade Date", suffixes=("_bili","_tc"))
    m = portfolio_bt(merged["pred_bili"].values, merged["Close (HKD)_bili"].values,
                      merged["pred_tc"].values, merged["Close (HKD)_tc"].values,
                      merged["Trade Date"].values)
    m["scenario"] = name; m["bili_mse"] = bili_mse; m["bili_r2"] = bili_r2
    m["tc_mse"] = tc_mse; m["tc_r2"] = tc_r2
    all_results.append(m)
    print(f"  Portfolio: Cum={m['port_cum']:.4%}, Ann={m['port_ann']:.4%}, Sharpe={m['port_sharpe']:.4f}, MDD={m['port_mdd']:.4%}")
    print(f"  BILI:      Cum={m['a1_cum']:.4%}, Sharpe={m['a1_sharpe']:.4f}")
    print(f"  TC:        Cum={m['a2_cum']:.4%}, Sharpe={m['a2_sharpe']:.4f}")

# === 对比表 ===
print("\n" + "="*70); print("4 场景对比"); print("="*70)
rows = []
for r in all_results:
    rows.append({
        "场景": r["scenario"],
        "组合累计收益": f"{r['port_cum']:.4%}",
        "组合年化": f"{r['port_ann']:.4%}",
        "组合波动": f"{r['port_vol']:.4%}",
        "组合夏普": f"{r['port_sharpe']:.4f}",
        "组合最大回撤": f"{r['port_mdd']:.4%}",
        "BILI累计": f"{r['a1_cum']:.4%}",
        "BILI夏普": f"{r['a1_sharpe']:.4f}",
        "TC累计": f"{r['a2_cum']:.4%}",
        "TC夏普": f"{r['a2_sharpe']:.4f}",
        "换手率": f"{r['port_turnover']:.4f}",
    })
df_cmp = pd.DataFrame(rows)
print(df_cmp.to_string(index=False))
df_cmp.to_csv(f"{OUT_DIR}/scenario_comparison.csv", index=False)

# === 两个 TC 替换效果对比（S2: PB→exp vs S3: PE ROE→exp）===
print("\n" + "="*70); print("TC 两个替换效果对比 (S2: PB→exp vs S3: PE ROE→exp)"); print("="*70)
s1 = all_results[0]; s2 = all_results[1]; s3 = all_results[2]
print(f"\nTC 基线 (S1): TC累计={s1['a2_cum']:.4%}, Sharpe={s1['a2_sharpe']:.4f}")
print(f"TC PB→exp (S2): TC累计={s2['a2_cum']:.4%}, Sharpe={s2['a2_sharpe']:.4f} (Δ vs基线={s2['a2_cum']-s1['a2_cum']:.4%})")
print(f"TC PE ROE→exp (S3): TC累计={s3['a2_cum']:.4%}, Sharpe={s3['a2_sharpe']:.4f} (Δ vs基线={s3['a2_cum']-s1['a2_cum']:.4%})")
print(f"\n组合对比:")
print(f"  S1 (TC基线):      组合={s1['port_cum']:.4%}, Sharpe={s1['port_sharpe']:.4f}")
print(f"  S2 (PB→exp):      组合={s2['port_cum']:.4%}, Sharpe={s2['port_sharpe']:.4f}")
print(f"  S3 (PE ROE→exp):  组合={s3['port_cum']:.4%}, Sharpe={s3['port_sharpe']:.4f}")

# === NAV 曲线 ===
fig, ax = plt.subplots(figsize=(12,6))
colors = ["gray","blue","green","red"]
for i, r in enumerate(all_results):
    ax.plot(r["dates"], r["nav"], label=r["scenario"], color=colors[i], linewidth=1.5)
ax.set_title("NAV Curve: 4 Scenarios (BILI + TC)")
ax.set_xlabel("Date"); ax.set_ylabel("NAV"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/nav_comparison.png", dpi=100); plt.close()
print(f"\n[Saved] {OUT_DIR}/")
