"""
BILI + KS + TC 三资产组合回测
BILI: PB DAUg→exp
KS:   PB V60→sin
TC:   PE ROE→sin
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

BILI_SRC = "/workspace/.uploads/d4d06402-aa6f-40b0-b7c8-c4d694a048eb_BILI_factor_data_extended.xlsx"
KS_SRC = "/workspace/.uploads/c6f91cd6-d922-4098-84e0-2cad2a717f51_KS_factor_data_extended.xlsx"
TC_SRC = "/workspace/.uploads/35657faa-847c-4c1d-b404-5b5165eb83ec_TC_factor_data_extended.xlsx"
OUT_DIR = "/workspace/backtest_3asset"
os.makedirs(OUT_DIR, exist_ok=True)

SLIPPAGE=0.0005; FEE=0.0003; POS_LIMIT=0.90

# BILI
BILI_FACTORS=["PB Ratio","Momentum 120d","DAU Growth (YoY %)","MAU Growth (YoY %)","Momentum 60d","Rel. Strength vs HSTECH"]
BILI_SHORT={"y (5d Log Return)":"y","PB Ratio":"PB","Momentum 120d":"M120","DAU Growth (YoY %)":"DAUg","MAU Growth (YoY %)":"MAUg","Momentum 60d":"M60","Rel. Strength vs HSTECH":"RS"}
BILI_SELECTED=['PB DAUg','M120 DAUg','M120 MAUg','M120 RS','DAUg RS','MAUg RS']
BILI_REPLACE={"PB DAUg":"exp"}

# KS
KS_FACTORS=["PB Ratio","Momentum 60d","20d Momentum","20d Volatility","Volatility 60d (ann.)","Relative Turnover","PE Ratio"]
KS_SHORT={"y (5d Log Return)":"y","PB Ratio":"PB","Momentum 60d":"M60","20d Momentum":"M20","20d Volatility":"V20","Volatility 60d (ann.)":"V60","Relative Turnover":"RT","PE Ratio":"PE"}
KS_SELECTED=['PB','PE','PB^2','PB V60','PB PE','M60 M20','V60 PE','PE^2']
KS_REPLACE={"PB V60":"sin"}

# TC
TC_FACTORS=["PB Ratio","PE Ratio","ROE (%)","20d Momentum","MAU Growth (YoY %)"]
TC_SHORT={"y (5d Log Return)":"y","PB Ratio":"PB","PE Ratio":"PE","ROE (%)":"ROE","20d Momentum":"M20","MAU Growth (YoY %)":"MAUg"}
TC_SELECTED=['PB','ROE','M20','MAUg^2','PE ROE','PE MAUg','ROE MAUg']
TC_REPLACE={"PE ROE":"sin"}

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
BASIS={"exp":g_exp,"ln1pz":g_ln1pz,"sin":g_sin,"tanh":g_tanh,"cos":g_cos,"sec":g_sec,"pow_0.5":g_pow05,"pow_2":g_pow2,"pow_-1":g_powm1}

def classify(v):
    if " " in v: return "interaction"
    elif "^2" in v: return "square"
    else: return "linear"

def build_design(Xr,Xp,selected,reps,ridx,fnames):
    cols=[]
    for v in selected:
        if v in reps:
            bn=reps[v]; vt=classify(v)
            if vt=="square":
                base=v.replace("^2",""); x=Xr[:,ridx[base]]
                cols.append(BASIS[bn](x))
            elif vt=="linear":
                x=Xr[:,ridx[v]]; cols.append(BASIS[bn](x))
            else:
                idx=list(fnames).index(v); z=Xp[:,idx]
                cols.append(BASIS[bn](z))
        else:
            idx=list(fnames).index(v); cols.append(Xp[:,idx])
    return np.column_stack(cols)

def prepare(src,factors,short,selected):
    s1=pd.read_excel(src,sheet_name="Sheet1")
    s2=pd.read_excel(src,sheet_name="Sheet2")
    df=pd.merge(s1[["Trade Date","y (5d Log Return)","Close (HKD)"]],
                s2[["Trade Date"]+factors],on="Trade Date").rename(columns=short)
    fcols=[short[c] for c in factors]
    df=df.dropna().reset_index(drop=True)
    n=len(df); n_tr=int(n*0.8)
    tr=df.iloc[:n_tr].copy(); te=df.iloc[n_tr:].copy()
    Xr_tr=tr[fcols].values; Xr_te=te[fcols].values
    y_tr=tr["y"].values; y_te=te["y"].values
    lo=np.quantile(Xr_tr,0.01,axis=0); hi=np.quantile(Xr_tr,0.99,axis=0)
    Xr_tr_w=np.clip(Xr_tr,lo[None,:],hi[None,:]); Xr_te_w=np.clip(Xr_te,lo[None,:],hi[None,:])
    poly=PolynomialFeatures(degree=2,include_bias=False)
    Xp_tr=poly.fit_transform(Xr_tr_w); Xp_te=poly.transform(Xr_te_w)
    fn=poly.get_feature_names_out(fcols)
    sp=StandardScaler(); Xp_tr_s=sp.fit_transform(Xp_tr); Xp_te_s=sp.transform(Xp_te)
    sr=StandardScaler(); Xr_tr_s=sr.fit_transform(Xr_tr_w); Xr_te_s=sr.transform(Xr_te_w)
    ridx={f:i for i,f in enumerate(fcols)}
    return {"tr":tr,"te":te,"y_tr":y_tr,"y_te":y_te,
            "Xr_tr_s":Xr_tr_s,"Xr_te_s":Xr_te_s,"Xp_tr_s":Xp_tr_s,"Xp_te_s":Xp_te_s,
            "fn":fn,"ridx":ridx,"selected":selected}

def fit_predict(d,reps,name):
    Xtr=build_design(d["Xr_tr_s"],d["Xp_tr_s"],d["selected"],reps,d["ridx"],d["fn"])
    Xte=build_design(d["Xr_te_s"],d["Xp_te_s"],d["selected"],reps,d["ridx"],d["fn"])
    Xi=sm.add_constant(Xtr,has_constant='add'); Xti=sm.add_constant(Xte,has_constant='add')
    ols=sm.OLS(d["y_tr"],Xi).fit()
    yp=ols.predict(Xti)
    tmse=mean_squared_error(d["y_te"],yp); tr2=r2_score(d["y_te"],yp)
    print(f"[{name}] train R²={ols.rsquared:.4f}, test MSE={tmse:.6f}, test R²={tr2:.4f}")
    return yp,tmse,tr2

def portfolio_bt_3asset(preds, closes, dates, asset_names):
    """3 资产组合回测"""
    df=pd.DataFrame({"d":dates})
    for i,an in enumerate(asset_names):
        df[f"p_{an}"]=preds[i]; df[f"c_{an}"]=closes[i]
    df=df.dropna().reset_index(drop=True)
    for an in asset_names:
        df[f"r_{an}"]=df[f"c_{an}"].pct_change()
    df=df.dropna().reset_index(drop=True)
    n=len(df)
    # 仓位：相对强度加权
    strengths=[np.abs(df[f"p_{an}"].values) for an in asset_names]
    tot=np.sum(strengths,axis=0); tot=np.where(tot<1e-10,1.0,tot)
    positions=[]
    for i,an in enumerate(asset_names):
        pos=np.sign(df[f"p_{an}"].values)*strengths[i]/tot*POS_LIMIT
        positions.append(pos)
    # lag 一天
    pos_lag=[np.concatenate([[0],p[:-1]]) for p in positions]
    returns=[df[f"r_{an}"].values for an in asset_names]
    contribs=[pos_lag[i]*returns[i] for i in range(len(asset_names))]
    gross=np.sum(contribs,axis=0)
    # turnover
    turn=0
    for i in range(len(asset_names)):
        turn+=np.abs(np.diff(np.concatenate([[0],positions[i]])))
    tc=turn*(SLIPPAGE+FEE)
    net=gross-tc
    nav=np.cumprod(1+net)
    dd=nav/np.maximum.accumulate(nav)-1
    cum=nav[-1]-1 if len(nav)>0 else 0
    ann=(1+cum)**(252/len(net))-1 if len(net)>0 else 0
    vol=np.std(net,ddof=1)*np.sqrt(252) if len(net)>1 else 0
    sharpe=ann/vol if vol>0 else 0
    mdd=np.min(dd) if len(dd)>0 else 0
    wr=np.mean(net>0) if len(net)>0 else 0
    av_turn=np.mean(turn)
    # 单资产
    asset_metrics={}
    for i,an in enumerate(asset_names):
        c=contribs[i]
        nav_a=np.cumprod(1+c); cum_a=nav_a[-1]-1 if len(nav_a)>0 else 0
        ann_a=(1+cum_a)**(252/len(c))-1 if len(c)>0 else 0
        vol_a=np.std(c,ddof=1)*np.sqrt(252) if len(c)>1 else 0
        sh_a=ann_a/vol_a if vol_a>0 else 0
        dd_a=nav_a/np.maximum.accumulate(nav_a)-1
        mdd_a=np.min(dd_a) if len(dd_a)>0 else 0
        asset_metrics[an]={"cum":cum_a,"ann":ann_a,"vol":vol_a,"sharpe":sh_a,"mdd":mdd_a}
    return {"port_cum":cum,"port_ann":ann,"port_vol":vol,"port_sharpe":sharpe,
            "port_mdd":mdd,"port_winrate":wr,"port_turnover":av_turn,
            "assets":asset_metrics,"nav":nav,"dates":df["d"].values}

print("="*70); print("3-Asset Backtest (BILI+KS+TC)"); print("="*70)

bili_d=prepare(BILI_SRC,BILI_FACTORS,BILI_SHORT,BILI_SELECTED)
ks_d=prepare(KS_SRC,KS_FACTORS,KS_SHORT,KS_SELECTED)
tc_d=prepare(TC_SRC,TC_FACTORS,TC_SHORT,TC_SELECTED)
print(f"BILI: train={len(bili_d['tr'])}, test={len(bili_d['te'])}")
print(f"KS:   train={len(ks_d['tr'])}, test={len(ks_d['te'])}")
print(f"TC:   train={len(tc_d['tr'])}, test={len(tc_d['te'])}")

# 各自最优替换
bili_pred,bili_mse,bili_r2=fit_predict(bili_d,BILI_REPLACE,"BILI")
ks_pred,ks_mse,ks_r2=fit_predict(ks_d,KS_REPLACE,"KS")
tc_pred,tc_mse,tc_r2=fit_predict(tc_d,TC_REPLACE,"TC")

# 对齐 test 期
bili_te=bili_d["te"][["Trade Date","Close (HKD)","y"]].copy(); bili_te["pred"]=bili_pred
ks_te=ks_d["te"][["Trade Date","Close (HKD)","y"]].copy(); ks_te["pred"]=ks_pred
tc_te=tc_d["te"][["Trade Date","Close (HKD)","y"]].copy(); tc_te["pred"]=tc_pred

# 3 路 merge
m=pd.merge(bili_te,ks_te,on="Trade Date",suffixes=("_bili","_ks"))
m=pd.merge(m,tc_te,on="Trade Date",suffixes=("","_tc"))
m=m.rename(columns={"Close (HKD)":"Close (HKD)_tc","pred":"pred_tc","y":"y_tc"})

preds=[m["pred_bili"].values,m["pred_ks"].values,m["pred_tc"].values]
closes=[m["Close (HKD)_bili"].values,m["Close (HKD)_ks"].values,m["Close (HKD)_tc"].values]
dates=m["Trade Date"].values
asset_names=["BILI","KS","TC"]

result=portfolio_bt_3asset(preds,closes,dates,asset_names)

print("\n"+"="*70); print("3-Asset Portfolio Results"); print("="*70)
print(f"Test period: {dates[0]} → {dates[-1]} ({len(dates)} days)")
print(f"\nPortfolio:")
print(f"  Cumulative Return: {result['port_cum']:.4%}")
print(f"  Annualized Return: {result['port_ann']:.4%}")
print(f"  Annualized Vol:    {result['port_vol']:.4%}")
print(f"  Sharpe:            {result['port_sharpe']:.4f}")
print(f"  Max Drawdown:      {result['port_mdd']:.4%}")
print(f"  Win Rate:          {result['port_winrate']:.4%}")
print(f"  Avg Turnover:      {result['port_turnover']:.4f}")

print(f"\n各资产单独表现:")
for an in asset_names:
    a=result["assets"][an]
    print(f"  {an}: cum={a['cum']:.4%}, ann={a['ann']:.4%}, vol={a['vol']:.4%}, sharpe={a['sharpe']:.4f}, mdd={a['mdd']:.4%}")

# 对比表
rows=[{
    "Asset":"Portfolio",
    "Cum Return":f"{result['port_cum']:.4%}",
    "Ann Return":f"{result['port_ann']:.4%}",
    "Ann Vol":f"{result['port_vol']:.4%}",
    "Sharpe":f"{result['port_sharpe']:.4f}",
    "Max DD":f"{result['port_mdd']:.4%}",
    "Win Rate":f"{result['port_winrate']:.4%}",
    "Turnover":f"{result['port_turnover']:.4f}",
}]
for an in asset_names:
    a=result["assets"][an]
    rows.append({
        "Asset":an,
        "Cum Return":f"{a['cum']:.4%}",
        "Ann Return":f"{a['ann']:.4%}",
        "Ann Vol":f"{a['vol']:.4%}",
        "Sharpe":f"{a['sharpe']:.4f}",
        "Max DD":f"{a['mdd']:.4%}",
        "Win Rate":"-",
        "Turnover":"-",
    })
df_cmp=pd.DataFrame(rows)
print("\n对比表:")
print(df_cmp.to_string(index=False))
df_cmp.to_csv(f"{OUT_DIR}/3asset_comparison.csv",index=False)

# benchmark: 等权 buy and hold
bh_returns=[]
for an in asset_names:
    r=pd.Series(m[f"Close (HKD)_{an.lower() if an!='TC' else 'tc'}"].values).pct_change().dropna().values
    bh_returns.append(r)
# 等权
min_len=min(len(r) for r in bh_returns)
bh_returns=[r[:min_len] for r in bh_returns]
bh_eq=np.mean(bh_returns,axis=0)
bh_nav=np.cumprod(1+bh_eq)
bh_cum=bh_nav[-1]-1
bh_ann=(1+bh_cum)**(252/min_len)-1
bh_vol=np.std(bh_eq,ddof=1)*np.sqrt(252)
bh_sharpe=bh_ann/bh_vol if bh_vol>0 else 0
print(f"\nBenchmark (等权 buy&hold): cum={bh_cum:.4%}, ann={bh_ann:.4%}, sharpe={bh_sharpe:.4f}")

# NAV 曲线
fig,ax=plt.subplots(figsize=(12,6))
ax.plot(result["dates"],result["nav"],label="3-Asset Portfolio",color="red",linewidth=2)
ax.plot(result["dates"][:min_len],bh_nav,label="Equal-weight B&H",color="gray",linewidth=1.5,linestyle="--")
nd=len(result["dates"])
for an in asset_names:
    nav_a=np.cumprod(1+pd.Series(m[f"Close (HKD)_{an.lower() if an!='TC' else 'tc'}"].values).pct_change().fillna(0).values)
    ax.plot(result["dates"],nav_a[:nd],label=f"{an} B&H",alpha=0.5,linewidth=1)
ax.set_title("3-Asset Portfolio NAV vs Benchmark")
ax.set_xlabel("Date"); ax.set_ylabel("NAV"); ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/nav_3asset.png",dpi=100); plt.close()
print(f"\n[Saved] {OUT_DIR}/")
