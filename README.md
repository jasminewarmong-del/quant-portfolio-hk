# quant-portfolio-hk

港股多资产量化选股策略：基于多项式因子工程 + 泰勒展开基函数替换 + 多资产组合回测 + 动态风控。

> ⚠️ Disclaimer: 本项目为学术练习，不构成投资建议。
> 策略 OOS 表现：组合 Sharpe ~1.15，但单资产 test R² 均为负，
> 实盘可行性存疑。仅供方法论学习参考。

## 项目概述

本项目对三支港股（Bilibili 9626.HK、Kuaishou 1024.HK、Tencent 9961.HK）构建多因子选股策略，完整流程包括：

1. **因子工程**：原始因子 → 二次多项式（含交互项）→ winsorize + 标准化
2. **变量选择**：Ridge/Lasso + TimeSeriesSplit CV + top-k 稳定性筛选
3. **基函数替换**：用泰勒展开的等价基函数（exp、sin、ln(1+z)、tanh、cos 等）替换多项式中的线性/平方/交互项，降低过拟合
4. **组合回测**：相对强度加权仓位 + 交易成本（滑点+手续费）+ 多资产分散
5. **动态风控**：连续亏损降仓 / deadband 过滤小信号 / 事件风控（极端行情中途打断）

策略最终版本（BILI + TC 两资产）：

- **BILI**: PB DAUg 交互项 → exp 替换
- **2TC**: PE ROE 交互项 → sin 替换
- 仓位：相对强度加权，总仓位 ≤ 0.9
- 风控：TC 连续亏损 ≥5 日降半仓、≥7 日暂停新增；连续盈利 ≥2 日恢复 50%、≥5 日恢复 100%
- Deadband：BILI |pred|<0.001 归零，TC |pred|<0.002 归零

测试期（2025-12-22 → 2026-09-02，170 天）核心指标：

| 方案              | Cum Return | Ann Return | Sharpe | Max DD  |
| --------------- | ---------- | ---------- | ------ | ------- |
| Baseline        | 21.22%     | 33.24%     | 0.98   | -14.08% |
| + Risk Control  | 22.84%     | 35.90%     | 1.13   | -13.24% |
| + RC + Deadband | 23.59%     | 37.15%     | 1.15   | -13.24% |

## 文件结构

```
quant-portfolio-hk/
├── README.md                              # 本文件
├── data/                                  # 原始因子数据 (xlsx)
│   ├── BILI_factor_data_extended.xlsx     # Bilibili 因子数据
│   ├── KS_factor_data_extended.xlsx       # Kuaishou 因子数据
│   └── TC_factor_data_extended.xlsx       # Tencent 因子数据
│
├── v1_stage123_BILI_lasso/                # BILI 阶段1-3 早期 Lasso/Ridge 版
├── v2_stage1234_BILI_tss/                 # BILI 阶段1-4 TimeSeriesSplit 合并版（最终）
├── v3_stage123_KS_lasso/                  # KS 阶段1-3 Lasso 版
├── v4_stage123_KS_ols/                    # KS 阶段1-3 OLS 版（不加 Lasso）
├── v5_stage123_TC/                        # TC 阶段1-3（固定 alpha=1.0）
├── v6_stage4_BILI_taylor/                 # BILI 阶段4 泰勒替换
├── v7_stage4_KS_taylor/                   # KS 阶段4 泰勒替换（3 个版本）
├── v8_stage4_TC_taylor/                   # TC 阶段4 泰勒替换
├── v9_backtest_BILI_KS/                   # 早期 BILI+KS 双资产回测（3 个迭代）
├── v10_backtest_BILI_TC_scenarios/        # BILI+TC 多场景对比（选 TC 最优替换）
├── v11_backtest_2asset_final/             # BILI+TC 最终版（RC + Deadband）
├── v12_backtest_3asset/                   # BILI+KS+TC 三资产组合
└── v13_backtest_event_risk/               # 中途打断版（含合成压力测试期）
```

## 各版本详细说明

### data/ — 原始数据

三个 xlsx 文件，每个含两个 sheet：

- **Sheet1**: Trade Date, y (5d Log Return) 未来 5 日对数收益率, Close (HKD) 收盘价
- **Sheet2**: Trade Date + 各资产因子值

| 资产   | 因子                                                                                                       |
| ---- | -------------------------------------------------------------------------------------------------------- |
| BILI | PB Ratio, Momentum 120d, DAU Growth (YoY %), MAU Growth (YoY %), Momentum 60d, Rel. Strength vs HSTECH   |
| KS   | PB Ratio, Momentum 60d, 20d Momentum, 20d Volatility, Volatility 60d (ann.), Relative Turnover, PE Ratio |
| TC   | PB Ratio, PE Ratio, ROE (%), 20d Momentum, MAU Growth (YoY %)                                            |

***

### v1\_stage123\_BILI\_lasso/ — BILI 阶段1-3 早期 Lasso/Ridge 版

**代码**：

- `stage123_polynomial_lasso.py` — LassoCV(5折) + 1-SE rule 变量选择
- `stage123_polynomial_ridge.py` — RidgeCV + top-k 稳定性筛选版本

**输出** (`output/`)：

- `stage2_ridge_topk_stability.csv` — Ridge top-k 稳定性
- `stage2_selection_frequency.csv` — 变量选择频率
- `stage3_metrics.csv` — OLS 重拟合指标
- `stage3_ols_coefficients.csv` — OLS 系数表
- `stage3_ridge_metrics.csv` — Ridge 指标
- `stage3_ridge_ols_coefficients.csv` — Ridge 选出后 OLS 系数
- `stage4_candidate_models.csv` — 泰勒替换候选模型（早期版本）

**说明**：早期用 KFold(shuffle=True) 有时间序列泄露问题，后续版本改用 TimeSeriesSplit。

***

### v2\_stage1234\_BILI\_tss/ — BILI 阶段1-4 TimeSeriesSplit 合并版（最终）

**代码**：`stage1234_BILI_tss.py`

**流程**：

1. Stage 1：数据合并 + dropna + 8/2 划分 + winsorize + poly(degree=2) + 标准化
2. Stage 2：RidgeCV(TimeSeriesSplit, 5折) → Ridge top-10 + 5折稳定性（频率≥3/5）→ 选出 6 个变量
3. Stage 3：OLS 重拟合 + CV MSE
4. Stage 4：单项泰勒替换 ablation，TimeSeriesSplit CV → 最优 `PB DAUg → exp`

**选出变量**：`['PB DAUg', 'M120 DAUg', 'M120 MAUg', 'M120 RS', 'DAUg RS', 'MAUg RS']`

**输出** (`output_BILI_tss/`)：

- `stage2_selection_frequency.csv`
- `stage3_ols_coefficients.csv`
- `stage4_candidate_models.csv`

***

### v3\_stage123\_KS\_lasso/ — KS 阶段1-3 Lasso 版

**代码**：`stage123_KS.py`

**流程**：LassoCV(TimeSeriesSplit, 5折) + 1-SE rule 变量选择 + OLS 重拟合

**输出** (`output_KS/`)：

- `stage2_selection_frequency.csv`
- `stage3_metrics.csv`
- `stage3_ols_coefficients.csv`

***

### v4\_stage123\_KS\_ols/ — KS 阶段1-3 OLS 版（不加 Lasso）

**代码**：`stage123_KS_ols.py`

**说明**：对第一个 OLS 式子直接 fit，然后做泰勒替换可替换项。此版本不做 Lasso 变量选择。

**输出** (`output_KS_ols/`)：

- `stage3_metrics.csv`
- `stage3_ols_coefficients.csv`

***

### v5\_stage123\_TC/ — TC 阶段1-3（固定 alpha=1.0）

**代码**：`stage123_TC.py`

**流程**：

1. 数据合并 + dropna + 8/2 划分 + winsorize + poly + 标准化
2. **固定 alpha=1.0**（不用 RidgeCV，避免 CV MSE 极端值）+ Ridge top-10 + 5折 TSS 稳定性 → 选出 7 个变量
3. OLS 重拟合 + 系数表 + metrics

**选出变量**：`['PB', 'ROE', 'M20', 'MAUg^2', 'PE ROE', 'PE MAUg', 'ROE MAUg']`

**输出** (`output_TC/`)：

- `stage2_selection_frequency.csv`
- `stage3_ols_coefficients.csv`
- `stage3_metrics.csv`

***

### v6\_stage4\_BILI\_taylor/ — BILI 阶段4 泰勒替换

**代码**：`stage4_BILI_taylor.py`

**依赖**：使用 v2 选出的 6 个变量（硬编码在 SELECTED 中）

**流程**：对每个选出变量做单项泰勒替换 ablation（linear→exp/pow\_0.5/pow\_2/pow\_-1, square→cos/sec, interaction→ln1pz/sin/tanh/exp），TimeSeriesSplit CV → 最优 `PB DAUg → exp`

**输出** (`output_BILI_stage4/`)：

- `stage4_candidate_models.csv` — 所有候选模型 CV MSE、Test MSE、改善百分比

***

### v7\_stage4\_KS\_taylor/ — KS 阶段4 泰勒替换（3 个版本）

**代码**：

- `stage4_KS_fix.py` — 早期 fix 版
- `stage4_KS_taylor.py` — 标准 KFold 版
- `stage4_KS_taylor_tss.py` — TimeSeriesSplit 版（最终）

**依赖**：使用 v3/v4 选出的 KS 变量

**输出**：

- `output_KS_stage4/stage4_candidate_models.csv` — 标准 KFold 版结果
- `output_KS_stage4/stage4_fix_models.csv` — fix 版结果
- `output_KS_stage4_tss/stage4_candidate_models.csv` — TSS 版结果（最终，最优 `PB V60 → sin`）

***

### v8\_stage4\_TC\_taylor/ — TC 阶段4 泰勒替换

**代码**：`stage4_TC_taylor.py`

**依赖**：使用 v5 选出的 7 个变量

**流程**：单项泰勒替换 ablation，TimeSeriesSplit CV → 用户最终指定采用 `PE ROE → sin`

**输出** (`output_TC_stage4/`)：

- `stage4_candidate_models.csv`

***

### v9\_backtest\_BILI\_KS/ — 早期 BILI+KS 双资产回测（3 个迭代）

**代码**：`backtest.py`

**说明**：最早的双资产组合回测，BILI + KS，3 个迭代版本（v1/v2/v3），主要调试回测框架、成本参数、敏感性分析。

**输出**：

- `backtest_results_v1/` — 第 1 版结果
- `backtest_results_v2/` — 第 2 版结果
- `backtest_results_v3/` — 第 3 版结果

每个目录含：

- `BILI_KUAISHOU_backtest_results.xlsx` — 综合结果
- `performance_metrics.csv`, `portfolio_daily.csv`, `trade_history.csv`
- `portfolio_nav_curve.png`, `portfolio_drawdown_curve.png`
- `benchmark.csv`, `cost_sensitivity.csv`, `position_sensitivity.csv`, `model_predictions.csv`

***

### v10\_backtest\_BILI\_TC\_scenarios/ — BILI+TC 多场景对比

**代码**：`backtest_BILI_TC.py`

**说明**：BILI 替换固定（PB DAUg→exp），对比 TC 三种替换方案（基线 / PB→exp / PE ROE→exp），选定 TC 最终替换。

**调仓函数**：相对强度加权，总仓位 0.9，无风控。

**输出** (`output/`)：

- `scenario_comparison.csv` — 3 场景对比表
- `nav_comparison.png` — NAV 曲线

***

### v11\_backtest\_2asset\_final/ — BILI+TC 最终版（RC + Deadband）

**代码**：`backtest_2asset.py`

**最终配置**：

- BILI: `PB DAUg → exp`
- TC: `PE ROE → sin`（v10 对比后选定）
- 调仓：相对强度加权 ×0.9
- **Risk Control（仅 TC）**：连续亏损 ≥5 日降半仓、≥7 日暂停新增；连续盈利 ≥2 日恢复 50%、≥5 日恢复 100%
- **差异化 Deadband**：BILI |pred|<0.001 归零，TC |pred|<0.002 归零
- 成本：滑点 5bp + 手续费 3bp

**输出** (`output/`)：

- `baseline_vs_rc_comparison.csv` — 3 方案（Baseline / RC / RC+Deadband）Portfolio+TC 对比
- `tc_detail_baseline_vs_rc.csv` — TC 的 long/short/flat、avg/max/min position
- `tc_risk_state.csv` — 逐日 risk\_state、loss\_streak、profit\_streak、base\_pos、adj\_pos
- `portfolioA_trade_history.csv` — 组合A 逐日交易记录（14 核心列 + 12 单资产明细列）
- `portfolioA_trade_summary.csv` — 交易汇总（long/short/flat、avg holding period 等）
- `nav_baseline_vs_rc.png` — 3 条 NAV 曲线
- `tc_position_baseline_vs_rc.png` — TC 仓位对比 + risk state 切换点
- `2asset_comparison.csv` — 早期 2 方案对比

***

### v12\_backtest\_3asset/ — BILI+KS+TC 三资产组合

**代码**：`backtest_3asset.py`

**配置**：

- BILI: `PB DAUg → exp`
- KS: `PB V60 → sin`
- TC: `PE ROE → sin`
- 调仓：三资产相对强度加权 ×0.9

**说明**：三资产组合测试，结论是 KS 拖累严重（年化 -33%），后续改回两资产（v11）。

**输出** (`output/`)：

- `3asset_comparison.csv` — 三资产 + 组合 + 基准对比
- `nav_3asset.png` — NAV 曲线

***

### v13\_backtest\_event\_risk/ — 中途打断版（含合成压力测试期）

**代码**：`backtest_2asset_event_risk.py`

**依赖**：基于 v11 (`backtest_2asset.py`) 复制 + 扩展，逻辑非 import。

**新增模块**：

1. **合成压力测试期数据生成** (`generate_synthetic_stress_period`)：63 个交易日，含 6 阶段事件（正常期/单日闪崩/连续下跌/波动飙升/系统性事件/恢复期）
2. **事件风控模块** (`compute_event_risk_state`)：三个触发器
   - A 极端日跌幅：ret < train 期 5% 分位
   - C 快速回撤：5 日累计收益 < train 期 5% 分位
   - B 波动率飙升：20d vol > train 均值 × 2.5
   - 1 触发器 → ×0.5，≥2 触发器 → ×0.0
   - 3 日冷却线性恢复
   - 严格无前视：t-1 日数据判定，t 日仓位

**4 方案对比**：

- Baseline
- Risk-Controlled
- RC + Deadband
- RC + Deadband + Event Risk

**输出** (`output/`)：

- `baseline_vs_full_comparison.csv` — 4 方案对比表
- `event_risk_state.csv` — 逐日 multiplier、仓位、is\_synth 标记
- `tc_detail_full.csv` — TC 详细对比
- `nav_full_comparison.png` — 4 条 NAV 曲线（含合成期起点+事件日标注）
- `event_multiplier.png` — BILI 和 TC 的 multiplier 逐日曲线
- `synthetic_data_preview.csv` — 合成期数据预览

**⚠️ 重要说明**：合成压力测试期使用合成数据，**不代表真实策略表现**，仅用于验证事件风控模块在极端行情下是否按规则触发。

***

## 依赖关系

```
data/ (xlsx)
  ↓
v1/v2 (BILI stage123)  →  v6 (BILI stage4)  →  v10/v11/v12/v13 (backtest)
v3/v4 (KS stage123)    →  v7 (KS stage4)    →  v12 (3asset)
v5 (TC stage123)       →  v8 (TC stage4)    →  v10/v11/v12/v13 (backtest)
```

**依赖方式**：所有依赖通过**硬编码**实现（变量名、替换规则直接写在代码里），不通过 import 导入。例如 `backtest_2asset.py` 直接写死 `BILI_REPLACE = {"PB DAUg":"exp"}`，这个结果来自 v6 的 stage4 输出。

***

## 运行环境

- Python 3.10+
- 依赖包：`numpy pandas scikit-learn statsmodels matplotlib openpyxl`

## 关键参数

| 参数            | 值                       | 说明               |
| ------------- | ----------------------- | ---------------- |
| 多项式阶数         | 2                       | 含主效应、平方、交互项      |
| Train/Test 划分 | 8:2                     | 时间序列顺序，无 shuffle |
| CV 方式         | TimeSeriesSplit(5折)     | 避免时间序列泄露         |
| Winsorize     | 1%/99% 分位               | 仅用 train 期分位     |
| 滑点            | 0.0005 (5bp)            | 线性比例             |
| 手续费           | 0.0003 (3bp)            | 线性比例             |
| 仓位上限          | 0.90                    | 总仓位              |
| RC 阈值         | 5/7/2/5                 | 连续亏损/恢复          |
| Deadband      | BILI=0.001, TC=0.002    | 差异化              |
| 事件风控阈值        | 5% 分位 / 2.5×vol / 3 日冷却 | 基于 train 期       |

## 方法论说明

### 泰勒基函数替换

多项式项（如 x², xy）可用泰勒展开的等价基函数替换，降低过拟合：

| 项类型   | 可替换基函数                         | 数学基础                  |
| ----- | ------------------------------ | --------------------- |
| 线性 x  | exp, pow\_0.5, pow\_2, pow\_-1 | 一阶泰勒展开                |
| 平方 x² | cos, sec                       | 二阶泰勒展开（cos(x)≈1-x²/2） |
| 交互 xy | ln(1+z), sin, tanh, exp        | 多元泰勒展开                |

### Risk Control（连续亏损风控）

- 统计单位：连续亏损交易日（非笔数）
- 亏损日定义：扣除成本后净 P\&L < 0
- **用 baseline position 假设持有判定 streak**（避免与 RC 自身循环依赖）
- 严格无前视：t-1 日 P\&L → t 日仓位

### 事件风控（中途打断）

- 三个触发器：极端日跌幅 / 快速回撤 / 波动率飙升
- 用价格极端变动作为战争/政策/极端事件的代理
- 阈值基于 train 期，不被合成期污染
- 严格无前视：t-1 日数据 → t 日 multiplier

## License

MIT
