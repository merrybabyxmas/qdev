#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/artifacts/quant_resource_bundle}"
CANONICAL_PAPER_ROOT="${CANONICAL_PAPER_ROOT:-$ROOT_DIR/docs/references/papers}"
PRIORITY_MAX="${PRIORITY_MAX:-3}"
KIND_FILTER="${KIND_FILTER:-all}"
MODE="${MODE:-download}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ensure_paper_root() {
  mkdir -p "$CANONICAL_PAPER_ROOT"
  if [[ -d "$ARTIFACT_ROOT/papers" && ! -L "$ARTIFACT_ROOT/papers" ]]; then
    shopt -s nullglob
    local existing_papers=("$ARTIFACT_ROOT/papers"/*.pdf)
    shopt -u nullglob
    if ((${#existing_papers[@]} > 0)); then
      mv "${existing_papers[@]}" "$CANONICAL_PAPER_ROOT"/
    fi
    rmdir "$ARTIFACT_ROOT/papers" 2>/dev/null || true
  elif [[ -e "$ARTIFACT_ROOT/papers" && ! -L "$ARTIFACT_ROOT/papers" ]]; then
    rm -rf "$ARTIFACT_ROOT/papers"
  fi
  if [[ ! -e "$ARTIFACT_ROOT/papers" ]]; then
    ln -s "$CANONICAL_PAPER_ROOT" "$ARTIFACT_ROOT/papers"
  fi
}

ensure_paper_root
mkdir -p "$ARTIFACT_ROOT/repos" "$ARTIFACT_ROOT/pypi" "$ARTIFACT_ROOT/logs"

success_count=0
skip_count=0
fail_count=0
paper_count=0
repo_count=0
package_count=0

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

should_run() {
  local priority="$1"
  local kind="$2"
  [[ "$priority" -le "$PRIORITY_MAX" ]] || return 1
  [[ "$KIND_FILTER" == "all" || "$KIND_FILTER" == "$kind" ]] || return 1
  return 0
}

run_paper() {
  local priority="$1"
  local label="$2"
  local dest_rel="$3"
  local source="$4"
  local dest="$ARTIFACT_ROOT/$dest_rel"
  if ! should_run "$priority" paper; then
    ((skip_count+=1))
    return 0
  fi
  if [[ -s "$dest" && "$FORCE" != "1" ]]; then
    log "paper skip: $label"
    ((skip_count+=1))
    ((paper_count+=1))
    return 0
  fi
  log "paper fetch: $label"
  if wget -q --show-progress --tries=3 --timeout=60 --waitretry=2 -U 'Mozilla/5.0' -O "$dest" "$source"; then
    ((success_count+=1))
  else
    ((fail_count+=1))
  fi
  ((paper_count+=1))
}

run_repo() {
  local priority="$1"
  local label="$2"
  local dest_rel="$3"
  local source="$4"
  local dest="$ARTIFACT_ROOT/$dest_rel"
  if ! should_run "$priority" repo; then
    ((skip_count+=1))
    return 0
  fi
  if [[ -e "$dest" && "$FORCE" != "1" ]]; then
    log "repo skip: $label"
    ((skip_count+=1))
    ((repo_count+=1))
    return 0
  fi
  if [[ -e "$dest" && "$FORCE" == "1" ]]; then
    rm -rf "$dest"
  fi
  mkdir -p "$(dirname "$dest")"
  log "repo fetch: $label"
  if GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --single-branch "$source" "$dest"; then
    ((success_count+=1))
  else
    ((fail_count+=1))
  fi
  ((repo_count+=1))
}

run_package() {
  local priority="$1"
  local label="$2"
  local dest_rel="$3"
  shift 3
  local dest="$ARTIFACT_ROOT/$dest_rel"
  if ! should_run "$priority" package; then
    ((skip_count+=1))
    return 0
  fi
  if [[ "$label" == "tensorflow" && "${ALLOW_TENSORFLOW:-0}" != "1" ]]; then
    log "package skip: $label (torch-first policy)"
    ((skip_count+=1))
    ((package_count+=1))
    return 0
  fi
  if [[ -e "$dest" && "$FORCE" != "1" ]]; then
    log "package skip: $label"
    ((skip_count+=1))
    ((package_count+=1))
    return 0
  fi
  mkdir -p "$dest"
  log "package fetch: $label"
  if [[ "$MODE" == "install" ]]; then
    if "$PYTHON_BIN" -m pip install --disable-pip-version-check "$@"; then
      ((success_count+=1))
    else
      ((fail_count+=1))
    fi
  else
    if "$PYTHON_BIN" -m pip download --no-deps --disable-pip-version-check --dest "$dest" "$@"; then
      ((success_count+=1))
    else
      ((fail_count+=1))
    fi
  fi
  ((package_count+=1))
}

# Duplicate notes preserved from the generator.
# - removed duplicate repo: https://github.com/microsoft/qlib.git (alias qlib)
# - removed duplicate paper: https://arxiv.org/pdf/1708.07469.pdf (alias 4_10_Deep_Learning_for_High_Dimensional_PDEs.pdf)
# - removed duplicate paper: https://arxiv.org/pdf/2304.07619.pdf (alias 5_05_Can_ChatGPT_Forecast_Stock_Price_Movements.pdf)

# Priority 1
# Papers
run_paper 1 'Deep Portfolio Theory' papers/2_01_Deep_Portfolio_Theory.pdf https://arxiv.org/pdf/1605.02654.pdf
run_paper 1 'RL Framework for Portfolio Management' papers/2_02_RL_Framework_for_Portfolio_Management.pdf https://arxiv.org/pdf/1706.10059.pdf
run_paper 1 'Deep Learning for Portfolio Optimization' papers/2_03_Deep_Learning_for_Portfolio_Optimization.pdf https://arxiv.org/pdf/2005.13665.pdf
run_paper 1 'Deep Reinforcement Learning for Portfolio Empirical' papers/2_04_Deep_Reinforcement_Learning_for_Portfolio_Empirical.pdf https://arxiv.org/pdf/1904.04963.pdf
run_paper 1 'Multi Agent RL for Portfolio Management' papers/2_05_Multi_Agent_RL_for_Portfolio_Management.pdf https://arxiv.org/pdf/2002.05268.pdf
run_paper 1 'Portfolio Optimization based on Neural Networks' papers/2_06_Portfolio_Optimization_based_on_Neural_Networks.pdf https://arxiv.org/pdf/1911.04941.pdf
run_paper 1 'Deep Q Learning for Portfolio Management' papers/2_07_Deep_Q_Learning_for_Portfolio_Management.pdf https://arxiv.org/pdf/1808.09911.pdf
run_paper 1 'Risk sensitive Deep RL for Portfolio' papers/2_08_Risk_sensitive_Deep_RL_for_Portfolio.pdf https://arxiv.org/pdf/2007.13606.pdf
run_paper 1 'Portfolio Management with Graph Neural Networks' papers/2_09_Portfolio_Management_with_Graph_Neural_Networks.pdf https://arxiv.org/pdf/2010.13000.pdf
run_paper 1 'Machine Learning Approach to Portfolio Risk' papers/2_10_Machine_Learning_Approach_to_Portfolio_Risk.pdf https://arxiv.org/pdf/1802.10469.pdf
run_paper 1 'Continuous Time Mean Variance Portfolio RL' papers/2_11_Continuous_Time_Mean_Variance_Portfolio_RL.pdf https://arxiv.org/pdf/1909.09571.pdf
run_paper 1 'Online Portfolio Selection A Survey' papers/2_12_Online_Portfolio_Selection_A_Survey.pdf https://arxiv.org/pdf/1212.2129.pdf
run_paper 1 'FinRL Deep RL for Automated Stock Trading' papers/3_01_FinRL_Deep_RL_for_Automated_Stock_Trading.pdf https://arxiv.org/pdf/2011.09607.pdf
run_paper 1 'DeepLOB Deep CNN for Limit Order Books' papers/3_02_DeepLOB_Deep_CNN_for_Limit_Order_Books.pdf https://arxiv.org/pdf/1808.03668.pdf
run_paper 1 'RL in Quantitative Finance A Review' papers/3_03_RL_in_Quantitative_Finance_A_Review.pdf https://arxiv.org/pdf/2105.10178.pdf
run_paper 1 'Algorithmic Trading with Deep RL' papers/3_04_Algorithmic_Trading_with_Deep_RL.pdf https://arxiv.org/pdf/1908.08272.pdf
run_paper 1 'Practical Deep RL Approach for Stock Trading' papers/3_05_Practical_Deep_RL_Approach_for_Stock_Trading.pdf https://arxiv.org/pdf/1811.07522.pdf
run_paper 1 'RL for Optimal Execution' papers/3_06_RL_for_Optimal_Execution.pdf https://arxiv.org/pdf/1812.06600.pdf
run_paper 1 'Universal features of price formation DL' papers/3_07_Universal_features_of_price_formation_DL.pdf https://arxiv.org/pdf/1803.06917.pdf
run_paper 1 'High Frequency Trading in Limit Order Book' papers/3_08_High_Frequency_Trading_in_Limit_Order_Book.pdf https://arxiv.org/pdf/1310.1140.pdf
run_paper 1 'Deep RL Automated Trading Ensemble' papers/3_09_Deep_RL_Automated_Trading_Ensemble.pdf https://arxiv.org/pdf/2005.01518.pdf
run_paper 1 'Limit Order Book Modelling with DL' papers/3_10_Limit_Order_Book_Modelling_with_DL.pdf https://arxiv.org/pdf/1701.08170.pdf
run_paper 1 'Multi agent RL for Liquidation Strategy' papers/3_11_Multi_agent_RL_for_Liquidation_Strategy.pdf https://arxiv.org/pdf/1906.11046.pdf
run_paper 1 'Stock Market Prediction using RL' papers/3_12_Stock_Market_Prediction_using_RL.pdf https://arxiv.org/pdf/2101.07107.pdf
# Repos
run_repo 1 OpenBB repos/openbb-finance__openbb https://github.com/OpenBB-finance/OpenBB.git
run_repo 1 qlib repos/microsoft__qlib https://github.com/microsoft/qlib.git
run_repo 1 QUANTAXIS repos/yutiansut__quantaxis https://github.com/yutiansut/QUANTAXIS.git
run_repo 1 awesome-quant repos/wilsonfreitas__awesome-quant https://github.com/wilsonfreitas/awesome-quant.git
run_repo 1 awesome-ai-in-finance repos/georgezouq__awesome-ai-in-finance https://github.com/georgezouq/awesome-ai-in-finance.git
run_repo 1 Stock-Prediction-Models repos/huseinzol05__stock-prediction-models https://github.com/huseinzol05/Stock-Prediction-Models.git
run_repo 1 Stock_Price_Prediction repos/yashveersinghsohi__stock_price_prediction https://github.com/yashveersinghsohi/Stock_Price_Prediction.git
run_repo 1 arch repos/bashtage__arch https://github.com/bashtage/arch.git
run_repo 1 PyPortfolioOpt repos/robertmartin8__pyportfolioopt https://github.com/robertmartin8/PyPortfolioOpt.git
run_repo 1 Riskfolio-Lib repos/dcajasn__riskfolio-lib https://github.com/dcajasn/Riskfolio-Lib.git
run_repo 1 machine-learning-for-trading repos/stefan-jansen__machine-learning-for-trading https://github.com/stefan-jansen/machine-learning-for-trading.git
run_repo 1 awesome-portfolio-management repos/letianwang0__awesome-portfolio-management https://github.com/LetianWang0/awesome-portfolio-management.git
run_repo 1 DeepPortfolio repos/arthurhxz__deepportfolio https://github.com/ArthurHxz/DeepPortfolio.git
run_repo 1 PGPortfolio repos/zhengyaojiang__pgportfolio https://github.com/ZhengyaoJiang/PGPortfolio.git
run_repo 1 cvxpy repos/cvxpy__cvxpy https://github.com/cvxpy/cvxpy.git
run_repo 1 scikit-portfolio repos/scikit-portfolio__scikit-portfolio https://github.com/scikit-portfolio/scikit-portfolio.git
run_repo 1 freqtrade repos/freqtrade__freqtrade https://github.com/freqtrade/freqtrade.git
run_repo 1 hummingbot repos/hummingbot__hummingbot https://github.com/hummingbot/hummingbot.git
run_repo 1 Lean repos/quantconnect__lean https://github.com/QuantConnect/Lean.git
run_repo 1 jesse repos/jesse-ai__jesse https://github.com/jesse-ai/jesse.git
run_repo 1 vnpy repos/vnpy__vnpy https://github.com/vnpy/vnpy.git
run_repo 1 OctoBot repos/drakkar-software__octobot https://github.com/Drakkar-Software/OctoBot.git
run_repo 1 backtrader repos/mementum__backtrader https://github.com/mementum/backtrader.git
run_repo 1 polynote repos/polynote__polynote https://github.com/polynote/polynote.git
run_repo 1 tensortrade repos/tensortrade-org__tensortrade https://github.com/Tensortrade-org/tensortrade.git
run_repo 1 Python-for-Finance-Cookbook repos/packtpublishing__python-for-finance-cookbook https://github.com/PacktPublishing/Python-for-Finance-Cookbook.git
run_repo 1 py4fi2nd repos/yhilpisch__py4fi2nd https://github.com/yhilpisch/py4fi2nd.git
run_repo 1 research_public repos/quantopian__research_public https://github.com/quantopian/research_public.git
run_repo 1 alpaca-trade-api-python repos/alpacahq__alpaca-trade-api-python https://github.com/alpacahq/alpaca-trade-api-python.git
run_repo 1 binance-spot-api-docs repos/binance__binance-spot-api-docs https://github.com/binance/binance-spot-api-docs.git
run_repo 1 shrimpy-python repos/shrimpy-python__shrimpy-python https://github.com/shrimpy-python/shrimpy-python.git
run_repo 1 algo-trading-python repos/bquanttrading__algo-trading-python https://github.com/bquanttrading/algo-trading-python.git
# Packages
run_package 1 yfinance pypi/yfinance yfinance
run_package 1 pandas-datareader pypi/pandas-datareader pandas-datareader
run_package 1 ccxt pypi/ccxt ccxt
run_package 1 finnhub-python pypi/finnhub-python finnhub-python
run_package 1 polygon-api-client pypi/polygon-api-client polygon-api-client
run_package 1 alpha_vantage pypi/alpha_vantage alpha_vantage
run_package 1 investpy pypi/investpy investpy
run_package 1 efinance pypi/efinance efinance
run_package 1 statsmodels pypi/statsmodels statsmodels
run_package 1 pmdarima pypi/pmdarima pmdarima
run_package 1 arch pypi/arch arch
run_package 1 ta pypi/ta ta
run_package 1 PyPortfolioOpt pypi/pyportfolioopt PyPortfolioOpt
run_package 1 Riskfolio-Lib pypi/riskfolio-lib Riskfolio-Lib
run_package 1 ffn pypi/ffn ffn
run_package 1 pyfolio pypi/pyfolio pyfolio
run_package 1 alphalens pypi/alphalens alphalens
run_package 1 empyrical pypi/empyrical empyrical
run_package 1 quantstats pypi/quantstats quantstats
run_package 1 scipy pypi/scipy scipy
run_package 1 cvxpy pypi/cvxpy cvxpy
run_package 1 cvxopt pypi/cvxopt cvxopt
run_package 1 Py-FS pypi/py-fs Py-FS
run_package 1 scikit-portfolio pypi/scikit-portfolio scikit-portfolio
run_package 1 backtrader pypi/backtrader backtrader
run_package 1 zipline-reloaded pypi/zipline-reloaded zipline-reloaded
run_package 1 bt pypi/bt bt
run_package 1 vectorbt pypi/vectorbt vectorbt
# Priority 2
# Papers
run_paper 2 'Attention Is All You Need' papers/1_01_Attention_Is_All_You_Need.pdf https://arxiv.org/pdf/1706.03762.pdf
run_paper 2 'Informer Beyond Efficient Transformer' papers/1_02_Informer_Beyond_Efficient_Transformer.pdf https://arxiv.org/pdf/2012.07436.pdf
run_paper 2 'Autoformer Decomposition Transformers' papers/1_03_Autoformer_Decomposition_Transformers.pdf https://arxiv.org/pdf/2106.13008.pdf
run_paper 2 'TFT Temporal Fusion Transformers' papers/1_04_TFT_Temporal_Fusion_Transformers.pdf https://arxiv.org/pdf/1912.09363.pdf
run_paper 2 'DeepAR Probabilistic Forecasting' papers/1_05_DeepAR_Probabilistic_Forecasting.pdf https://arxiv.org/pdf/1704.04110.pdf
run_paper 2 'NBEATS Neural basis expansion' papers/1_06_NBEATS_Neural_basis_expansion.pdf https://arxiv.org/pdf/1905.10437.pdf
run_paper 2 'PatchTST Time Series Worth 64 Words' papers/1_07_PatchTST_Time_Series_Worth_64_Words.pdf https://arxiv.org/pdf/2211.14730.pdf
run_paper 2 'DLinear Are Transformers Effective for TS' papers/1_08_DLinear_Are_Transformers_Effective_for_TS.pdf https://arxiv.org/pdf/2205.13504.pdf
run_paper 2 'TSMixer All MLP Architecture' papers/1_09_TSMixer_All_MLP_Architecture.pdf https://arxiv.org/pdf/2303.06053.pdf
run_paper 2 'Stock Price Prediction Attention LSTM' papers/1_10_Stock_Price_Prediction_Attention_LSTM.pdf https://arxiv.org/pdf/1811.12587.pdf
run_paper 2 'Advancing Financial Market Prediction using DL' papers/1_11_Advancing_Financial_Market_Prediction_using_DL.pdf https://arxiv.org/pdf/2305.12200.pdf
run_paper 2 'Time Series Representation Learning via Contrasting' papers/1_12_Time_Series_Representation_Learning_via_Contrasting.pdf https://arxiv.org/pdf/2106.14112.pdf
run_paper 2 'Deep Hedging' papers/4_01_Deep_Hedging.pdf https://arxiv.org/pdf/1802.03042.pdf
run_paper 2 'Neural Stochastic Differential Equations' papers/4_02_Neural_Stochastic_Differential_Equations.pdf https://arxiv.org/pdf/1905.09883.pdf
run_paper 2 'Rough Volatility' papers/4_03_Rough_Volatility.pdf https://arxiv.org/pdf/1410.3394.pdf
run_paper 2 'Deep Galerkin Method Solving PDEs' papers/4_04_Deep_Galerkin_Method_Solving_PDEs.pdf https://arxiv.org/pdf/1708.07469.pdf
run_paper 2 'Pricing Options and Computing Implied Vol NN' papers/4_05_Pricing_Options_and_Computing_Implied_Vol_NN.pdf https://arxiv.org/pdf/1901.08943.pdf
run_paper 2 'Arbitrage Free Neural Networks for Pricing' papers/4_06_Arbitrage_Free_Neural_Networks_for_Pricing.pdf https://arxiv.org/pdf/1909.00902.pdf
run_paper 2 'Machine Learning for Pricing American Options' papers/4_07_Machine_Learning_for_Pricing_American_Options.pdf https://arxiv.org/pdf/1904.09732.pdf
run_paper 2 'Rough Paths and Predictive Models' papers/4_08_Rough_Paths_and_Predictive_Models.pdf https://arxiv.org/pdf/1912.01046.pdf
run_paper 2 'A Neural Network Approach to Volatility Surface' papers/4_09_A_Neural_Network_Approach_to_Volatility_Surface.pdf https://arxiv.org/pdf/1805.00611.pdf
run_paper 2 'TimeGAN Time series Generative Adversarial Networks' papers/4_11_TimeGAN_Time_series_Generative_Adversarial_Networks.pdf https://arxiv.org/pdf/1912.12260.pdf
run_paper 2 'GANs for Financial Time Series' papers/4_12_GANs_for_Financial_Time_Series.pdf https://arxiv.org/pdf/1907.06673.pdf
run_paper 2 'FinBERT Financial Sentiment Analysis' papers/5_01_FinBERT_Financial_Sentiment_Analysis.pdf https://arxiv.org/pdf/1908.10063.pdf
run_paper 2 'FinGPT Open Source Financial LLMs' papers/5_02_FinGPT_Open_Source_Financial_LLMs.pdf https://arxiv.org/pdf/2306.06031.pdf
run_paper 2 'Instruct FinGPT Financial Sentiment' papers/5_03_Instruct_FinGPT_Financial_Sentiment.pdf https://arxiv.org/pdf/2306.14041.pdf
run_paper 2 'ChatGPT for Financial Prediction' papers/5_04_ChatGPT_for_Financial_Prediction.pdf https://arxiv.org/pdf/2304.07619.pdf
run_paper 2 'FinEval Financial Knowledge Evaluation LLM' papers/5_06_FinEval_Financial_Knowledge_Evaluation_LLM.pdf https://arxiv.org/pdf/2308.09958.pdf
run_paper 2 'InvestLM Financial Domain LLM' papers/5_07_InvestLM_Financial_Domain_LLM.pdf https://arxiv.org/pdf/2309.13064.pdf
run_paper 2 'Graph Neural Networks for Financial Forecasting' papers/5_08_Graph_Neural_Networks_for_Financial_Forecasting.pdf https://arxiv.org/pdf/2112.15310.pdf
run_paper 2 'Multi Modal Deep Learning Finance' papers/5_09_Multi_Modal_Deep_Learning_Finance.pdf https://arxiv.org/pdf/2001.01234.pdf
run_paper 2 'Attention Networks for Financial Data' papers/5_10_Attention_Networks_for_Financial_Data.pdf https://arxiv.org/pdf/1909.00000.pdf
run_paper 2 'Large Language Models in Finance A Survey' papers/5_11_Large_Language_Models_in_Finance_A_Survey.pdf https://arxiv.org/pdf/2308.05027.pdf
run_paper 2 'BloombergGPT Finance LLM Concept' papers/5_12_BloombergGPT_Finance_LLM_Concept.pdf https://arxiv.org/pdf/2303.17564.pdf
# Repos
run_repo 2 QuantLib repos/lballabio__quantlib https://github.com/lballabio/QuantLib.git
run_repo 2 dx repos/yhilpisch__dx https://github.com/yhilpisch/dx.git
run_repo 2 Financial-Models-Numerical-Methods repos/cantaro86__financial-models-numerical-methods https://github.com/cantaro86/Financial-Models-Numerical-Methods.git
run_repo 2 Stochastic-Calculus repos/darios__stochastic-calculus https://github.com/DarioS/Stochastic-Calculus.git
run_repo 2 Probabilistic-Programming-and-Bayesian-Methods-for-Hackers repos/camdavidsonpilon__probabilistic-programming-and-bayesian-methods-for-hackers https://github.com/camdavidsonpilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers.git
run_repo 2 Kalman-and-Bayesian-Filters-in-Python repos/rlabbe__kalman-and-bayesian-filters-in-python https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.git
run_repo 2 dawp repos/yhilpisch__dawp https://github.com/yhilpisch/dawp.git
run_repo 2 quant-ecosystem repos/sonntagsgesicht__quant-ecosystem https://github.com/sonntagsgesicht/quant-ecosystem.git
run_repo 2 FinRL repos/ai4finance-foundation__finrl https://github.com/AI4Finance-Foundation/FinRL.git
run_repo 2 FinGPT repos/ai4finance-foundation__fingpt https://github.com/AI4Finance-Foundation/FinGPT.git
run_repo 2 ElegantRL repos/ai4finance-foundation__elegantrl https://github.com/AI4Finance-Foundation/ElegantRL.git
run_repo 2 stockpredictionai repos/borisbanushev__stockpredictionai https://github.com/borisbanushev/stockpredictionai.git
run_repo 2 gluon-ts repos/awslabs__gluon-ts https://github.com/awslabs/gluon-ts.git
run_repo 2 darts repos/unit8co__darts https://github.com/unit8co/darts.git
run_repo 2 handson-ml3 repos/ageron__handson-ml3 https://github.com/ageron/handson-ml3.git
run_repo 2 fastbook repos/fastai__fastbook https://github.com/fastai/fastbook.git
run_repo 2 d2l-ko repos/d2l-ai__d2l-ko https://github.com/d2l-ai/d2l-ko.git
run_repo 2 pytorch-beginner repos/l1aoxingyu__pytorch-beginner https://github.com/L1aoXingyu/pytorch-beginner.git
# Packages
run_package 2 QuantLib-Python pypi/quantlib-python QuantLib-Python
run_package 2 sympy pypi/sympy sympy
run_package 2 vollib pypi/vollib vollib
run_package 2 py_vollib pypi/py_vollib py_vollib
run_package 2 option-price pypi/option-price option-price
run_package 2 stochastic pypi/stochastic stochastic
run_package 2 numba pypi/numba numba
run_package 2 jax pypi/jax jax
run_package 2 dask pypi/dask dask
run_package 2 ray pypi/ray ray
run_package 2 gekko pypi/gekko gekko
run_package 2 filterpy pypi/filterpy filterpy
run_package 2 tensorflow pypi/tensorflow tensorflow
run_package 2 'torch torchvision' pypi/torch-torchvision torch torchvision
run_package 2 transformers pypi/transformers transformers
run_package 2 darts pypi/darts darts
run_package 2 sktime pypi/sktime sktime
run_package 2 tslearn pypi/tslearn tslearn
run_package 2 statsforecast pypi/statsforecast statsforecast
run_package 2 neuralforecast pypi/neuralforecast neuralforecast
run_package 2 tsai pypi/tsai tsai
# Priority 3
# Repos
run_repo 3 awesome-quant-finance repos/chatchavan__awesome-quant-finance https://github.com/chatchavan/awesome-quant-finance.git
# Packages
run_package 3 scikit-learn pypi/scikit-learn scikit-learn

summary_path="$ARTIFACT_ROOT/run_summary.json"
cat > "$summary_path" <<EOF
{
  "downloaded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "priority_max": $PRIORITY_MAX,
  "kind_filter": "$KIND_FILTER",
  "mode": "$MODE",
  "success_count": $success_count,
  "skip_count": $skip_count,
  "fail_count": $fail_count,
  "paper_count": $paper_count,
  "repo_count": $repo_count,
  "package_count": $package_count
}
EOF
log "summary written to $summary_path"
log "success=$success_count skip=$skip_count fail=$fail_count"
