# Capstone Project Brief: Regime Detection and RL-based ETF Allocation

## 1. Introduction & Objectives

This capstone project integrates **macroeconomic regime detection** with **reinforcement learning (RL) for portfolio allocation**. Students will explore how identifying historical analogs (“historical twins”) in macro-financial data can inform investment strategy, and how an RL agent can use these regime signals to dynamically allocate across assets. The project is academically relevant – extending literature on regime-based investing – and practically valuable for global macro portfolio management.

**Learning Outcomes:** By the end of this project, you will be able to:

* Construct a **regime-detection framework** using macroeconomic indicators (e.g. yield curve, inflation, credit spreads) to classify market regimes.
* Implement a **nearest-neighbor similarity** approach to find historical analogs of current conditions (“historical twins”) and derive a regime **signal**.
* Design and train a **reinforcement learning agent** (using PyTorch-based tools) that incorporates the regime signal in its state to allocate a portfolio of Exchange-Traded Funds (ETFs).
* Evaluate performance with appropriate metrics (returns, Sharpe ratio, drawdowns) and conduct robustness tests (including stress periods like 2008 and 2020).

**Project Deliverables:** The project has two key deliverables:

1. **Regime-Detection Framework:** Recreate the methodology from *“Finding Your Historical Twins”* – using macro-financial data transformed into z-scores and a nearest-neighbors distance metric to identify similar historical regimes. Produce a clearly defined **regime signal** (e.g. a label or index indicating the current macro regime) based on this analysis.
2. **RL-based ETF Allocation Model:** Develop a reinforcement learning model that uses the above regime signal as part of its state to inform asset allocation. The agent will allocate across six ETFs (ACWI, EEM, BNDX, HYLB, DBC, VNQI) with the goal of maximizing risk-adjusted returns. This deliverable includes the RL environment design, training of the agent, and performance evaluation against benchmarks.

These deliverables mimic real-world quantitative finance workflows: first understanding the market context (regime detection) and then making tactical allocation decisions (via RL) informed by that context. By clearly separating these two components, the project ensures you gain deep insights into each before ultimately combining them in a cohesive strategy.

## 2. Background & Literature

**Regime Detection & Historical Analogs:** The concept of macro-financial “regimes” refers to periods of persistent economic conditions (e.g. growth vs. recession, inflationary vs. deflationary). Identifying the current regime can improve forecasting and asset allocation. Traditional approaches like Markov switching models (e.g. Hamilton 1989) infer regimes statistically, while recent machine learning approaches use clustering to find regimes from data. The *“Finding Your Historical Twins”* approach (popularized by research from Verdad and others) is an intuitive, data-driven method: convert each period’s macro indicators into a standardized vector, then find **nearest historical neighbors** of the current vector to gauge similarity. If current conditions closely resemble past episodes, those “twins” can offer insight into what might come next.

A recent example by Verdad (*Analogous Market Moments, 2024*) constructed a macro similarity measure using indicators like high-yield credit spreads, inflation, stock–bond correlation, and the yield curve. Each month’s data becomes a point in multi-dimensional space, and **Euclidean distance** between points measures how alike two periods are. Clustering similar points allowed them to define distinct economic regimes (e.g. “Growth”, “Inflation”, “Precarious”, “Crisis”). This shows how nearest-neighbor analysis and clustering can identify meaningful regimes. In our project, we will build on these ideas, using z-scores of macro variables and distance metrics to categorize the current environment and produce a regime label or score.

**Reinforcement Learning in Portfolio Allocation:** In parallel, finance researchers and practitioners have begun applying deep reinforcement learning to portfolio management. RL is well-suited for sequential decision problems like trading, where an agent continuously reallocates assets to optimize long-term returns. Recent studies demonstrate that RL-based portfolios can outperform traditional strategies like mean-variance optimization, achieving better risk-adjusted returns. Notable applications include RL for stock trading and multi-asset allocation (e.g. using Deep Q-Networks, Policy Gradient methods, or actor-critic algorithms). For example, Jiang et al. (2017) showed a deep RL agent can learn trading strategies that beat baselines, and more recent work compares RL strategies favorably against Markowitz’s mean-variance portfolios across various metrics (return, Sharpe, drawdown).

However, a challenge in financial RL is incorporating **market context**. Many early approaches relied only on price technical indicators as state, which may lead to myopic behavior. Including macroeconomic regime information can potentially improve the policy by guiding the agent with a high-level market phase signal. Oliveira et al. (2025) highlight the value of regime-aware allocation, integrating macro regime classification with portfolio optimization to outperform static and random baselines. Our project will contribute to this emerging area by explicitly providing the RL agent with a regime signal derived from the “historical twin” framework. This hybrid approach—marrying economic intuition (regimes) with modern AI (RL)—is at the cutting edge of quantitative finance research.

**Related Literature:** To ground your work, consult resources like:

* *Ang & Bekaert (2004)* on how regime shifts affect asset returns (early regime-switching models).
* *Verdad Research (2024)* on macro analogs and clustering regimes (accessible summary by Swedroe, 2024).
* *Oliveira et al. (2025)* on using FRED-MD macro data for regime-based tactical allocation.
* *FinRL (2020)* – an open-source library demonstrating deep RL for trading.
* *Moody & Saffell (1999)* – an early application of RL (Neuro-Dynamic Programming) for financial portfolios.
* *Jiang et al. (2017)* – deep reinforcement learning for portfolio management using deep Q-learning.
  These works provide background on why regime detection is useful and how RL can be applied in finance, helping to justify design choices in this project.

## 3. Data Requirements & Sources

**Macro-Financial Time Series:** We will use a set of macroeconomic and financial indicators to characterize market regimes. All data will be from free public sources such as the St. Louis Fed’s FRED database and Prof. Robert Shiller’s dataset. The exact series and their identifiers (FRED mnemonics or other sources) include:

* **Interest Rates & Yield Curve:** e.g. 10-Year Treasury Yield and 3-Month T-Bill rate. From these, compute the *yield curve slope* (10y minus 3m, FRED series e.g. `DGS10` and `DTB3` or the spread `T10Y3M`). This slope indicates economic outlook (inversions often precede recessions).
* **Inflation:** e.g. annual Consumer Price Index inflation rate (FRED CPIAUCSL for CPI, then compute year-over-year % change). Captures price stability vs. inflationary pressures.
* **Credit Spread:** e.g. the high-yield corporate bond spread over Treasuries (FRED series like **BofA US High Yield Index Option-Adjusted Spread**, ticker `BAMLH0A0HYM2` or similar). Reflects risk appetite and financial stress.
* **Stock-Bond Correlation:** a rolling correlation between equities and bonds (could be computed from historical S\&P 500 and 10-year Treasury returns). This is a more advanced indicator; alternatively, use a simpler risk metric like equity market volatility.
* **Equity Market Volatility:** e.g. the VIX index (CBOE Volatility Index for S\&P 500) as a proxy for risk aversion. Available from Yahoo Finance (`^VIX`) – if using, align its frequency with other data (monthly average or latest value each month).
* **Economic Growth:** e.g. unemployment rate (FRED `UNRATE`) or GDP growth (quarterly real GDP growth, FRED `A191RL1Q225SBEA`). If using GDP (quarterly), we can interpolate or hold values constant for intervening months.
* **Additional (optional):** Equity valuation metric like Shiller CAPE ratio (from Shiller’s data), or money supply growth, etc., if desired to enrich the macro state.

All series will be gathered at a **monthly frequency** (most macro data are monthly or quarterly; quarterly series can be converted to monthly by carrying forward values). Monthly frequency balances detail and stability for regime identification. We will restrict to a broad time horizon (e.g. 1960–present) to have plenty of historical samples for the nearest-neighbor analysis. For consistency, convert each series to a **z-score** (standardized value) over the sample period or a rolling window – details in Section 4.

**Asset Price Data (ETFs):** The six ETFs for the allocation model are:

* **ACWI** – iShares MSCI ACWI, a global equity index fund (developed + emerging markets).
* **EEM** – iShares MSCI Emerging Markets ETF (emerging markets equities).
* **BNDX** – Vanguard Total International Bond ETF (investment-grade bonds outside the US, USD-hedged).
* **HYLB** – Xtrackers USD High Yield Corporate Bond ETF (USD high-yield bonds).
* **DBC** – Invesco DB Commodity Index Tracking Fund (broad commodities futures).
* **VNQI** – Vanguard Global ex-U.S. Real Estate ETF (international real estate equity).

We will use **Yahoo Finance** (via the `yfinance` Python API or similar) to download historical prices for these tickers. Daily price data is available for each; however, since some funds started after 2010, our price history likely runs from \~2011 to present for all six. We’ll use **adjusted close prices** to account for dividends. Depending on how we structure the RL environment, we may use **daily returns** or **monthly returns** for these assets:

* *For RL training:* Using daily data provides more time steps (improving training sample efficiency), but macro signals update monthly. We can handle this by assuming the macro regime indicator remains constant within each month.
* *For backtesting the baseline strategy:* monthly S\&P 500 returns (or ACWI as a global proxy) will be used, possibly supplemented by the long-run **Shiller S\&P 500** data for extended history. Shiller’s dataset provides monthly S\&P 500 levels, dividends, and earnings back to 1871, which is useful for regime studies spanning many decades.

**Data Cleaning & Preparation:** Steps to prepare the data include:

* **Alignment:** Merge all macro series by date (inner join on month), dropping any month that is missing data for key series. Ensure the time index is uniform (e.g., end-of-month).
* **Handling Missing Values:** Some series might have occasional missing entries or start later. For missing months in a series, we can forward-fill if gaps are small, or drop those periods. For series with different start dates, we may restrict the analysis to the common period or use proxy data to backfill (e.g., before HYLB’s inception, a high-yield index could proxy).
* **Frequency Conversion:** Convert daily ETF prices to monthly (e.g., use last trading day of each month prices to compute monthly returns) if doing monthly regime switching. Alternatively, keep daily for RL, but then need to up-sample macro data (assign each trading day of a month the same macro values).
* **Return Calculation:** Compute log returns or percentage returns for the ETFs. For daily data, use daily log returns; for monthly, monthly log or simple returns. Also compute returns for the S\&P 500 (or ACWI) for use in the baseline strategy evaluation.
* **Standardization:** Create z-scores for each macro feature. Commonly, we subtract the mean and divide by the standard deviation of each series. We can use the full history to compute mean and std, or a rolling window (to avoid look-ahead bias in real-time calculations – perhaps use an expanding window up to the current time when computing historical z-scores). These z-scores form the feature vector for regime detection.

All data sources are public: FRED (no-cost API for macro series), Yahoo Finance (free daily data), and Shiller’s data (downloadable Excel/CSV). This ensures the project is reproducible without proprietary data. In the code, we will likely use `pandas_datareader` or `fredapi` for FRED, and `yfinance` for ETF prices.

## 4. Feature Engineering & Regime Signal

In this section, we transform raw data into features for our regime model and derive the **regime signal** that will later feed the RL agent. The core idea is to represent each time period by a vector of standardized macro-financial indicators, and then define regimes based on similarity in this feature space.

**Macro Z-Score Computation:** For each macro series (inflation, yield curve, etc.), compute a z-score to put variables on a comparable scale. Below is pseudocode illustrating this process:

```python
import pandas as pd

# Assume macro_df is a DataFrame with columns ['Date', 'Inflation', 'YieldCurve', 'CreditSpr', ...]
macro_df.set_index('Date', inplace=True)

# Compute z-scores for each column
zscore_df = macro_df.copy()
for col in macro_df.columns:
    mu = macro_df[col].mean()
    sigma = macro_df[col].std()
    zscore_df[col] = (macro_df[col] - mu) / sigma
```

This yields a DataFrame `zscore_df` where each row is a date (month) and columns are standardized indicators. Each row can be viewed as a point in an *n*-dimensional feature space (n = number of indicators). For example, a row might be `(Inflation_z, YieldCurve_z, CreditSpr_z, ...)`. Periods with similar economic conditions will lie near each other in this space.

**Nearest-Neighbors Similarity:** To find “historical twins” for a given period, we calculate distances between the period’s feature vector and all others in history. We will primarily use Euclidean distance (as a straight-line measure in multi-dimensional space) as our similarity metric:

```python
import numpy as np

# Get feature matrix and dates
X = zscore_df.values  # shape: [T, n_features]
dates = zscore_df.index

# Example: find the 5 nearest neighbors for the last date in the dataset
target = X[-1]  # feature vector for latest period
distances = np.linalg.norm(X - target, axis=1)  # Euclidean distance to all periods
nearest_idx = np.argsort(distances)[1:6]  # indices of 5 closest (excluding itself at index -1)
nearest_dates = dates[nearest_idx]
print("Nearest historical analogs for {}:".format(dates[-1]), nearest_dates.tolist())
```

In practice, we will apply this for each time period (especially for out-of-sample testing) to identify its nearest neighbors. A smaller distance means a more similar macro environment. This enables us to define a **Similarity Score** = *negative distance* (so higher score = more similar to past) or simply use the distance itself as an “unprecedentedness” measure (high distance = regime is unlike anything seen recently).

**Defining Regimes:** There are multiple ways to turn these similarities into discrete regimes:

* **Clustering:** We can cluster the feature vectors (using K-means, Gaussian Mixture Model, etc.) into a few groups, which become labeled regimes. For example, a GMM with 4 clusters might produce regimes akin to the Growth/Inflation/Precarious/Crisis categories identified by Verdad. Each month then gets a regime label (1, 2, 3, or 4). If we take this route, our regime signal for the RL agent could be a one-hot encoded vector or an integer {1,…,4}.
* **Threshold on Distance:** Alternatively, we can use the nearest-neighbor distance directly. For instance, define two regimes: “Similar” vs. “Dissimilar” based on whether the distance to the closest historical point is below or above some threshold. A more refined approach could define *k*-nearest neighbors and look at average distance: if the current state is well within the cloud of historical points (high similarity), call it a “familiar regime”; if it’s an outlier (far from all historical points), call it an “unprecedented regime”. This binary regime indicator could be part of the state (e.g. 0 = normal, 1 = unusual).

For better richness, we will likely adopt a clustering approach to get multiple regimes. For example, using K = 3 or 4 clusters on the historical z-score data:

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
labels = gmm.fit_predict(X)  # assign each time point to one of 4 clusters
zscore_df['RegimeLabel'] = labels  # add regime label to the dataframe
```

This unsupervised learning will group periods with similar macro conditions. We can then interpret the clusters (e.g., one cluster might correspond to high-inflation periods, another to recessions, etc., based on the feature means). The “regime signal” to be used in the RL model could be either this categorical label or, if we prefer continuous features, we might incorporate the actual macro z-scores and/or the nearest-neighbor distance as part of the state.

**Sanity Check & Visualization:** It’s useful to validate the regime classification. We can:

* Plot time series of the regime label over history to see if it aligns with known periods (e.g., cluster “crisis” lights up during 1973-4, 2008, 2020, etc. – as expected from).
* Examine the average macro values in each regime (to ensure they make intuitive sense, similar to the regime definitions in literature).
* These steps help confirm our regime detection is capturing meaningful structure.

By the end of this stage, we will have a function or module that, given current macro data, outputs a **regime indicator**. This could be as simple as `get_regime_state(current_macro_vector) -> regime_id or (distances)` that we will integrate into the RL environment as part of the observation.

## 5. Baseline Backtest & Validation

Before delving into RL, we will develop a baseline trading strategy to **validate the regime signal** and provide a performance benchmark. The proposed baseline is a *long/short strategy on the S\&P 500 (or ACWI equity index) based on similarity*. In essence: **go long equities when the current macro state is similar to historically strong periods, and short (or underweight) equities when the state is dissimilar (unprecedented or similar to weak periods).**

**Strategy Rules:** Using the nearest-neighbor outputs from Section 4, we define rules such as:

* Compute the distance from the current month’s macro vector to its nearest historical neighbor. Let *D(t)* be this minimum distance for month *t*.

* Determine a threshold *D*<sub>*crit*</sub> (e.g., the median or 75th percentile of all such distances in the training sample). If *D(t)* is **below** *D*<sub>*crit*</sub> (meaning the state is *similar* to past observations), we interpret the regime as “familiar/benign” and take a **long** position in S\&P 500 for the next period. If *D(t)* is **above** *D*<sub>*crit*</sub> (“dissimilar” or extreme macro state), take a **short** position (or move to cash) on the S\&P 500 for the next period.

  * *Rationale:* When the macro backdrop is something we’ve seen before and presumably have data on how markets reacted, there is more confidence to be invested (especially if many of those analogs yielded positive returns). Conversely, if the environment is unlike anything in the dataset (e.g., 2008 credit freeze, or unprecedented inflation surge), it may pay to be defensive (short or out of equities), expecting higher risk or volatility.

* A refinement: use **k-nearest neighbors** and look at their average subsequent S\&P 500 returns. For instance, find the 5 most similar historical months to the current month. If the average 12-month return of the S\&P 500 after those 5 analog months was positive, go long; if negative, go short. This leverages actual forward outcomes from analogous periods rather than a fixed threshold. It’s effectively a **nearest-neighbors prediction** approach.

We will backtest this strategy over a long history (several decades) using monthly data:

* Position can be represented as +100% long S\&P, -100% short S\&P, or 0% (if we allow a neutral stance instead of shorting).
* Use **Shiller’s S\&P 500 data** (with price and dividends) for a continuous long-term series, or use ACWI from 1988 onward (MSCI ACWI index history) if global context is desired.
* Each month, decide the position based on the prior month’s macro data and regime analysis. Track the portfolio value assuming no transaction costs for simplicity (though we only trade when signal flips, which should be infrequent if regimes persist).

**Performance Metrics:** After running the backtest, compute key metrics:

* **CAGR (annualized return)** and **volatility**.
* **Sharpe Ratio:** risk-adjusted return using (return – risk-free) / volatility.
* **Maximum Drawdown:** the largest peak-to-trough loss, to assess downside risk.
* **Calmar or Sortino Ratio:** additional risk-adjusted measures (Calmar = CAGR/MaxDrawdown).
* **Hit Rate:** percentage of months with positive return, and maybe compare average returns in long vs short months.
* We will compare these metrics against simple benchmarks: always long S\&P 500 (buy-and-hold), and maybe a 50/50 long-short or always flat strategy.

**Robustness Checks:** To ensure the strategy is not overfitted:

* **Vary the Similarity Threshold or k:** Test different values of *k* in the k-NN analog method (e.g., using 3 or 10 nearest neighbors) and different threshold criteria, to see if performance is consistent.
* **Changing Macro Features:** Try excluding one of the macro features or using an alternate indicator (e.g., use unemployment vs GDP growth) and observe if the strategy’s qualitative performance holds. Consistency would indicate the regime signal is robust.
* **Out-of-Sample Testing:** If possible, calibrate the threshold or cluster regimes on data up to year X, then test the long/short strategy from X+1 onward. This simulates how the strategy would perform in “real-time” without peeking ahead.
* **Turnover and Transaction Costs:** While our base test ignores trading costs (assuming monthly reallocations have minimal cost), we can check how many trades occurred and ensure turnover isn’t extreme. If it is, incorporate a small transaction cost and see if the strategy still adds value.

This baseline not only provides a yardstick for the RL agent’s performance, but also deepens understanding of the regime signal’s efficacy. For example, if we find the long-similar/short-dissimilar strategy yields a higher Sharpe ratio than buy-and-hold and we observe sensible timing (e.g., it went short during 2008, avoiding part of the crash), that validates the approach. Any insights from the backtest (such as shortcomings or lags in the signal) can inform how we incorporate the regime indicator into the RL model.

## 6. RL Environment Design

With a validated regime signal, we proceed to design the **reinforcement learning environment** for the ETF allocation task. We will construct a custom environment following the OpenAI Gym interface (with `reset()` and `step()` methods) so it can interact with standard RL algorithms. Key design elements include:

* **State (Observation Space):** This will include both macro-regime information and possibly recent portfolio information. At a minimum, the state at time *t* will contain:

  * The latest **macro feature z-scores** (as defined in section 4) or the **regime label**. For example, we might include the regime as a one-hot vector of length 4 (if using 4 regimes) or include a continuous “similarity score”. This gives the agent awareness of the broad economic regime.
  * Recent **asset market data**, if needed. We could include, say, the last one-month return of each of the 6 ETFs or some technical indicators, to allow the agent to consider momentum or short-term signals. However, this complicates state and isn’t strictly required by the prompt – the focus is on the regime signal. To keep things simple and emphasize macro context, the primary state will be the macro regime indicators.
  * We do not necessarily include the agent’s current holdings in state, because many RL portfolio implementations assume full observability of the environment (which is the market). However, including last action or current weights can sometimes help the agent learn about transaction costs or momentum in allocation. Since we’ll likely allow frictionless trading, we can omit portfolio composition from state.

  *State representation example:* a vector `[Inflation_z, YieldCurve_z, CreditSpr_z, Volatility_z, RegimeOneHot_1, RegimeOneHot_2, RegimeOneHot_3, RegimeOneHot_4]`. This could be e.g. 4 continuous values + 4 binary values = 8-dim state. If using continuous macro features directly, that might be enough and a separate discrete regime label isn’t needed.

* **Action Space:** The action is the portfolio weight allocation across the 6 ETFs. We define this as a **continuous action space** in ℝ^6. An action is a 6-dimensional vector `a = (w_ACWI, w_EEM, w_BNDX, w_HYLB, w_DBC, w_VNQI)` representing the portfolio weights for each asset *at the start of the next period*. We enforce the constraints that weights are ≥ 0 (no shorting, for simplicity) and sum to 1 (fully invested, no leverage or cash). To handle this in an RL algorithm:

  * One approach is to let the agent output an unconstrained vector and then normalize it to sum to 1. For example, the environment’s `step(action)` can internally do: `weights = softmax(action_raw)` or `action_normalized = action_raw / sum(action_raw)` (after ensuring non-negativity).
  * Another approach is to use a custom action space like `gym.spaces.Box(low=0, high=1, shape=(6,))` and then project to the simplex (e.g. by normalization). The agent (especially continuous-action methods like DDPG, PPO) can naturally produce outputs in \[0,1].

  We will likely implement the action as a float vector and perform normalization inside the environment. This ensures the portfolio always sums to 100%. We assume monthly rebalancing (if using monthly steps) or daily rebalancing (if using daily steps). Frequent rebalancing is allowed since we’re ignoring transaction costs, but the agent may learn to not churn too much if it doesn’t improve reward.

* **Reward Function:** We choose a reward that aligns with the investor’s goal of **high risk-adjusted returns with low drawdowns**. A well-considered reward at each time step could be:

  * **Excess Return** minus a **Penalty for Volatility/Drawdown**. For example, `reward_t = R_portfolio(t) - λ * (Volatility_t)` for some λ. But volatility is usually measured over a period, not single step, so we might approximate it.
  * **Sharpe Ratio as episodic reward:** At episode end, compute portfolio Sharpe ratio = (mean\_ret / vol) and use that as the total reward. However, most RL algorithms need stepwise rewards to propagate credit. So instead, we can shape the reward to encourage Sharpe-like behavior:

    * Use **return** as the base reward each step, but also include a penalty whenever the portfolio hits a new drawdown. For instance, keep track of the portfolio’s running maximum value; if the current value drops below the max by X%, give a negative bonus proportional to that drop. This pushes the agent to avoid large drawdowns.
    * Or use a rolling Sharpe: e.g., `reward_t = R_t - 0.5 * (R_t / σ_last_k)^2` or similar, though this gets tricky.
  * A simpler approach: reward = next period portfolio return **minus** a penalty for large negative returns. For example:

    ```python
    if portfolio_return < 0:
         reward = portfolio_return - 0.5 * (portfolio_return**2)
    else:
         reward = portfolio_return
    ```

    This way, moderate negative returns incur small penalty, but large negative returns (drawdowns) incur larger penalty (because square term). This is a heuristic that penalizes downside volatility more than upside. The agent thus indirectly maximizes a Sortino-like ratio.

  We will likely test a couple of reward formulations. The key is to incorporate **risk-adjustment** so the agent doesn’t just chase raw return with high volatility. For interpretability, we might stick with **“Sharpe ratio with drawdown penalty”** as described. One concrete instantiation:

  ```
  reward_t = R_t - (0.1 * max(0, drawdown_t))
  ```

  where `drawdown_t` is the current drawdown from peak in percentage and 0.1 is a chosen penalty factor. This means if the portfolio hits, say, a 5% drawdown, we subtract 0.005 from reward that step.

* **Observation and Transition Dynamics:**

  * If using monthly steps: At the end of each month, the environment observes the macro data for that month (which gives the regime), and the agent chooses weights for the next month. The environment then applies those weights to the returns of the next month to compute reward.
  * If using daily steps: We will incorporate intra-month steps. One idea: macro data updates on the first trading day of each month, but stays constant through the month. So for days within a month, the macro state part of observation remains the same. The agent could potentially adjust daily, but that might be unrealistic (we expect monthly rebalancing). To simplify, we might actually stick to monthly time steps in the environment.
  * **Episode Termination:** We can define an episode as a finite period of time in the historical data. For training, an episode could span e.g. 5 years of data (60 steps if monthly). After 60 steps, the episode ends and environment resets to a new start year. Alternatively, use the entire history as one episode and just let the algorithm iterate through it repeatedly. But multiple shorter episodes can improve learning by providing more start points.

* **Initial State and Reset:** The `reset()` function of the environment will initialize the portfolio (e.g., start with an even weight or a given default allocation) at a random start date in the historical data (to provide variation for training episodes). It will return the initial state observation (macro features at that start date, regime label, etc.). For deterministic evaluation, we can also set `reset(start_date=...)` to a fixed date.

* **Integration of Regime in State:** For the regime to influence the agent, it must be part of the observation fed into the policy network. We will ensure the state vector explicitly contains either the discrete regime (as one-hot) or the underlying macro features that define the regime. This way, the agent’s neural network can learn different behaviors for different regimes (effectively conditioning the policy on the regime). For example, it might learn to allocate more to equities (ACWI, EEM) in a “Growth” regime, but shift to bonds or commodities in a “Crisis” regime, aligning with historical performance in those regimes.

In summary, the environment will simulate an **asset allocation process** where each step the agent sees the current economic conditions (state), adjusts its portfolio (action), then experiences the portfolio’s return (reward). This design will be implemented in Python (using `gymnasium` interface for compatibility with Stable-Baselines3 algorithms).

## 7. Agent Selection & Training Plan

We will employ state-of-the-art deep RL algorithms to train the portfolio agent. Given the continuous action space and the need for stability, we consider the following algorithms (using PyTorch implementations via Stable-Baselines3 or similar libraries):

* **Proximal Policy Optimization (PPO):** PPO is a policy-gradient method known for stable learning via clipped surrogate objective. It can handle continuous actions by outputting a Gaussian distribution for each action dimension (or using a Beta distribution for bounded actions). PPO is a good starting point due to its robustness and ease of use. It’s on-policy, meaning it collects fresh experience data each training iteration, which can be fine for our environment.
* **Deep Deterministic Policy Gradient (DDPG):** An off-policy actor-critic algorithm suited for continuous control. DDPG learns a deterministic policy with a Q-value critic. It can be powerful but sometimes less stable; however, with proper tuning or enhancements like TD3 (Twin Delayed DDPG) or SAC (Soft Actor-Critic), it can perform well. DDPG (or TD3) might be considered if PPO struggles to converge to good allocations, as off-policy methods can better reuse experience and handle continuous action fine-grained adjustments.
* **Stable-Baselines3 (SB3) Library:** We plan to use SB3’s implementations of PPO and DDPG (as well as variants like A2C, SAC if needed), since SB3 provides reliable, well-tested code in PyTorch. This saves us from writing algorithms from scratch and allows focusing on environment and tuning.

**Model Architecture:** The agent’s policy network will take the state vector (macro features/regime + possibly some price info) as input. We will configure a neural network with 2-3 hidden layers (for example, 64 or 128 neurons each, with ReLU activations). The output layer depends on algorithm:

* For PPO (Gaussian policy), output a mean and std for each of the 6 actions (so 6\*2 outputs if independent gaussians). SB3 handles this automatically if action space is Box(6).
* For DDPG, output 6 continuous values (which we’ll normalize to weights).
  Given the state size is small (under 10 features), a moderate network suffices.

**Training Process:**

* **Training Horizon:** Decide how many timesteps constitute training. If using monthly data for \~14 years (2011-2025, \~168 months), one epoch through data is 168 steps. We can train over many epochs (many episodes cycling through data). If we use multiple starting points, the total timesteps = episodes \* steps per episode. We might aim for, say, 10,000 training steps as a baseline (which could be \~60 episodes of 168 steps each).
* **Episode Sampling:** During training, we can randomly choose start years for each episode to augment experience. For example, episode 1: Jan 2011–Dec 2015, episode 2: Jul 2012–Jun 2017, etc. This gives the agent exposure to different market conditions in each episode and avoids always starting in the same regime.
* **Reward Scaling:** Ensure the reward magnitudes are neither too small nor too large for stable learning. If using percentage returns (\~0.01 typical) as reward, PPO/DDPG can handle that. If we incorporate Sharpe or drawdown penalties, we might need to scale appropriately (e.g., Sharpe ratios \~0.5 to 1.5, we can use them directly or multiply by 100 if needed to be around 1–10 range).
* **Hyperparameter Tuning:** We will experiment with key hyperparams:

  * *Learning rate:* Start with default (e.g., 3e-4 for PPO) and adjust if learning is too slow/unstable.
  * *Batch size (for PPO)*: PPO uses batches of time steps. Given our dataset is not huge, we might set batch size equal to the episode length or a fraction of it.
  * *Discount factor (γ):* We can keep γ \~ 0.99 to emphasize long-term rewards. Even though episodes are relatively short, a high γ encourages the agent to consider the effect of allocation on eventual outcomes (like end-of-episode performance).
  * *Exploration noise (for DDPG):* Ensure sufficient exploration in continuous actions, possibly by adding Ornstein-Uhlenbeck noise (SB3 does this by default for DDPG).
  * *Seed and Trials:* Try multiple random seeds to ensure results are consistent.

**Software and Tools:** We will use Python with **PyTorch** and **Stable-Baselines3** for the RL algorithms. The environment will likely be a custom class inheriting from `gym.Env`. We can use Jupyter notebooks or scripts to run training. We will monitor training progress via:

* Logging episode rewards (e.g., average reward per episode).
* Perhaps using TensorBoard (SB3 can output logs) to watch for convergence.
* Saving intermediate models in `models/` directory to evaluate and possibly do early stopping if one performs particularly well on validation data.

**Training/Validation Split:** Although RL isn’t trained in a supervised sense, we can still set aside some “out-of-sample” period for testing. For example, use data up to 2020 for training, then reserve 2021–2025 for evaluating the final policy (the agent won’t train on those). This way, we ensure the agent’s performance claims hold on unseen data. We’ll incorporate this in the training loop by simply not starting any training episodes in the test period, and only using the test period in a final simulation.

**Algorithm Choice Rationale:** We expect PPO to be a solid choice to start with, due to its stability with a variety of tasks. If the action space normalization or the reward design poses challenges, we may try DDPG/TD3 since they directly optimize continuous actions and might find extreme allocations that PPO’s exploratory nature might not. We will document which algorithm and parameters ended up performing best. The goal isn’t to exhaustively compare every RL algorithm, but to pick one or two that are reasonable and ensure the agent learns a sensible allocation strategy that uses the regime signal effectively.

## 8. Evaluation & Sensitivity Analysis

After training the RL agent, we will rigorously evaluate its performance relative to benchmarks and test how sensitive the results are to various factors. The evaluation has multiple facets:

**Benchmark Comparison:** We will compare the RL-driven portfolio to:

* **Buy-and-Hold Benchmark:** An equally weighted static portfolio of the six ETFs (rebalanced annually or not at all). This represents a naive diversified strategy.
* **60/40 or Risk-Parity Benchmark:** Perhaps a traditional portfolio like 60% ACWI (global equity) / 40% BNDX (global bonds) rebalanced periodically, or a risk-parity allocation (which would put more weight on bonds/commodities to equalize risk). These give context on risk-adjusted returns.
* **Regime-Naive RL:** As an ablation test, train a second RL agent *without the regime input*. For instance, give it just recent asset returns or basic indicators as state, but no macro features. Compare its performance to our regime-aware agent. This will highlight the value added by the regime signal in the decision process.
* **Baseline Strategy:** The long-similar/short-dissimilar S\&P strategy from Section 5, if it can be extended to multi-asset (maybe it only guides equity exposure). We can compare the RL agent’s equity allocation behavior to this heuristic strategy qualitatively.

Key metrics for comparison:

* Annualized **Return**, **Volatility**, **Sharpe Ratio**.
* **Max Drawdown** and **Calmar Ratio**.
* **Sortino Ratio** (with MAR = 0).
* **Portfolio turnover** (how much trading is done; lower is generally better if too high costs would eat profits).
* These will be computed on the test dataset (e.g., 2021–2025) or over the full run if we use a rolling training scheme.

We expect (hope) the RL agent outperforms equal-weight and other static allocations in Sharpe ratio, and matches or improves on the downside risk metrics due to its drawdown-aware reward. If the agent significantly beats benchmarks, that validates the approach; if not, it may indicate either the regime signal or the reward needed adjustments.

**With vs. Without Regime Signal:** A core analysis is to isolate the impact of the regime feature:

* If the regime-aware agent consistently allocates differently in different regimes (e.g., it learns to shift to bonds in the “Crisis” regime and to equities in “Growth” regime), and this improves performance, it shows the regime signal was useful. We will present examples of the agent’s policy: for instance, *policy heatmaps* showing average weight in each asset conditional on regime. This can be done by feeding the trained policy a set of representative states (e.g., typical state from each regime) and recording the action.
* We will also look at the performance of the agent when the regime input is scrambled or fixed. For example, run the trained agent on the test period but feed it a dummy constant regime (so it can’t see true regimes) – performance should drop if the agent was truly leveraging that info.

**Volatility Window Sensitivity:** The term “volatility windows” likely refers to how we calculate volatility for Sharpe or in the reward. We will test different window lengths for any moving volatility estimate used:

* If we used a certain period for volatility (say 6-month rolling vol for reward shaping), try 3-month or 12-month and see if agent behavior changes.
* We can also examine the agent’s allocation sensitivity to market volatility regimes (e.g., perhaps include VIX in state; if not, we can still observe if it de-leverages in high VIX periods implicitly).

**Stress Test Scenarios:** Finally, we will conduct targeted evaluation on known stress periods:

* **2008 Global Financial Crisis:** Although our ETF data might not go back to 2008 for all assets, we can simulate the agent’s behavior if such a scenario occurred. One approach: take the macro data from 2008 (which we have), feed it through our regime model to get regime state (likely “Crisis”), and simulate what the agent *would have done* (maybe using proxies for assets since some ETFs didn’t exist, e.g. use S\&P for ACWI, EMB for EM bonds etc.). We expect a good agent to rotate into safer assets (bonds, or even go to minimal equity) ahead of or during the drawdown.
* **2020 COVID Crash:** March 2020 saw an extreme shock. Our data covers this. We can replay the period 2019–2020 to see if the agent recognized rising risk (maybe via credit spreads spiking, yield curve changes) and adjusted. Perhaps the regime model would label early 2020 as “Crisis” and the agent should reduce equity and increase bonds/commodities. We will chart the allocations over 2019–2021 to inspect this.
* If the agent navigated these well (better than a static strategy), it’s evidence of success. If not, we’ll analyze why – maybe the regime signal lagged (in COVID, things unfolded very rapidly).

**Parameter Sensitivity:** We will note how sensitive the results are to:

* **Reward Penalty Coefficient:** e.g., if we change the drawdown penalty λ, does the agent take on significantly different risk levels?
* **Regime Granularity:** If we used 4 regimes, what if we try 3 or 5? The cluster structure might change. We can try an alternative regime classification and retrain the agent to see if performance is similar, indicating the approach is robust to the exact regime definitions.
* **Length of Training Data:** If we train the agent only on a shorter history vs the full history, does it generalize? This could test overfitting: if an agent trained on 2011–2018 performs poorly on 2019–2025 compared to one trained on 2011–2020, it suggests more training data helped it generalize to unforeseen regimes (like the pandemic).

Our analysis will include tables of performance metrics and perhaps a few plots:

* **Equity Curve plot:** showing growth of \$1 for the RL strategy vs benchmarks over the test period.
* **Drawdown plot:** highlighting max drawdown periods for each strategy.
* **Allocation-over-time plot:** showing how the RL agent’s weights in the 6 ETFs evolved over a few years, and marking regime changes on the timeline to illustrate correspondence (e.g., in “Inflation” regime, the agent tilts to commodities (DBC) and EM equity (EEM) possibly).

All these evaluations will be documented to demonstrate the efficacy of the RL allocator and the importance of the regime signal. If results are mixed, we’ll discuss possible reasons and improvements (see Section 11 on challenges).

## 9. Project Milestones & Timeline

To manage this project over a semester (\~12-14 weeks), we break it into milestones with specific deliverables. Below is a timeline with \~7 milestones:

1. **Week 1-2: Project Proposal & Literature Review** – Refine project scope and read key references. **Deliverable:** a brief proposal (1-2 pages) summarizing the approach and listing data sources and tools. Also, an annotated bibliography of at least 5 relevant sources (papers, articles).
2. **Week 3-4: Data Collection & Pipeline Setup** – Write code to fetch and store data from FRED, Yahoo Finance, Shiller’s website. Begin initial exploration of data. **Deliverable:** Jupyter Notebook `01_DataCollection.ipynb` that pulls the required macro series and ETF prices, and performs initial cleaning (handling missing values, aligning dates). By end of Week 4, the cleaned dataset (CSV or parquet) should be ready.
3. **Week 5: Feature Engineering & Regime Detection** – Implement the z-score transformation and nearest neighbor calculations. Experiment with clustering to define regimes. **Deliverable:** Notebook `02_RegimeDetection.ipynb` showing the computation of macro z-scores, distance matrix (or nearest neighbor findings), and resulting regime classification (e.g., cluster assignments for each date). Include some plots (regime timeline, cluster centers) and commentary interpreting the regimes.
4. **Week 6-7: Baseline Strategy Backtest** – Using the regime signals from Week 5, code the long-similar/short-dissimilar S\&P 500 strategy and backtest it. **Deliverable:** Notebook `03_BaselineBacktest.ipynb` with the strategy logic and performance analysis (metrics, charts). By Week 7, we should have results for the baseline and a clear sense of whether the regime signal has predictive power.
5. **Week 8-9: RL Environment & Model Development** – Develop the Gym environment for the multi-ETF allocation. Write the `PortfolioEnv` class (or similar) implementing state, action, reward as per design. Configure the RL algorithm (PPO or DDPG) and conduct initial training runs (on a subset of data or fewer episodes to ensure code works). **Deliverable (Week 9):** Python script or notebook `04_RL_Environment_and_Training.ipynb` that defines the environment and runs a short training loop (e.g., 1000 steps) to verify the agent and environment interact correctly.
6. **Week 10-11: RL Training & Tuning** – Run full training for the agent using the chosen algorithm and full data (excluding test period). Possibly try a couple of variants (with/without regime in state, different reward settings). Monitor convergence. Once trained, evaluate the agent on the test period. **Deliverable:** Updated `04_RL_Environment_and_Training.ipynb` with final training logs, and a saved model file (e.g., `agent_final.zip`). Also, start compiling results for comparison (store the agent’s decisions and returns in a DataFrame for analysis).
7. **Week 12: Evaluation & Sensitivity Analysis** – Perform the comparisons and stress tests described in Section 8. This includes generating performance metric tables and plots for RL vs benchmarks, and analyzing the agent’s behavior. **Deliverable:** Notebook `05_Evaluation.ipynb` with all analyses. By end of Week 12, preliminary conclusions should be drawn.
8. **Week 13-14: Report Writing & Presentation** – Compile the findings, methodology, and code into a coherent report (which could be this document expanded with results) and prepare a presentation. **Deliverable:** Final project report in Markdown/PDF and a slide deck for presentation. This report should include an introduction, methodology (regime detection and RL), results, discussion, and references – essentially a polished version of all prior notebooks’ contents combined with narrative. The presentation will highlight key insights (e.g., how regimes were identified, how the RL agent performed in different regimes, etc.).

Each milestone builds on the previous, ensuring steady progress. Regular check-ins with the instructor (say, brief updates each week) will help stay on track. There is some buffer in the timeline (especially Week 13-14) for catch-up if earlier tasks slip or if additional tuning is needed. By structuring the work this way, you’ll tackle the project in manageable pieces, reducing last-minute rush and allowing time for reflection and troubleshooting at each stage.

## 10. Resources & Reading List

To successfully complete the project, the following resources are recommended:

* **Original Paper / Concept Reference: “Finding Your Historical Twins”:** While not a formal journal paper, refer to the concept as described by Verdad’s research. *Analogous Market Moments (Verdad, 2024)* and its summary by Larry Swedroe provide insight into using macro similarities for regime clustering. This will guide Deliverable 1.
* **Macroeconomic Regime Detection Literature:**

  * *Oliveira et al. (2025), “Tactical Asset Allocation with Macroeconomic Regime Detection”* – arXiv paper demonstrating improved portfolio performance using macro regime classification.
  * *Ang & Bekaert (2004), “How Do Regimes Affect Asset Allocation?”* – discusses regime-switching models in finance.
  * *Hamilton (1989)* on Markov switching econometrics for context on older regime modeling.
  * *Chen et al. (2023)* – example of modern ML (GANs/RNNs) to identify macroeconomic states.
* **Data Sources Documentation:**

  * *FRED (Federal Reserve Economic Data)* – documentation on how to use their API (e.g., Python `pandas_datareader` for FRED).
  * *FRED-MD Database* – McCracken & Ng (2016) paper on FRED-MD explains transformations for macro series (e.g., taking logs, differences before standardizing). The FRED-MD repository might also have a list of series that could be insightful.
  * *Robert Shiller’s Online Data* – the website  where one can download the Excel file containing long-term stock market and interest rate data.
  * *Yahoo Finance API (yfinance)* – usage examples for downloading historical data for ETFs.
* **Reinforcement Learning Texts/Tutorials:**

  * *Sutton & Barto’s “Reinforcement Learning: An Introduction”* – foundational text to understand RL concepts (policy, reward, episodes, etc.).
  * *OpenAI Spinning Up in Deep RL* – an online guide with how-to for policy gradient and actor-critic methods, including pseudocode for PPO and DDPG.
  * *Stable-Baselines3 Documentation* – particularly the tutorial on continuous actions and the documentation for `SB3` RL algorithms (PPO, DDPG, etc.).
  * *FinRL Library & Papers:* FinRL is an open-source project for financial RL. Their documentation and example code (on GitHub/Colab) can provide templates for structuring the environment and state space for portfolio problems.
* **Portfolio Management & RL Papers:**

  * *Jiang, Xu, and Liang (2017), “Deep Reinforcement Learning for Portfolio Management”* – introduced a portfolio vector memory approach.
  * *Yu et al. (2019), “Deep Reinforcement Learning for Asset Allocation”* – another perspective with continuous action.
  * *Buehler et al. (2019), “Deep Hedging”* – uses RL for risk management, relevant for reward design (drawdown penalties).
  * *Academic surveys:* e.g., *“Reinforcement Learning in Finance”* (IEEE, 2020) for an overview of recent advances.
* **Code Repositories and Examples:**

  * *GitHub – OpenAI Gym Trading Environments:* e.g., `Noterik/stock-trading-environment` or similar projects for inspiration on reward design and state representation.
  * *GitHub – FinRL codes:* which often include Jupyter notebooks demonstrating training an agent on stock index ETFs with technical indicators.
  * *WorldQuant University Capstone (if available):* The search result suggests a similar capstone repo (Musonda2day/Asset-Portfolio-Management-usingDeep...), which might have useful code structure.
* **Tool Documentation:**

  * *PyTorch* – for building any custom neural network or if debugging SB3’s policies.
  * *Pandas & NumPy* – general Python data manipulation libraries (for computing z-scores, etc.). The official documentation and Stack Overflow are handy for specific tasks (e.g., handling missing time series data).
  * *Matplotlib/Seaborn* – for plotting results (to visualize regime classification and performance charts).

By reviewing these resources, you will gain both the theoretical background and practical guidance needed. It’s highly encouraged to read the papers in the Background & Literature to understand the rationale behind regime detection and RL methods, and use the code examples to accelerate development (while writing your own original code for the project). Document any external code or ideas you incorporate to maintain academic integrity.

## 11. Potential Challenges & Risk Mitigation

Implementing this project may surface several challenges. Being aware of them upfront allows us to plan mitigations:

* **Data Gaps and Quality:** Some macro series may have missing data or require transformations (e.g., yield curve spread needs interest rates, which might have different day conventions). *Mitigation:* Carefully inspect each series for NAs and outliers. Use forward-fill or backward-fill for short gaps; for longer gaps, consider replacing with a similar indicator. For instance, if one credit spread series is unavailable in early years, use another (like BAA-AAA spread) as proxy. Leverage documentation like FRED-MD’s guidelines on transforming series (differencing, logging) to ensure stationarity before z-scoring. Additionally, ensure that the macro indicators we choose indeed cover the major dimensions of economic conditions; if something important is missing (say, no indicator of market volatility), the regimes might be less meaningful.
* **Limited History for ETFs:** Several ETFs (BNDX, HYLB, VNQI) have <15 years of data. This limits the length of time the RL agent can train on. *Mitigation:* We can extend the history by using index proxies. For example, use MSCI ACWI Index back to 1988 for ACWI, EMB (EM bond ETF) as a proxy for HYLB before 2016, etc., or even create a synthetic series by splicing similar funds. Alternatively, accept the 2011–2025 period but augment training by randomizing start points as discussed, so the agent effectively sees multiple cycles by restart. We could also lower the frequency to weekly data to get more steps out of the same period (though macro still monthly, weekly sampling might not add info).
* **Regime Model Uncertainty:** Clustering regimes is part art and science. If we pick 4 regimes but the reality is more complex, we might misclassify some periods or overfit regimes to history. *Mitigation:* Validate regimes with known events (check if “crisis” regime coincides with known crises; if not, adjust features or number of clusters). Possibly implement a simple alternative (like a binary recession indicator based on NBER dates or yield curve inversion) to cross-check that our regimes make sense. We should also be cautious not to use *future data* in defining regimes at time *t* – our approach inherently looks at full history to define clusters, which is a bit look-ahead. A true real-time regime classification would be adaptive (clustering expanding window). We may note this caveat, or simulate a rolling clustering if time permits.
* **Reinforcement Learning Instability:** Training deep RL can be fickle – the agent might diverge (especially with a poorly scaled reward or an overly complex network for the amount of data). *Mitigation:* Use relatively simple networks and try the more stable PPO first. Monitor training curves: if the reward collapses or oscillates wildly, adjust hyperparameters (lower learning rate, increase entropy regularization to encourage exploration, etc.). It’s also important to normalize inputs (our state features are mostly z-scores \~N(0,1), which is good; any other inputs like price returns could be scaled to similar range). We will also set a reasonable episode length to avoid extremely long trajectories – this ensures the PPO updates happen frequently enough.
* **Overfitting to Historical Path:** The agent is trained on one historical path (even if we randomize start, it’s still fragments of the same timeline). This is unlike typical RL where the environment can generate endless new scenarios. Thus, there’s a risk the agent just memorizes a sequence of actions tied to chronological events rather than truly learning a generalizable strategy. *Mitigation:* We partition data into train vs test periods to detect overfitting – if performance is great in train and poor in test, that’s a red flag. To combat overfitting, we can:

  * Limit the complexity of the model (small network, fewer train iterations – don’t train until it perfectly fits training data).
  * Add some randomness to training environment: e.g., random noise in returns, or randomize the sequence of macro states slightly. Some approaches generate synthetic data via bootstrapping or resampling to give the agent varied experiences. We might shuffle the order of some historical periods (though that might break realism) or train on slightly altered histories (e.g., drop 2008 once to see if it still learns caution in high credit spreads).
  * Use early stopping: evaluate the agent on the validation set periodically and stop training when that score peaks.
* **Reward Design Issues:** If the reward isn’t capturing what we want, the agent might learn an unintended behavior (e.g., maximizing return by leveraging into one asset but incurring huge unseen risk). The drawdown penalty needs to be balanced – too high and the agent will be ultra-conservative (maybe sitting in bonds all the time), too low and it might ignore drawdowns. *Mitigation:* Trial different reward formulations and examine agent behavior. For instance, if we see the agent always 100% in one asset, perhaps we add a small penalty for concentration or for large changes in allocation (to mimic risk of rebalancing). We will tune the λ in the Sharpe/drawdown reward to get a reasonable trade-off (target a certain volatility or max DD). If needed, test the agent on a few known scenarios during training to ensure its reward aligns with desired outcomes (e.g., does it get penalized appropriately in a simulated crash?).
* **Computational Constraints:** Training an RL agent, even on a small environment, might be time-consuming if we do many episodes or hyperparameter searches. Each episode involves iterating through potentially hundreds of time steps and neural net computations. *Mitigation:* Since our state and action spaces are modest, this should be manageable on a standard laptop or Colab environment. We will also save intermediate models so we can restart or analyze without retraining from scratch. If compute is limited, we might reduce the number of training episodes or simplify the task (e.g., use monthly rather than daily steps to reduce total steps).
* **Interpretability and Verification:** RL policies can be black boxes. We need to ensure the learned strategy makes sense (we don’t want it exploiting some data quirk). *Mitigation:* After training, analyze the policy: e.g., how does it allocate in each regime (as mentioned in evaluation). Also, check if any one input dominates decisions by doing a sensitivity analysis (slightly perturb an input and see if action changes drastically). This can reveal if the agent latched onto something trivial. We can also compare the agent’s actions against the baseline strategy: if they align in obvious scenarios (like agent de-risked in 2008 similarly to baseline short signal), that’s a good sanity check.
* **Integration of Parts:** This project has many components (data, regime model, backtest, RL). Ensuring they work together is a task – for example, the regime classification must feed into the environment correctly. *Mitigation:* Plan integration tests. For instance, once regime labels are ready, simulate a tiny loop: feed a sequence of states with regime info to a dummy agent to see if the pipeline runs. Or after RL training, confirm that using the regime model on test data matches what the agent expects. Keeping the code modular (separate functions for data prep, for computing regime, for environment step logic) will help isolate bugs.

By anticipating these issues, we can address them proactively. It’s likely that some iteration will be needed – for example, we might need to revisit the regime features if the first attempt yields a poor baseline strategy, or we may need to retune the reward after seeing the agent’s initial training. Building in time for such iteration (as per the timeline) is important. Ultimately, documenting these challenges and our solutions will be a valuable part of the project report, demonstrating deep learning (no pun intended) and problem-solving, which is a key objective of a capstone.

## 12. Appendices

**Appendix A: Project Folder Structure** – Organizing files is crucial for a project of this scope. Below is a suggested layout for the project repository:

```
finance-capstone-project/
├── data/
│   ├── raw/                  # original downloaded data files
│   │   ├── fred_macros.csv   # e.g., raw macro series from FRED
│   │   ├── shiller_sp500.csv # Shiller’s data
│   │   └── etf_prices.csv    # raw historical prices for ETFs
│   ├── processed/
│   │   └── master_dataset.csv  # cleaned & merged data (features and returns)
├── notebooks/
│   ├── 01_DataCollection.ipynb
│   ├── 02_RegimeDetection.ipynb
│   ├── 03_BaselineBacktest.ipynb
│   ├── 04_RL_Environment_and_Training.ipynb
│   └── 05_Evaluation.ipynb
├── src/                      # Python modules for reusability
│   ├── data_loader.py        # functions to load data from FRED, Yahoo
│   ├── features.py           # functions to compute z-scores, regime labels
│   ├── regime_model.py       # perhaps code for clustering or nearest neighbor logic
│   ├── env/portfolio_env.py  # the Gym environment class for the RL agent
│   └── train_agent.py        # script to train the RL agent (could use notebooks instead)
├── models/
│   └── saved_agent.zip       # saved RL model (policy weights)
├── reports/
│   └── capstone_report.md    # the final report (possibly generated from notebooks)
├── requirements.txt          # list of Python packages used (pandas, numpy, torch, etc.)
└── README.md                 # overview of the project and instructions
```

This structure separates raw vs processed data (to avoid confusion and enable reproducibility), notebooks for exploratory work and reporting, and a `src` directory for code that might be reused across notebooks. You can adjust as needed, but keep things modular (for example, if a function to fetch data is in `data_loader.py`, the notebook can import it, which is cleaner than rewriting the logic in the notebook).

**Appendix B: Configuration Files** – To facilitate experimentation, consider using a config file (YAML or JSON) to store parameters like which macro series to use, what dates to train/test, algorithm settings, etc. For instance, a `config.yaml` could look like:

```yaml
data:
  macro_series:
    - name: "CPI Inflation YoY"
      source: "FRED"
      code: "CPIAUCSL"
      transform: "pct_change_yr"   # indicate how to transform raw series
    - name: "Yield Curve Slope"
      source: "FRED"
      code: "T10Y3M"
      transform: null
    - name: "High Yield OAS"
      source: "FRED"
      code: "BAMLH0A0HYM2"
      transform: null
    - name: "Unemployment Rate"
      source: "FRED"
      code: "UNRATE"
      transform: null
  etf_series: ["ACWI", "EEM", "BNDX", "HYLB", "DBC", "VNQI"]
  start_date: "2000-01-01"
  end_date: "2025-12-31"
regime_detection:
  method: "GMM"        # or "KNN_threshold"
  n_clusters: 4
  similarity_metric: "euclidean"
rl_model:
  algorithm: "PPO" 
  train_start: "2011-01-01"
  train_end: "2020-12-31"
  test_start: "2021-01-01"
  test_end: "2025-12-31"
  reward_mode: "sharpe_drawdown"
  reward_params:
    lambda_drawdown: 0.1
    target_return: 0.0   # if needed for excess return calc
```

This is just an illustrative snippet. The code would parse this config to set up data and model parameters, making it easier to tweak experiments without hardcoding values in the code.

**Appendix C: Code Snippets and Notebook Skeletons** – Below we outline what each key notebook will contain, along with any tricky code segments:

* **01\_DataCollection.ipynb:**

  * *Objective:* fetch all required data and save to CSV.
  * *Sections:*

    1. Import libraries (`pandas`, `yfinance`, `pandas_datareader`).
    2. Define the list of FRED series and use `pandas_datareader.DataReader` to fetch them. Example:

       ```python
       import pandas_datareader.data as web
       fred_series = {"CPIAUCSL": "CPI", "T10Y3M": "YieldCurve", "BAMLH0A0HYM2": "HY_OAS", "UNRATE": "Unemployment"}
       macro_dfs = []
       for code, name in fred_series.items():
           series = web.DataReader(code, 'fred', start='1960-01-01', end='2025-12-31')
           series.columns = [name]
           macro_dfs.append(series)
       macro_df = pd.concat(macro_dfs, axis=1, join='inner')
       ```
    3. Download ETF price data via `yfinance`:

       ```python
       import yfinance as yf
       tickers = "ACWI EEM BNDX HYLB DBC VNQI"
       etf_data = yf.download(tickers, start="2010-01-01", end="2025-12-31", auto_adjust=True)
       prices = etf_data['Adj Close']  # DataFrame of prices
       ```
    4. Save raw data to CSV in `data/raw/`. (We might save macro and ETF data separately.)

* **02\_RegimeDetection.ipynb:**

  * *Objective:* turn raw data into regime labels.
  * *Sections:*

    1. Load the raw data (macro\_df from CSV, ensure it’s monthly frequency).
    2. Compute transformations: e.g., CPI YoY from CPI index (if needed):

       ```python
       macro_df['CPI_YoY'] = macro_df['CPI'].pct_change(12) * 100
       ```

       and then drop the raw CPI if not needed.
    3. Compute z-scores for each column (as shown in Section 4 code snippet).
    4. Perform clustering or nearest neighbor analysis:

       * If GMM: use `sklearn.mixture.GaussianMixture` to fit on the z-score matrix and predict labels.
       * If K-means: use `sklearn.cluster.KMeans` similarly.
       * If manual threshold: compute nearest distances and label those above percentile as “Unusual”.
    5. Analyze the clusters: compute mean of each feature per regime cluster, print them out or bar chart to interpret regime characteristics.
    6. Plot regime over time (e.g., color a time series plot of one indicator by regime, or just a separate plot of regime label vs time).
    7. Save the resulting DataFrame with regime labels to `data/processed/master_dataset.csv` for use in later notebooks.

* **03\_BaselineBacktest.ipynb:**

  * *Objective:* implement and evaluate the long-similar/short-dissimilar strategy.
  * *Sections:*

    1. Load `master_dataset.csv` which contains macro z-scores, regime info, and also ensure S\&P 500 or ACWI returns are present.
    2. Define the trading signal: For simplicity, maybe use the binary approach. For each date *t*, determine:

       ```python
       threshold = distances.quantile(0.75)  # for example, 75th percentile as cut-off
       signal[t] = 1 if min_distance[t] < threshold else -1
       ```

       Or use cluster label if we have a specific “bad regime” label (like if regime 4 = crisis, then signal = -1 when in regime 4, else +1).
    3. Simulate strategy:

       ```python
       initial_cash = 1.0
       equity = initial_cash
       equity_curve = []
       for t in range(start, end):
           if signal[t] == 1:
               ret = sp500_ret[t]  # go long
           elif signal[t] == -1:
               ret = -sp500_ret[t]  # short
           else:
               ret = 0
           equity *= (1 + ret)
           equity_curve.append(equity)
       ```

       (This assumes no leverage beyond 1x short.)
    4. Calculate performance metrics from equity\_curve and compare with `sp500` buy-and-hold.
    5. Plot the equity curve for strategy vs S\&P.
    6. Print metrics like Sharpe, drawdown, etc. This notebook will clearly show if the strategy beat the market or not.

* **04\_RL\_Environment\_and\_Training.ipynb:**

  * *Objective:* define the RL environment class and train the agent.
  * *Sections:*

    1. Define the custom Gym environment:

       ```python
       import gym
       from gym import spaces
       import numpy as np

       class PortfolioEnv(gym.Env):
           def __init__(self, data, train_mode=True):
               super().__init__()
               self.data = data  # this would be a DataFrame including features and asset returns
               self.train_mode = train_mode
               self.current_step = 0
               # Define observation space (for example, length = number of features + maybe prev weights)
               num_features = data.features.shape[1]  # e.g., 4 macro features or 4 regime one-hot
               self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
               # Define action space (6 weights between 0 and 1)
               self.action_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
               self.weights = np.array([1/6]*6)  # start with equal weights
               self.initial_capital = 1.0
               self.portfolio_value = self.initial_capital

           def reset(self):
               self.current_step = 0
               self.portfolio_value = self.initial_capital
               self.weights = np.array([1/6]*6)
               # Optionally randomize start step in training
               if self.train_mode:
                   self.current_step = np.random.randint(0, len(self.data.index) - 2)
               # Get state
               return self._get_observation()

           def _get_observation(self):
               # Get macro features at current_step
               obs = self.data.features.iloc[self.current_step].values.astype(np.float32)
               return obs

           def step(self, action):
               # Normalize action to weights
               action = np.array(action)
               if action.sum() <= 0:
                   # if all zeros (unlikely due to random init), default to equal
                   weights = np.array([1/6]*6)
               else:
                   weights = action / action.sum()
               # Compute reward as portfolio return next step
               next_step = self.current_step + 1
               returns = self.data.returns.iloc[next_step].values.astype(np.float32)  # vector of returns for each asset
               portfolio_ret = np.dot(weights, returns)  # weighted sum of asset returns
               # Update portfolio value
               prev_value = self.portfolio_value
               self.portfolio_value *= (1 + portfolio_ret)
               # Compute drawdown (if needed for reward shaping)
               # For simplicity, we maintain the running max
               if next_step == 0:
                   self.running_max = self.portfolio_value
               else:
                   self.running_max = max(self.running_max, self.portfolio_value)
               drawdown = (self.running_max - self.portfolio_value) / self.running_max
               # Calculate reward (Sharpe approx: excess return - penalty * drawdown)
               reward = portfolio_ret - 0.1 * drawdown
               # Move to next step
               self.current_step = next_step
               done = (self.current_step == len(self.data.index) - 2)
               return self._get_observation(), reward, done, {}
       ```

       (The above is simplified and may need tweaks – e.g., handling episode end properly, maybe include a time penalty or something to encourage shorter episodes.)
    2. Initialize environment with training data subset:

       ```python
       train_data = ...  # DataFrame for train period with features and returns
       env = PortfolioEnv(train_data, train_mode=True)
       ```
    3. Use Stable-Baselines3 to create and train agent:

       ```python
       import stable_baselines3 as sb3
       model = sb3.PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
       model.learn(total_timesteps=5000)
       model.save("models/ppo_agent")
       ```

       (Adjust timesteps and perhaps use `eval_env` with a callback for early stopping on validation.)
    4. If time permits, train a second agent variant (e.g., without regime in state).
    5. After training, load the model and run it on validation data in a loop to collect actions and rewards for analysis.

* **05\_Evaluation.ipynb:**

  * *Objective:* compare the RL agent’s performance with benchmarks.
  * *Sections:*

    1. Load the trained model (e.g., `model = sb3.PPO.load("models/ppo_agent")`).
    2. Create an environment for the **test dataset** (201\[?]–2025) with `train_mode=False` so it doesn’t random reset.
    3. Run a loop:

       ```python
       obs = test_env.reset() 
       done = False
       history = []
       while not done:
           action, _states = model.predict(obs, deterministic=True)
           obs, reward, done, info = test_env.step(action)
           history.append((test_env.current_step, test_env.portfolio_value, action, reward))
       ```

       This collects the agent’s decisions and portfolio value over the test period.
    4. Compute test performance metrics from `history` (or from stored returns in env).
    5. Compare with benchmarks: we can simulate equal-weight portfolio over the same period easily, and any other baseline.
    6. Plot the results:

       * Equity curve of \$1 for RL vs equal-weight vs any other strategy.
       * Perhaps a bar chart of Sharpe ratios for each.
    7. Regime conditioning analysis: using the test data (which has macro/regime features):

       * Extract the regime label each period from test\_data.
       * Group the agent’s actions by regime to compute average weight per asset in each regime.
       * Display that in a table or heatmap (assets vs regime, values = avg weight). This shows how the agent reallocates in different environments.
    8. Stress test: Focus on 2020:

       * Plot a zoom-in of portfolio value Jan 2020–Dec 2020 for RL vs benchmark, and annotate when COVID crash happened.
       * Print the agent’s allocation in Jan 2020, Feb 2020, Mar 2020 to see if it dropped equity exposure.
       * Do similar for any other notable period (maybe 2015–2016 if EM crisis, etc., depending on data).
    9. Summarize findings in text: did RL outperform, was the regime useful, etc.

**Appendix D: Hardware/Software Setup** – (If needed, describe using Google Colab or local Python, ensuring the correct versions of libraries. But likely not needed in the brief unless specific.)
