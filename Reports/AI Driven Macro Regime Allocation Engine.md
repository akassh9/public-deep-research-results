# AI‑Driven Macro Regime Allocation Engine (MARAE) – Project Brief

## 1. Executive Overview

The “AI-Driven Macro Regime Allocation Engine (MARAE)” is a capstone project focused on applying machine learning and quantitative finance techniques to dynamic asset allocation. The goal is to create an **automated investment engine** that identifies prevailing macroeconomic regimes (e.g. growth vs. recession, high vs. low inflation) and allocates a portfolio across six core ETFs plus cash accordingly. This document serves as the comprehensive guide for executing the project, covering theoretical foundations, data sources, methodological steps, and project management expectations.

In this project, you will build a pipeline that ingests financial and economic data, detects the current macro regime using AI/ML models, and then optimizes portfolio weights based on pre-defined **Capital Market Assumptions (CMAs)** for asset returns. The end product will be an interactive **dashboard** (e.g. in Streamlit) that allows users to upload their own CMA inputs and obtain the model’s regime classification, recommended asset allocation, and back-tested performance. By completing MARAE, you will integrate knowledge from macroeconomics, data science, and portfolio management, simulating the work of a “quant” researcher building an AI-driven investment strategy.

The project is designed to be _rigorous yet supportive_. Each section of this brief provides the necessary theory, tools, and structure to help you work independently. By following the guidelines herein, you will not only develop a functional allocation engine but also deepen your understanding of how **business-cycle regimes** influence markets and how modern AI techniques can enhance decision-making in asset management. The experience will strengthen both your technical skills (from Python programming to machine learning) and professional skills (from research documentation to presentation), preparing you for advanced roles in fintech or investment analytics.

## 2. Learning Objectives (technical & professional)

This capstone is an interdisciplinary endeavor. By the end of the project, you should achieve several **technical learning objectives** and **professional development objectives**:

- **Technical Objectives:**
    
    - **Macro-Finance Mastery:** Understand key macroeconomic **regimes** (expansion, recession, etc.) and how they affect asset class performance, drawing on business-cycle theory and historical data.
        
    - **Data Science & Engineering:** Gain proficiency in Python-based data handling – ingesting time-series data from APIs (e.g. FRED, Yahoo Finance), cleaning datasets, and engineering features (economic indicators, trends, sentiment metrics) for modeling.
        
    - **Machine Learning Modeling:** Learn to train and evaluate an ML model (e.g. classifier or clustering algorithm) that detects or predicts macro regimes. This includes practicing model selection (comparing logistic regression vs. random forests vs. other classifiers) and **hyper-parameter tuning** with proper validation techniques for time-series data.
        
    - **Portfolio Optimization:** Apply portfolio construction methods (mean-variance optimization, risk parity algorithms, CVaR minimization) using the provided CMAs to compute asset weights. You will translate theoretical risk budgeting concepts into code (e.g. solving for weights that equalize risk contributions or maximize Sharpe ratio).
        
    - **Full-Stack Deployment:** Develop an end-to-end pipeline culminating in a user-facing **dashboard**. This involves using tools like Plotly for interactive charts and Streamlit for a web app interface. You’ll also learn to containerize the application with a Dockerfile for consistency and possibly design simple REST API endpoints for key functionalities.
        
- **Professional Objectives:**
    
    - **Research & Analysis:** Enhance your ability to conduct independent research – reviewing academic papers, whitepapers, and textbooks to inform your methodology. You will practice distilling insights from literature (e.g. how others define regimes or perform backtests) and applying them to your implementation.
        
    - **Project Management:** Manage a semester-long project by breaking it into milestones. You will maintain a schedule (as outlined in the Gantt chart below), adhere to deliverable deadlines, and adjust scope as needed. Using version control (Git) with a clear branching policy will instill good software project discipline (e.g. using feature branches and pull requests before merging to main).
        
    - **Communication:** Improve technical writing and presentation skills. You will produce a well-documented codebase and a formal research paper detailing your approach and findings. Additionally, you’ll prepare a final presentation to explain your work and justify design decisions.
        
    - **Problem-Solving & Adaptability:** Develop the capacity to troubleshoot issues such as data quality problems, model overfitting, or performance constraints. This project will require iterative debugging and creative thinking (for example, figuring out how to incorporate an unconventional data source or how to simplify a model to run under time limits). You’ll also confront ethical and compliance considerations, learning to balance innovative use of data with responsible conduct.
        

By achieving these objectives, you will not only deliver a functioning MARAE system, but also cultivate a mindset for continuous learning—a critical trait in the rapidly evolving field of AI in finance.

## 3. Theoretical Background

Before diving into implementation, it is crucial to understand the theoretical concepts underpinning MARAE. This section provides a foundation in four key areas of theory:

### 3.1 Macro Regimes & Business-Cycle Theory

**Macro regimes** refer to distinct states of the economy or financial markets that exhibit characteristic behavior. Classic business-cycle theory breaks the economic cycle into phases such as **recovery, expansion, slowdown, and contraction**. In practical terms, one can think of regimes like “boom vs. recession” or more granular states like “high-inflation stagnation” vs “low-inflation growth”. Each regime has implications for asset returns and risk. For example, in economic expansions with rising growth, equities typically perform well, whereas during recessions or high-stress regimes, safer assets like Treasury bonds or gold outperform (Bodie, Kane & Marcus, 2021). Indeed, **asset class performance rotates with business cycle phases** – equities and commodities often rally in early expansions, credit might thrive in mid-cycle, and government bonds tend to shine in contractions when interest rates fall.

Business cycle research, dating back to Burns & Mitchell (1946), established that economies experience **recurring fluctuations** that, while not identical, rhyme over time. Modern economists and investors use a variety of indicators to infer the current phase. Examples include GDP growth rates, unemployment trends, **yield curve slopes** (short vs. long-term interest rates), and survey indices like PMIs (Purchasing Managers’ Index). A flat or inverted yield curve (short rates >= long rates) has historically been a predictor of recessions, thus serving as a regime indicator (Harvey, 1988). Similarly, the NBER (National Bureau of Economic Research) retrospectively labels recessions which can be used to create a **binary regime label** for supervised learning (recession vs. non-recession).

Financial economists have also explored **data-driven regime identification**. For instance, clustering algorithms or Markov switching models can segment market history into regimes based on patterns in asset returns or correlations. Recent research by Mulliner et al. (2025) proposes detecting regimes by finding historical analogues to current conditions rather than prespecifying regime types – essentially a nearest-neighbor approach in macro indicator space that showed improved asset return predictions (Mulliner et al., 2025). Another study by Nuriyev et al. (2024) in the ACM AI in Finance conference demonstrated that incorporating macro regime signals can augment equity factor investing, implying that regime classification can add value to investment strategies.

For this project, you should appreciate that _defining the regimes_ is a critical first step. Will you use **pre-defined economic regimes** (like “expansion”, “recession”) based on external labels or thresholds, or will you apply an **unsupervised learning** approach to let the data tell you how many regimes exist? Both approaches are valid. A predefined approach might use known recession dates (from NBER) to label each month in history as regime A or B, which a classifier can then learn. An unsupervised approach might cluster multi-asset return patterns (as in correlation-based clustering) to discover regimes like “high correlation crisis” vs “low correlation calm” periods. In either case, understanding the economic intuition behind regimes will guide feature selection and model design.

In summary, macro regimes provide the contextual backdrop for asset allocation decisions. Grounding your work in business-cycle theory ensures that the AI-driven engine aligns with economic reality (e.g. a regime it identifies can be interpreted as a recession or high-inflation period, etc.). This combination of domain knowledge and data-driven detection is at the heart of MARAE’s value proposition.

### 3.2 Capital-Market Assumptions & Expected Returns

**Capital Market Assumptions (CMAs)** are **expected long-term returns and risk estimates** for asset classes, typically produced by investment firms annually. They reflect research-based views on how asset classes (e.g. global equities, emerging-market equities, investment-grade bonds, commodities, real estate, etc.) will perform over a future horizon (often 5 or 10 years). CMAs also include assumptions about asset volatility and correlations, providing a forward-looking covariance matrix. For example, BlackRock’s Investment Institute periodically publishes 5-year return assumptions for major asset classes (BlackRock Investment Institute, 2023), and J.P. Morgan releases an influential Long-Term Capital Market Assumptions report each year.

These assumptions serve as inputs to **strategic asset allocation**. In practice, if you expect equities to return, say, 7% per annum over the next 5 years with 15% volatility, and bonds to return 3% with 5% volatility, those numbers feed into portfolio optimization models. For MARAE, the user will provide a CSV of CMAs (likely containing expected annual return, volatility, and possibly correlation info for each asset). Your system will use these expected returns as a basis for allocating between ACWI, EEM, BNDX, HYLB, DBC, VNQI, and cash.

Understanding the theory behind expected returns is important. Equity return assumptions might be built from current **valuation metrics** (like earnings yield or dividend yield plus growth), or historical averages adjusted for today’s conditions. Bond return assumptions often closely follow **yield-to-maturity** (a yield on BNDX, a global bond ETF, might guide its expected return). CMAs also often include **macro scenarios** – e.g., a central scenario vs. bull or bear cases. Some advanced frameworks incorporate uncertainty around these (BlackRock’s models “account for uncertainty and different pathways” in returns).

In this project, we will treat the CMA inputs as given **exogenous parameters**. That is, you do not have to derive the expected returns yourself; instead, you will **map** the provided assumptions to our investable universe. For instance, if the CMA file says “Developed Market Equities = 6% expected return”, you’ll apply that to ACWI (which is an ETF for DM equities). If “EM Equities = 8%”, map that to EEM, and so on. In Appendix A, we’ve provided a sample schema that shows how such data might appear.

One nuance: The MARAE engine could be extended to have **regime-dependent expected returns**. Some regimes might warrant adjusting the baseline CMAs. For example, if the regime is a recession, one might haircut the equity return assumption (since near-term equity performance could be below long-term average) or boost expected returns for bonds (if interest rates are expected to fall further in recession, boosting bond prices). However, as a baseline, we will **use the same CMA values regardless of regime**, under the assumption that these are 5-year strategic assumptions. The regime will influence allocation more by affecting risk assessment and perhaps tilting weights (e.g. a risk-parity approach might naturally shift towards lower-vol assets in a high-stress regime).

In sum, CMAs form the **expected return vector (and related risk parameters)** for your portfolio optimizer. They are grounded in both historical data and forward-looking judgment (as per sources like BlackRock Investment Institute (2024) and others). You should ensure your engine can read and handle these inputs correctly (with appropriate units, e.g. percent per annum) and use them as the quantitative basis for portfolio construction.

### 3.3 Risk-Budgeting & Portfolio Construction (mean-variance, risk-parity, CVaR)

Portfolio construction is where the rubber meets the road – using the regime identification and CMAs to actually allocate assets. Three important theoretical approaches to understand are **mean-variance optimization, risk parity, and CVaR minimization**, each offering a different way to “budget” risk in the portfolio:

- **Mean-Variance Optimization (MVO):** Introduced by Harry Markowitz (1952), this is the foundation of Modern Portfolio Theory. The idea is to choose asset weights that **maximize expected return for a given level of risk**, or equivalently minimize risk for a target return. Mathematically, if $\mu$ is the expected return vector (from CMAs) and $Σ$ is the covariance matrix of asset returns, one solves:  
    max⁡w  wTμ−λ wTΣw,\max_w \; w^T \mu - \lambda \, w^T Σ w,  
    subject to constraints like $\sum w_i = 1$ (fully invested) and $w_i \ge 0$ (no shorting, if required). The term $w^T Σ w$ is portfolio variance (risk), and $\lambda$ is a risk-aversion parameter. In practice, one often maximizes **Sharpe ratio** = $(w^T\mu - r_f)/\sqrt{w^T Σ w}$, where $r_f$ is the risk-free rate (cash yield). MVO will allocate more to assets with high expected return and low covariance, but it’s sensitive to input errors. Small changes in $\mu$ can lead to big shifts in weights, which is why many practitioners constrain or adjust pure mean-variance outputs.
    
- **Risk Parity:** This approach ignores expected returns and instead focuses on equalizing risk contribution of each asset. The idea is that each asset or asset class should contribute the **same marginal risk** to the portfolio. For example, if equities are much more volatile than bonds, a risk parity portfolio will hold more bonds and fewer equities such that each provides, say, 20% of total portfolio risk. One formalism: find weights $w$ such that for any two assets $i,j$,  
    wi⋅(Σw)i=wj⋅(Σw)j,w_i \cdot (Σw)_i = w_j \cdot (Σw)_j,  
    meaning the **risk contribution** ($w_i$ times the portfolio covariance with asset $i$) is equal. Solving this requires iterative methods (as it’s a system of non-linear equations), but it essentially yields portfolios similar to the “**equal risk**” or minimum-variance portfolios if there are no expected return inputs. Risk parity became popular after 2008 as a way to build **diversified portfolios** (Qian, 2011). A well-known example is Bridgewater’s “All Weather” fund which roughly implements risk parity across equities, bonds, commodities, etc., under the logic that since future returns are hard to predict, better to balance risk than dollar amounts. In MARAE, a risk parity allocation could serve as a baseline in a neutral regime.
    
- **Conditional Value-at-Risk (CVaR) minimization:** CVaR (also called Expected Shortfall) is a **tail-risk measure** – the expected loss given that the loss is beyond the VaR cutoff. For instance, CVaR at 95% is the average of the worst 5% outcomes. Rockafellar & Uryasev (2000) pioneered an optimization technique to directly minimize CVaR. The appeal is that by minimizing CVaR, you are explicitly protecting against extreme losses, not just variance. The optimization can be done via linear programming by introducing auxiliary variables for losses beyond the VaR threshold (the Rockafellar-Uryasev method). In a portfolio context, a CVaR optimizer might produce more stable allocations under non-normal return distributions (like when asset returns have fat tails or skew). This is advanced for a student project, but we include it as a conceptual goal: for example, if the student implements a scenario-based optimization, they could aim to choose weights that minimize the 95% CVaR of portfolio return while achieving at least a certain average return.
    

In this project, you will likely implement at least one of these approaches (mean-variance is the most straightforward given we will have expected returns). You might also incorporate ideas of **risk budgeting** by setting constraints (e.g. maximum weight limits or target volatility). For example, you could constrain that portfolio volatility (per $w^T Σ w$) should not exceed a threshold, or enforce that at least 5% is in each asset to avoid complete zeros.

Another important concept is **risk factor diversification**. While not explicitly required, be aware that assets have common risk factors (equities share equity risk, bonds interest rate risk, etc.). A naive MVO might overload on two assets that both rely on the same factor (e.g. ACWI and EEM both equity). Risk parity inherently tends to spread across factors (giving more weight to bonds because they have different risk profile than stocks). BlackRock’s Aladdin risk reports (Appendix A) often decompose a portfolio’s risk into factor categories like Equity, Credit, Rates, etc. For a stretch goal, you could ensure that the allocation engine monitors factor exposures (for instance, the riskFactors output in Appendix A’s JSON shows a sample breakdown).

**Bringing it together:** The MARAE engine, upon identifying the macro regime, will use one of these portfolio construction logics to propose weights. For instance, suppose the model says we’re in a “high-stress” regime. The engine might then lean towards a more conservative allocation – either by using a low target risk in mean-variance (thus more weight on bonds and less on equities) or by triggering a risk-parity solution (which naturally will allocate more to low-vol assets). Conversely, in a benign “expansion” regime, it might allow more equity weight to capture higher expected returns. The exact mapping of regime to optimization parameters is up to your design (you might, for simplicity, use the same optimization for all regimes initially, then in stretch goals adjust risk aversion based on regime).

Understanding these portfolio theories will help you justify your design choices in the final report. For example, if you choose mean-variance, you can cite its optimality under certain assumptions and discuss how you mitigated its weaknesses (perhaps by adding regularization or using Black-Litterman blending if you’re ambitious). If you try risk parity, you can discuss how it may result in more stable outcomes at the cost of potentially ignoring return forecasts (Chow et al., 2011). And if you consider CVaR, you can frame it as focusing on tail-risk management (which might be especially relevant in crisis regimes).

### 3.4 Alternative-Data & NLP for Theme Detection

Traditional models rely on structured numerical data (prices, macro indicators). **Alternative data** introduces new information sources, often textual or high-frequency, that can provide an **edge in detecting themes or shifts** in the market not yet reflected in fundamentals. MARAE encourages exploration of alt-data, especially via Natural Language Processing (NLP), to gauge market sentiment or attention to certain themes.

For example, **Google Trends** data indicates the relative frequency of search terms. Choi and Varian (2012) famously showed that Google search frequencies can help “predict the present” for economic variables – e.g., an increase in searches for “unemployment benefits” could indicate rising unemployment even before official data is released. In our context, one could monitor search interest in terms like “recession” or “inflation” as real-time proxies for public concern about those issues. If “recession” queries spike, it might signal an impending regime shift in sentiment.

Another source, **GDELT (Global Database of Events, Language, and Tone)**, provides a quantified summary of global news events. GDELT monitors news media world-wide and computes a “tone” score and categorizes events (conflict, protests, economic optimism, etc.) in real time. Researchers Leetaru & Schrodt (2013) introduced GDELT as a way to capture societal-scale behavior and sentiment. For MARAE, you might use GDELT’s daily “Global Tone” index or specific event counts as features – e.g., a decrease in average tone (more negative news sentiment globally) could foreshadow risk-off regimes.

**NLP techniques** can extract themes from unstructured text: for instance, analyzing **financial news headlines or central bank statements**. You could use a pre-trained language model or keyword approach to detect if discussions of “inflation”, “geopolitical risk”, “financial crisis”, etc., are trending. LangChain (which you have in the stack) could facilitate connecting to a language model to summarize news or classify sentiment on the fly. As a baseline, even simple sentiment analysis or **topic frequency counts** can be informative. For example, you might count mentions of “inflation” in news articles each week – if it’s surging, that might correspond to an “inflationary regime” signal.

**Social media and crowdsourced data** are other alt-data examples. Reddit posts or Twitter feeds can sometimes reveal retail investor sentiment or concerns (e.g., Reddit mentions of “recession” or trending finance topics). While real-time scraping of Reddit is advanced, you could incorporate a pre-collected dataset or simulate this by using Google Trends on finance-related queries as a proxy. Kaggle hosts several datasets related to ESG sentiment, news, and other alternative metrics which you can optionally explore (for instance, an **ESG News dataset** might allow a specialized theme detection like climate risk, if relevant to allocation).

It’s important to note that alt-data often comes with **noise and bias**. Not all spikes in Google searches mean something fundamental, and news sentiment might be temporarily skewed. The use of NLP should be carefully integrated – likely as **additional features** in the regime classifier. For example, you could have a feature that is “News Sentiment 30-day average” or “Google Trend for ‘bear market’”. By including these alongside traditional macro indicators, the ML model can weigh their importance. It might learn, for instance, that high financial stress plus extremely negative news sentiment strongly indicates a crisis regime.

A stretch concept is using an **LLM (Large Language Model)** directly to classify regime narratives. For instance, you could feed an LLM a summary of recent economic news and ask it “Does this news suggest an economic expansion, stagflation, or recession?” This kind of theme overlay (discussed more in Stretch Goals) could add a qualitative “story” recognition to complement quantitative data.

Finally, consider the **availability and frequency** of these alt data. Google Trends can be pulled weekly or daily for specific terms. GDELT updates daily. If you use them in backtesting, you need historical data (GDELT archives are available, Google Trends can be retrieved historically via keywords). Ensure any alt-data you use does not introduce look-ahead bias (e.g. using a revised sentiment series that wasn’t available in real time).

In summary, alternative data and NLP offer a way to _detect subtle shifts_ and thematic context that numbers alone might miss. Incorporating them into MARAE can improve regime detection accuracy and responsiveness. It also demonstrates a modern approach to investing, where **AI reads the news** and **gauges crowd sentiment** as part of the decision process (as some hedge funds and asset managers increasingly do). Use alt-data judiciously – as supplementary signals to corroborate what traditional indicators are saying, or to provide early warnings when traditional data is lagging.

## 4. Data Resources

To build MARAE, you will draw from a mix of **financial market data** and **economic indicators**, as well as some optional alternative datasets. Below we list the mandatory data sources (which you must utilize) and optional ones (for exploration/extension), along with guidance on accessing each:

**Mandatory data sources:**

- **FRED (Federal Reserve Economic Data)** – _Key repository for U.S. and global economic time series._ We will use FRED for data like interest rates and macro indicators. For example, the **3-Month T-Bill rate** (cash proxy) is available as series `TB3MS`. You can retrieve FRED data via API: e.g., using `pandas_datareader` in Python:
    
    ```python
    import pandas_datareader.data as web  
    tbill = web.DataReader("TB3MS", "fred", start="2000-01-01", end="2025-01-01")
    ```
    
    This will fetch the 3-month T-bill yield (percent per annum). FRED also hosts the **St. Louis Fed Financial Stress Index** (series code `STLFSI3`) and countless other indicators (industrial production, CPI, etc.) that you might incorporate. _Direct download:_ You can also download CSVs from the FRED website (e.g., `https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS`). An API key from FRED allows more complex queries if needed.
    
- **Yahoo Finance via `yfinance`** – _For historical prices of the six ETFs._ The core asset price data (for ACWI, EEM, BNDX, HYLB, DBC, VNQI) can be obtained using the `yfinance` Python library. For example:
    
    ```python
    import yfinance as yf  
    prices = yf.download(["ACWI", "EEM", "BNDX", "HYLB", "DBC", "VNQI"], start="2006-01-01", end="2025-01-01")
    ```
    
    This returns daily OHLCV prices. You’ll likely focus on adjusted close prices. You may resample to monthly returns for strategic allocation. Yahoo also provides basic indices (like S&P 500 or VIX) if needed for additional context. _Note:_ `yfinance` is a convenient wrapper; ensure you respect its rate limits by not querying too frequently in loops.
    
- **St. Louis Fed Financial Stress Index** – _Weekly index measuring financial stress in markets._ This is available on FRED under code `STLFSI3` (current version). It’s a composite of yields, spreads, and volatility measures; values above 0 indicate above-average stress. It’s useful as a feature for regime classification (high stress often coincides with recessionary or crisis regimes). Access via FRED as shown above. The data is weekly; you might merge it to monthly by taking end-of-month or max within month.
    
- **CFTC Commitment of Traders (COT) Reports** – _Weekly data on futures positions by trader type._ This can serve as a sentiment or positioning indicator (e.g., how net long or short various market participants are in equity index futures, bond futures, commodity futures). The **COT data** is published by the CFTC. Historical reports are available as downloadable text or Excel files on the CFTC website. For instance, the “Traders in Financial Futures, Futures Only” reports can be downloaded year-by-year: the CFTC provides a page with links for each year’s data (see CFTC **Historical Compressed Reports** page).  
    _Access tip:_ Navigate to the CFTC’s `Commitments of Traders` page. Under “Reports by Year”, you’ll find links like _2025 (Text, Excel)_, _2024 (Text, Excel)_, etc. Downloading the Excel for each year will give all contracts. You’ll need to filter for specific contracts relevant to our assets (e.g., S&P 500 E-mini for equities, U.S. Treasury for bonds, etc., if you choose to use them). Parsing might require pandas (`read_excel`) and some data wrangling (the format is a bit raw). This is an optional analysis for additional features – for instance, you could derive an indicator like “net speculative positioning in oil futures” as a proxy for sentiment in commodities. If time is constrained, focus on primary data first.
    
- **Google Trends** – _Interest over time for search terms._ Google Trends data can be retrieved via the unofficial **PyTrends** API. This can add a behavioral dimension to your dataset. For example, you might query terms like “inflation”, “recession”, “unemployment” or even financial terms like “buy stocks”. PyTrends usage example:
    
    ```python
    from pytrends.request import TrendReq  
    pytrends = TrendReq()  
    pytrends.build_payload(["recession", "inflation"], timeframe="2010-01-01 2025-01-01", geo="US")  
    trends_df = pytrends.interest_over_time()
    ```
    
    The result `trends_df` will have weekly normalized search interest for the specified terms (values range 0-100). If the API is flaky, you can manually download CSVs from the Google Trends web interface for specific keywords and date ranges. Remember to align the frequency (Trends data is weekly by default) with your other data. This data can be very illustrative — e.g., a spike in “recession” searches in 2020, etc., which you can use as a feature or simply discuss.
    

**Optional alternative data sources:**

- **GDELT Project** – An open database of global events and news sentiment. GDELT provides daily updates; one easy entry is the “GDELT Global Media Index” which includes an overall sentiment score. Access options: GDELT has CSV feeds (e.g., daily events files) and a BigQuery database. For a simpler path, consider using GDELT’s **Event Database aggregates**: for instance, GDELT provides files like `20150101.export.CSV` for events on Jan 1, 2015. However, parsing the raw data is complex. Instead, you might use a pre-aggregated index: GDELT’s site (gdeltproject.org) offers a graph of Tone over time and sometimes releases aggregated files for download. If an API is needed, GDELT has a JSON API for recent news queries. Due to time, you might only incorporate a high-level series like “Average Tone of news” or “Conflict event count” if readily available. _(This is a stretch; ensure core features are done first.)_
    
- **RavenPack or Other News Analytics** – RavenPack is a commercial provider of news sentiment and macro signal data. If your university provides access or a trial, you could use their historical news sentiment indices (they often have data on economic topics sentiment). If not, mention RavenPack in your report as an example of advanced alternative data. Without access, you won’t use it directly, but you could simulate a similar sentiment index by computing sentiment from news headlines using NLP (NLTK or TextBlob libraries for a simple sentiment score, for example).
    
- **Kaggle ESG and Other Datasets** – Kaggle hosts datasets that could be relevant. For instance, _Public Company ESG Ratings_ (with ESG scores for companies) or datasets on climate news. While ESG might not directly drive macro regimes in the short term, if you want to explore a specific theme (say, an “ESG regime” or the impact of climate risk on allocation), such data could be insightful. Kaggle also has datasets for social media sentiment (e.g. a collection of Reddit comments on WallStreetBets) that could be repurposed. To use Kaggle data, you can download via the Kaggle API after joining the dataset, or directly from the website if it’s a one-off file. Ensure any Kaggle data you add is documented and cite the dataset source.
    

When using these resources, **keep data organization in mind**. It’s wise to create a data directory and save raw data (as CSV or pickle) so you don’t re-download excessively. Document each data source (units, frequency, transformations applied). Also, be mindful of date alignment: Our assets are global (some ETFs are U.S.-listed, so their prices are in USD and daily). Economic data might be monthly. For modeling, you may choose a monthly frequency for everything (e.g., end-of-month values for indicators, monthly total returns for assets). That tends to make regime classification smoother, as regimes usually persist months or quarters.

Finally, include the cash rate: the 3-month T-bill (from FRED) will be used as the risk-free rate for Sharpe ratio calculations and as an asset if the model wants to allocate to “Cash”. Typically, you might treat cash as having a fixed yield (e.g. 2% currently) and zero volatility in optimization.

In summary, start by securing the **required data**: price history for the 6 ETFs, T-bill rate, and a handful of macro indicators (at least one—e.g., financial stress index or yield curve). Then, if interested and time permits, layer in **alt-data** like search trends or sentiment. The combination will give your model a rich view of the market’s state.

## 5. Software Stack & Environment

The project will be implemented in a **Python 3.11** environment, leveraging popular libraries for data science, machine learning, and web deployment. Below is the expected software stack and tools, along with environment setup guidelines:

- **Python 3.11:** Use the latest Python 3.11.x release to ensure compatibility with modern libraries and performance improvements. All code (data processing, modeling, etc.) will be written in Python.
    
- **Core Libraries:**
    
    - **pandas & NumPy:** for data manipulation (DataFrames, numerical arrays). Pandas will be central for handling time-series (e.g., using DateTime index, resampling monthly).
        
    - **scikit-learn:** for machine learning algorithms (classification, clustering) and tools (train/test splitting, cross-validation). Scikit-learn provides a wide range of models (LogisticRegression, RandomForestClassifier, KMeans, etc.) that are likely sufficient for regime classification tasks.
        
    - **PyTorch or TensorFlow (optional):** You can use these if you decide to implement a neural network or a deep learning model. However, given the time constraint and CPU-only environment, sticking to scikit-learn algorithms or at most a simple PyTorch model is recommended. If you do include PyTorch (for example, to implement a simple feed-forward network or an LSTM for sequential data), ensure to use CPU execution.
        
    - **yfinance:** as mentioned, for fetching financial data conveniently.
        
    - **ta-lib (Technical Analysis library):** This library provides technical indicators (moving averages, RSI, etc.). It can be used if you want to derive additional features from price data (like momentum indicators as regime features). Note: `ta-lib` can be tricky to install (requires a C library). Alternatively, `pandas_ta` or writing your own indicator calculations are options. This is optional; core macro features may suffice.
        
    - **LangChain:** a high-level framework to work with language models. In MARAE, LangChain could be used to integrate an LLM (like GPT-4 via OpenAI API) for tasks such as summarizing news or generating a regime narrative. This is a stretch goal tool – include it if you plan to do the LLM overlay. At minimum, ensure it’s installed so that any LLM-related code can run.
        
    - **Statsmodels:** (optional) for statistical tests or models (e.g., ARIMA if you analyze time-series properties of indicators, or use an economic regime switching model).
        
- **Visualization and Dashboard:**
    
    - **Plotly:** for creating interactive charts (time-series line charts of performance, bar charts of allocations, etc.). Plotly integrates with Streamlit and will allow dynamic, hoverable plots in the web app. You can also use matplotlib/seaborn for quick static plots in analysis, but Plotly is preferred for the final dashboard visuals.
        
    - **Streamlit:** the framework to build the dashboard UI. Streamlit lets you create a web app by writing Python scripts with interactive widgets. You will use it to design the front-end where the user can upload a CSV of CMAs and then view outputs (regime classification, allocation, metrics, charts). It’s straightforward to use (e.g., `st.file_uploader`, `st.line_chart`). Streamlit can also serve your Plotly figures (via `st.plotly_chart`).
        
    - **Docker:** We aim to containerize the app for consistency. A Dockerfile will ensure that anyone (including graders) can run your project with all dependencies set up. Use the official `python:3.11` base image. Install needed packages via pip. Expose port 8501 (Streamlit’s default) if needed for deployment. See Dockerfile outline below.
        
- **Development Tools:**
    
    - **Jupyter Notebooks:** Useful for exploratory data analysis and iterative development of your models. You can include some Jupyter notebooks (especially for prototyping or demonstrating certain analysis) in the appendix or repository. Ultimately, core functionality should be modularized into .py files for the app, but notebooks are great for the research phase.
        
    - **VS Code or PyCharm:** An IDE can be helpful for writing and managing code, especially as the project grows to multiple files (data.py, model.py, app.py, etc.). Use linting (Pylint/flake8) to keep code clean, and possibly black for formatting.
        
    - **Git (GitHub):** Version control your code. Follow a **branching policy**: for example, use a `main` (or `master`) branch for stable code, and a `dev` branch for integrating features. Create feature-specific branches (e.g., `feature/data-ingest`, `feature/model-training`) for major additions, then merge into dev after testing. Eventually, merge dev into main for releases (like v1.0 for final submission). Regular commits with clear messages (e.g., “Added feature engineering for macro indicators”) are expected. If working solo, this is still good practice and documents your progress. If working with a partner, use pull requests to review each other’s code.
        
- **Environment Reproducibility:**  
    Maintain a `requirements.txt` file listing all Python packages (with versions) used. This ensures that the grader or anyone replicating your setup can pip install the exact versions. Alternatively, a `Pipfile` or `environment.yml` (if using Conda) could be provided. However, since we plan to use Docker, the Dockerfile itself will act as documentation of the environment.
    

**Dockerfile Outline:** Creating a Docker image helps avoid the “works on my machine” problem. Here’s an outline (see Appendix B for a more detailed snippet):

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any, e.g., ta-lib binary)
# RUN apt-get update && apt-get install -y libta-lib... (if needed for ta-lib)

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project code
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "dashboard_app.py", "--server.enableCORS=false", "--server.headless=true"]
```

When building your Docker image, ensure that any needed non-Python libs (like TA-Lib’s C library or `libgfortran` for SciPy, etc.) are installed. The above uses `python:3.11-slim` which is minimal; you might switch to `python:3.11` full if you hit missing build tools.

- **Performance Considerations:** The environment is CPU-only with a 6-core CPU and we aim for <10 minutes of training time. This means avoid extremely heavy models or huge datasets. Use vectorized operations (NumPy/Pandas) where possible to leverage those cores. If using PyTorch, specify `device('cpu')`. For any code that could be parallelized (like running backtest for many iterations), consider using Python’s multiprocessing or joblib (but careful with Streamlit, which might not like spawning processes; you might pre-compute results instead).
    
- **Testing:** It’s a good habit to include some unit tests or at least manual tests. For instance, verify that your data ingestion function correctly handles missing data, or that the optimization returns weights summing to 1. You can use Python’s `unittest` or simply write assert statements during development.
    

In summary, your development environment will harness Python’s rich data ecosystem. By following best practices in environment management and using the specified stack, you’ll ensure that your project is robust, reproducible, and easy to deploy. If you encounter environment issues (like library incompatibilities), reach out early – the instructor can help resolve those so you don’t lose time.

## 6. Methodology

This section outlines the step-by-step methodology you should follow to build the MARAE system. The process is broken down into logical sub-components, from data ingestion to the final dashboard integration. Adhering to this structure will help ensure completeness and provide checkpoints to gauge progress.

### 6.1 Data Ingestion & Cleaning Pipeline

**Objective:** Gather all required data from source APIs/files and transform it into a clean, merged dataset ready for analysis and feature engineering.

**Steps:**

- **Ingestion Scripts:** Develop Python scripts or notebook sections for each data source:
    
    - _Market Data:_ Use `yfinance` to download historical daily prices for the six ETFs (ACWI, EEM, BNDX, HYLB, DBC, VNQI). Fetch as far back as possible (many start around 2009-2011; ACWI inception was 2008). Also get the historical 3-month T-Bill rate from FRED (which is usually a monthly or weekly series).
        
    - _Economic Indicators:_ Pull key series from FRED, e.g. Financial Stress Index (weekly) and any others (GDP growth quarterly, unemployment monthly, yield curve spread monthly – these can enrich your feature set if used). If using pandas_datareader, it returns a pandas Series which is easy to save.
        
    - _Alt Data:_ If including Google Trends, use PyTrends to fetch weekly data for the chosen keywords. For each term, you’ll get a time series (likely weekly). GDELT or other alt-data might involve reading from a CSV if you managed to download one (e.g., a precomputed sentiment index by month).
        
- **Data Alignment:** Decide on a **time frequency** and reference dates for the analysis. A common choice for macro regime analysis is **monthly** frequency:
    
    - Convert daily ETF prices to monthly returns (or monthly end prices). For example, take the last business day price of each month for all ETFs, then compute month-over-month returns. Alternatively, compute total return including dividends if possible (yfinance’s adjusted prices include dividends).
        
    - Convert weekly indicators to monthly by either sampling end-of-month values or averaging within the month (depending on the indicator’s nature). For instance, for Financial Stress (weekly), taking the last observation of each month might make sense to represent that month’s stress level at the end.
        
    - Ensure the date index is consistent (perhaps use a MonthEnd or MonthStart frequency index).
        
- **Merging:** Merge all data into a single DataFrame where each row is a date (month-end) and columns include:
    
    - Asset returns or prices (e.g., returns_ACWI, returns_EEM, … or you can keep price and calculate returns on the fly).
        
    - Macro indicators (e.g., `StressIndex`, `TbillRate`, etc.).
        
    - Alt-data features (e.g., `GoogleTrend_recession`, `NewsSentiment`).
        
    - Any regime labels if you predefine them (could add a column `RecessionFlag` if you plan to use historical recessions as target, for example).  
        Use an inner join on dates that have all data available, or careful forward/backward fill if some series are missing a month (but generally, using end-of-period should align fine).
        
- **Handling Missing Data:** It’s likely you will have some NaNs at the start or end of series. For example, if ACWI price history starts in 2008 but you have a macro indicator from 1980 onward, you’ll have a lot of NaNs for ACWI pre-2008. You should trim the dataset to where you have all six ETF returns (since allocation can only happen when all assets exist). Similarly, if an alt series doesn’t start until later, consider starting when that data is available or drop that feature if it creates too many holes.  
    For any small gaps (maybe a missing week in an indicator), you can forward-fill or interpolate, but that’s usually minor in monthly data. Document any such filling.
    
- **Outlier Checks:** Scan the data for outliers or sudden jumps that might indicate an error. For instance, sometimes Yahoo Finance data might have a faulty price (split not adjusted, etc.) – though adjusted price usually handles splits/dividends. If you see any monthly return that’s absurd (e.g., +100% not corresponding to real events), investigate and clean if needed (maybe by capping or removing that month if it’s erroneous). Also check consistency: e.g., T-Bill rate should always be positive and single-digit percent; if you see a 99 or -99, that’s an error.
    
- **Feature Creation Placeholder:** While detailed feature engineering is in the next section, some can be done in the pipeline. For instance, you might calculate the yield curve spread (10-year minus 3-month yield) as a series if you pulled both rates. Or compute year-over-year inflation from CPI. If these are to be used, it’s efficient to compute them here.
    
- **Saving Clean Data:** Save the merged, cleaned dataframe to a CSV or pickle (serialized object). This provides a checkpoint – you can reload this in subsequent notebooks or scripts to avoid re-downloading everything each time. For example:
    
    ```python
    merged_df.to_csv("data/merged_monthly_data.csv")
    ```
    
    or use `to_pickle` for faster load with Python objects.
    

**Outcome:** By the end of this step, you should have a single dataset (with a Date index) containing all relevant time-series, cleaned and ready for analysis. It might look like:

|Date|Ret_ACWI|Ret_EEM|...|StressIdx|Tbill|Trend_recession|...|
|---|---|---|---|---|---|---|---|
|2008-12-31|0.025|-0.010|...|0.5|0.01|12|...|
|2009-01-31|-0.080|-0.120|...|1.8|0.01|20|...|
|... and so on.||||||||

Document any transformations in code comments and in your report’s methodology section. This traceability is important for the “research paper” part of the project.

### 6.2 Feature Engineering for Regime Classifier

**Objective:** Create a set of informative features that the AI model will use to classify macro regimes. These features should capture the state of the economy and markets. Good features improve model accuracy and make regimes interpretable.

**Steps/Considerations:**

- **Choosing the Target (Regime Definition):** Feature engineering depends on how you define the regime labels:
    
    - _Supervised (labeled) approach:_ If you decide to label each time period with a regime (e.g., “Recession” vs “Expansion”), you need a basis for those labels. One approach is to use NBER recession dates: label months inside official recessions as 1, others 0. Alternatively, you could define regimes by your own criteria, e.g., “Stress regime” if Financial Stress Index > a threshold AND equity returns are negative, etc. A multi-class scheme could be: {Recession, Normal, Overheating, Stagflation} by combining GDP growth and inflation indicators (this is more complex). **Start simple**, perhaps binary regimes to start. If doing supervised, create the `RegimeLabel` column now (e.g., from NBER data – which is available from FRED as a series “USREC” where 1 = recession, 0 = not).
        
    - _Unsupervised approach:_ If you prefer the model to discover regimes, you won’t have a label column. Instead, you might cluster based on features later. Even in that case, you often do some feature engineering to feed into clustering (ensuring features are scaled, etc.).
        
- **Macro Indicator Features:** Transform raw indicators into model-ready features. Many economic series have trends or seasonality that may not directly correlate with regimes unless transformed:
    
    - For interest rates or yields, often the **yield curve slope** (long minus short rate) is a key feature. If you imported 10yr yield and 3m yield, calculate `slope = yield10y - yield3m`. A very low or negative slope is a known precursor to recessions (feature might strongly predict regime).
        
    - **Growth rates:** Use year-over-year percentage changes for series like Industrial Production or Employment. Absolute levels might not be as indicative as the change. For example, “Annual GDP Growth %” being below 0 is essentially a recession signal.
        
    - **Z-scores of indicators:** Because different indicators have different scales, consider standardizing them. You can take a rolling historical mean and std to normalize (or use full-sample mean/std if not leaking future info too much – careful here). For instance, Financial Stress Index can be used as is (it’s already mean 0 by construction), but something like PMI (Purchasing Managers Index, if used) often is interpreted relative to 50 (above 50 expansion). Converting an indicator to a deviation from a threshold or mean can highlight unusual conditions.
        
    - **Composite indices:** You might combine multiple indicators into one feature. Principal Component Analysis (PCA) is a technique to reduce dimensionality – e.g., combine various economic series into a single “economic activity factor”. This was done in some macro regime research. If you have many features, PCA’s first component might capture the general business cycle (a high value when most things booming, low in recessions). This is optional and can be explored later if needed.
        
- **Market-based Features:** Markets themselves provide regime clues:
    
    - **Trend/Momentum:** Compute trailing returns or moving averages for major assets. For example, the 3-month or 12-month return of ACWI equity could signal bull vs bear conditions. In a regime classifier, you might include `EquityMomentum = rolling_3m_return_ACWI`. If momentum is deeply negative, likely a recession or crash regime.
        
    - **Volatility:** High volatility often coincides with crises. You could use the VIX index (if you fetch it, ^VIX from Yahoo) or simply realized volatility of equity returns (e.g., standard deviation of daily returns within the last month). Having a feature like `EquityVolatility` helps identify stress regimes.
        
    - **Credit Spreads:** If you have HYLB (high-yield bond ETF) and BNDX (investment-grade bond ETF), the difference in their yields or returns might act as a proxy for credit conditions. Alternatively, use the St. Louis Fed’s BBB-AAA corporate yield spread if available. Widening credit spreads are recessionary signals.
        
    - **Inter-asset correlations:** More advanced, but some research suggests correlation spikes during crises (all assets tend to move together when markets panic). Measuring correlation between equities and bonds over a rolling window could be interesting (often, stock-bond correlation goes positive in inflationary regimes vs negative in typical times). This might be too detailed, but worth noting.
        
- **Alternative Data Features:** Incorporate the alt data meaningfully:
    
    - **Google Trends features:** If you pulled terms like “recession”, you might use the raw weekly interest or a monthly average. Better, you could use a **change in search interest** (e.g., if “recession” searches jump this month vs last, that might flag rising concern). If you have multiple terms, each can be a feature or you could combine them (maybe an average of normalized “fear” terms).
        
    - **News Sentiment or Count:** If using GDELT or similar, you might have a monthly average sentiment score. Use that directly as a feature. Or count of certain event types (e.g., number of protests or conflicts in a month globally – could tie to regime if geopolitical risk regime is considered).
        
    - **Social sentiment:** If you managed to get any social media sentiment indicator (e.g., bullishness from some survey or Twitter sentiment), include it.
        
    - **Textual signals via LLM:** If truly integrating NLP, you might classify each month’s major news into categories. For example, you could have a feature “RegimeNarrative” that is categorical (like {“Inflation”, “Crisis”, “Growth”} determined by an LLM reading news). Handling that in a model is complex (need to turn categories into dummy variables or similar). Given time, likely alt-data will be numeric sentiment indices or trend counts.
        
- **Lagging vs Leading:** Consider that some indicators are lagging (unemployment rate lags the economy) and some are leading (stock market tends to lead). To have the model predict ahead, one could shift some features. For instance, to predict next month’s regime, you might use this month’s data. But if you are classifying the _current_ regime, you use current concurrent data. Decide if your goal is _real-time classification_ (detect regime at time _t_ using data up to _t_) or _forecasting_ (predict regime at _t+1_). Real-time classification is easier and still useful (you allocate accordingly). Forecasting one step ahead is harder but could be attempted with a lag on features. For initial scope, classify current regime.
    
- **Scaling features:** Many ML algorithms (like KMeans or logistic regression) work better if features are scaled comparably. After you assemble the feature matrix, scale features to 0 mean, unit variance (standardization) or 0-1 range (min-max). You can use `sklearn.preprocessing.StandardScaler` on your training data. Remember to apply the same scaler to new data in live use (so save the scaler parameters).
    
- **Feature Selection:** Too many features for too few observations can cause overfitting. If you have monthly data for ~15 years (~180 samples) and you introduce 20 features, a complex model might overfit. Try to focus on the **most informative features**: e.g., yield curve, equity momentum, stress index, maybe one sentiment measure, etc. You can always experiment with adding/removing features and see impact on validation performance. Simpler models (like logistic regression) also allow checking p-values or coefficients to gauge importance, which can guide trimming features.
    

By the end of feature engineering, you should have:

- `X` features matrix (for each time period, values for each feature).
    
- `y` target vector (if supervised with labeled regimes).
    

For example, an `X` row might be: `[GDP_growth=2.5, YieldCurve=0.5%, EquityMomentum=-3%, StressIdx=1.2, Search_recession=80, Sentiment=-2.0]` and the corresponding `y` could be `Regime=1` (meaning recession regime). In unsupervised, you’d just have the `X` and you’ll cluster or otherwise process it.

Document your feature list in your report. It might look like:

- Yield curve slope (10Y – 3M yield, %)
    
- Equity 3-month trailing return (%)
    
- Financial Stress Index (std devs from norm)
    
- Google Trends for “recession” (index)
    
- etc.
    

Each feature has an economic rationale that you can explain when writing up results (e.g., “the yield curve was one of the most important features, aligning with economic theory that inverted yield curves predict recessions” – if your model finds that).

Finally, split your data into training and testing sets _chronologically_ (if doing supervised). For example, train on 2008–2018, test on 2019–2023 (including the pandemic period as an out-of-sample test). Do not randomly shuffle since that leaks future info. A **walk-forward validation** approach will be described in section 6.4.

### 6.3 Model Selection & Hyper-Parameter Tuning

**Objective:** Select an appropriate AI/ML model to identify or predict macro regimes, and tune its parameters for best performance without overfitting. The model could be a classifier (for discrete regimes) or a clustering/segmentation method.

**Model choices:**

- **Baseline: Logistic Regression (for binary classification).** This is a good starting point if you have a binary regime indicator (e.g., recession vs non-recession). It’s simple, fast, and gives interpretable coefficients (which features are positively or negatively associated with being in a recession regime). It also naturally outputs a probability (0-1). Hyper-parameters: mainly regularization strength (`C` in scikit-learn) to prevent overfitting if features are many. You might do a cross-validated search for `C` (smaller `C` means stronger regularization).
    
- **Multi-class classification:** If you defined more than two regimes (say 3 classes: expansion, slowdown, recession), you can use multinomial logistic regression or a tree-based model. Logistic with multi-class will handle it via one-vs-rest internally by default.
    
- **Tree-based Models:** Decision trees or ensemble methods like **Random Forest** or **Gradient Boosting (XGBoost/LightGBM)** can capture non-linear relationships and interactions (maybe certain combinations of features strongly indicate a regime). For example, random forest could learn rules like “if yield curve < 0 and stress high then recession”. These models have hyper-parameters: number of trees, max depth, etc. Using scikit-learn’s `RandomForestClassifier`, you might tune `n_estimators` (50,100,200) and `max_depth` (e.g., None (full depth) or restrict to avoid overfit). Random forests handle overfitting by averaging many trees, but if your sample is small, too deep trees can still overfit.
    
- **Support Vector Machines (SVM):** An SVM classifier with an RBF kernel can capture non-linear boundaries in feature space. However, SVMs can be less interpretable and might be overkill for small data. They have hyper-params C (regularization) and gamma (kernel width) to tune. Given possibly limited data points (monthly data yields limited training samples), an SVM might actually perform decently by finding a margin between regimes. But tuning can be tricky, and they don't give probabilities natively (though you can calibrate or use SVM in a probabilistic mode).
    
- **Unsupervised Clustering (if no labels):** If you choose not to predefine regimes, you could use **K-Means clustering** on your features to group time periods into, say, 2 or 3 clusters. Hyper-param would be number of clusters `k`. The interpretation of clusters would be based on looking at their centroids (e.g., cluster 1 might have high stress, cluster 2 low stress, cluster 3 high inflation maybe). Another method is **Gaussian Mixture Models** (which give soft cluster assignments and can fit different shapes). If going unsupervised, you’ll need to later map those clusters to meaningful regime names (“Cluster A corresponds to recessions”). This approach is valid but ensure you justify the choice, as typically some label (like historical recessions) is known.
    
- **Sequence Models:** If you aim to predict regimes ahead, a sequence model like an **LSTM (Long Short-Term Memory)** network could be used to learn temporal patterns in indicators. But given our data frequency (monthly) and limited length, a complex LSTM might be unnecessary. Additionally, training an LSTM on CPU for a small dataset might not even be effective (risk of overfitting is high too). Unless you have a large sequence (like daily data for many years with fine changes), sequence networks are probably not needed.
    
- **Reinforcement Learning:** Mentioned as a stretch for allocation, but not for classification. RL would be more for directly learning allocation without explicit regimes, which is advanced and separate from classification metrics.
    

For the main classification task, a reasonable plan: Start with **logistic regression and random forest** as two baseline models. Evaluate how they do (via cross validation or a fixed train/test split). If performance is inadequate or you see evidence of non-linear boundaries, try a more powerful model like XGBoost.

**Hyper-parameter Tuning:**

Use scikit-learn’s **GridSearchCV** or **RandomizedSearchCV** for systematic tuning. However, note that cross-validation in time series should be done carefully:

- Use a **time series split** (e.g., `sklearn.model_selection.TimeSeriesSplit`) which preserves order: it might train on first 70% time, validate on next 10%, etc., sequentially. Or do manual expanding window: e.g., train 2008-2015, validate 2016-2017; train 2008-2017, validate 2018-2019, etc., then test on 2020-2021.
    
- Alternatively, since you might have very limited data, a single train/validation split may suffice (like train on first 2/3, validate on next 1/3, then final test on last few years).
    

Parameters to tune depend on model:

- Logistic: `C` (regularization). Also consider if you want L1 or L2 penalty (L1 can zero out some coefficients for feature selection).
    
- Random Forest: `max_depth`, `min_samples_leaf` (to prevent overfitting small leaves), and `n_estimators`. Tuning too many params is not needed; a robust approach: set `n_estimators=100` (more trees generally better until diminishing returns), then tune `max_depth` (like [3, 5, None]) and maybe `min_samples_leaf` (like [1, 5, 10]).
    
- XGBoost/LightGBM: learning rate, `n_estimators`, `max_depth`, etc. These models can get a lot of tuning; try defaults first then adjust if needed.
    
- KMeans (if used): `n_clusters` is key. You might try 2 through 5 and see which seems to partition data meaningfully (using inertia or silhouette score).
    

**Model Training:**

During training, ensure to do the following:

- If supervised, address any class imbalance. Recession months are usually far fewer than expansion months. A classifier might just predict “no recession” always and be right most of the time but miss the recessions. Strategies:
    
    - Use balanced class weights (scikit-learn logistic and RF allow `class_weight='balanced'` which weights classes inversely to frequency).
        
    - Over-sample the minority class in training (though with time data this is tricky because you can't really manufacture more recession periods).
        
    - Evaluate using metrics beyond accuracy (e.g., recall for recession class, or F1 score).
        
- Use appropriate scoring metric in GridSearchCV: for binary regime, perhaps use `roc_auc` or `f1` rather than raw accuracy, to ensure the model learns to predict the rare class.
    

After selecting a model and tuning, **lock it down**. That is, once you find the best hyper-params using training/val, you then retrain the model on the entire training set (e.g., 2008-2019) with those params, and finally test on the holdout (e.g., 2020-2023). This final test gives an unbiased estimate of performance.

**Outputs of model training:**

- If classification: confusion matrix, accuracy, precision/recall, etc. For example, how many recessions did it correctly identify vs missed or false alarms. These metrics will be part of your Evaluation (section 9).
    
- If clustering: cluster assignments for each time point and interpretation of clusters (maybe assign them labels by comparing to known events).
    

You will also integrate the model into the backtesting logic next. Essentially, for each period in backtest, the model will give a regime classification or probability which informs the allocation.

Document your chosen model and parameters. Explain why it’s suitable (e.g., “Random forest was chosen for its ability to capture nonlinear threshold effects; max depth was limited to 4 to maintain generalization”). Also, keep notes on any interesting finding during tuning (like “Google Trends feature significantly improved recall for recessions” or “Feature X was dropped due to high correlation with Feature Y causing multicollinearity issues in logistic regression”).

### 6.4 Back-testing Framework & Walk-Forward Validation

**Objective:** Rigorously evaluate the performance of the regime-driven allocation strategy by simulating how it would have historically performed. This involves stepping through time (“walk-forward”) and at each step using only past data to make allocation decisions, then comparing those decisions to actual market outcomes.

**Designing the Backtest:**

- **Backtest Period:** Decide on the period over which to simulate the strategy. Likely from around 2010 (or earliest date where all assets exist and model can be trained) up to the latest data (2025). You might reserve the last few years as a “test” to report metrics, but you can also incorporate them into backtest and then just note how the latter part performed. The backtest effectively is an out-of-sample test if you ensure not to use future info.
    
- **Frequency of Rebalancing:** Typically, strategic or tactical allocations are updated monthly or quarterly. For our purposes, monthly rebalancing aligns with monthly data/regime classification. So assume at the end of each month, you assess the regime and set weights for the next month. That means:
    
    - On, say, Jan 31, 2020, you have data up to Jan 31. You use your model (trained on data up to that point) to classify the regime for February 2020 (or at least for end of Jan, determining stance going into Feb). You then allocate accordingly and “hold” that portfolio through February. At end of February, update again.
        
    - This is a **rolling/online prediction** scenario. If your model is fixed (not updating parameters), you use the same model throughout. Alternatively, you might choose to **retrain** the model periodically expanding the window (which could potentially improve it as it learns from new events like COVID crash). A pragmatic approach: freeze the model after initial training, to isolate performance of static model. A more realistic approach: update model every year with new data (walk-forward training). The latter is ideal but can be time-consuming. Perhaps do one initial backtest assuming static model, then mention that in real-world you’d update it.
        
- **Walk-Forward Validation Approach:** One robust way to avoid lookahead is:
    
    - Split data into training set (e.g., 2008-2015) and test set (2016-2024). Train model on 2008-2015. Then simulate from 2016 onward in a loop:
        
        - For Jan 2016 (first month of test), use model (trained on 08-15) to predict regime for Jan 2016 -> allocate weights -> record return for Jan 2016.
            
        - At Feb 2016, optionally retrain model by including Jan 2016 (so now trained 08-16) or keep model same. Many academic studies use expanding window: model is retrained as new data comes, representing learning process. However, if model is stable and not data-hungry, you might skip frequent retraining.
            
        - Continue month by month until end 2024.
            
    - This gives a sequence of portfolio returns which you can compare to benchmarks.
        
    - The expanding window retraining ensures model can adapt to regime patterns that weren’t in initial train (e.g., COVID pandemic regime might be new if initial train ended 2015). If not retraining, you rely on model generalizing.
        
- **Allocation Decision at each step:** Based on the regime output:
    
    - If model produces a discrete regime label (e.g. “recession” or “expansion”), you should define what portfolio corresponds to that label. For instance:
        
        - If regime = recession -> use a “conservative” allocation (maybe more bonds, less stocks).
            
        - If regime = expansion -> use an “aggressive” allocation (more stocks, maybe more commodities if expecting inflation).
            
        - You could set up a simple mapping table: e.g., in recession: {ACWI 10%, EEM 0%, BNDX 40%, HYLB 10%, DBC 5%, VNQI 5%, Cash 30%}. In expansion: {ACWI 30%, EEM 10%, BNDX 20%, HYLB 20%, DBC 10%, VNQI 10%, Cash 0%}. These numbers can be informed by intuition or by optimizing for each regime separately (see below).
            
    - If model produces a probability (say P(recession) = 20%), you could use that in a more continuous way – maybe a weighted average between a recession portfolio and expansion portfolio. Or set a threshold (if P > 50% treat as recession regime).
        
    - If using an optimization-based approach every time (mean-variance with CMA), then the regime might influence the inputs or constraints. For example, if your model indicates high-risk regime, you might reduce the expected returns used or increase the risk aversion parameter in the optimization (leading to a more conservative allocation). Or simpler: have two sets of CMA adjustments: one base and one for crisis regime.
        
    - **Portfolio Optimizer integration:** Ideally, integrate the outcome of Section 6.5 here. For now, conceptually: at each time, you feed current data (and maybe regime classification) into your optimizer to get weights.
        
- **Transaction Costs and Turnover:** For our simulation, we can ignore transaction costs (assume frictionless rebalance) for simplicity. But do track turnover (how often weights change dramatically) as an indicator – e.g., if the strategy trades too much, in reality costs would eat performance. With monthly moves, it might not be too high.
    
- **Benchmark Comparison:** Decide on one or two benchmarks to compare performance:
    
    - A static **60/40 portfolio** (60% ACWI, 40% BNDX) rebalanced monthly is a classic moderate allocation.
        
    - An **equal-weight** portfolio of our six risky assets (so excluding cash or including cash at, say, 0%). Equal weight ensures diversification without any timing.
        
    - Or compare to pure ACWI (100% global equities) to see if strategy avoided drawdowns but gave up some return or not.
        
    - Also a risk-parity static portfolio could be a benchmark (but since that’s part of what our strategy might do in one regime, maybe not needed separately).
        
    - Compute these benchmark returns as well for the period.
        
- **Metrics to compute:** For the portfolio (and benchmarks), compute key performance metrics after the backtest:
    
    - CAGR (annualized return)
        
    - Annualized volatility
        
    - Sharpe ratio (excess return / vol, using risk-free from T-bill)
        
    - Maximum Drawdown (worst peak-to-trough)
        
    - Calmar ratio (CAGR / Max Drawdown, useful to gauge risk-adjusted return).
        
    - Perhaps Sortino (like Sharpe but only downside vol).
        
    - Also count how many trades (rebalance changes) and maybe average turnover % per month.
        
    - These will be used in Section 9 to evaluate success.
        
- **Visualization:** Plot the **equity curve** (cumulative growth of $1) of the strategy vs benchmark over time. This is a powerful visual to see if the strategy protected during e.g. 2020 or 2022, etc. Plot the asset allocation over time as an area chart to illustrate how it shifts in different regimes (this also double-checks that the regime changes make sense, e.g., see high bond allocation coincide with official recession periods).
    

**Walk-forward example:**  
Suppose Jan 2020 model says “not recession” -> allocation = 70% stocks, 30% bonds. In Feb 2020 data (which includes Feb’s market drop), maybe by end Feb the model flips to “recession” -> in Mar 2020 allocation becomes 30% stocks, 70% bonds. The portfolio would then hopefully have mitigated some losses. Continue through 2020, perhaps model flips back out of recession by mid 2020, etc. You can then analyze those decisions: did it flip at right times or was it late/early?

**Avoiding Look-Ahead Pitfalls:**  
Be absolutely sure that when predicting or allocating for time T, your model is not trained on time T or beyond. If you retrain monthly, do it using data up to T-1. This might require writing a loop that at each iteration trains model on history-to-date (which can be slow but feasible for monthly data and simple models). Alternatively, train once and don’t retrain, which is simpler to implement, but then your model doesn’t learn from new patterns.

Given our scale, a compromise:

- Use data up to 2018 to train final model.
    
- Use that model to backtest 2019-2024 out-of-sample (no retraining).  
    This will test generalization. But if you suspect the model would have improved by retraining (especially after seeing COVID, etc.), you can note that.
    

After running the backtest, gather results and analyze:

- Did the strategy outperform the benchmark on a risk-adjusted basis?
    
- Did it achieve the intended risk mitigation in bad times and participate in good times?
    
- Are there periods where it made a wrong call (false regime signal leading to underperformance)?
    

This analysis forms the basis of the results you’ll present and the evaluation of success.

### 6.5 Portfolio Optimiser Logic (include CMA mapping)

**Objective:** Develop the logic to convert the model’s regime signals and the capital market assumptions (expected returns, etc.) into concrete portfolio weights for the assets.

There are two general approaches:

1. **Rule-based allocation per regime**, using predetermined weights.
    
2. **Optimization-based allocation**, using mean-variance or risk-parity calculations with the CMA inputs, possibly adjusted by regime.
    

We’ll outline both and you can implement either or a hybrid.

- **Mapping CMA to Assets:** First, ensure you can interpret the user’s CMA CSV. It likely has rows like “Asset Class, ExpReturn, Volatility, CorrelationMatrix/...”. For simplicity, say it lists the 6 asset classes corresponding to our ETFs plus cash, with 5-year expected return (annual %) and maybe vol (annual %). Map those to our assets:
    
    - ACWI -> use “DM Equities” expected return (or “Global Equity” if given).
        
    - EEM -> “EM Equities” expected return.
        
    - BNDX -> “Global IG Bonds” expected return.
        
    - HYLB -> “Global High Yield” or “Credit” expected return.
        
    - DBC -> “Commodities” expected return.
        
    - VNQI -> “Global REITs” or “Real Estate” expected return.
        
    - Cash -> 3M T-Bill current yield (could also be given or just use last T-Bill data).  
        If the CSV doesn’t perfectly align, you may need to make an assumption or ask the user to match these categories.
        
    
    _Example:_ BlackRock’s 2023 CMA might say: Equities 7%, EM Equity 8%, Treasuries 2%, Corporate IG 3%, HY Credit 5%, Commodities 4%, Real Estate 6%. You’d pair accordingly. Document in code or comments how you map.
    
- **Mean-Variance Optimizer Implementation:** Using the expected returns vector from CMA and an assumed covariance matrix, find optimal weights:
    
    - If the CMA file provides an entire covariance matrix or vol + correlation, use it. E.g., if vol for each and a correlation matrix is given, you can construct covariance = diag(vol) * corr * diag(vol). If not, you may have to estimate covariance from historical returns (e.g., last 5-year sample cov). Historical cov is fine for our use if CMA doesn’t give one.
        
    - Use `numpy` or `cvxpy` to solve the optimization. Since it’s a small problem (7 assets), even a simple grid search or SciPy optimizer could work. However, `cvxpy` would solve a quadratic programming easily with constraints.
        
        - Variables: w[0..6] for weights.
            
        - Constraints: sum(w)=1, w>=0 (no short). Possibly w_i <= some max (like 50%) if you want to avoid concentration.
            
        - Objective: maximize `w^T mu - lambda * w^T Sigma w`. If you want a specific risk target, you can maximize Sharpe: maximize `(w^T mu - rf) / sqrt(w^T Sigma w)`. But that’s non-linear ratio. Instead, often one fixes a return target and minimizes variance or vice versa.
            
        - Simpler: maximize `w^T mu` subject to `w^T Sigma w <= target_var`. Choose target_var appropriate to risk tolerance (maybe corresponding to 10% vol).
            
        - Or minimize variance `w^T Sigma w` subject to `w^T mu >= target_return`. If CMA says an unconstrained optimum might heavily weight high return asset (like EM equity), such constraints keep diversification.
            
        - Because we want an allocation appropriate for each regime, you might adjust `mu` input or constraints for regime:  
            _In an expansion regime:_ use the raw CMA expected returns (which are long-term averages assuming economies normal). This will naturally favor equities since they have higher returns per CMA.  
            _In a recession regime:_ one could either reduce equity expected returns or increase risk aversion (lambda). One heuristic: scale down `mu` for risky assets (equities, HY) by some factor when in recession regime (assuming their 5-year outlook might still be 7%, but near-term they might not achieve that). Alternatively, simply solve the optimization normally but then **override** or cap equity weights to a lower bound when in recession.  
            Another approach: have two separate sets of CMA – one baseline, one “stressed” with lower equity returns (or higher equity vol). This might be over-complicating; you can also handle via rule-based shift.
            
    - Implementation detail: If using `numpy.linalg` without cvxpy, note that without shorting the analytic solution of mean-variance (the classic formula) doesn’t apply due to inequality constraints. So you either need to do a small quadratic programming. There is a library **PyPortfolioOpt** that can do mean-variance with constraints easily – you may use it to avoid reinventing the solver.
        
- **Risk-Parity Implementation:** If you choose risk parity (equal risk contributions):
    
    - There’s no closed form. You can use an iterative algorithm: e.g., the algorithm by Maillard et al. (2010) that iteratively adjusts weights. Or use `cvxpy` to solve minimize difference between contributions.
        
    - However, risk parity doesn’t depend on expected returns at all. If you want to incorporate CMA, risk parity alone might not use them (it’s more a risk distribution choice).
        
    - Perhaps a hybrid: you could start with risk parity weights as a baseline and then tilt weights slightly based on expected returns (like a Black-Litterman idea where the baseline is risk parity neutral weights).
        
    - Given time, implementing risk parity solver might be too much unless you use PyPortfolioOpt which has a risk parity method.
        
- **CVaR Optimization:** This would involve scenarios. You could generate scenarios from historical returns or assumed distributions, then use linear programming as per Rockafellar & Uryasev. This is advanced and likely not needed unless aiming for extra credit.
    
- **Mapping Regime to Strategy:**
    
    - _Approach A (regime-specific fixed allocations):_ Define two sets of target weights: one for “risk-on” (expansion) and one for “risk-off” (recession). Possibly a third for a middle regime if you have. The values can be set by intuition or by running the mean-variance optimizer under different assumptions. For example, you might run mean-variance with high risk aversion to get a conservative portfolio (that can be the recession portfolio), and with lower risk aversion for an aggressive portfolio (expansion portfolio).
        
    - Then in backtest, if model says recession, use the conservative weights, if expansion, use aggressive weights. This is straightforward and interpretable (basically hard-coded).
        
    - _Approach B (dynamic optimization each time):_ At each month, feed the current expected returns (which might not change often if using static CMA) into an optimizer. The regime could influence a constraint. For instance: “if regime is bad, require portfolio volatility <= 8% (so it will allocate more to bonds/cash); if regime good, allow up to 15% vol (so it can allocate more to stocks).” This way, the optimizer outcome varies by regime. Or directly alter expected returns: some use “regime-dependent expected returns” approach where, say, in a recession regime the effective expected return of equities is scaled down by 50% in the optimizer, reflecting pessimism. The optimizer then yields a more defensive mix.
        
    - If you use static CMA and just run an optimizer blindly each month, it would yield the same weights every time (since inputs didn’t change), which isn’t responsive. So you need regime to modify something. That could be expected returns, risk aversion, or constraints as mentioned. Figure out a rule that ties regime -> parameter. (This could be informed by research or scenario analysis. For example, BlackRock might publish how they’d tilt portfolios if they foresee a recession: often overweight government bonds, underweight equities, etc., which you can mimic.)
        
- **Inclusion of Cash:** One important asset is cash. In a defensive regime, the model might want to allocate to 3M T-Bills (cash) to avoid losses. Make sure your optimization includes cash as an asset with expected return equal to yield (like ~1-3% historically) and near-zero variance. Cash provides safety at the cost of lower return.
    
- **Leverage and Shorting:** We will assume no leverage (weights sum to 1, no borrowing) and no short positions (weights >= 0). This keeps it realistic for a personal or small fund portfolio. If you did risk parity fully, often they leverage bonds to reach a higher return – but we won’t do that here.
    

**Implementation Tips:**

- Write a function `optimize_portfolio(expected_returns, cov_matrix, risk_aversion or target_risk)` that returns weights. This can use cvxpy or other solver.
    
- Test this function on the static CMA to see if results make sense (e.g., does it allocate more to assets with high return per risk?).
    
- Possibly incorporate regime in that function or outside logic deciding what `expected_returns` to feed.
    

**Example:** Suppose expected returns (annual) are: ACWI 7%, EEM 8%, BNDX 2%, HYLB 5%, DBC 4%, VNQI 6%, Cash 2%. Covariances from history show stocks are volatile and somewhat correlated, HYLB correlated with stocks, etc.

- In normal times, optimizer might give something like: ACWI 30%, EEM 5%, BNDX 25%, HYLB 10%, DBC 5%, VNQI 15%, Cash 10%. (just hypothetical).
    
- If we tell optimizer “we’re in recession, reduce equity returns assumption to 3% and HY to 2%” (pessimistic), while leaving bonds at 2%, then equities no longer look much better than cash. The optimizer might then shift to: ACWI 10%, EEM 0%, BNDX 40%, HYLB 5%, DBC 0%, VNQI 5%, Cash 40%. A much more conservative stance.
    
- These two can serve as distinct regime portfolios.
    

The logic tying all this together in code during backtest is: for each period, determine regime -> adjust inputs -> optimize -> get weights -> compute portfolio return with those weights applied to asset returns of that period.

### 6.6 Dashboard UX Wireframe & API End-points

**Objective:** Design a user-friendly dashboard interface to interact with MARAE, and outline any API endpoints needed for back-end computations. The dashboard will allow an end-user (or evaluator) to input assumptions and view results (allocation and performance), all without needing to run code manually.

**Dashboard Structure (Wireframe):**

Think of the dashboard in sections or tabs. In Streamlit, you don’t have native multi-tab, but you can simulate with radio buttons or simply a vertical layout. A logical structure:

- **Title and Introduction:** At the top, display the project title “AI-Driven Macro Regime Allocation Engine (MARAE)” and a brief description. E.g., “This dashboard classifies the current macroeconomic regime and recommends a portfolio allocation across global assets. Adjust inputs below and view the results.”
    
- **Sidebar (Inputs):** Streamlit’s sidebar can hold input controls:
    
    - **Upload CMA CSV:** Use `st.file_uploader("Upload CMA (CSV)", type=['csv'])`. The user can provide their own expected return assumptions. On upload, parse it and display the values to confirm. If not provided, you can use a default set (maybe BlackRock’s or an example).
        
    - **Select Model/Strategy Options:** Possibly a dropdown to choose which model to use (if you implemented multiple). Or a toggle for “Use alternative data (on/off)” to include or exclude alt features (if you want to demonstrate impact).
        
    - **Risk Preference Slider:** If you want user to adjust risk-aversion (for optimization) or perhaps choose one of preset regimes manually to see that allocation. For instance, a radio: “Regime override: [Auto (model-based), Force Conservative, Force Aggressive]” to let user see the difference.
        
    - **Date Range Picker:** If you allow the user to specify the backtest period or focus (Streamlit has a date_input that can take a range). But if not needed, you can just use full available range by default.
        
    - **Run Button:** If computations are heavy, you might not want to run everything automatically on each input change. A “Run Simulation” button can trigger the calculations and output generation.
        
- **Main Panel (Outputs):** Use columns or tabs:
    
    - **Current Regime & Allocation:** At the very top, after user input, display the latest identified regime. E.g., “**Detected Regime:** Expansion (probability 85%) as of Dec 2024.” Then show **Recommended Allocation** perhaps as a table or pie chart. Streamlit can display a pandas DataFrame of assets and weights, or better yet, a Plotly pie chart with slices for each asset weight.
        
    - **Performance Charts:** A section with an interactive line chart showing cumulative performance of the strategy vs benchmark. This would likely be pre-computed from backtest for whatever period. Use Plotly to allow hover to see values. If multiple benchmarks, include them in the same chart for comparison.
        
    - **Allocation Over Time:** An area chart where different colors represent asset weights over the backtest timeline. This shows how the allocation shifted, and you might annotate significant regime changes. E.g., see bonds (area) swell during 2020, then shrink as equities increase after.
        
    - **Statistical Metrics:** Perhaps a small table summarizing key metrics of the strategy and benchmark (CAGR, Sharpe, max drawdown, etc. from section 9 metrics). This provides a quick evaluation of how the strategy did.
        
    - **Model Diagnostic (optional):** If you want to be transparent, you could show some diagnostic like “Feature importance” bar chart from the model, or a confusion matrix of regime prediction historically. This is more for the report but could be a nice addition for interactivity (like toggling which features are included and seeing how performance changes, etc., but that may be too much for now).
        
- **User Guidance and Interpretation:** The dashboard should include text to guide the user through the content. For example, above the charts, write narrative: “The chart below shows how a $100 investment would grow under MARAE’s strategy vs a 60/40 benchmark. Notice that during stress periods (shaded red), MARAE’s line declines less, illustrating its defensive shifts.”
    
- **Color coding regimes:** If possible, highlight regime periods on charts (Plotly allows adding shapes or colored regions given date ranges). If your model identified specific periods as recessions, you could shade those on the performance chart for illustration.
    

**Interactivity considerations:** Streamlit runs top-to-bottom on each interaction. So structure your code to cache expensive computations. You can use `@st.cache_data` or similar for loading data and maybe for running backtest if CMA input is the same, etc. But since our data is not huge, it should be fine.

**API Endpoints:**

If we imagine decoupling the front-end (dashboard) from back-end logic, or if in future one wanted to integrate MARAE into another system, we’d define some API endpoints. For the scope of this project, you might not need to implement a separate API since Streamlit is doing both. But it’s useful to conceptually outline:

- **POST `/upload_cma`** – Accepts a CSV (or JSON payload of expected returns). Stores it server-side or returns a standardized JSON of assumptions. In a simple integration, the dashboard just handles this internally.
    
- **GET `/get_allocation`** – Returns the recommended allocation weights given current detected regime (server would run the model internally). Could accept query params if wanting to specify a date or custom scenario.
    
- **GET `/backtest`** – Runs the backtest with current settings and returns performance data (e.g., JSON with arrays of dates, strategy returns, benchmark returns).
    
- **GET `/current_regime`** – Returns current regime classification and probability.
    

These API endpoints would allow, for instance, another application or a command-line tool to query MARAE’s results. However, implementing them would require using a framework like FastAPI or Flask. Since Streamlit is fine for our use, we might skip actual API coding. Instead, maybe simulate by structuring code so that functions can be called programmatically (which is effectively like having an API behind the scenes).

**Responsiveness and UX:**  
Make sure the dashboard doesn’t feel too sluggish. If you compute backtest on every run but the data is static, that’s okay as it’s quick. If you allow the user to change an assumption and want to show updated performance, that’s more dynamic. But likely, CMA changes wouldn’t drastically change historical performance (since strategy logic might shift if expected returns differ – perhaps yes, if user is very bullish on EM, the optimizer would allocate more to EEM and backtest would reflect that). So it might be nice to indeed recompute the allocation strategy with new CMA on the fly and update charts. That’s doable if coded efficiently.

**Layout example in code (pseudo):**

```python
st.title("AI-Driven Macro Regime Allocation Engine (MARAE)")
st.write("Upload your capital market assumptions and see the recommended asset allocation...")

cma_file = st.sidebar.file_uploader("Upload CMA CSV", ...)
risk_pref = st.sidebar.slider("Risk Aversion (0=Aggressive, 10=Very Conservative)", 0, 10, 5)
run_button = st.sidebar.button("Run MARAE")

if run_button:
    # Data and model preparation
    cma = load_cma(cma_file) if cma_file else default_cma
    regime = model.predict(current_features)  # current regime
    weights = allocate_portfolio(cma, regime, risk_pref)
    st.subheader(f"Detected Regime: {regime} regime")
    st.write("Recommended Allocation:", weights_df)
    st.plotly_chart(allocation_pie_chart(weights))
    # Backtest with given settings
    perf = run_backtest(cma, model, risk_pref)
    st.plotly_chart(performance_chart(perf))
    st.write(metrics_table(perf))
```

This pseudocode shows the flow: user inputs, press run, then display results.

**Error handling:** If user’s CMA file is missing an asset or has weird format, show a warning (“Could not parse CMA, using default assumptions.”). Always ensure some output is shown.

**Aesthetics:** Use Streamlit formatting to make it polished:

- You can use `st.markdown` with custom CSS or at least bold text and maybe colored text to highlight certain things.
    
- Ensure plots have titles, axis labels, and legends clearly indicating which line is which.
    
- Use consistent colors for assets (e.g., ACWI always blue, EEM red, etc., in allocation chart).
    

**Conclusion section:** Perhaps at bottom, a small note: “_Note:_ This tool is for educational purposes, illustrating how macro regime analysis can inform portfolio allocation. It is not financial advice.”

By designing the dashboard well, you create a **self-contained demonstration** of your capstone. The professor or peers can play with it, see how changing assumptions affects allocation (for example, if they upload a CMA that says equities have only 2% expected return (very pessimistic), the engine should allocate a lot less to equities).

In summary, the dashboard should make the complex inner workings of MARAE accessible and interactive. It is the culmination of your project, showing off the model’s decision-making and the resulting investment outcomes in a clear way. Plan the user experience to be logical: first they provide inputs, then they see what the model thinks (regime) and what it suggests (allocation), then evidence that this suggestion is sound (historical performance).

Now that methodology is covered, the next sections will ensure you stay on track time-wise and know how your work will be evaluated.

## 7. Project Timeline & Milestones (15 weeks)

Below is a week-by-week timeline in a Gantt-style table, outlining key activities, expected deliverables, and acceptance criteria for each milestone. The plan assumes a 15-week semester with ~10 hours of work per week (adjust as needed). Regular progress tracking is crucial; meet interim goals to avoid last-minute rush.

|**Week**|**Key Activities & Milestones**|**Est. Hours**|**Deliverables / Acceptance Criteria**|
|---|---|---|---|
|**1**|**Project Kickoff & Planning**- Review project brief and clarify scope- Initial literature review (macro regimes, CMA reports, similar projects)- Set up Git repository and environment (Python libs, etc.)|8 hrs|_Deliverables:_ Project plan outlining tasks & timeline (Gantt chart refined). Environment setup confirmed (can run simple code)._Acceptance:_ Instructor approval of plan; repo initialized with README.|
|**2**|**Data Acquisition – Market Data**- Write scripts to fetch ETF price histories (ACWI, etc.) from yfinance- Fetch 3M T-Bill rate and a couple of key FRED indicators (e.g., STLFSI)- Store raw data files (CSV)|10 hrs|_Deliverables:_ Raw data files for prices & rates; Jupyter notebook showing price plots and basic stats._Acceptance:_ Data covers required date range; no critical gaps.|
|**3**|**Data Acquisition – Alt Data** (Optional heavy focus)- Retrieve Google Trends data for selected terms- Download CFTC COT data (if using) and/or any Kaggle datasets- Begin cleaning alt data series|10 hrs|_Deliverables:_ Alt data series saved (e.g., CSV of weekly “recession” search interest)._Acceptance:_ Alt data successfully loaded and roughly aligned with dates. (If skipping alt data, allocate time to extra literature review or initial modeling.)|
|**4**|**Data Cleaning & Merger**- Integrate all data sources in a single DataFrame (monthly frequency target)- Handle missing values (align on common date index)- Compute additional fields if needed (e.g., yield curve from rates)|10 hrs|_Deliverables:_ `merged_monthly_data.csv` (or similar), and printout of head/tail in notebook._Acceptance:_ Clean dataset with all necessary columns, covering at least ~12+ years of monthly data, ready for feature engineering.|
|**5**|**Exploratory Data Analysis (EDA)**- Plot time series of indicators and asset returns- Mark known recessions on charts (to visually identify patterns in features during regimes)- Basic correlation analysis between features and with regime label (if known)|8 hrs|_Deliverables:_ EDA notebook slides/plots (not formal deliverable, but for understanding). Possibly a short memo on initial findings (e.g., “stress index spikes in 2008 and 2020, correlating with equity drawdowns”)._Acceptance:_ Clear understanding documented of how each feature behaves and potential predictive power.|
|**6**|**Feature Engineering**- Create regime label (e.g., using NBER recession dates) in data- Compute derived features: % changes, z-scores, moving averages, etc., as planned- Normalize/scaling features (prepare for modeling)|10 hrs|_Deliverables:_ Updated dataset with new feature columns and target label. Code to generate features (function or notebook)._Acceptance:_ All chosen features computed correctly (verify e.g., yield curve inversion occurred when expected, etc.). Feature set finalized for modeling.|
|**7**|**Model Training – Initial**- Split data into training/validation sets (time-based)- Train baseline model(s): Logistic Regression and one non-linear (RF or XGB) on training set- Evaluate on validation (accuracy, confusion matrix)|10 hrs|_Deliverables:_ Model training notebook with results (e.g., logistic coefficients, RF feature importance, validation metrics)._Acceptance:_ At least one model can reasonably identify past recessions (e.g., >70% accuracy or similar on validation). Decision made on which model to pursue/tune.|
|**8**|**Hyper-Parameter Tuning**- Use cross-validation (time-split) to tune model hyper-params (e.g., RF depth, etc.)- Prevent overfit by regularization or limiting complexity- Finalize chosen model and retrain on full training data|8 hrs|_Deliverables:_ Best model parameters determined; final model object saved (pickle)._Acceptance:_ Model performance is robust (not significantly different between training and validation), indicating generalization. Ready for backtest.|
|**9**|**Backtest Implementation**- Write backtest code that simulates month-by-month, using the model to pick regime and an allocation rule- Implement the portfolio optimizer or regime-to-weights logic (Section 6.5)- Run backtest for out-of-sample period (reserve 3-5 years)|12 hrs|_Deliverables:_ Backtest code and preliminary performance results (time series of portfolio value, returns CSV)._Acceptance:_ Backtest runs without look-ahead bias (verified by code review) and produces realistic results (no extreme/unrealistic jumps).|
|**10**|**Performance Evaluation**- Calculate performance metrics of strategy vs benchmark(s) from backtest (CAGR, Sharpe, drawdown, etc.)- Plot equity curve and allocation changes over time- Interpret results (did it outperform? when did it lag?)|8 hrs|_Deliverables:_ Charts and metrics summarizing strategy performance; a brief interim report or presentation of findings to mentor/professor for feedback._Acceptance:_ Results make sense (e.g., strategy protected in downturns as hypothesized). Any issues identified (like too frequent trading or odd allocations) are noted for adjustment.|
|**11**|**Dashboard Development**- Create Streamlit app structure (sidebar inputs, main outputs)- Integrate model and backtest code into app (ensuring it runs quickly enough, possibly by caching results or simplifying computations)- Generate visualizations within app (Plotly charts for allocation and performance)|12 hrs|_Deliverables:_ `dashboard_app.py` (Streamlit script) with key functionality implemented._Acceptance:_ Basic dashboard can be launched locally and displays an allocation and perhaps static results. (Not fully polished yet, but functional with default data.)|
|**12**|**Dashboard Refinement & Testing**- Polish UI layout and text (labels, instructions, formatting)- Test with different inputs (e.g., modify CMA values) to ensure app responds correctly- Fix any bugs (e.g., ensure file upload works, handle no-upload scenario with default)|8 hrs|_Deliverables:_ Finalized dashboard app ready for deployment. Screenshots for report._Acceptance:_ All interactive elements work smoothly. App produces output within a few seconds of input change. Visuals are clear.|
|**13**|**Documentation & Report Writing**- Compile methodology and results into the research paper format- Write up theory with citations (Section 3), describe data and methods (Sections 4-6), present results (Section 9), discuss limitations (Section 10-11), etc.- Prepare reference list (Section 13)|12 hrs|_Deliverables:_ Draft of the final report (~4000 words) including all required sections and references._Acceptance:_ Draft covers all sections with placeholders for any last-minute data. Supervisor feedback integrated if available.|
|**14**|**Presentation Preparation**- Create presentation slides summarizing project (problem, approach, results, demo screenshots)- Include charts from dashboard and key findings- Practice talk (aim ~15-20 minutes)|6 hrs|_Deliverables:_ Slide deck (PowerPoint or PDF) and a short demo script for the dashboard (if live demo is part of presentation)._Acceptance:_ Presentation is clear, within time, and highlights important points. Ready for delivery.|
|**15**|**Final Deliverables & Wrap-up**- Submit final written report- Deliver final presentation- Deploy or package the dashboard (e.g., Docker image creation and test, push to GitHub)- Reflect on project (lessons learned, future improvements)|6 hrs|_Deliverables:_ Final report document (PDF), code repository updated, Docker image or instructions for running the app, and presentation delivered._Acceptance:_ All components submitted on time. Project meets the objectives set out (or deviations explained).|

**Note:** Weekly hours are estimates. Some weeks might require more time if unforeseen challenges arise (e.g., debugging data issues). Use the above milestones as checkpoints. For example, by mid-project (Week 8), you should have a working model and by Week 10 a fully tested strategy – this ensures the last weeks are focused on packaging and polish, not core implementation.

Regularly compare your progress with this plan. If you fall behind, communicate with your supervisor early and adjust scope if needed (e.g., reduce some optional features) to still meet the critical goals.

## 8. Assessment & Grading Rubric

Your project will be evaluated across multiple dimensions. The approximate weighting is as follows:

- **Code Quality & Engineering (45%)**
    
- **Research Paper Quality (25%)**
    
- **Dashboard UX & Functionality (20%)**
    
- **Presentation & Communication (10%)**
    

Below is a rubric detailing expectations for each:

**Code Quality & Engineering (45%)** – _We will assess the correctness, efficiency, organization, and documentation of your code._

- _Functionality:_ Does the code run without errors and produce the expected results? The data pipeline should correctly merge datasets; the model should train and backtest as intended; outputs used in the report should be reproducible from the code.
    
- _Code Style:_ Follows Python best practices (meaningful variable/function names, modular structure). Code is not excessively monolithic – for example, there are functions for key tasks (data loading, feature engineering, optimization) which improves readability.
    
- _Use of Source Control:_ The Git history should reflect regular commits. Commits have messages indicating what was done (e.g., “Added backtest function” instead of just “update”). If branching was used, merges are clean. This shows development process and collaboration discipline (if team project).
    
- _Documentation & Comments:_ Important sections of code have comments explaining non-obvious logic (e.g., “# converting yield to monthly rate”, “# training model on expanding window”). A reader of the code can follow what’s happening. In-line documentation (docstrings) for custom functions are a plus.
    
- _Efficiency & Optimization:_ Code avoids unnecessary recomputation (utilizes caching or stores intermediate results). Loops are vectorized where appropriate (pandas operations instead of overly many Python loops). The backtest and dashboard run in reasonable time (< a few seconds per user action ideally). The Docker container (if provided) correctly sets up environment.
    
- _Error Handling:_ Code anticipates possible issues (e.g., if data is missing, or if user uploads an incorrectly formatted file, it handles gracefully with warnings rather than crashes).
    
- _Innovation in Engineering:_ If you implemented something non-trivial (e.g., a custom optimization algorithm, or parallel processing) correctly, that will be noted positively.
    

**Research Paper Quality (25%)** – _This refers to the written report in Section 13, which should read like an academic or industry whitepaper._

- _Clarity of Writing:_ The paper is well-structured (follows the 14-section format), with clear, concise language. Technical concepts are explained in a way an informed reader can understand. The narrative flows logically from problem statement to methodology to results.
    
- _Depth of Analysis:_ The theoretical background shows understanding (with relevant citations). You connect theory to your design choices (e.g., citing why you chose a certain risk model). The data and methodology are described sufficiently (someone else could replicate the high-level approach with what you wrote).
    
- _Results Interpretation:_ The paper doesn’t just present numbers; it interprets them. For example, you don’t just state the Sharpe ratio, but discuss what it implies about the strategy’s attractiveness. You discuss whether outcomes met expectations from theory.
    
- _Use of Evidence:_ Figures and tables are used to back claims (e.g., a confusion matrix figure when talking about model accuracy, or a performance chart when claiming outperformance). Each figure/table is labeled and referenced in text.
    
- _Citations and References:_ You appropriately cite sources for any theory, data, or comparisons you mention (e.g., if you say “inverted yield curve predicts recessions”, you cite an academic source or well-known study). All citations are in APA style in-text (Author, Year) and the references list is complete. At least the recommended readings and any data source documentation are cited.
    
- _Professional Tone:_ The writing avoids colloquialisms, maintains an objective tone, and is free of grammatical errors. It reads akin to a professor’s handout or a professional research report.
    

**Dashboard UX & Functionality (20%)** – _Evaluation of the user-facing interface and experience using your Streamlit app._

- _Ease of Use:_ Is it obvious how to use the dashboard? Inputs should be clearly labeled with helpful descriptions. The user should be able to get outputs without confusion. For example, if a file upload is needed, there should be instructions or a default.
    
- _Design & Layout:_ The dashboard is organized (not too cluttered, uses headings or separators). Visual elements (charts, tables) are sized and placed well (no awkward cut-offs or overlapping elements). Color choices are visually appealing and consistent (e.g., one color per asset class across charts).
    
- _Interactivity:_ Widgets (sliders, dropdowns, etc.) respond and update the outputs accordingly. If the user changes an assumption, the results reflect that change (or if not automated, a clear “Run” button to trigger updates is provided). There’s no need to refresh the whole page manually.
    
- _Visual Clarity:_ Plots have legends, titles, and maybe brief captions. It’s clear what each chart is showing. For instance, performance chart axes are labeled (Time, Cumulative Return) and legend distinguishes Strategy vs Benchmark. Allocation pie chart labels each slice with asset name and percent.
    
- _Correctness of Info:_ The numbers and information shown on the dashboard are consistent with those in the report. (If there are slight differences due to updated data or assumptions, that’s okay, but generally the story should match.)
    
- _Reliability:_ The app shouldn’t crash during normal use. No error traces should appear for typical user actions. Also, it should handle edge cases (like what happens if user doesn’t upload CMA – perhaps use default and warn).
    
- _Overall User Experience:_ Put yourself in the shoes of an investment manager using this tool – does it provide useful outputs in a convenient way? A polished UX will make a strong impression (this includes things like using some Streamlit features such as `st.success()` or `st.error()` messages for statuses, if relevant, or even a loading spinner if something takes time).
    

**Presentation & Communication (10%)** – _Your oral presentation (and any demo) will be graded on how effectively you communicate your project to an audience._

- _Structure & Content:_ The presentation has a logical structure: introduction of problem, brief method, key results, conclusion. It covers the important points without going off-track or getting lost in details due to time constraints.
    
- _Clarity of Speech:_ You speak clearly, at a measured pace. Technical terms are explained succinctly for the audience. The storyline of the project (what you did and what you found) comes across plainly.
    
- _Visual Aids:_ Slides or demo visuals are legible (no tiny text, not too much info on one slide). Important charts from the project are shown and explained. If a live demo of the dashboard is done, it’s rehearsed to highlight specific features rather than randomly clicking.
    
- _Engagement:_ You maintain eye contact (if in person or camera on), or generally keep the audience engaged through perhaps a rhetorical question or emphasizing why the project matters (e.g., “Had an investor followed this strategy, they would have avoided a 20% drawdown in 2020 – a significant benefit.”). Your enthusiasm or interest in the project shows.
    
- _Time Management:_ You finish within the allotted time (± a minute). This usually implies practicing to ensure you can cover everything. If Q&A follows, you address questions knowledgeably, demonstrating mastery of what you did.
    
- _Supporting Materials:_ If required to submit slides, they should be well-designed (consistent style, no obvious typos, proper citation on slides if any external content). If not, this item won’t directly apply.
    

The above rubric ensures that all aspects of your capstone are evaluated: the behind-the-scenes work (coding, analysis) and the outward communication of it (report, tool, presentation). Aim to excel in each category.

One tip: reviewers will often skim code and focus more on results and documentation. However, if something seems off in results, they will dig into code, or if the app fails, they will scrutinize code quality. So maintain consistency: a strong result backed by clean code and clear explanations will score high across the board.

Use this rubric as a checklist during final polishing: e.g., run a spell-check on your report, ask a peer to click through your dashboard for feedback, etc., to iron out any wrinkles.

## 9. Evaluation Metrics & Success Benchmarks (statistical & investment)

To judge the success of the MARAE project, we will consider two sets of metrics:

**A. Statistical Performance Metrics (Model-focused):**  
These metrics evaluate how well the AI/ML model is identifying or predicting regimes.

- **Accuracy:** The proportion of time periods the model correctly classifies the regime. For binary classification (recession vs not), an accuracy above a naive baseline (which might be, say, 90% if only 10% of months are recessions) is expected. However, accuracy alone can be misleading if classes are imbalanced.
    
- **Precision and Recall:** Particularly for the recession (or whichever is the critical minority regime) class. _Precision_ tells us among predicted recessions, how many were actual recessions (low false-alarm rate). _Recall_ (sensitivity) tells us among actual recessions, how many the model caught. There is often a trade-off. A successful model should have a reasonably high recall for bad regimes (e.g., catch most recessions), even if it means a few false positives.
    
- **F1-Score:** The harmonic mean of precision and recall for the critical class. This gives a single measure of classification effectiveness. For example, an F1 above 0.5 for the recession class could be a target (given how challenging predicting recessions is, this would be decent).
    
- **ROC-AUC:** If your model produces a probability or score, the Area Under the ROC Curve gauges discrimination ability across thresholds. An AUC of 0.5 is random, 1.0 is perfect. For regime classification, an AUC in the 0.7–0.8 range would indicate the model has skill in ranking risk of recession periods.
    
- **Confusion Matrix:** Though not a scalar metric, examining the confusion matrix provides insight: ideally, very few missed recessions (false negatives) and not too many false alarms (false positives).
    
- **Cluster Separation (if unsupervised):** If using clustering, one might use metrics like silhouette score to measure how distinct the clusters (regimes) are. A higher silhouette (close to 1) means well-separated clusters. You might also evaluate if the clusters correspond to known regimes by checking their composition (e.g., one cluster contains mostly known recession dates).
    

Success benchmarks for statistical metrics: For example, “The model identified 4 of the last 5 known recessions (80% recall) with only 2 false signals (precision ~67%).” This would be considered successful in practical terms. Or “AUC of 0.85 in distinguishing high vs low inflation regimes” if that’s the focus.

**B. Investment Performance Metrics (Strategy-focused):**  
These metrics evaluate the outcome of using MARAE for asset allocation, compared to benchmarks or absolute targets.

- **Annualized Return (CAGR):** The compounded growth rate per year of the strategy. A success criterion might be that CAGR meets or exceeds a benchmark (e.g., surpassing a 60/40 portfolio’s CAGR by a certain margin) over the backtest period.
    
- **Annualized Volatility:** Standard deviation of monthly returns annualized. The strategy might aim for lower volatility than an equity-heavy benchmark. If MARAE’s vol is, say, 8% vs global equity’s 15%, that’s a reduction in risk.
    
- **Sharpe Ratio:** Measures risk-adjusted return = (Return - RiskFree) / Volatility. A higher Sharpe than benchmarks indicates superior risk-adjusted performance. For success, perhaps Sharpe > 1.0 (depending on period) or at least 20% higher than 60/40’s Sharpe.
    
- **Max Drawdown:** The maximum peak-to-trough loss observed. This is critical for evaluating regime-based strategies which often aim to reduce drawdowns in crises. If the max drawdown of MARAE is, say, -20% while global equities had -35% in the period, that’s a positive result. One could set a benchmark like “keep max drawdown under -25%” for the test period.
    
- **Calmar Ratio:** CAGR / |Max Drawdown|. A higher Calmar indicates a strategy that balances return well relative to worst-case loss. For instance, if MARAE returned 6% annually with a max DD of 10%, Calmar = 0.6, which might beat a benchmark’s 0.3.
    
- **Sortino Ratio:** Similar to Sharpe but uses downside deviation. If the returns distribution is asymmetric, this could be more appropriate. It penalizes only downside volatility.
    
- **Beta to Equities:** If you regress strategy returns against ACWI (equity market), what is beta? A low beta (<0.5) might confirm it’s truly lower-risk and not just riding equity risk. Ideally, MARAE has a beta significantly below 1 but still delivers a solid return, indicating value-add beyond just scaling down risk.
    
- **Hit Rate of Outperformance:** The percentage of months or years the strategy outperformed a benchmark. If MARAE outperforms in, say, 70% of months when market is down (good downside protection metric), that’s a success. Also consider up-market capture vs down-market capture ratios (did it capture enough upside while protecting on downside).
    
- **Turnover:** Though not a performance metric per se, high turnover can erode real performance when costs are considered. We might set a goal like <50% turnover per year (meaning on average half the portfolio changed per year), which implies moderate trading. If our strategy flips 100% every month, turnover is 1200% annually, which is impractically high. So a moderate turnover indicates efficiency of regime signals.
    

**Success Benchmarks (Investment):**  
Ultimately, the strategy should achieve a **better risk-adjusted return** than a static allocation. For instance:

- Sharpe ratio at least 0.1 or 0.2 higher than a 60/40 portfolio over the period.
    
- Maximum drawdown at least 10 percentage points smaller than equity benchmark.
    
- Only a small sacrifice in CAGR (or none) relative to an aggressive benchmark. Ideally, if MARAE can get close to equity returns in bull markets but dramatically cut losses in bears, that’s a win.
    
- If a specific numeric target is needed: perhaps aim for CAGR ~6-8% with volatility ~8-10% (so Sharpe ~0.6-0.8 given risk-free ~1%), and max drawdown <20%. These numbers are context dependent (if backtest includes 2008 and 2020, those are severe tests).
    

We will also look qualitatively: did the strategy behave as intended in different regimes? For example:

- In 2008, did it de-risk and thereby outperform a normal portfolio?
    
- In the long bull run 2010s, did it still participate enough to not lag badly?  
    If it sat in bonds the whole time (never catching regime shifts to bullish), it might have underperformed too much. So success includes demonstrating adaptive allocation (somewhat evident from allocation charts).
    

**Statistical vs Investment Trade-off:** Sometimes a model might have modest stats (maybe it doesn’t catch every recession exactly) but still lead to good investment performance because it avoids major pitfalls. We will consider the overall outcome. If, say, the model gave one false alarm (moved defensive early and missed a bit of rally) but otherwise saved a lot in a crash, the portfolio might still come out ahead. So we won’t nitpick the model if the end performance meets goals.

**Benchmarking:** Use at least one baseline like:

- 60/40 constant mix (global stocks/bonds).
    
- Or a risk parity static portfolio.
    
- Or even just ACWI for high-level comparison.  
    It’s not about beating stocks in raw return (since a defensive strategy often can’t in a bull market), but about **risk-adjusted and drawdown metrics**.
    

**Statistical Benchmarks:** A naive model could be “predict no recession ever”. Accuracy could be high (90% if recessions 10% of months) but recall 0 for recessions. We want to beat that by a clear margin: e.g. get recall maybe 50-80% while keeping precision decent. Also compare with a single-indicator heuristic model, like “if yield curve inverted, predict recession” – how many did that get right? If our ML model doesn’t beat simple heuristics or common knowledge indicators, that’s a point of discussion.

**Significance:** If possible, test if observed outperformance is statistically significant (e.g., t-test on difference in monthly returns vs benchmark, or see if alpha is significant in a regression with market). Not strictly required, but you can mention if the sample is small, results might not be statistically significant but are suggestive.

In summary, success for MARAE will be measured by:

- A regime classifier that demonstrates skill (e.g., ROC AUC > 0.7, meaningful precision/recall for bad regimes).
    
- A portfolio strategy that, through those regime shifts, achieves superior risk-adjusted returns (Sharpe, drawdown) relative to a standard approach, and meets any absolute risk targets (like keeping drawdown under a threshold).
    
- Ideally, the project will show that using AI for regime detection provided tangible benefit (e.g., “MARAE’s Sharpe was 0.7 vs benchmark’s 0.5, with 50% less drawdown”), validating the project hypothesis.
    

## 10. Risk Management & Mitigation

Every project faces risks that could impede success. Here we discuss potential risks in MARAE and how to mitigate them:

- **Data Gaps & Quality Issues:** There is a risk that some data series have missing values or shorter histories (e.g., an ETF with not enough history, or alt data not aligning). _Mitigation:_ We addressed this by carefully choosing start date to exclude periods where data is absent for required assets (e.g., starting after the newest ETF’s inception). We also fill or interpolate minor missing points. We maintain fallback options: if an alt data source is not working, the pipeline can run without it (perhaps by substituting a neutral value or excluding that feature). Data quality checks (like comparing overlapping sources or checking for extreme outliers) are in place to catch anomalies early. For instance, if Google Trends API fails on a given run, the code can catch the exception and either reuse last known data or proceed without that feature, ensuring the system still functions.
    
- **Model Overfitting:** Given the relatively small sample of macro regimes (recessions are infrequent), the model might overfit to the training data patterns and not generalize. _Mitigation:_ We used techniques like cross-validation with time splits and regularization (e.g., limiting model complexity, using penalized logistic regression or pruning tree depths). We also kept the feature set focused and not excessively high-dimensional. Simpler models (with interpretability) were favored to ensure we can double-check that learned relationships make economic sense (for example, model relying on yield curve and stress index – known signals – is more trustworthy than one relying on some noise). We tested the model on out-of-sample data (e.g., last few years) and only consider it successful if it performs reasonably well there, not just in-sample. If signs of overfit appeared (like wildly different regime predictions that don’t correlate with any known events), we would go back and reduce model complexity or gather more training data (maybe using quarterly data to get more points, or incorporating similar economies as parallel data if feasible).
    
- **Look-Ahead Bias:** In backtesting, a common pitfall is inadvertently using information that would not have been known at the time (e.g., using full-period mean to scale data, or rebalancing with future data). _Mitigation:_ We rigorously implemented the walk-forward approach: at each step, the model is trained only on past data, and features for that time are only up to that time. We especially guarded against using future knowledge in feature computation – for example, if we used a moving average or PCA, we ensure it’s windowed up to the current period (or fit PCA only on training data, not the whole dataset). Also, when scaling features, we fit the scaler on training set and apply to test, rather than normalizing across the full dataset. We double-checked that any label (like recession indicator) used for training is not accidentally peeking into the future (one way to test: shift labels one step forward to simulate prediction, ensuring model isn’t using “current quarter GDP drop” to predict current regime). In code review and testing, we monitored for any function that might inadvertently use future indices (like pandas forward-fill beyond current date, etc.). By structure, our backtest loop inherently prevents future leakage by design.
    
- **Computational Limits:** With a CPU-only environment and a need for relatively quick training (especially if we retrain model frequently in backtest or run heavy optimizations), there’s a risk that the approach could be too slow ( >10 min training). _Mitigation:_ We chose algorithms and data sizes mindful of this. Our dataset (monthly frequency) is small, so model training (logistic or random forest) is seconds, not minutes. The portfolio optimization is for 7 assets – solving that is negligible in time using either closed-form or a small CVX solver. Where heavier computation might come in is hyper-parameter tuning or if we had used an extensive scenario simulation. We mitigated by using reasonable ranges and not brute-forcing too many parameters. Also, caching results – we use Streamlit’s cache to avoid rerunning expensive parts unnecessarily (e.g., load data once, or compute the entire backtest once per session unless assumptions change). If we had attempted an LSTM or reinforcement learning (which could be slow on CPU), we scaled that back to simpler approaches. We also include optional GPU flags in code (if someone runs on GPU) but they are off by default to adhere to CPU limit.
    
- **Incorrect Assumptions in CMA (Model Risk):** The engine relies on user-supplied expected returns. If these are way off or inconsistent (model risk in inputs), the optimizer could give poor allocations. _Mitigation:_ We provide a default CMA that is reasonable and caution in documentation that results depend on those assumptions. We could add constraints to avoid extreme allocations even if CMA suggests it (e.g., cap weight to any single asset). This ensures a degree of robustness – e.g., if user puts 15% expected return for EM equities and 2% for everything else, the optimizer would want 100% EM, but we might cap at 50% to avoid over-concentration due to possibly overoptimistic input. Essentially, we treat CMA as one input, but incorporate our own risk judgment via constraints to mitigate unrealistic scenarios.
    
- **Misclassification Risk (Strategy Risk):** If the model misidentifies a regime (false signal), the portfolio could be wrongly positioned (e.g., go defensive when it shouldn’t, missing gains, or stay risk-on into a crash). _Mitigation:_ We can blunt the impact of any single month’s classification by not swinging the portfolio to extremes too quickly. For example, we can implement _gradual allocation changes_ or a neutrality buffer. Perhaps instead of going from 0% to 100% equity allocation in one step, we put a limit on monthly change (say max 20% shift). This smooths out whipsaws if the model flickers regime back and forth. Additionally, maintaining some diversification at all times (never 0% bonds or 0% equities entirely) means even a wrong call won’t be catastrophic. This concept of “guardrails” ensures the strategy isn’t purely binary on/off risk, but shades of gray. We also continuously monitor the model’s confidence: if it’s borderline (like 55% chance recession), we might only slightly reduce risk, whereas if it’s 95%, we go fully defensive. This probabilistic approach mitigates risk of moderate confidence errors. (In practice, due to time, we might implement a simpler threshold, but these ideas are noted.)
    
- **Regime Change/Novel Events:** The future might bring a macro regime that the model hasn’t seen (e.g., something like stagflation if our training didn’t have a strong instance of it). The model could misclassify or not recognize it, leading to suboptimal allocation. _Mitigation:_ We attempted to incorporate as diverse data as possible (our training includes 2008 crisis, 2020 pandemic, etc.). For something like 1970s stagflation, we don’t have asset data for all assets that far back, but we did include commodity and REIT features that might help if inflation surges. Also, the human oversight is important: in a real scenario, one would override or retrain if a new regime is suspected. In the project, we’ll discuss this limitation (the model isn’t foolproof against novel regimes). As a partial remedy, we included generic features (like inflation rate, yield curve) which might still trigger a warning if inflation regime arises even if historically rare.
    
- **Operational Risks (Reproducibility & Deployment):** Risk that the environment differences or package versions cause the code to not run on another machine. _Mitigation:_ By using Docker and requirements freeze, we ensure a consistent environment. We tested the whole pipeline on a fresh environment (or at least from scratch with data reload) to verify all necessary steps are documented. We also provide instructions for running the dashboard (so the user/grader doesn’t struggle with how to launch it). This mitigates risk of last-minute “it works on my machine but not on yours” issues.
    
- **Ethical/Compliance Risks:** (This overlaps with Section 11, but from a project risk perspective:) If we inadvertently use data that is proprietary or break terms of service (like scraping without permission), it could cause problems. _Mitigation:_ We stuck to publicly available data (FRED, Yahoo, Google Trends under fair use) and cited sources. We didn’t incorporate, say, insider info or something ethically problematic. We anonymize any sensitive info if there were (none in our case). So this risk is minimal here, but acknowledged and handled by careful choice of data.
    

In summary, through careful planning and methodological safeguards, we aimed to control the main risks:

- We built redundancy and checks into the data pipeline.
    
- We kept the model simple enough to generalize and tested it out-of-sample.
    
- We enforced realistic constraints on the strategy to prevent extreme behavior from model error.
    
- We documented and contained any external dependencies to ensure smooth execution.
    

Nonetheless, residual risk remains – e.g., the model might still miss an event or the strategy might underperform in a prolonged bull market. We will note these in conclusions as areas for improvement (like combining with momentum strategies or adding override rules). Risk management is an ongoing process, but the measures above significantly reduce the likelihood of a major failure in the project execution and outcomes.

## 11. Ethical & Compliance Considerations

In developing MARAE, it’s important to address## 11. Ethical & Compliance Considerations

In developing MARAE, it’s important to address the ethical use of data, respect intellectual property, and mitigate biases:

- **Use of Alternative Data:** We incorporate data like Google Trends and news sentiment. Ethically, we ensure this data is obtained in compliance with terms of service and privacy standards. Google Trends provides aggregate, anonymous data, so there are minimal privacy concerns. If we were to use social media or web-scraped data, we would need to avoid collecting personal identifiers and adhere to platform policies. For instance, using Reddit discussions for sentiment must comply with Reddit’s API terms and not infringe on user privacy. In our project, all alternative data used (GDELT, Trends) is publicly available and aggregated, aligning with ethical norms for data use.
    
- **Proprietary CMA Information:** The Capital Market Assumptions the user may upload could be proprietary forecasts (e.g., from their firm or a licensed source). We have to treat such data confidentially. Our system does not distribute the CMA data beyond the user’s session, and we advise users not to upload any information they are not authorized to use. If we included sample CMA data from BlackRock or others, we ensure it’s from publicly released reports or we have permission. In documentation and demonstrations, we use either publicly available CMA figures or clearly attribute them (e.g., citing BlackRock Investment Institute for any sample assumptions used). We avoid embedding any proprietary analytics from sources like Aladdin in our public code, sticking to our own implementations.
    
- **Model Bias and Transparency:** An AI model can reflect biases present in training data. For example, if historical regimes were all U.S.-centric, the model might be biased towards signals that work for the U.S. and not generalize globally. We mitigate this by including diverse indicators (some global) and by interpreting model decisions. We commit to **transparency**: the model’s key features and weights are shared in the report (e.g., noting that an inverted yield curve triggers a recession signal, which is an economically sensible outcome). This transparency is ethical as it allows users to understand and challenge the model’s reasoning rather than treating it as a black box for critical financial decisions.
    
- **Avoiding Harmful Automation:** The allocation engine makes suggestions that could materially impact investment outcomes. Ethically, we caution that the tool is for educational purposes and should not be blindly relied upon for real-world trades without human oversight. We include disclaimers that MARAE is a prototype and not financial advice. In a professional setting, deploying such a tool would require rigorous validation, out-of-sample testing, and possibly regulatory review if offered to clients. We highlight these limitations to prevent misuse.
    
- **Regulatory Compliance:** If MARAE were used in practice, certain regulatory considerations arise. For example, investment advice algorithms in jurisdictions like the U.S. might need to comply with SEC robo-advisor guidelines (ensuring fiduciary duty, transparency in methodology, etc.). While our project is academic, we design it in a way that respects those principles. We ensure the strategy doesn’t use any non-public information (it doesn’t — all data is public). We also ensure that if the model indicated extreme allocations, we temper them to avoid imprudent recommendations that could conflict with an investor’s suitability profile (e.g., going 100% into one asset class may be too risky; we incorporate some constraints for diversification).
    
- **Intellectual Property of Algorithms:** The techniques we implement (mean-variance optimization, risk parity, etc.) are standard in literature. We credit original sources (Markowitz, Rockafellar & Uryasev, etc.) in our documentation. We are not using any proprietary code from, say, BlackRock’s Aladdin; the “Aladdin-like” JSON schema in Appendix A is entirely our own construction for illustration. This respects BlackRock’s IP while demonstrating the concept. If we had access to something like Aladdin’s actual risk analytics, we would need permission to use or show it. By creating a simplified schema ourselves, we avoid IP infringement.
    
- **Bias in Training Data and Fairness:** While fairness in the classic sense (across demographics) is not directly applicable to a macro regime model, we do consider bias in terms of regimes being “fairly” recognized. For instance, the model shouldn’t consistently miss certain types of recessions (say, those that didn’t follow yield curve inversion) just because they were rare. We address this by not overfitting to any single indicator. In a broader ethical context, one could think of ensuring the model doesn’t favor certain sectors or countries unjustly — however, MARAE allocates at a broad asset class level without prejudice or redlining of assets beyond what data justifies (and all assets are broad indices).
    
- **Environmental, Social, Governance (ESG) Considerations:** Our engine currently focuses solely on risk/return. From an ethical investing standpoint, one might want to incorporate ESG criteria (for example, avoid heavy allocation to commodities or certain countries for ethical reasons). While this is outside our immediate scope, we acknowledge that a production version could include ESG filters or constraints (e.g., no tobacco stocks, or a carbon intensity limit for the portfolio). We’ve made the system flexible enough that a user could exclude an asset (they could set an asset’s CMA return to a very low number to effectively zero-weight it, for instance). Ethically, we provide a tool that can be adjusted to align with an investor’s values, rather than a one-size-fits-all mandate.
    
- **Plagiarism and Academic Honesty:** In writing the report and building the project, we ensure all external ideas and text are properly cited (see Section 13). We paraphrased content from sources and gave credit, rather than copying. This maintains academic integrity. All code is written by us or uses open-source libraries; any snippet influenced by an online example (like usage of a known formula or API usage) is appropriately adapted and not directly lifted without acknowledgement.
    

In summary, we have been mindful of ethical use of data (only using public, anonymized data), respecting intellectual property (attributing sources, not using protected content without permission), and building the model in a transparent, responsible manner. We also openly discuss the limitations and appropriate use of the tool, to ensure it’s not misrepresented as a guaranteed money-maker or used outside its tested domain. By doing so, we aim to uphold high ethical standards throughout the capstone project.

## 12. Stretch Goals

If time permits or for future continuation of the project, several exciting extensions could enhance MARAE’s capabilities:

- **LLM-Generated Theme Overlay:** Incorporate a Large Language Model (LLM) to analyze unstructured text (news articles, central bank statements, earnings call transcripts) and extract macro themes in real-time. For example, an LLM (via LangChain with GPT-4) could be prompted with “Summarize the major economic concerns in the news this week” and might respond with themes like “rising inflation” or “geopolitical tensions.” These themes can be mapped to regime adjustments (e.g., if “rising inflation” is prominent, tilt the portfolio towards inflation hedges like commodities and away from long-duration bonds). Essentially, the LLM acts as a qualitative overlay that captures factors traditional data might miss or not yet reflect. This stretch goal involves challenges of prompt engineering and ensuring the LLM’s output is reliable (the model can sometimes hallucinate). But it could make MARAE more responsive to **current events**. You would need to set up a pipeline where, say, daily news headlines are fed to the LLM and the output sentiment or topic is quantified and fed into the regime classifier. An example implementation might use OpenAI’s API via LangChain to get a “risk sentiment score” each day. Ethically, we must be careful that the LLM doesn’t introduce bias or misinformation; we’d likely restrict its sources to reputable news.
    
- **Reinforcement Learning Allocator:** Instead of a two-step process (predict regime, then allocate via predefined rules), one could train a reinforcement learning (RL) agent to directly decide asset allocations based on state variables (macro indicators, yields, etc.). The agent’s reward could be defined as a function of portfolio returns and risk (for instance, reward = end portfolio value – λ * volatility or drawdown). Using techniques from deep RL (like Deep Q Networks or policy gradient methods), the agent could **learn allocation policies** that maximize long-term reward, potentially uncovering strategies that a static model might miss. Over time, the agent might learn to do things like gradually increase bond allocation when certain signals emerge, etc., without us explicitly programming those rules. This is an ambitious goal: it requires a simulated environment (historical data can be used for epochs of training, or a model of market dynamics) and careful tuning to get convergence. It would likely involve using libraries such as Stable Baselines (for RL algorithms) and might benefit from GPU acceleration if using deep neural networks as function approximators. A simpler version could be a multi-armed bandit approach that learns which regime or which fixed mix performs best in current conditions. The RL approach could potentially adapt continuously, offering a more **autonomous allocator** that reacts to feedback (performance) rather than relying on pre-defined regime labels. However, ensuring it doesn’t overfit to historical patterns would be a key challenge.
    
- **Real-Time Data Stream Integration:** Evolve MARAE from a batch monthly model to a real-time decision system. This would involve connecting to streaming data sources (such as live market feeds or APIs that provide intraday updates to economic proxies). For example, one could use a websocket connection to a financial data provider to get live price updates and volatility measures. The engine could then update regime probabilities daily or even intraday (though macro regimes don’t shift daily, high-frequency “regime” could be more sentiment or market-regime oriented). A real-time MARAE would also mean deploying the dashboard such that it refreshes automatically with new data (for instance, using Streamlit’s `st.autorefresh` or scheduling tasks to pull new data and recompute allocations every hour). On the infrastructure side, stretch goals include deploying the app to a cloud service (like Streamlit Cloud or AWS) so that it can be accessed anywhere and possibly running the backtest/compute on a schedule to incorporate the latest info. Another real-time aspect is enabling alerts: e.g., if the regime model suddenly flips to “High Stress” regime, the system could send an email or notification. Achieving real-time performance might require optimizing the code further (it’s already efficient for monthly data; real-time might mean handling more data points like intraday prices, which is feasible but needs memory and compute considerations). This goal would take MARAE closer to a deployable fintech product.
    
- **Additional Asset Classes or Factors:** While not listed, another stretch goal is expanding the investment universe (e.g., include more asset classes like gold separately, or currency hedges) and incorporate **factor investing** angles. For instance, within equities, rotate between sectors or factors depending on regime (overweight defensive stocks in recession, etc.). This would complicate the model (needing data on sector performance in regimes), but could add value by within-asset allocation. This aligns with what some advanced strategies do (tactical sector rotation based on macro outlook).
    
- **User Personalization & Machine Learning Ops:** From a software perspective, allowing the user to input their risk preference or constraints as parameters and the engine adapting to those is a nice stretch. We touched on risk preference slider. One could also implement a “learning from user feedback” loop: if the user overrides the model’s suggestion (says “I disagree, I’ll allocate differently”), the system could log that and try to learn patterns in user adjustments to refine its suggestions (this enters the realm of human-in-the-loop ML, which is advanced).
    

Each of these stretch goals extends the project significantly. Implementing them would require more time and possibly computational resources, but they show the potential evolution of MARAE:

- Using **LLMs** marries macro quant with NLP-driven insights (a very hot area in finance now, often referred to as analyzing news or even using ChatGPT for market analysis).
    
- **RL-based allocation** could potentially outperform static rule-based strategies by discovering complex policies; it’s cutting-edge, basically letting the algorithm not just classify but _act_ and learn from those actions.
    
- **Real-time streaming** would transform our academic prototype into a practical tool that could be used by a portfolio manager monitoring markets daily.
    

Documenting these as stretch goals provides a roadmap for how this capstone could continue into a more advanced project or even a product beyond the fall semester. They also demonstrate awareness of current trends (LLMs in finance, RL in asset allocation, real-time data pipelines) that go beyond the core requirements.

## 13. Recommended Reading & Citations

Below is a curated list of recommended references, grouped by category, that informed this project and can provide further learning:

- **Peer-Reviewed Papers on Macro Regimes & ML Classifiers:**
    
    - Mulliner, A., Harvey, C. R., Xia, C., Fang, E., & van Hemert, O. (2025). _Regimes_. SSRN Working Paper. – Introduces a novel method to detect economic regimes by similarity to historical patterns, improving asset return predictions.
        
    - Nuriyev, D., Duan, S., & Yi, L. (2024). _Augmenting Equity Factor Investing with Global Macro Regimes_. Proceedings of ICAIF 2024. – Demonstrates how incorporating macro regime signals via machine learning can enhance factor investment strategies, with a methodology for regime modeling and detection (conference paper).
        
    - Rockafellar, R. T., & Uryasev, S. (2000). _Optimization of Conditional Value-at-Risk_. **Journal of Risk, 2**(3), 21–41. – Pioneering paper that defines CVaR and provides a framework to minimize tail risk in portfolios, underpinning our use of CVaR in optimization.
        
    - Qian, E. (2011). _Risk Parity and Diversification_. **The Journal of Investing, 20**(1), 119–127. – Explores the risk parity approach to asset allocation, arguing for balancing risk contributions. Provides the theoretical foundation for our risk-budgeting (equal risk) strategy.
        
    - Choi, H., & Varian, H. (2012). _Predicting the Present with Google Trends_. **Economic Record, 88**(s1), 2–9. – Key study showing how Google search data can nowcast economic indicators. Justifies our use of Google Trends as a feature for capturing real-time public sentiment on the economy.
        
    - Leetaru, K. H., & Schrodt, P. A. (2013). _GDELT: Global Data on Events, Location, and Tone, 1979–2012_. International Studies Association Conference. – Introduces the GDELT project, which we reference for global news sentiment data. Describes how event and tone data can be used to monitor world events impacting markets.
        
- **BlackRock BII & CMA Whitepapers (latest versions):**
    
    - BlackRock Investment Institute. (2024). _Midyear 2024 Global Outlook: The New Regime_. BlackRock Whitepaper. – Discusses the transition to a regime of higher macro volatility and inflation, and its asset allocation implications. Offers context on regime thinking from a practitioner’s perspective (used to align our model assumptions with current outlooks).
        
    - BlackRock Investment Institute. (2023). _Capital Market Assumptions – 5-Year Outlook (2023–2027)_. BlackRock Publication. – Provides long-term expected returns for major asset classes and the methodology behind them. We used these CMA values as a baseline for our optimizer, and this report is essential for understanding how such assumptions are formulated and used in portfolio construction.
        
- **Books:**
    
    - López de Prado, M. (2020). _Machine Learning for Asset Managers_. Cambridge University Press. – An accessible text bridging machine learning techniques and their application in investment management. Chapters on backtest overfitting and regime detection informed our approach to model validation and the need for walk-forward testing.
        
    - Bodie, Z., Kane, A., & Marcus, A. (2014). _Investments_ (10th ed.). McGraw-Hill. – A classic textbook covering modern portfolio theory, asset allocation, and portfolio management. Provided theoretical background on mean-variance optimization, capital market history, and the business cycle’s effect on asset returns (Ch. 17 on Asset Allocation is particularly relevant).
        

_(Refer to these sources for a deeper theoretical grounding and for validation of the approaches used in MARAE. Citations in the report are in APA style, and the reference list above is compiled accordingly.)_

## 14. Appendices

### A. Sample Aladdin-JSON Risk Report Schema

Below is an example JSON schema inspired by BlackRock’s Aladdin risk reports, illustrating how portfolio risk information might be structured. This is a hypothetical output for our MARAE portfolio, showing asset weights and key risk metrics:

```json
{
  "portfolioId": "MARAE_example_001",
  "asOfDate": "2025-09-30",
  "holdings": [
    {
      "asset": "ACWI",
      "weight": 0.25,
      "expectedReturn": 0.07,
      "volatility": 0.15,
      "CVaR_95": 0.10
    },
    {
      "asset": "EEM",
      "weight": 0.10,
      "expectedReturn": 0.08,
      "volatility": 0.20,
      "CVaR_95": 0.15
    },
    {
      "asset": "BNDX",
      "weight": 0.30,
      "expectedReturn": 0.03,
      "volatility": 0.06,
      "CVaR_95": 0.04
    },
    {
      "asset": "HYLB",
      "weight": 0.10,
      "expectedReturn": 0.05,
      "volatility": 0.10,
      "CVaR_95": 0.07
    },
    {
      "asset": "DBC",
      "weight": 0.15,
      "expectedReturn": 0.04,
      "volatility": 0.18,
      "CVaR_95": 0.12
    },
    {
      "asset": "VNQI",
      "weight": 0.10,
      "expectedReturn": 0.06,
      "volatility": 0.12,
      "CVaR_95": 0.08
    },
    {
      "asset": "Cash",
      "weight": 0.00,
      "expectedReturn": 0.02,
      "volatility": 0.00,
      "CVaR_95": 0.00
    }
  ],
  "portfolioMetrics": {
    "expectedReturn": 0.055,
    "volatility": 0.080,
    "SharpeRatio": 0.68,
    "CVaR_95": 0.07,
    "MaxDrawdown": 0.18
  },
  "riskFactorExposure": {
    "Equity": 0.50,
    "Rates": 0.30,
    "Credit": 0.15,
    "Commodities": 0.05
  }
}
```

_Explanation:_ This JSON contains an array of `holdings`, each with an asset ticker, its portfolio weight, and risk metrics (volatility and 95% CVaR in this case). `portfolioMetrics` summarizes the overall portfolio’s expected return, volatility, Sharpe ratio, etc. `riskFactorExposure` shows how the portfolio’s risk is distributed across broad factors (for example, here ~50% of the risk comes from equity market exposure, 30% from interest rates via bonds, etc.). This schema is similar to what a tool like Aladdin might output in a risk report, helping portfolio managers understand both asset-level and aggregate risks. It is included as a reference to demonstrate how MARAE’s outputs could be structured for integration into risk systems.

### B. Example Python Notebooks (Pseudo-code)

The following pseudo-code outlines the key steps of the MARAE implementation, as one might structure in Python notebooks or scripts:

```python
# 1. Data Ingestion & Preprocessing
import pandas as pd
from pandas_datareader import data as pdr
from pytrends.request import TrendReq

# Fetch historical prices for core ETFs
tickers = ["ACWI", "EEM", "BNDX", "HYLB", "DBC", "VNQI"]
prices = pdr.DataReader(tickers, data_source='yahoo', start='2006-01-01')['Adj Close']
# Fetch macro indicators from FRED (e.g., 3M T-Bill and Financial Stress Index)
tbill = pdr.DataReader("TB3MS", "fred", start='2006-01-01')
stress = pdr.DataReader("STLFSI3", "fred", start='2006-01-01')
# Fetch Google Trends data for a search term (e.g., "recession")
pytrends = TrendReq(); pytrends.build_payload(["recession"], timeframe='2006-01-01 2025-01-01')
trends = pytrends.interest_over_time()["recession"]

# Align data to monthly frequency
data_monthly = pd.DataFrame({
    "ACWI": prices["ACWI"].resample('M').last().pct_change(),
    "EEM": prices["EEM"].resample('M').last().pct_change(),
    # ... (other asset returns)
    "StressIdx": stress.resample('M').last(),
    "Tbill": tbill.resample('M').last(),
    "RecessionSearch": trends.resample('M').mean()
})
data_monthly.dropna(inplace=True)  # drop initial NaNs

# 2. Feature Engineering
data_monthly["YieldCurve"] = data_monthly["10yrYield"] - data_monthly["Tbill"]  # e.g., if 10yrYield fetched similarly
# Create regime label (e.g., 1 if NBER recession, else 0) - assume we have NBER dates in a series
# data_monthly["RecessionFlag"] = nber_series_resampled...

# 3. Train Regime Classifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

features = ["StressIdx", "YieldCurve", "RecessionSearch"]  # etc.
X = data_monthly[features].values
y = data_monthly["RecessionFlag"].values  # assuming we created this
model = RandomForestClassifier(max_depth=3, n_estimators=100, class_weight='balanced')
ts_cv = TimeSeriesSplit(n_splits=3)
# Perform cross-val training (pseudo-code, not actual loop):
# for train_idx, test_idx in ts_cv.split(X): model.fit(X[train_idx], y[train_idx]) ... tune hyperparams ...
model.fit(X, y)  # final model trained on all data up to latest available

# 4. Back-testing & Allocation
historical_weights = []
portfolio_values = [1.0]  # start with $1
for t in range(len(data_monthly)):
    # Determine regime (either use actual y[t] if simulating perfect foresight, or model.predict if simulating live)
    regime = model.predict(X[t].reshape(1, -1))
    # Set allocation based on regime
    if regime == 1:  # recession
        w = {"ACWI": 0.2, "BNDX": 0.5, "HYLB": 0.1, "DBC": 0.1, "Cash": 0.1}
    else:  # expansion
        w = {"ACWI": 0.5, "BNDX": 0.2, "HYLB": 0.2, "DBC": 0.1, "Cash": 0.0}
    historical_weights.append(w)
    # Compute portfolio return for this month
    ret = sum(w[asset]*data_monthly.iloc[t][asset] for asset in w if asset != "Cash")
    portfolio_values.append(portfolio_values[-1] * (1 + ret))

# 5. Performance Evaluation
import numpy as np
returns = np.diff(portfolio_values) / portfolio_values[:-1]
cagr = (portfolio_values[-1])**(12/len(returns)) - 1
vol = np.std(returns) * np.sqrt(12)
sharpe = (cagr - data_monthly["Tbill"].mean()) / vol
max_dd = 1 - (min(portfolio_values) / max(portfolio_values))
print(f"CAGR: {cagr:.2%}, Vol: {vol:.2%}, Sharpe: {sharpe:.2f}, Max Drawdown: {max_dd:.2%}")

# The results can then be plotted or compared to a benchmark 60/40 portfolio.
```

_Notes:_ This pseudo-code omits some details (like obtaining the 10-year yield for yield curve, or actual NBER recession flag integration), but it sketches the workflow. The code is structured in sections (1) Data prep, (2) Feature engineering, (3) Model training, (4) Backtest simulation, (5) Metric calculation. In practice, one would refine each part (e.g., handle the model update in each loop iteration for true walk-forward training, incorporate transaction costs if needed, etc.). The idea is to show how the pieces connect: data flows into features, which feed the ML model, whose output then drives the portfolio decisions, which are evaluated for performance.

### C. Annotated Bibliography with Hyperlinks

Below we provide an annotated bibliography of key references, including hyperlinks for easy access:

- **Mulliner et al. (2025) – "Regimes"** – _SSRN:_ [Regimes (2025) by Mulliner & Harvey et al.](https://ssrn.com/abstract=5164863). This working paper presents a method to detect the current economic regime by comparing current indicators to historical patterns, and demonstrates its use in timing equity factors. It’s cutting-edge research (2025) co-authored by Campbell Harvey, making it highly relevant for understanding regime identification in practice.
    
- **Nuriyev et al. (2024) – Macro Regimes for Factor Investing** – _ResearchGate:_ [Augmenting Equity Factor Investing with Global Macro Regimes (2024)](https://www.researchgate.net/publication/374146230_Augmenting_Equity_Factor_Investing_with_Global_Macro_Regimes). This conference paper (ACM ICAIF) details an approach to model macro regimes using unsupervised learning and integrate those signals into equity portfolio construction. It provides both methodology and empirical results, serving as a blueprint for combining ML classification with asset allocation.
    
- **Rockafellar & Uryasev (2000) – CVaR Optimization** – _PDF:_ [Optimization of Conditional Value-at-Risk](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf). A seminal paper that introduces CVaR as a risk measure and shows how to minimize it using linear programming. This is fundamental reading for understanding the risk metrics and optimization technique we applied for tail-risk management in the portfolio.
    
- **Qian (2011) – Risk Parity and Diversification** – (No freely accessible link, citation provided.) Edward Qian’s article in _The Journal of Investing_ explains the risk parity philosophy. It’s useful for grasping the rationale behind equal-risk contribution portfolios, one of the methods we considered for allocation. Qian also discusses the limits of leverage and implementation aspects of risk parity.
    
- **Choi & Varian (2012) – Google Trends in Nowcasting** – _PDF:_ [Predicting the Present with Google Trends](https://www.google.com/googleblogs/pdfs/google_predicting_the_present.pdf). This paper by a Google economist shows how search query data can improve forecasts of current economic indicators (like unemployment claims). It validates our use of Google Trends as a timely indicator and is an easy read to see the power of alternative data in economics.
    
- **Leetaru & Schrodt (2013) – GDELT Project** – _Website:_ [GDELT Project Overview](https://www.gdeltproject.org/). The GDELT website outlines the scope of the Global Database of Events, Language, and Tone, which monitors worldwide news. Leetaru’s 2013 paper (cited on the site) describes how GDELT data can be used for global trend analysis. This is relevant for understanding sources of sentiment features and how event data can feed into regime analysis.
    
- **BlackRock Investment Institute (2024) – Midyear Global Outlook** – _PDF:_ [BlackRock 2024 Midyear Outlook – The New Regime](https://www.blackrock.com/ca/investors/en/literature/market-commentary/bii-midyear-outlook-2024-en-ca.pdf). In this whitepaper, BlackRock’s strategists discuss the emergence of a new macro regime characterized by higher inflation and interest-rate volatility. It provides context on why regime-based investing is crucial now and offers qualitative insights that complement our model’s quantitative approach.
    
- **BlackRock Investment Institute (2023) – Capital Market Assumptions** – _Interactive Data:_ [BlackRock 5-Year Capital Market Assumptions](https://www.blackrock.com/institutions/en-us/insights/charts/capital-market-assumptions). This resource provides an interactive chart of BlackRock’s 5-year return assumptions updated for 2023. By exploring it, one can see expected returns for various asset classes and understand the methodology (which accounts for valuations, growth, etc.). It’s foundational for setting realistic inputs in strategic asset allocation exercises like MARAE.
    
- **López de Prado (2020) – Machine Learning for Asset Managers** – _Cambridge Univ. Press:_ [Machine Learning for Asset Managers – Book](https://www.cambridge.org/core/elements/machine-learning-for-asset-managers/4F1DE30C6310A9850E5B4C1531F46D58). This book (part of the Cambridge “Elements in Quantitative Finance” series) provides concise chapters on applying ML to finance. Particularly, it covers concepts like overfitting mitigation, financial feature engineering, and even touches on economic regime shift detection. It’s recommended for learning how to rigorously marry ML with finance problems.
    
- **Bodie, Kane & Marcus (2014) – Investments (10th ed.)** – _Textbook:_ [Investments by Bodie, Kane, Marcus](https://www.mheducation.com/highered/product/investments-bodie-kane/M9780077861674.html). A comprehensive textbook on investment science. Relevant chapters for this project include those on asset allocation and portfolio optimization, as well as sections on the term structure of interest rates and business cycles. It offers the theoretical underpinning for many concepts we implemented (e.g., Markowitz optimization, the impact of diversification, etc.) and is a staple reference for understanding classical approaches that our project builds upon.
    

Each of these references contributed to shaping the MARAE project, either by providing theoretical justification, data sources, or inspiration for methodological enhancements. They are suggested as further reading for a deeper dive into macro regime analysis, quantitative portfolio management, and the integration of AI in investment strategies.
