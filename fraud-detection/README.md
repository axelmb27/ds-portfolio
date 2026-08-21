# Transactional Fraud Detection

**Business question:** How can we accurately identify fraudulent mobile-money transactions in a large dataset (6 million records), minimizing financial losses?

Built on ~6.36 million simulated mobile-money transactions (PaySim1, Kaggle). The emphasis here is **identifying behavioral patterns through SQL-based exploratory data analysis (EDA)** and **preventing target leakage** to build a robust machine learning model. The full pipeline involved: `SQL EDA` -> `Python EDA` -> `PostgreSQL Connection` -> `Model Training & Evaluation`.

---
## Data Dictionary

| Variable | Description | Type |
|---|---|---|
| `isFraud` | **Target.** 1 if the transaction is actual fraud, 0 otherwise | integer |
| `step` | Time unit. 1 step = 1 hour. The dataset simulates ~30 days (744 steps) | integer |
| `type` | Transaction type: `CASH_IN`, `CASH_OUT`, `TRANSFER`, `PAYMENT`, `DEBIT` | string |
| `amount` | Transaction amount | float |
| `nameOrig` | Originating customer | string |
| `oldbalanceOrg` | Origin account balance before the transaction | float |
| `newbalanceOrig` | Origin account balance after the transaction | float |
| `nameDest` | Destination customer/merchant | string |
| `oldbalanceDest` | Destination account balance before the transaction | float |
| `newbalanceDest` | Destination account balance after the transaction | float |
| `isFlaggedFraud` | Simulator's internal anti-fraud flag (marks transfers > 200,000) | integer |

*Source: "PaySim1", Kaggle.*

**Note: Balance columns were strictly excluded from the final model to prevent target leakage (see findings below).**

## The Headline for the Business

**The trained model successfully captures ~80% of potentially lost-to-fraud money while catching only 53% of fraud events — by prioritizing high-value fraud (money, not event count)** With a ROC-AUC of 0.8451 and a PR-AUC of 0.5425 on highly imbalanced data, the model efficiently separates fraudulent actions from legitimate volume. This can be increased depending on the business ability of dealing with false positives (transactions incorrectly flagged as fraud).

---

## Why the Data Work is the Point

Every cleaning and feature engineering decision was made by querying the raw data in PostgreSQL to uncover mechanisms of fraud. Rare is not the same as wrong, but identifying data generation mechanics is key to avoiding over-optimistic models.

### Finding 1 — Fraud is highly concentrated by transaction type

![Fraud Concentration](figures/02_fraud_by_type.png)

Fraud is strictly present in only two transaction types: `TRANSFER` (0.77% fraud rate) and `CASH_OUT` (0.18% fraud rate). `TRANSFER` is 4 times riskier. No other type contained any fraud. 
**Action:** The dataset was filtered to train only on `TRANSFER` and `CASH_OUT` to simplify the model with zero loss of fraud examples.

### Finding 2 — A cyclical temporal pattern exposes fraud hours

![Cyclical Pattern](figures/03_cyclical_pattern.png)

While the absolute number of fraudulent transactions is roughly constant throughout the day, the *rate* of fraud spikes drastically during low-activity hours (hours 2-6, relative to the simulation start). The fraud rate surges from ~0.06% during peak hours to up to 22% during these "night" hours. 
**Action:** Engineered a new binary feature `is_night`.

### Finding 3 — Target leakage in account balances

An analysis of the balance columns (`oldbalanceOrg` - `amount` vs `newbalanceOrig`) revealed severe target leakage. For legitimate transactions, the math mostly reconciles. However, fraudulent rows show massive balance "errors" (e.g., median error of 399k vs 164k). According to PaySim documentation, balances are altered after a fraud is cancelled. 
**Action:** Excluded all balance columns from the model to prevent it from simply learning the leakage. 

### Finding 4 — The built-in rule is useless

The simulator's built-in `isFlaggedFraud` rule (which flags any `TRANSFER` > 200,000) catches almost no real fraud. This mathematically confirms the need for an advanced machine learning approach over simple heuristics.

### Finding 5 — Customers vs Merchants

Every fraudulent transaction occurs between two Customer accounts (starting with 'C'). Merchants (starting with 'M') are never the origin or destination of fraud in this dataset. Furthermore, destination "mule" accounts are virtually single-use. 

### Finding 6 — Amount vs Step has signal

 ![Amount Signal](figures/04_amount.png)

 Consistent with Finding 2, fraud transactions distributions tend to be pretty stochastic in the step dimension, however in amount, they are mostly bounded [10^4-10^7].

---

## Feature Engineering & Model Training

Based on the EDA, the final feature set was intentionally kept small, robust, and LEAKAGE-FREE:
- `amount` (strong legitimate signal)
- `type` (TRANSFER or CASH_OUT)
- `step`
- `hour_of_day` (extracted as `step % 24`)
- `is_night` (flag for hours 2-6)

### Model Training Strategy
Because the dataset is heavily imbalanced (fraud represents < 0.2% of transactions), standard accuracy is misleading. The model was trained and evaluated using **PR-AUC (Precision-Recall Area Under Curve)** over ROC-AUC, as PR-AUC is more sensitive to improvements in the minority (fraud) class. 

### Threshold Tuning & Business Impact
A standard machine learning classification threshold (0.5) is rarely optimal for real-world fraud detection. By evaluating the model's probabilities, we can adjust the decision threshold to align with the business's risk tolerance:

- **High Threshold (Conservative):** Prioritizes Precision. Catches less fraud, but ensures almost no legitimate customers are blocked (low false positives).
- **Optimized Threshold (Business-driven):** By lowering the threshold, we increase Recall, successfully capturing **~80% of potentially lost-to-fraud money**. The trade-off is an increase in false positives (legitimate transactions flagged as fraud), which would need to be handled via friction (e.g., 2FA prompts) or manual review queues.

**Confusion Matrices: Default vs. Business-Tuned Threshold**

*(Below, you can see how adjusting the threshold impacts the raw number of caught fraud vs. false alerts for the two ML models evaluated: RF vs XGBoost)*

![Confusion Matrix comparison - Threshold](figures/05_confusion_matrix_comp.png)

**Performance Highlights:**
**Business Favored Model recommendation: Random Forest with 0.3-0.1 threshold**
- **ROC-AUC:** 0.8451
- **PR-AUC:** 0.5425
- **Money saved:** 80%-84.3%
---

## Repo

```
fraud-detection/
├── README.md
├── .gitignore
├── fraud_detection_eda_and_fe.sql  # PostgreSQL EDA and Feature Engineering
├── Fraud_detection.ipynb           # Python EDA, DB connection, Modeling
└── figures/                        
```

**Data:** Not committed due to size limits. Download from the [Kaggle competition](https://www.kaggle.com/datasets/ealaxi/paysim1) or via `kagglehub`. Cite: E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, psycopg2

---

# Part 2 — Stakeholder Dashboard (Power BI)

Part 1 proves the model works. Part 2 answers a different question: **can a non-technical fraud-risk stakeholder see the value and act on it?** This layer translates the model output into an operating decision, built on four pre-aggregated tables exported from the notebook — no row-level data, no leakage columns.

> The `.pbix` is included. The screenshots below are the deliverable.

## Page 1 — Monitoring: impact → where → when

![Dashboard Page 1](powerbi/screenshots/page1.png)

The header asserts a single operating point (Random Forest @ threshold 0.30): **$3.64B at stake → 80% of the money recovered, catching only 53% of fraud *events*, while flagging just 0.28% of clients.** The model is optimized for money, not event count — it prioritizes the expensive frauds. The lower visuals justify the modeling scope (fraud lives only in TRANSFER and CASH_OUT) and the `is_night` feature (fraud rate spikes overnight while volume collapses).

## Page 2 — Decision analysis: which model, which threshold

![Dashboard Page 2](powerbi/screenshots/page2.png)

**Trade-off curve (right):** money saved vs. % of clients flagged, one line per model. The x-axis is **operational cost (flagged %), not the raw threshold** — the only honest way to compare RF and XGBoost, whose probability scales differ (XGBoost uses another scale, so an identical threshold means different things per model). Compared at equal cost, **RF dominates across the entire realistic operating region (≤2% flagged).** The curve is cropped to that region on purpose: XGBoost only "wins" further right by flagging 5–8% of clients — a cost no antifraud team pays (Remember the total number of test set transactions is on the hundreds of thousands). Threshold labels (0.5 / 0.3 / 0.04) make the operating point explicit.

**Gains curve (left):** if the review team can only inspect a fraction of transactions, ranking by model score and reviewing the top ~2% captures ~70% of fraud *events*. The "Random (no model)" diagonal is the baseline — the gap above it is the model's lift.

## What these two charts show *together* (the core insight)

The gains curve shows **XGBoost catching slightly more fraud events** in the top 2%; the trade-off curve shows **RF saving more money** at equal cost. XGBoost catches more *cases*; RF catches more *dollars*, because it ranks high-value frauds higher. **This result is the reason this project optimizes for money saved rather than event-count recall, a real business decision.** A model tuned to maximize caught-fraud *count* would recommend XGBoost and increase operational cost by flagging ~10^4 legitimate transactions for review.

## On the tool

For a static, single-analyst portfolio, Power BI is included to demonstrate the communication layer: turning a defensible model into an operating recommendation a risk stakeholder can act on.