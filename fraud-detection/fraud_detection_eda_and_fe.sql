-- =============================================================================
--  PROJECT : Transactional Fraud Detection  (PaySim, using PostgreSQL)
--  FILE    : fraud_detection_eda_and_fe.sql
--  AUTHOR  : Axel Morales Buendia
--  PURPOSE : Exploratory analysis of fraud patterns and feature engineering.
--  DATA    : PaySim1 (Kaggle, ealaxi/paysim1). ~6.36M mobile-money transactions.
-- =============================================================================


-- #############################################################################
-- ##  PART 1 — EXPLORATORY DATA ANALYSIS                                      ##
-- #############################################################################

-- -----------------------------------------------------------------------------
-- 1. Fraud concentration by transaction type
--    FINDING: Fraud is only present in type = TRANSFER (0.77%) and CASH_OUT (0.18%).
--    TRANSFER is 4 times riskier by rate. No other type is ever fraud.
--    => we can train the model with these two types with zero fraud loss.
-- -----------------------------------------------------------------------------
SELECT
    type,
    COUNT(*)                                              AS total_transactions,
    COUNT(*) FILTER (WHERE isfraud = 1)                   AS fraud_count,
    ROUND(COUNT(*) FILTER (WHERE isfraud = 1) * 100.0 / COUNT(*), 4) AS fraud_pct,
    ROUND(SUM(amount), 2)                                 AS total_amount,
    ROUND(SUM(amount) FILTER (WHERE isfraud = 1), 2)      AS total_fraud_amount,
    ROUND(SUM(amount) FILTER (WHERE isfraud = 1) * 100.0 / SUM(amount), 4) AS amount_loss_pct
FROM transactions
GROUP BY type
ORDER BY fraud_count DESC;


-- -----------------------------------------------------------------------------
-- 2. Impact of fraud in total portfolio
--    FINDING: ~12.06 billion lost to fraud, fraud represents 1% across the entire data portfolio.
--    Small, but a large absolute amount
-- -----------------------------------------------------------------------------
SELECT
    SUM(amount) FILTER (WHERE isfraud = 1)               AS total_fraud_loss,
    SUM(amount) FILTER (WHERE isfraud = 0)               AS total_legit_volume,
    ROUND(
        SUM(amount) FILTER (WHERE isfraud = 1) * 100.0 /
        NULLIF(SUM(amount) FILTER (WHERE isfraud = 0), 0), 4
    )                                                    AS pct_lost_to_fraud
FROM transactions;


-- -----------------------------------------------------------------------------
-- 3. Cyclical temporal pattern (hour of day = module(step, 24)
--    FINDING: Fraud RATE grows in hours 2-6 (4% - 22%) vs ~0.06% during peak hours
--    (~10^2 times  higher). Fraud is ~constant by hour,
--    but the RATE of fraud/total spikes in the low-activity (presumably night) window,
--    when the number of legitimate transactions decrease.
--    NOTE: hour is relative to the simulation start, not wall-clock midnight.
--    => motivates the is_night logic feature.
-- -----------------------------------------------------------------------------
SELECT
    step % 24                                            AS hour_of_day,
    COUNT(*)                                             AS total_transactions,
    COUNT(*) FILTER (WHERE isfraud = 1)                  AS fraud_count,
    ROUND(COUNT(*) FILTER (WHERE isfraud = 1) * 100.0 / COUNT(*), 4) AS fraud_pct
FROM transactions
GROUP BY step % 24
ORDER BY fraud_pct DESC;


-- -----------------------------------------------------------------------------
-- 4. Only Customers (C...), never Merchants (M...), are involved in fraud
--    FINDING: every fraudulent transaction is between C-type accounts;
--    no Merchant is ever the origin or destination of fraud.
-- -----------------------------------------------------------------------------
SELECT DISTINCT isfraud, nameorig, namedest
FROM transactions
WHERE isfraud = 1
  AND NOT (nameorig LIKE 'M%' OR namedest LIKE 'M%');


-- -----------------------------------------------------------------------------
-- 5. Mule accounts are single-use  (SELF-JOIN)
--    Q: does an account that RECEIVES fraud money later re-appear SENDING money?
--    Exploration (sample):
-- -----------------------------------------------------------------------------
SELECT
    a.nameDest AS mule_account,
    a.amount   AS amount_received,
    b.amount   AS amount_resent,
    b.type     AS resend_type,
    a.isfraud  AS a_isfraud,
    b.isfraud  AS b_isfraud
FROM transactions AS a
JOIN transactions AS b
    ON a.nameDest = b.nameOrig
WHERE a.isfraud = 1
  AND b.step >= a.step
LIMIT 50;

--    FINDING: of 8,213 fraud transactions, destination accounts re-appear as
--    origin only 8 times, and NONE of those re-sends is fraud.
--    Mule accounts are single-use -> maybe the simulator cancells
--    fraudulent transactions. (In real data, mule-chain tracing would matter.)
SELECT
    COUNT(*)                        AS total_resends,
    COUNT(DISTINCT a.nameDest)      AS unique_mule_accounts,
    SUM(b.isfraud)                  AS resends_that_are_fraud
FROM transactions AS a
JOIN transactions AS b
    ON a.nameDest = b.nameOrig
WHERE a.isfraud = 1
  AND b.step >= a.step;


-- -----------------------------------------------------------------------------
-- 6. isFlaggedFraud is a useless rule
--    FINDING: the simulator's built-in flag (TRANSFER > 200,000) catches almost
--    no real fraud -> confirms need for a real model, very poor indicator.
-- -----------------------------------------------------------------------------
SELECT
    isFlaggedFraud,
    isFraud,
    COUNT(*)      AS total_tx,
    MIN(amount)   AS min_amount,
    MAX(amount)   AS max_amount
FROM transactions
WHERE type = 'TRANSFER' AND amount > 200000
GROUP BY isFlaggedFraud, isFraud;


-- #############################################################################
-- ##  PART 2 — DATA QUALITY / TARGET-LEAKAGE CHECK                            ##
-- ##  These balance columns are NOT used as model features. PaySim docs state ##
-- ##  balances are altered after fraud is cancelled -> using them = leakage.  ##
-- ##  Kept here only to DEMONSTRATE the leakage, then excluded from the model.##
-- #############################################################################

-- -----------------------------------------------------------------------------
-- 7. Origin balance consistency ( oldbalanceOrg - amount  vs  newbalanceOrig )
--    FINDING: legitimate rows mostly reconcile; fraud rows show large balance
--    "errors" -> the balance columns encode the label. This is target leakage.
-- -----------------------------------------------------------------------------
SELECT
    isfraud,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (
        WHERE ROUND((oldbalanceorg - amount)::numeric, 2) = ROUND(newbalanceorig::numeric, 2)
    ) AS orig_exact_match_count,
    COUNT(*) FILTER (
        WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2)
    ) AS orig_mismatch_count,
    ROUND(
        (COUNT(*) FILTER (
            WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2)
        ) * 100.0 / NULLIF(COUNT(*), 0))::numeric, 2
    ) AS orig_mismatch_pct,
    ROUND(
        AVG(ABS((oldbalanceorg - amount) - newbalanceorig)) FILTER (
            WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2)
        )::numeric, 2
    ) AS avg_inconsistency_amount
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isfraud;


-- -----------------------------------------------------------------------------
-- 8. Distribution (percentiles) of the balance error
--    FINDING: fraud error is far larger at every percentile (median 399k vs
--    164k; p95 9.9M vs 980k) -> quantifies how strongly balances leak the label.
-- -----------------------------------------------------------------------------
SELECT
    isfraud,
    COUNT(*) FILTER (
        WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2)
    ) AS total_mismatches,
    ROUND(MIN(ABS((oldbalanceorg - amount) - newbalanceorig)) FILTER (
        WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2)
    )::numeric, 2) AS min_error,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ABS((oldbalanceorg - amount) - newbalanceorig))
        FILTER (WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2))::numeric, 2) AS p25_error,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ABS((oldbalanceorg - amount) - newbalanceorig))
        FILTER (WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2))::numeric, 2) AS median_error,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ABS((oldbalanceorg - amount) - newbalanceorig))
        FILTER (WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2))::numeric, 2) AS p75_error,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS((oldbalanceorg - amount) - newbalanceorig))
        FILTER (WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2))::numeric, 2) AS p95_error,
    ROUND(MAX(ABS((oldbalanceorg - amount) - newbalanceorig)) FILTER (
        WHERE ROUND((oldbalanceorg - amount)::numeric, 2) <> ROUND(newbalanceorig::numeric, 2)
    )::numeric, 2) AS max_error
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isfraud;


-- #############################################################################
-- ##  PART 3 — FEATURE ENGINEERING                                           ##
-- #############################################################################

-- -----------------------------------------------------------------------------
-- 9. Model-ready feature view (LEAKAGE-FREE by design)
--    Design decisions, all justified by the EDA above:
--      * Restrict to TRANSFER + CASH_OUT   (query 1: fraud lives only here)
--      * amount                            (strong legitimate signal)
--      * hour_of_day = step % 24           (temporal signal)
--      * is_night flag for hours 2-6       (query 3: fraud rate ~350x higher)
--      * NO balance columns, NO account IDs -> avoids the leakage of queries 7-8
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS fraud_features;

CREATE VIEW fraud_features_2 AS
SELECT
    amount,
    type,
    step,
    step % 24 AS hour_of_day,
    CASE WHEN (step % 24) IN (2,3,4,5,6) THEN 1 ELSE 0 END AS is_night,
    isfraud
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT');

-- Sanity check
SELECT * FROM fraud_features LIMIT 10;