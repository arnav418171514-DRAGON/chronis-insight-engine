# Architectural Decisions & Methodology

## 1. Grounding in Clinical Baselines
To prevent the engine from relying on arbitrary developer assumptions (e.g., "a 10% drop in sleep is bad"), anomaly thresholds are cross-referenced with authorized medical guidelines (WHO, CDC, NIH). For example, a sleep anomaly is only flagged if it represents a statistical deviation *and* drops below the recommended 7.0-hour minimum.

## 2. Statistical Methodology
* **Anomaly Detection:** Utilizes a Rolling Z-Score (default 7-day window). A Z-score threshold of > 2.0 is used to ensure events fall outside the ~95% confidence interval of the user's normal behavior, preventing alert fatigue.
* **Pattern Discovery:** Utilizes Ordinary Least Squares (OLS) Linear Regression over a 14-day window. Trends are only surfaced if the slope has a statistically significant p-value (p < 0.05).

## 3. Evidence Sufficiency (Abstention Logic)
The system is built to explicitly refuse to generate insights if the data is sparse. It requires a minimum volume of historical logs and checks for null values. Confidence scores are dynamically penalized based on the ratio of missing data in the observation window.

## 4. Known Failure Modes & Limitations
* **Linear Assumption:** The OLS regression assumes behavioral changes occur linearly. It may fail to accurately capture cyclical patterns (e.g., behaviors that systematically spike only on weekends).
* **Z-Score Sensitivity:** Standard deviation calculations on highly rigid habits (e.g., logging exactly 8.0 hours of sleep every night) can result in a near-zero denominator, making minor normal deviations appear mathematically massive. Handled computationally via `np.where`, but analytically it remains a blind spot.