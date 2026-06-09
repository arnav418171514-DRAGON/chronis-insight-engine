# chronis-insight-engine
# Chronis Behavioral Insight Engine

An end-to-end data pipeline designed to parse daily behavioral logs, detect statistically significant anomalies, and discover long-term trends using clinical baselines.

## System Components
1. **Pattern Discovery:** Uses OLS Linear Regression (p < 0.05) to find 14-day behavioral trends.
2. **Anomaly Detection:** Uses 7-day Rolling Z-Scores combined with WHO/CDC clinical baselines to flag unusual daily deviations.
3. **Evidence Sufficiency Gate:** A strict verification layer that forces the system to abstain if data volume or density is compromised.

## How to Run Locally

1. Clone the repository and navigate into the project directory.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy scipy
3. Run the master insight generator script:
   ```bash
   python main.py
4. The generated insights will output directly to the terminal, detailing the behavior status, confidence scores, and mathematical reasoning for all users.

## Running the Tests

To verify the engine's statistical accuracy, clinical baseline triggers, and evidence sufficiency gates, a complete test suite is provided.

Run the tests using pytest:
```bash
pytest tests/test_analyzer.py -v