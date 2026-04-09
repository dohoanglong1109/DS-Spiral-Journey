# Cycle 02: Telco customer churn

- Main purpose: Predict how many percentage that this customer will leave (churn)
- Baseline model: Logistic Regression
- Project's structure:
├── data
│   ├── 01_raw
│   ├── 02_interim
│   └── 03_processed
├── notebooks           <- Quick test notebooks file
├── src
│   ├── __init__.py
│   ├── data            <- Loading and saving data
│   ├── features        <- Feature Engineering
│   ├── models          <- Chứa script training và inference (dự báo)
│   └── visualization   <- Matplotlib, Seaborn
├── models              <- Saving trained data (file .pkl, .joblib)
├── configs             <- config.yaml (parameters, direction)
├── reports
│   └── figures
├── requirements.txt
└── README.md
