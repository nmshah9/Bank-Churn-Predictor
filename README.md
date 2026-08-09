# Bank Customer Churn Prediction — End-to-End Deep Learning Project
**by nmshah9**

An end-to-end ML project that predicts bank customer churn using an Artificial Neural Network
(ANN), tuned with KerasTuner, and benchmarked against classical ML models — plus a branded
Streamlit app for live predictions.

## 📁 Project Structure
```
bank_churn/
├── Bank_Customer_Churn_Prediction.ipynb   # Full EDA -> ANN -> KerasTuner -> comparison notebook
├── app.py                                  # Streamlit app (custom nmshah9 branding)
├── banner.png                              # Your branding banner (used in the app)
├── requirements.txt
├── data/
│   └── Churn_Modeling.csv                  # Source dataset (10,000 customers)
├── src/                                    # Modular, reusable Python source
│   ├── preprocessing.py                    # Cleaning, feature engineering, encoding, scaling
│   ├── model_builder.py                    # Baseline + KerasTuner-tunable ANN architectures
│   ├── train_baseline.py                   # Trains baseline ANN (Dropout + EarlyStopping + Checkpoint)
│   ├── tune_model.py                       # KerasTuner RandomSearch + retrains best config
│   └── compare_models.py                   # Benchmarks LogReg / RandomForest / both ANNs
├── models/                                 # Generated after training (scaler, encoders, .keras models)
└── plots/                                  # Generated EDA & training-curve plots (from the notebook)
```

## 🚀 Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 📓 Step 1 — Run the Notebook (trains everything, saves all artifacts)

Open `Bank_Customer_Churn_Prediction.ipynb` in Jupyter and **Run All**:

```bash
jupyter notebook Bank_Customer_Churn_Prediction.ipynb
```

This notebook:
1. Loads & explores the data (EDA + visualizations, saved to `plots/`)
2. Checks distributions/skewness and shows how transformations would be applied
3. Detects & caps outliers (IQR method), engineers new features, encodes categoricals
4. Scales features and splits train/test
5. Builds a multi-hidden-layer ANN with **Dropout** regularization
6. Trains it with **EarlyStopping** + **ModelCheckpoint** (+ ReduceLROnPlateau)
7. Runs a **KerasTuner** `RandomSearch` over layers/units/activation/optimizer/learning rate
8. Retrains the best-found architecture (again with EarlyStopping + ModelCheckpoint)
9. Compares Logistic Regression, Random Forest, Baseline ANN, and Tuned ANN
10. Saves everything the Streamlit app needs into `models/`

Alternatively, run the equivalent scripts directly from the project root:
```bash
python src/train_baseline.py
python src/tune_model.py
python src/compare_models.py
```

## 🖥️ Step 2 — Run the Streamlit App

Once `models/` contains `scaler.pkl`, `encoders.pkl`, `feature_columns.pkl`, and at least one
`.keras` model file (produced by Step 1), launch the app from the project root:

```bash
streamlit run app.py
```

The app shows your `banner.png` branding at the top, lets you enter a customer profile in the
sidebar, choose which trained ANN to use, and get a live churn probability with a
color-coded verdict — plus the full model comparison table.

## 📊 Results Summary (this run)

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------------|----------|-----------|--------|----------|---------|
| Random Forest            | 0.868    | 0.842     | 0.432  | 0.571    | 0.863   |
| Baseline ANN              | 0.869    | 0.788     | 0.484  | 0.600    | 0.861   |
| KerasTuner-Tuned ANN       | 0.859    | 0.802     | 0.408  | 0.541    | 0.856   |
| Logistic Regression        | 0.814    | 0.616     | 0.229  | 0.333    | 0.782   |

The **Baseline ANN** and **Random Forest** perform best overall; both ANN variants and the
Random Forest substantially outperform plain Logistic Regression, confirming non-linear feature
interactions drive churn (e.g. Age × NumOfProducts × IsActiveMember).

## 🔧 Notes
- Deployed to Streamlit Community Cloud? Make sure `data/Churn_Modeling.csv` and the `models/`
  folder (or the retraining scripts) are included in your GitHub repo, since Streamlit Cloud
  runs from a clean checkout.
- `models/*.keras`, `models/*.pkl` and `plots/*.png` are generated artifacts — safe to
  `.gitignore` if you'd rather have the app/notebook regenerate them on first run.
