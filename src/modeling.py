import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import os

# Ensure dirs
os.makedirs('reports', exist_ok=True)

# Data Gen/Load (self-contained; comment gen for real)
np.random.seed(42)
n_rows = 10000
provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal',
             'Eastern Cape', 'Mpumalanga', 'Limpopo', 'North West']
province_weights = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
historical_months = pd.date_range(
    start='2014-02-01', end='2015-08-31', freq='ME').strftime('%Y-%m')
intro_months = pd.date_range(
    start='1980-01-01', end='2015-12-31', freq='ME').strftime('%Y-%m')

df = pd.DataFrame({
    'UnderwrittenCoverID': np.random.randint(10000, 99999, n_rows),
    'PolicyID': np.random.randint(1000, 9999, n_rows),
    'TransactionMonth': np.random.choice(historical_months, n_rows),
    'IsVATRegistered': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    'Citizenship': np.random.choice(['SA', 'Other'], n_rows, p=[0.95, 0.05]),
    'LegalType': np.random.choice(['Individual', 'Company'], n_rows, p=[0.9, 0.1]),
    'Title': np.random.choice(['Mr', 'Mrs', 'Ms'], n_rows),
    'Language': np.random.choice(['English', 'Afrikaans'], n_rows, p=[0.7, 0.3]),
    'Bank': np.random.choice(['ABSA', 'Standard Bank', 'FNB'], n_rows),
    'AccountType': np.random.choice(['Cheque', 'Savings'], n_rows),
    'MaritalStatus': np.random.choice(['Single', 'Married'], n_rows, p=[0.4, 0.6]),
    'Gender': np.random.choice(['M', 'F'], n_rows, p=[0.6, 0.4]),
    'Country': ['South Africa'] * n_rows,
    'Province': np.random.choice(provinces, n_rows, p=province_weights),
    'PostalCode': np.random.randint(1000, 9999, n_rows),
    'MainCrestaZone': np.random.choice(['Urban', 'Rural'], n_rows, p=[0.7, 0.3]),
    'SubCrestaZone': np.random.choice(['A', 'B', 'C'], n_rows),
    'ItemType': ['Vehicle'] * n_rows,
    'Mmcode': np.random.randint(100000, 999999, n_rows),
    'VehicleType': np.random.choice(['Sedan', 'Hatchback', 'SUV'], n_rows, p=[0.5, 0.3, 0.2]),
    'RegistrationYear': np.random.randint(2000, 2016, n_rows),
    'Make': np.random.choice(['Toyota', 'VW', 'BMW', 'Ford'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
    'Model': ['Generic Model'] * n_rows,
    'Cylinders': np.random.choice([4, 6, 8], n_rows, p=[0.7, 0.2, 0.1]),
    'Cubiccapacity': np.random.normal(2000, 500, n_rows).astype(int),
    'Kilowatts': np.random.normal(100, 30, n_rows).astype(int),
    'Bodytype': ['Passenger Car'] * n_rows,
    'NumberOfDoors': np.random.choice([2, 4, 5], n_rows, p=[0.1, 0.8, 0.1]),
    'VehicleIntroDate': np.random.choice(intro_months, n_rows),
    'CustomValueEstimate': np.random.normal(150000, 50000, n_rows),
    'AlarmImmobiliser': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
    'TrackingDevice': np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
    'CapitalOutstanding': np.random.normal(50000, 20000, n_rows),
    'NewVehicle': np.random.choice([0, 1], n_rows, p=[0.2, 0.8]),
    'WrittenOff': np.random.choice([0, 1], n_rows, p=[0.95, 0.05]),
    'Rebuilt': np.random.choice([0, 1], n_rows, p=[0.98, 0.02]),
    'Converted': [0] * n_rows,
    'CrossBorder': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
    'NumberOfVehiclesInFleet': np.random.choice([1, 2, 3], n_rows, p=[0.8, 0.15, 0.05]),
    'SumInsured': np.random.normal(200000, 50000, n_rows),
    'TermFrequency': np.random.choice(['Annual', 'Monthly'], n_rows, p=[0.7, 0.3]),
    'CalculatedPremiumPerTerm': np.random.normal(6000, 2000, n_rows),
    'ExcessSelected': np.random.choice([1000, 2000, 5000], n_rows),
    'CoverCategory': ['Private Vehicle'] * n_rows,
    'CoverType': np.random.choice(['Comprehensive', 'Third Party'], n_rows, p=[0.7, 0.3]),
    'CoverGroup': ['Motor'] * n_rows,
    'Section': ['Section 1'] * n_rows,
    'Product': ['Standard Policy'] * n_rows,
    'StatutoryClass': ['A'] * n_rows,
    'StatutoryRiskType': ['Private'] * n_rows
})

df['TotalPremium'] = (df['SumInsured'] * 0.03) + \
    np.random.normal(0, 1000, n_rows)
claim_prob = 0.2
has_claim = np.random.binomial(1, claim_prob, n_rows)
severity = np.random.exponential(1000, n_rows) * has_claim
df['TotalClaims'] = np.clip(severity, 0, 10000)

# For real: df = pd.read_csv('data/raw/insurance_data.csv')

# Data Prep: Subset for severity (claims >0)
df_sev = df[df['TotalClaims'] > 0].copy()

# Feature Eng
df_sev['VehicleAge'] = 2015 - df_sev['RegistrationYear']
df_sev['IntroAge'] = pd.to_datetime(
    df_sev['VehicleIntroDate'], format='%Y-%m').dt.year
df_sev['IntroAge'] = 2015 - df_sev['IntroAge']
df_sev['IsNewVehicle'] = (df_sev['VehicleAge'] <= 1).astype(int)
df_sev['HasSecurity'] = df_sev['AlarmImmobiliser'] + df_sev['TrackingDevice']

# Features
features = ['VehicleAge', 'IntroAge', 'SumInsured', 'Cylinders', 'Cubiccapacity', 'Kilowatts',
            'CustomValueEstimate', 'IsNewVehicle', 'HasSecurity', 'Province', 'Gender', 'VehicleType']
df_model = df_sev[features + ['TotalClaims']].dropna()

# Encoding
le_prov = LabelEncoder()
le_gender = LabelEncoder()
le_veh = LabelEncoder()
df_model['Province'] = le_prov.fit_transform(df_model['Province'])
df_model['Gender'] = le_gender.fit_transform(df_model['Gender'])
df_model['VehicleType'] = le_veh.fit_transform(df_model['VehicleType'])

X = df_model[features]
y = df_model['TotalClaims']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Models
models = {
    'LinearReg': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

results = pd.DataFrame(columns=['Model', 'RMSE', 'R2'])

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.loc[len(results)] = [name, rmse, r2]
    print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")

results.to_csv('reports/model_results.csv', index=False)

# Feature Importance (XGBoost)
xgb = models['XGBoost']
importance = pd.DataFrame({'Feature': features, 'Importance': xgb.feature_importances_}).sort_values(
    'Importance', ascending=False)
print("\nTop Features (XGBoost):\n", importance.head(5).to_string())
importance.to_csv('reports/feature_importance.csv', index=False)

# SHAP
explainer = shap.Explainer(xgb)
sample_test = X_test.sample(min(100, len(X_test)), random_state=42)
shap_values = explainer(sample_test)
shap.summary_plot(shap_values, sample_test, show=False)
plt.savefig('reports/shap_summary.png')
plt.close()
print("SHAP plot saved – e.g., SumInsured high = +R500 predicted claim.")

# Per-zip Linear Reg (sample 3 zips)
zip_sample = df_sev['PostalCode'].value_counts().head(3).index
zip_results = {}
for z in zip_sample:
    df_z = df_sev[df_sev['PostalCode'] == z].copy()
    if len(df_z) > 10:
        df_z_model = df_z[features + ['TotalClaims']].dropna()
        if len(df_z_model) > 5:
            X_z = df_z_model[features].copy()
            y_z = df_z_model['TotalClaims']
            # Encode for zip
            for col in ['Province', 'Gender', 'VehicleType']:
                le = LabelEncoder()
                X_z[col] = le.fit_transform(X_z[col].astype(str))
            lr_z = LinearRegression().fit(X_z, y_z)
            r2_z = r2_score(y_z, lr_z.predict(X_z))
            zip_results[z] = r2_z
print("\nPer-Zip R2 (sample):\n", zip_results)

# Premium Optimization: Binary Claim Prob
df_all = df.copy()
df_all['VehicleAge'] = 2015 - df_all['RegistrationYear']
df_all['IntroAge'] = pd.to_datetime(
    df_all['VehicleIntroDate'], format='%Y-%m').dt.year
df_all['IntroAge'] = 2015 - df_all['IntroAge']
df_all['IsNewVehicle'] = (df_all['VehicleAge'] <= 1).astype(int)
df_all['HasSecurity'] = df_all['AlarmImmobiliser'] + df_all['TrackingDevice']
df_all['ClaimProb'] = (df_all['TotalClaims'] > 0).astype(int)

features_bin = ['VehicleAge', 'IntroAge', 'SumInsured', 'Cylinders', 'Cubiccapacity', 'Kilowatts',
                'CustomValueEstimate', 'IsNewVehicle', 'HasSecurity', 'Province', 'Gender', 'VehicleType']
df_bin = df_all[features_bin + ['ClaimProb']].dropna()

le_prov_bin = LabelEncoder()
le_gender_bin = LabelEncoder()
le_veh_bin = LabelEncoder()
df_bin['Province'] = le_prov_bin.fit_transform(df_bin['Province'])
df_bin['Gender'] = le_gender_bin.fit_transform(df_bin['Gender'])
df_bin['VehicleType'] = le_veh_bin.fit_transform(df_bin['VehicleType'])

X_bin = df_bin[features_bin]
y_bin = df_bin['ClaimProb']
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=42)

rf_class = RandomForestClassifier(n_estimators=100, random_state=42)
rf_class.fit(X_train_bin, y_train_bin)
y_pred_bin = rf_class.predict_proba(X_test_bin)[:, 1]
claim_severity = df_all['TotalClaims'][df_all['TotalClaims'] > 0].mean()
risk_premium = np.mean(y_pred_bin * claim_severity) + 200  # + loading/margin
print(f"Sample Risk-Based Premium: {risk_premium:.2f} R")

print("Modeling complete; check reports/ for CSVs/plots.")
