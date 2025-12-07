import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure reports dir exists
os.makedirs('reports', exist_ok=True)

# Data Generation (Synthetic for demo; replace with pd.read_csv for real data)
np.random.seed(42)  # Reproducible
n_rows = 10000

# Core columns
provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal',
             'Eastern Cape', 'Mpumalanga', 'Limpopo', 'North West']
province_weights = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]  # Gauteng dominant

# Fixed TransactionMonth: Sample from historical 19 months (2014-02 to 2015-08)
historical_months = pd.date_range(
    start='2014-02-01', end='2015-08-31', freq='ME').strftime('%Y-%m')

# Fixed VehicleIntroDate: Sample from realistic 432 months (1980-01 to 2015-12)
intro_months = pd.date_range(
    start='1980-01-01', end='2015-12-31', freq='ME').strftime('%Y-%m')

df = pd.DataFrame({
    'UnderwrittenCoverID': np.random.randint(10000, 99999, n_rows),
    'PolicyID': np.random.randint(1000, 9999, n_rows),
    # Sample from historical months
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
    'Country': 'South Africa',
    'Province': np.random.choice(provinces, n_rows, p=province_weights),
    'PostalCode': np.random.randint(1000, 9999, n_rows),
    'MainCrestaZone': np.random.choice(['Urban', 'Rural'], n_rows, p=[0.7, 0.3]),
    'SubCrestaZone': np.random.choice(['A', 'B', 'C'], n_rows),
    'ItemType': 'Vehicle',
    'Mmcode': np.random.randint(100000, 999999, n_rows),
    'VehicleType': np.random.choice(['Sedan', 'Hatchback', 'SUV'], n_rows, p=[0.5, 0.3, 0.2]),
    'RegistrationYear': np.random.randint(2000, 2016, n_rows),
    'Make': np.random.choice(['Toyota', 'VW', 'BMW', 'Ford'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
    'Model': 'Generic Model',
    'Cylinders': np.random.choice([4, 6, 8], n_rows, p=[0.7, 0.2, 0.1]),
    'Cubiccapacity': np.random.normal(2000, 500, n_rows).astype(int),
    'Kilowatts': np.random.normal(100, 30, n_rows).astype(int),
    'Bodytype': 'Passenger Car',
    'NumberOfDoors': np.random.choice([2, 4, 5], n_rows, p=[0.1, 0.8, 0.1]),
    # Sample from realistic intro dates
    'VehicleIntroDate': np.random.choice(intro_months, n_rows),
    'CustomValueEstimate': np.random.normal(150000, 50000, n_rows),
    'AlarmImmobiliser': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
    'TrackingDevice': np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
    'CapitalOutstanding': np.random.normal(50000, 20000, n_rows),
    'NewVehicle': np.random.choice([0, 1], n_rows, p=[0.2, 0.8]),
    'WrittenOff': np.random.choice([0, 1], n_rows, p=[0.95, 0.05]),
    'Rebuilt': np.random.choice([0, 1], n_rows, p=[0.98, 0.02]),
    'Converted': 0,
    'CrossBorder': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
    'NumberOfVehiclesInFleet': np.random.choice([1, 2, 3], n_rows, p=[0.8, 0.15, 0.05]),
    'SumInsured': np.random.normal(200000, 50000, n_rows),
    'TermFrequency': np.random.choice(['Annual', 'Monthly'], n_rows, p=[0.7, 0.3]),
    'CalculatedPremiumPerTerm': np.random.normal(6000, 2000, n_rows),
    'ExcessSelected': np.random.choice([1000, 2000, 5000], n_rows),
    'CoverCategory': 'Private Vehicle',
    'CoverType': np.random.choice(['Comprehensive', 'Third Party'], n_rows, p=[0.7, 0.3]),
    'CoverGroup': 'Motor',
    'Section': 'Section 1',
    'Product': 'Standard Policy',
    'StatutoryClass': 'A',
    'StatutoryRiskType': 'Private'
})

# Financials: Premium ~3% of SumInsured + noise
df['TotalPremium'] = (df['SumInsured'] * 0.03) + \
    np.random.normal(0, 1000, n_rows)

# Claims: 20% have claims, severity exponential
claim_prob = 0.2
has_claim = np.random.binomial(1, claim_prob, n_rows)
severity = np.random.exponential(1000, n_rows) * has_claim
severity = np.clip(severity, 0, 10000)  # Cap outliers
df['TotalClaims'] = severity

# Ensure data/raw exists
os.makedirs('data/raw', exist_ok=True)

# Save raw data
df.to_csv('data/raw/insurance_data.csv', index=False)
print(
    f"Generated {len(df)} rows of synthetic data saved to data/raw/insurance_data.csv")


class InsuranceEDA:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        print(
            f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def data_summary(self):
        """Descriptive stats and structure check."""
        print("Data Types:\n", self.df.dtypes.head())  # Sample
        print("\nDescriptive Stats:\n", self.df[[
              'TotalPremium', 'TotalClaims', 'SumInsured']].describe())
        print("\nMissing Values:\n", self.df.isnull().sum().sum())  # Total

    def univariate_analysis(self):
        """Histograms and bar charts."""
        num_cols = ['TotalPremium', 'TotalClaims']
        for col in num_cols:
            plt.figure(figsize=(8, 4))
            self.df[col].hist(bins=50)
            plt.title(f'Distribution of {col}')
            plt.savefig(f'reports/{col}_hist.png')
            plt.close()

        cat_cols = ['Province', 'Gender', 'VehicleType']
        for col in cat_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 4))
                self.df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.savefig(f'reports/{col}_bar.png')
                plt.close()

    def bivariate_analysis(self):
        """Correlations and loss ratio by groups."""
        self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        print("Overall Loss Ratio:", self.df['LossRatio'].mean())

        # By Province, VehicleType, Gender
        print("\nLoss Ratio by Province:\n", self.df.groupby(
            'Province')['LossRatio'].agg(['mean', 'std']).round(4))
        print("\nLoss Ratio by VehicleType:\n", self.df.groupby(
            'VehicleType')['LossRatio'].agg(['mean', 'std']).round(4))
        print("\nLoss Ratio by Gender:\n", self.df.groupby(
            'Gender')['LossRatio'].agg(['mean', 'std']).round(4))

        # Scatter: Premium vs Claims by PostalCode (sample)
        sample_df = self.df.groupby('PostalCode').agg(
            {'TotalPremium': 'mean', 'TotalClaims': 'mean'}).reset_index().head(50)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sample_df, x='TotalPremium',
                        y='TotalClaims', hue='PostalCode', palette='tab10')
        plt.title('Avg Premium vs Claims by ZipCode')
        plt.savefig('reports/premium_claims_scatter.png')
        plt.close()

        # Correlation matrix
        corr_cols = ['TotalPremium', 'TotalClaims',
                     'SumInsured', 'RegistrationYear']
        corr = self.df[corr_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('reports/correlation_heatmap.png')
        plt.close()

    def outlier_detection(self):
        """Box plots for outliers."""
        num_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        self.df[num_cols].boxplot(figsize=(10, 6))
        plt.title('Outliers in Key Numerical Features')
        plt.savefig('reports/outliers_boxplot.png')
        plt.close()
        # Z-score
        z_scores = np.abs(stats.zscore(self.df['TotalClaims'].dropna()))
        outliers = (z_scores > 3).sum()
        print(f"Outlier count (Z > 3) in TotalClaims: {outliers}")

    def trends_geography(self):
        """Compare by geography."""
        geo_group = self.df.groupby(['Province', 'CoverType'])[
            'TotalPremium'].mean().unstack()
        geo_group.plot(kind='bar', figsize=(12, 6))
        plt.title('Avg Premium by Province and CoverType')
        plt.savefig('reports/premium_by_province_cover.png')
        plt.close()


# Run EDA
if __name__ == "__main__":
    eda = InsuranceEDA('data/raw/insurance_data.csv')
    eda.data_summary()
    eda.univariate_analysis()
    eda.bivariate_analysis()
    eda.outlier_detection()
    eda.trends_geography()

    # 3 Creative Plots
    # 1. Loss Ratio Heatmap
    heatmap_data = eda.df.groupby(['Province', 'Gender'])[
        'LossRatio'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', fmt='.2%')
    plt.title('Loss Ratio Heatmap: Province x Gender')
    plt.savefig('reports/loss_ratio_heatmap.png')
    plt.close()

    # 2. Claims Trends by Make (using TransactionMonth)
    eda.df['Month'] = pd.to_datetime(eda.df['TransactionMonth'])
    trends = eda.df.groupby(['Make', eda.df['Month'].dt.to_period('M')])[
        'TotalClaims'].mean().reset_index()
    trends['Period'] = trends['Month'].astype(str)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trends, x='Period', y='TotalClaims', hue='Make')
    plt.title('Claims Trends by Make Over Time')
    plt.xticks(rotation=45)
    plt.savefig('reports/claims_by_make_trends.png')
    plt.close()

    # 3. Violin by VehicleType
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=eda.df, x='VehicleType', y='TotalClaims')
    plt.title('Claims Distribution by VehicleType')
    plt.savefig('reports/claims_violin_vehicletype.png')
    plt.close()

    print("EDA complete: Plots saved to reports/")

# Temporal trend check (bonus)
print("\nTemporal Trends:\n", eda.df.groupby('TransactionMonth')
      ['TotalClaims'].agg(['mean', 'count']).head())
