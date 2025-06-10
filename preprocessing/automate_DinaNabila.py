import pandas as pd
# untuk load dataset
import urllib.request as urllib
from io import BytesIO
# untuk normalisasi fitur
from sklearn.preprocessing import MinMaxScaler


# ============
# LOAD DATASET
# ============
# buka file dataset dari url menggunakan urllib
dataset_url = 'https://raw.githubusercontent.com/dinanabila/Eksperimen_SML_DinaNabila/a11c4f623480c73695aa76c54d1854108e9d812b/raw-dataset/train_egg_sales.csv'
response = urllib.urlopen(dataset_url)
dataset = BytesIO(response.read())

# masukkan dataset ke dataframe pandas
df = pd.read_csv(dataset, sep=';')


# =================================
# TAMBAH KOLOM MONTH DAN YEAR_INDEX
# =================================
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['year_index'] = df['year'] - df['year'].min()
df = df.drop('year', axis=1)


# =====================================
# GANTI DATA OUTLIER DENGAN INTERPOLASI
# =====================================
# jadikan kolom Date sebagai index
df.set_index('Date', inplace=True)

# tandai data outlier (Maret s.d. April 2020) sebagai NaN
df.loc['2020-03-01':'2020-04-30', 'Egg Sales'] = None

# interpolasi secara time-based
df['Egg Sales'] = df['Egg Sales'].interpolate(method='time')


# ===========================
# NORMALISASI FITUR EGG SALES
# ===========================
# inisialisasi scaler minmax untuk normalisasi
scaler = MinMaxScaler()

# fit dan transform
df['Egg Sales'] = scaler.fit_transform(df[['Egg Sales']])


# ============================================
# BAGI DATASET MENJADI DATA LATIH DAN VALIDASI
# ============================================
# bagi dataset 70:30
SPLIT_TIME = int(len(df) * 0.7)
x_train = df[:SPLIT_TIME]
x_valid = df[SPLIT_TIME:]


# ======================================
# EXPORT DATASET HASIL PREPROCESS KE CSV
# ======================================
x_train.to_csv("preprocessing/x_train_preprocessing.csv", index=True, index_label="Date")
x_valid.to_csv("preprocessing/x_valid_preprocessing.csv", index=True, index_label="Date")