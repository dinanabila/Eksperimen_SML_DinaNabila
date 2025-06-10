import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
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
    x_train.to_csv("preprocessing/preprocessed-dataset/x_train_preprocessing.csv", index=True, index_label="Date")
    x_valid.to_csv("preprocessing/preprocessed-dataset/x_valid_preprocessing.csv", index=True, index_label="Date")


# ============
# LOAD DATASET
# ============
# load dataset dari repo github
df_telur = pd.read_csv('raw-dataset/train_egg_sales.csv', sep=';')


# ===============================
# JALANKAN FUNGSI PREPROCESS_DATA
# ===============================
preprocess_data(df_telur)