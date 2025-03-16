import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# dataFame = "data.csv"  # Thay bằng đường dẫn đúng của bạn
# df = pd.read_csv(dataFame)

# num_samples = 30  # Số mẫu cần tạo
# new_samples = {col: np.random.choice(df[col], size=num_samples) for col in df.columns}

# new_df = pd.DataFrame(new_samples)

# new_df.insert(0, 'DoanhNghiep', range(1, num_samples + 1))

# new_file_path = "data_test.csv"
# new_df.to_csv(new_file_path, index=False)

dataFame = "data_test.csv"  # Thay bằng đường dẫn file của bạn
df = pd.read_csv(dataFame)

feature_columns = [col for col in df.columns if col != 'Risk']
features = df[feature_columns]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled_features, columns=feature_columns)

if 'Risk' in df.columns:
    df_scaled['Risk'] = df['Risk']
    
scaled_file_path = "data_scaled.csv"
df_scaled.to_csv(scaled_file_path, index=False)