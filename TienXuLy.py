import pandas as pd
import numpy as np

# Đọc dữ liệu đầu vào từ 2 file audit_risk.csv và trial.csv
audit_risk = pd.read_csv("./audit_data/audit_risk.csv")
trial = pd.read_csv("./audit_data/trial.csv")

#print(audit_risk.info())
#print(trial.info())

# Đổi tên các cột đọc từ file trial.csv để giống với audit_risk.csv
trial.columns = ['Sector_score','LOCATION_ID', 'PARA_A', 'Score_A', 'PARA_B', 'Score_B',
                 'TOTAL', 'numbers', 'Marks',
                 'Money_Value', 'MONEY_Marks', 'District',
                 'Loss', 'LOSS_SCORE', 'History', 'History_score', 'Score', 'Risk_trial' ]

# Chuẩn hóa dữ liệu
# Do cột Score_A và Score_B trong file trial.csv được lưu trữ khác với audit_risk.csv
# nên cần chuẩn hóa lại bằng các cột Score_A và Score_B trong audit_risk.csv
print()
print("Các giá trị Score_A trong [trial]:", trial['Score_A'].unique())
print("Các giá trị Score_B trong [trial]:", trial['Score_B'].unique())
print()
print("Các giá trị Score_A trong [audit_risk]:", audit_risk['Score_A'].unique())
print("Các giá trị Score_B trong [audit_risk]:", audit_risk['Score_B'].unique())

trial['Score_A'] = trial['Score_A']/10
trial['Score_B'] = trial['Score_B']/10

# Tìm các cột giống nhau giữa 2 file, lưu vào same_columns
# Dòng lệnh này để tự động tìm ra các cột giống nhau, không cần tìm thủ công

same_columns = np.intersect1d(audit_risk.columns, trial.columns)
same_columns = list(same_columns)

# Kết hợp dữ liệu từ 2 file audit_risk và trial (dựa vào các cột giống nhau), vẫn dữ lại các cột  riêng của 2 file
merged_df = pd.merge(audit_risk, trial, how='outer', on = same_columns)

# print(merged_df.info())

# Xóa cột Risk_trial vì: Dữ liệu của Risk_trial không khớp với cột Risk(Nhãn trong audit_risk).
df = merged_df.drop(['Risk_trial'], axis = 1)

# Kiểm tra lại tập dữ liệu
# df.info()

# Cột Money_Value có một giá trị Null -> dùng giá trị trung bình để điền vào nơi thiếu
df['Money_Value'] = df['Money_Value'].fillna(df['Money_Value'].median())

# Hiển thị danh sách các dữ liệu khác nhau và số lượng của chúng trong mỗi trường dữ liệu
# for col in df.columns:
#     print(df[col].value_counts())

# Do tất cả dữ liệu trong Dectection_Risk là giống nhau nên cần xóa nó đi (không ảnh hưởng đến kết quả dự đoán).
df = df.drop(['Detection_Risk'], axis = 1)

# In ra kiểu dữ liệu của các cột
# print(df.dtypes)

# Do LOCATION_ID lưu đa số các dữ liệu dạng số nhưng đang ở kiểu object nên cần loại bỏ các giá trị object (không hợp lệ)
# In ra các giá trị duy nhất của cột LOCATION_ID
# print(df['LOCATION_ID'].value_counts())

# Do LOCATION_ID lưu đa số các dữ liệu dạng số nhưng đang ở kiểu object nên cần loại bỏ các giá trị object (không hợp lệ)
# In ra các giá trị duy nhất của cột LOCATION_ID
# print(df['LOCATION_ID'].value_counts())

# Loại bỏ LOHARU, NUH, SAFIDON trong cột LOCATION_ID
df = df[(df.LOCATION_ID != 'LOHARU')]
df = df[(df.LOCATION_ID != 'NUH')]
df = df[(df.LOCATION_ID != 'SAFIDON')]

## Chuyển dữ liệu thành kiểu float

df = df.astype(float)

# Kiểm tra lại kiểu dữ liệu của các cột
# print(df.dtypes)

# Xây dựng ma trận tương quan dữ liệu để loại bỏ các cột không cần thiết (tương quan cao)
import seaborn as sns
from sklearn.utils import resample
import matplotlib.pyplot as plt

correlation = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, annot=True, cmap='RdBu')
plt.subplots_adjust(left=0.08, right=1, top=0.992, bottom=0.158)
# plt.show()
plt.close()

# Xoá các cột tương quan cao ( >= 0.8) để tránh overfitting
df.drop(['Risk_A','Risk_B','TOTAL','Score','Audit_Risk','Score_B.1',
         'Risk_C','Marks','Risk_D','Inherent_Risk','Score_MV','RiSk_E',
         'District_Loss','PROB','Prob','Risk_F','History_score','LOSS_SCORE'], axis = 1,  inplace=True)

# Ma trận thể hiện độ tương quan sau khi xóa các trường dữ liệu tương quan cao
correlation = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, annot=True, cmap='RdBu')
plt.subplots_adjust(left=0.08, right=1, top=0.987, bottom=0.162)
# plt.show()
plt.close()

# print(df)

# Sau khi xem xét dữ liệu, ta thấy cột PARA_A và PARA_B sẽ suy ra được Score_A và Score_B
# PARA_A < 1 => Score_A = 2
# PARA_A >= 1 và <= 2 => Score_A = 4
# PARA_A > 2 => Score_A = 6
# Tương tự với Score_B

# Money_Value có giá trị < 6 thì Money_Marks = 2
# Money_Value có giá trị >= 6 và < 10 thì Money_Marks = 4
# Money_Value có giá trị >= 10 thì Money_Marks = 6

# # Kiểm tra xem Score_A có đúng theo quy tắc từ PARA_A hay không
# df['Score_A_computed'] = df['PARA_A'].apply(lambda x: 0.2 if x < 1 else (0.6 if x > 2 else 0.4))
# df['Score_B_computed'] = df['PARA_B'].apply(lambda x: 0.2 if x < 1 else (0.6 if x > 2 else 0.4))

# print(df[['Score_A', 'Score_A_computed']].dtypes)
# print(df[['Score_B', 'Score_B_computed']].dtypes)

# # So sánh với Score_A và Score_B gốc
# print(np.isclose(df['Score_A'], df['Score_A_computed']).all())  
# print(np.isclose(df['Score_B'], df['Score_B_computed']).all())

# mismatch_A = df[df['Score_A'] != df['Score_A_computed']]
# mismatch_B = df[df['Score_B'] != df['Score_B_computed']]

# print(mismatch_A[['PARA_A', 'Score_A', 'Score_A_computed']])
# print(mismatch_B[['PARA_B', 'Score_B', 'Score_B_computed']])

# # print(df)

# # Tính toán Money_Marks dựa trên quy tắc đã cho
# df['Money_Marks_computed'] = df['Money_Value'].apply(lambda x: 2 if x < 6 else (6 if x >= 10 else 4))

# # Kiểm tra xem Money_Marks có đúng với công thức không (dùng np.isclose để tránh sai số)
# print(np.isclose(df['MONEY_Marks'], df['Money_Marks_computed']).all())

# # Nếu có dòng nào sai lệch, in ra để kiểm tra
# mismatch = df[df['MONEY_Marks'] != df['Money_Marks_computed']]
# print(mismatch[['Money_Value', 'MONEY_Marks', 'Money_Marks_computed']])

# # Tính % dữ liệu không khớp giữa các cột
# diff_para_a = (df['Score_A'] != df['Score_A_computed']).mean()
# diff_para_b = (df['Score_B'] != df['Score_B_computed']).mean()
# diff_money = (df['MONEY_Marks'] != df['Money_Marks_computed']).mean()

# print(f"Phần trăm lệch PARA_A: {diff_para_a:.2%}")
# print(f"Phần trăm lệch PARA_B: {diff_para_b:.2%}")
# print(f"Phần trăm lệch Money_Value: {diff_money:.2%}")

df = df.drop(['PARA_A', 'PARA_B'], axis = 1)
df = df.drop(['Money_Value'], axis = 1)

# print(df)

# Chuẩn hóa miền giá trị:
df['Score_A'] = df['Score_A']*10
df['Score_B'] = df['Score_B']*10

# Sắp xếp lại các cột theo thứ tự mong muốn
desired_column_order = ['Sector_score', 'LOCATION_ID', 'Score_A', 'Score_B', 'numbers', 'CONTROL_RISK', 'MONEY_Marks', 'District', 'Loss', 'History', 'Risk']
df = df[desired_column_order]

# Kiểm tra nhãn 0/1 trong cột Risk
# Y chỉ chứa các nhãn
y = df["Risk"]

# Vẽ biểu đồ cột cho y.value_counts()
y_counts = y.value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=y_counts.index, y=y_counts.values)
plt.xlabel('Nhãn')
plt.ylabel('Số lượng')
plt.title('Số lượng nhãn 0 và 1 trong tập dữ liệu')
# plt.show()
plt.close()

# Cân bằng nhãn trong tập dữ liệu (cột Risk)
df_majority = df[df.Risk == 0]
df_minority = df[df.Risk == 1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=len(df_majority),    
                                 random_state=123) 

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Kiểm tra lại nhãn 0/1 sau khi cân bằng
y_upsampled = df_upsampled["Risk"]
y_counts_upsampled = y_upsampled.value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=y_counts_upsampled.index, y=y_counts_upsampled.values)
plt.xlabel('Nhãn')
plt.ylabel('Số lượng')
plt.title('Số lượng nhãn 0 và 1 trong tập dữ liệu sau khi cân bằng')
#plt.show()

#print(df.info())

df = df.sample(frac=1, random_state=123).reset_index(drop=True)

print(df.info())

# Lưu dữ liệu từ class_df vào một file .csv
df.to_csv('data.csv', index = False)
