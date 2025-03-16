import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('data.csv')

#Ccột cuối là nhãn
X = df.iloc[:, :-1].values  # Đặc trưng 
y = df.iloc[:, -1].values   # Nhãn 

#KNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("===============================================================")
print("                           KNN                                 ")
print("===============================================================")

# Tìm k tốt nhất
k_values = range(1, 21, 2)
variances = []

for k in k_values:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=0)
    # Chuẩn hóa dữ liệu với MinMaxScaler [0->1]
    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    KNN_model = KNeighborsClassifier(n_neighbors=k, p = 2)
    KNN_model.fit(X_train, y_train)
    y_pred = KNN_model.predict(X_test)
    variance = np.var(y_pred)
    variances.append(variance)

plt.figure(figsize=(10, 6))
plt.plot(k_values, variances, marker='o')
plt.title('Biểu đồ thể hiện tương quan giữa K và phương sai')
plt.xlabel('Giá trị của K')
plt.ylabel('Phương sai')
plt.xticks(np.arange(1, 21, 2))
plt.grid()
plt.show()
plt.close()

num_y_pred = []
accuracies = []
precisions = []
recalls = []
f1s = []

# Khởi tạo danh sách để lưu các giá trị dự đoán và nhãn thực tế
all_y_pred = []
all_y_test = []


# Chạy thử 10 lần
for i in range(10):
#Chia dữ liệu thành 2 phần: 1/3 dùng để test, 2/3 dùng để train
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state = i*2)
       # Chuẩn hóa dữ liệu với MinMaxScaler [0->1]
       sc_X = MinMaxScaler()
       X_train = sc_X.fit_transform(X_train)
       X_test = sc_X.transform(X_test)
       
       KNN_model = KNeighborsClassifier(n_neighbors=13, p = 2)
       KNN_model.fit(X_train, y_train)

       # Dự đoán trên tập dữ liệu kiểm tra
       y_pred = KNN_model.predict(X_test)
       num_y_pred.append(y_pred)

       # Tính accuracy, precision, recall, F1
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred)
       recall = recall_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred)
    
       print("Độ chính xác (Accuracy) trong lần chạy thứ ", i+1, " là: ", accuracy)
       print("Precision trong lần chạy thứ ", i+1, " là: ", precision)
       print("Recall trong lần chạy thứ ", i+1, " là: ", recall)
       print("F1 trong lần chạy thứ ", i+1, " là: ", f1)
       print("-------------------------------------------\n")


       # Lưu các giá trị dự đoán và nhãn thực tế
       all_y_pred.extend(y_pred)
       all_y_test.extend(y_test)

       accuracies.append(accuracy)
       precisions.append(precision)
       recalls.append(recall)
       f1s.append(f1)

#Vẽ biểu đồ ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix

# Tính ma trận nhầm lẫn tổng hợp
cm_total = confusion_matrix(all_y_test, all_y_pred)

# Vẽ ma trận nhầm lẫn tổng hợp
plt.figure(figsize=(10, 8))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues")
plt.title('Ma trận nhầm lẫn của K-Nearest Neighbors(KNN) qua 10 lần chạy')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()




# Tính độ chính xác trung bình
average_accuracy = np.mean(accuracies)
# Tính precision trung bình
average_precision = np.mean(precisions)
# Tính recall trung bình
average_recall = np.mean(recalls)
# Tính F1 trung bình
average_f1 = np.mean(f1s)

print("Độ chính xác trung bình (Average Accuracy) sau 10 lần chạy: ", average_accuracy)
print("Precision trung bình (Average Precision) sau 10 lần chạy: ", average_precision)
print("Recall trung bình (Average Recall) sau 10 lần chạy: ", average_recall)
print("F1 trung bình (Average F1) sau 10 lần chạy: ", average_f1)

print("===============================================================")
print("                        Bayes                                  ")
print("===============================================================")
#Bayes
from sklearn.naive_bayes import GaussianNB

num_y_pred = []
accuracies = []
precisions = []
recalls = []
f1s = []

all_y_pred = []
all_y_test = []

# Chạy thử 10 lần
for i in range(10):
# Chia dữ liệu thành 2 phần: 1/3 dùng để test, 2/3 dùng để train
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state = i*2)
       
       Bayes_model = GaussianNB()
    
       Bayes_model.fit(X_train, y_train)

       # Dự đoán trên tập dữ liệu kiểm tra
       y_pred = Bayes_model.predict(X_test)
       num_y_pred.append(y_pred)

       # Tính độ chính xác
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred)
       recall = recall_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred)
    
       print("Độ chính xác (Accuracy) trong lần chạy thứ ", i+1, " là: ", accuracy)
       print("Precision trong lần chạy thứ ", i+1, " là: ", precision)
       print("Recall trong lần chạy thứ ", i+1, " là: ", recall)
       print("F1 trong lần chạy thứ ", i+1, " là: ", f1)
       print("-------------------------------------------\n")

       # Lưu các giá trị dự đoán và nhãn thực tế
       all_y_pred.extend(y_pred)
       all_y_test.extend(y_test)

       accuracies.append(accuracy)
       precisions.append(precision)
       recalls.append(recall)
       f1s.append(f1)

#Vẽ biểu đồ ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix

# Tính ma trận nhầm lẫn tổng hợp
cm_total = confusion_matrix(all_y_test, all_y_pred)

# Vẽ ma trận nhầm lẫn tổng hợp
plt.figure(figsize=(10, 8))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues")
plt.title('Ma trận nhầm lẫn của Gaussian Navie Bayes qua 10 lần chạy')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# Tính độ chính xác trung bình
Bayes_average_accuracy = np.mean(accuracies)
# Tính precision trung bình
Bayes_average_precision = np.mean(precisions)
# Tính recall trung bình
Bayes_average_recall = np.mean(recalls)
# Tính F1 trung bình
Bayes_average_f1 = np.mean(f1s)

print("Độ chính xác trung bình (Average Accuracy) sau 10 lần chạy: ", Bayes_average_accuracy)
print("Precision trung bình (Average Precision) sau 10 lần chạy: ", Bayes_average_precision)
print("Recall trung bình (Average Recall) sau 10 lần chạy: ", Bayes_average_recall)
print("F1 trung bình (Average F1) sau 10 lần chạy: ", Bayes_average_f1)

print("===============================================================")
print("                        DeicisionTree                          ")
print("===============================================================")
#DeicisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Tìm giá trị tối ưu cho max_depth của Decision Tree
depth_range = range(1, 21)
depth_accuracies = []

for depth in depth_range:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    depth_accuracies.append(acc)

plt.figure(figsize=(10, 5))
plt.plot(depth_range, depth_accuracies, marker='s', linestyle='dashed', color='r')
plt.xlabel('Max Depth')
plt.ylabel('Độ chính xác')
plt.title(f'Độ chính xác của từng Max Depth')
plt.xticks(depth_range)
plt.grid()
plt.show()

# Tìm giá trị tối ưu cho min_samples_leaf của Decision Tree
leaf_range = range(1, 21)
leaf_accuracies = []

for leaf in leaf_range:
    tree_model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    leaf_accuracies.append(acc)


# Vẽ biểu đồ min_samples_leaf
plt.figure(figsize=(10, 5))
plt.plot(leaf_range, leaf_accuracies, marker='o', linestyle='dashed', color='b')
plt.xlabel('Giá trị Min Samples Leaf')
plt.ylabel('Độ chính xác')
plt.title(f'Độ chính xác của từng Min Samples Leaf')
plt.xticks(leaf_range)
plt.grid()
plt.show()

num_y_pred = []
accuracies = []
precisions = []
recalls = []
f1s = []

all_y_pred = []
all_y_test = []


for i in range(0,10):
# Chia dữ liệu thành 2 phần: 1/3 dùng để test, 2/3 dùng để train
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state = i*2)
       
       clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = i*2, max_depth=9, min_samples_leaf=8)
    
       clf_gini.fit(X_train, y_train)

       # Dự đoán trên tập dữ liệu kiểm tra
       y_pred = clf_gini.predict(X_test)
       num_y_pred.append(y_pred)

       # Tính độ chính xác
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred)
       recall = recall_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred)
    
       print("Độ chính xác (Accuracy) trong lần chạy thứ ", i+1, " là: ", accuracy)
       print("Precision trong lần chạy thứ ", i+1, " là: ", precision)
       print("Recall trong lần chạy thứ ", i+1, " là: ", recall)
       print("F1 trong lần chạy thứ ", i+1, " là: ", f1)
       print("-------------------------------------------\n")

       # Lưu các giá trị dự đoán và nhãn thực tế
       all_y_pred.extend(y_pred)
       all_y_test.extend(y_test)

       accuracies.append(accuracy)
       precisions.append(precision)
       recalls.append(recall)
       f1s.append(f1)

from sklearn.metrics import confusion_matrix

# Tính ma trận nhầm lẫn tổng hợp
cm_total = confusion_matrix(all_y_test, all_y_pred)

# Vẽ ma trận nhầm lẫn tổng hợp
plt.figure(figsize=(10, 8))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues")
plt.title('Ma trận nhầm lẫn của Decision Tree qua 10 lần chạy')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# Tính độ chính xác trung bình
DT_average_accuracy = np.mean(accuracies)
# Tính precision trung bình
DT_average_precision = np.mean(precisions)
# Tính recall trung bình
DT_average_recall = np.mean(recalls)
# Tính F1 trung bình
DT_average_f1 = np.mean(f1s)

print("Độ chính xác trung bình (Average Accuracy) sau 10 lần chạy: ", DT_average_accuracy)
print("Precision trung bình (Average Precision) sau 10 lần chạy: ", DT_average_precision)
print("Recall trung bình (Average Recall) sau 10 lần chạy: ", DT_average_recall)
print("F1 trung bình (Average F1) sau 10 lần chạy: ", DT_average_f1)

print("===============================================================")

import pickle

# Khởi tạo mô hình
knn = KNeighborsClassifier(n_neighbors=13)
nb = GaussianNB()
dt = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=0)

# Huấn luyện mô hình
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Đánh giá độ chính xác (giả sử đã có hàm tính accuracy)
knn_acc = knn.score(X_test, y_test)
nb_acc = nb.score(X_test, y_test)
dt_acc = dt.score(X_test, y_test)

# Lưu mô hình có độ chính xác cao nhất
models = [
    {"model": knn, "accuracy": average_f1},  # KNN
    {"model": nb, "accuracy": Bayes_average_f1},  # Naive Bayes
    {"model": dt, "accuracy": DT_average_f1}  # Decision Tree
]

best_model = max(models, key=lambda m: m["accuracy"])["model"]

# Lưu mô hình tốt nhất vào file model.pkl
if best_model:
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Mô hình tốt nhất ({best_model.__class__.__name__}) đã được lưu vào model.pkl")
