from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Tải mô hình
filename = 'D:/Code/Do An MHUD/model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Kiểm tra nếu người dùng tải tệp CSV
            if 'file-upload' in request.files:
                file = request.files['file-upload']
                print(f"Đã nhận tệp: {file.filename}")  # Kiểm tra tệp

                if not file.filename.endswith('.csv'):
                    return jsonify({'error': 'Vui lòng tải lên tệp CSV.'}), 400

                df = pd.read_csv(file)
                print("Dữ liệu từ CSV: ", df.head())  # Kiểm tra dữ liệu

                if df.empty:
                    return jsonify({'error': 'Tệp CSV không có dữ liệu.'}), 400

                # Kiểm tra số lượng đặc trưng mà mô hình mong đợi
                expected_features = model.n_features_in_
                print(f"Số lượng đặc trưng mà mô hình mong đợi: {expected_features}")

                # Lấy các cột từ cột thứ 2 đến đủ số lượng đặc trưng mong đợi
                input_columns = df.columns[1:1 + expected_features]  # Lấy cột từ thứ 2
                input_data = df[input_columns].values
                print(f"Số lượng đặc trưng trong dữ liệu: {input_data.shape[1]}")

                # Kiểm tra nếu số lượng đặc trưng có khớp với mô hình không
                if input_data.shape[1] != expected_features:
                    return jsonify({'error': f'Số lượng đặc trưng không khớp. Mong đợi {expected_features} đặc trưng, nhưng nhận được {input_data.shape[1]} đặc trưng.'}), 400

                # Dự đoán và thêm cột 'Prediction' vào DataFrame
                predictions = model.predict(input_data)
                df['Prediction'] = ["Không gian lận" if pred == 0 else "Gian lận" for pred in predictions]

                # Trả về kết quả dưới dạng JSON (Lấy các cột 'Doanh_Nghiep' và 'Prediction')
                results = df[['Doanh_Nghiep', 'Prediction']].to_dict(orient='records')
                return jsonify({'predictions': results})

            # Kiểm tra nếu người dùng gửi dữ liệu từ form
            elif request.form:
                # Nhận dữ liệu từ form và dự đoán cho một dòng
                Sector_score = float(request.form['Sector_score'])
                LOCATION_ID = float(request.form['LOCATION_ID'])
                Score_A = float(request.form['Score_A'])
                Score_B = float(request.form['Score_B'])
                numbers = float(request.form['numbers'])
                CONTROL_RISK = float(request.form['CONTROL_RISK'])
                MONEY_Marks = float(request.form['MONEY_Marks'])
                District = float(request.form['District'])
                Loss = float(request.form['Loss'])
                History = float(request.form['History'])

                # Dự đoán cho một dòng dữ liệu
                input_data = [[Sector_score, LOCATION_ID, Score_A, Score_B, numbers, CONTROL_RISK, MONEY_Marks, District, Loss, History]]
                prediction = model.predict(input_data)[0]
                
                # Diễn giải nhãn
                prediction_text = "Không gian lận" if prediction == 0 else "Gian lận"
                
                return jsonify({'prediction': prediction_text})

        except Exception as e:
            print(f"LỖI: {e}")  # Ghi log lỗi vào terminal
            return jsonify({'error': str(e)}), 400

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
