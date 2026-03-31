import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

def main():
    #Membaca data testing PlantVillage
    try:
        x_test = np.load('data/x_test.npy', allow_pickle=True)
        y_test = np.load('data/y_test.npy', allow_pickle=True) # Isinya teks asli
    except FileNotFoundError:
        print("Error: File testing tidak ditemukan!")
        return

    #Memuat model SVM & Encoder
    try:
        model = joblib.load('models/svm_plantvillage.pkl')
        le = joblib.load('models/label_encoder.pkl')
    except FileNotFoundError:
        print("Error: Salah satu model/encoder belum ada.")
        return

    #Ubah teks menjadi angka
    y_test_encoded = le.transform(y_test)

    print("Melakukan prediksi")
    y_pred_encoded = model.predict(x_test)

    #Kembalikan prediksi angka ke tekslabel penyakit
    y_pred_text = le.inverse_transform(y_pred_encoded)

    print("HASIL EVALUASI MODEL DETEKSI PENYAKIT DI TANAMAN")

    akurasi = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"Akurasi keseluruhan: {akurasi * 100:.2f}%\n")

    print("Laporan Detail Akurasi Tiap Penyakit:")
    print(classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_))

if __name__ == "__main__":
    main()
