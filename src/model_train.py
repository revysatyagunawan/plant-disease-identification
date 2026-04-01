import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

def main():
    #File ekstraksi dimuat
    try:
        x_train = np.load('data/x_train.npy', allow_pickle=True)
        y_train = np.load('data/y_train.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Error: File data tidak ada di folder 'data'")
        return

    # KONVERSI ANGKA JADI TEKS LABEL
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Menampilkan kelas apa saja yang dideteksi
    print(f"Kelas terdeteksi: {le.classes_}")

    # Penyiapan modul SVM
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, decision_function_shape='ovr', probability=True)
    )
    
    print("Mulai proses training")
    pipeline.fit(x_train, y_train_encoded)
    
    # Menyimpan model dan encoder
    os.makedirs('models', exist_ok=True)
    
    # Kita simpan model SVM dan kamus penerjemah labelnya
    joblib.dump(pipeline, 'models/svm_plantvillage.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("Selesai")

if __name__ == "__main__":
    main()