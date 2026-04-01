import os
import numpy as np

from preprocessing import preprocess
from segmentation import segmentation
from feature_extraction import extract_features

def process_folder(dataset_dir: str):
    X = []
    y = []
    
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        print(f"Memproses kategori: {category} di {dataset_dir}")
        
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            
            try:
                img = preprocess(file_path)                           
                mask = segmentation(img)                    
                features = extract_features(img, mask)                
                
                X.append(features)
                y.append(category)
            except Exception as e:
                print(f"Gagal memproses {file_path}: {e}")
                
    # Kembalikan fitur dalam bentuk 1D NumPy Array dan labelnya
    return np.array(X, dtype=np.float32), np.array(y)

def main():
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)

    x_train, y_train = process_folder('../data/train')
    np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)

    x_test, y_test = process_folder('../data/test')
    np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
if __name__ == "__main__":
    main()
