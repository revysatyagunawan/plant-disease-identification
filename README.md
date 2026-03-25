# Machine Learning-Based Plant Disease Identification
Plant Health Diagnostics using Support Vector Machine (SVM)

## Technical Implementation (Pipeline)
| Stage | Technique | Input | Output |
| :--- | :--- | :--- | :--- |
| 1. Preprocessing | Gaussian Blur -> LAB CLAHE | image_path: str | preprocessed: np.ndarray (H×W×C) |
| 2. Segmentation | K-Means clustering (background separation) | preprocessed | leaf_mask: np.ndarray |
| 3. Morphology | Opening & Closing for noise reduction | leaf_mask | clean_mask: np.ndarray |
| 4. Feature Extraction | Edge (Canny) + GLCM + LBP + Color Stat | img + clean_mask | feature_vector: np.ndarray (1D) |
| 5. Classification | Multiclass SVM (One-vs-Rest) | feature_vector | prediction: "Disease Name" |

## Feature Set (Vector Representation)
- Shape Features: Edge detection (Canny) for leaf shape deformation.
- Texture Features: GLCM Contrast, GLCM Homogeneity, LBP (Local Binary Pattern Histogram).
- Color Features: Average color channel values (RGB/HSV) to detect leaf pigment changes.

## Dataset
- Source: PlantVillage and New Plant Diseases Database.
- Total: >87,000 images (MVP subset format used: 3-5 classes for 1 plant type initially).
- Split: 80% Training, 20% Testing (stratified).
- Format: RGB Images.

## Project Structure
```
plant-disease-identification/
├── data/
│   ├── train/                 # 80% Training Data
│   └── test/                  # 20% Testing Data
├── models/
├── src/
│   ├── preprocessing.py       # Gaussian Blur & CLAHE module
│   ├── segmentation.py        # K-Means & Morphology module
│   ├── feature_extraction.py  # 1D NumPy array generation module
│   ├── model_train.py         # SVM training module (OvR)
│   └── model_test.py          # Model performance evaluation module
├── .github/
│   └── CODEOWNERS
├── requirements.txt
└── README.md
```

## Setup & Installation
### Prerequisites
- Python 3.10 or 3.11.
- Git.

### Installation Steps
1. Clone this repository: `git clone https://github.com/revysatyagunawan/plant-disease-identification.git`
2. Create virtual environment: `python -m venv venv`
3. Activate it:
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Model Performance
Classifier: Support Vector Machine (SVM).
- Multiclass Strategy: One-vs-Rest (OvR).
- Input Data: 1D NumPy Array (Result of mathematical feature extraction).
- Evaluation Metrics: Accuracy & Confusion Matrix.

## Development Workflow
### Branch Strategy
Always create a new branch from main before starting work (example: feature/preprocessing or feature/svm-training).

### Pull Request Process
1. Create PR from feature branch to main.
2. Merge can only be performed after receiving Approval
3. Delete the feature branch after a successful merge.

## Documentation
### Pipeline Modules
1. Preprocessing (src/preprocessing.py)
   - Gaussian Blur (for noise reduction).
   - CLAHE (the image is converted from BGR to LAB color space. CLAHE is applied only to the Luminance [L] channel).
2. Segmentation (src/segmentation.py)
   - K-Means clustering to isolate leaf pixel areas.
3. Morphology (src/segmentation.py)
   - Opening and Closing operations to close noise gaps in the leaf mask.
4. Feature Extraction (src/feature_extraction.py)
   - Edge detection (Canny).
   - GLCM features.
   - LBP histogram.
   - Color channels aggregation.

## Acknowledgments
- Dataset: PlantVillage & New Plant Diseases.
