# Hit Song Prediction Using Chorus Features

## Overview
This project predicts whether a song is a **Hit** or **Non-Hit** using acoustic features extracted from its chorus. It implements an end-to-end machine learning pipeline covering dataset preparation, chorus extraction, acoustic feature extraction, dimensionality reduction, model training, evaluation, and a Streamlit application for live demonstration.

## Project Pipeline
```text
Full Song
   ↓
Chorus Detection / Cached Chorus
   ↓
First 15 Seconds of Chorus
   ↓
518 Acoustic Features
   ↓
Standardization + PCA
   ↓
Machine Learning Classifier
   ↓
HIT / NON-HIT
```

## Dataset
The final dataset contains **548 songs** from seven artists:
- Taylor Swift
- Drake
- Morgan Wallen
- Ariana Grande
- Jason Aldean
- Ed Sheeran
- Luke Bryan

Songs are labelled **Hit (1)** or **Non-Hit (0)** according to the dataset construction procedure used in the project.

A fixed train-test split is used:
- **Training set:** 411 songs
- **Test set:** 137 songs

The split files are stored in `data/splits/train_songs.csv` and `data/splits/test_songs.csv`.

## Chorus Extraction
Song structure is detected using **All-In-One Music Structure Analyzer**. The first detected chorus is used, with a maximum of **15 seconds** analyzed per song.

For the demo, previously extracted choruses can be reused from the local cache. If an uploaded song is not already available, the application can run automatic chorus detection on the full song.

## Feature Extraction
Acoustic features are extracted from the chorus using **Librosa**. Each song is represented by a **518-dimensional feature vector**. The features are standardized and reduced using **Principal Component Analysis (PCA)** before classification.

## Models Evaluated
The project evaluates multiple models:
- Linear Discriminant Analysis (LDA)
- Shrinkage LDA
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Neural Network

Final comparison metrics are stored in `results/metrics/model_metrics.csv`.

## Final Model
**Standard Linear Discriminant Analysis (LDA)** was selected as the final model based on the overall evaluation results.

| Metric | Held-out Test Result |
| --- | ---: |
| Accuracy | **77.37%** |
| F1 Score | **0.8208** |

## Live Demo
The project includes a lightweight **Streamlit** application. The user uploads a full song and the application:
1. Checks for an already extracted chorus.
2. Uses the cached chorus when available.
3. Runs automatic chorus detection for an unseen song when required.
4. Extracts the same acoustic features used during model development.
5. Applies the trained PCA + LDA pipeline.
6. Displays a **HIT** or **NON-HIT** prediction and hit probability.
7. Allows playback of the chorus used for prediction.

## Repository Structure
```text
hit-song-chorus-ml/
├── app.py
├── data/
│   └── splits/
│       ├── train_songs.csv
│       └── test_songs.csv
├── results/
│   ├── logs/
│   └── metrics/
│       └── model_metrics.csv
├── src/
│   ├── archive/
│   ├── demo_utils.py
│   ├── save_train_test_split.py
│   ├── train_demo_model.py
│   ├── train_lda.py
│   ├── train_lda_shrinkage.py
│   ├── train_logistic.py
│   ├── train_neural_network.py
│   └── write_final_metrics.py
├── models/          # generated locally
├── demo_songs/      # local demo audio; not committed
└── README.md
```

Large/generated audio files, cached choruses, structure outputs, and local model artifacts may be excluded through `.gitignore`.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/divishag/Hit-Song-Prediction-Using-Chorus.git
cd Hit-Song-Prediction-Using-Chorus
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the main dependencies
```bash
pip install numpy pandas scikit-learn librosa joblib streamlit
```

The automatic chorus-detection fallback additionally requires the **All-In-One** environment/dependencies used by the project.

## Train the Demo Model
From the project root:
```bash
python src/train_demo_model.py
```

This creates the local model artifact required by the Streamlit application.

## Run the Demo
```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal and upload a supported full-song audio file.

## Results and Limitations
The experiments indicate that chorus-level acoustic features contain useful information for distinguishing hit and non-hit songs in this dataset. Standard LDA achieved the strongest overall classification performance and was selected for the demonstration.

The model is not expected to classify every song correctly. Commercial success also depends on factors outside the acoustic feature set, such as promotion, artist popularity, release timing, audience trends, and other contextual factors.

## Technologies Used
Python, Librosa, scikit-learn, All-In-One Music Structure Analyzer, Pandas, NumPy, Streamlit, and Joblib.

## Course
**UE24CS352A - Machine Learning**  
Mini-Project Assignment
