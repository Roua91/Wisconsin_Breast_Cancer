# Wisconsin Breast Cancer 

## Project Description

Breast cancer is the most common cancer in women and a significant public health issue. Early detection is crucial for effective treatment and improving survival rates. This project aims to leverage machine learning techniques to enhance the accuracy of breast cancer diagnosis using the Breast Cancer Wisconsin Diagnostic dataset. The project includes data preprocessing, exploratory data analysis, model training, evaluation of various supervised and unsupervised machine learning models, and deployment of the best-performing model using Streamlit.


## Introduction

Breast cancer presents a significant challenge due to its high incidence and severity. This project focuses on early detection using machine learning models to classify breast tumors as malignant or benign. By analyzing various features of cell nuclei computed from digitized images, this project aims to identify the most effective machine learning model for this classification task.

## Dataset

The dataset used in this study is the Breast Cancer Wisconsin Diagnostic dataset, collected by Wolberg et al. (1995). It contains features describing the characteristics of cell nuclei, such as radius, texture, perimeter, area, smoothness, compactness, concavity, and more. The dataset is preprocessed to remove irrelevant columns and handle missing values.

## Methodology

### Data Collection and Preprocessing

- **Data Cleaning**: Removal of irrelevant columns (e.g., ID, unnamed columns).
- **Data Encoding**: Encoding categorical variables (e.g., 'M' for malignant, 'B' for benign).
- **Feature Scaling**: Normalizing features to ensure they are on the same scale.

### Exploratory Data Analysis
- **Correlation Matrix**
- **Bivariate Analysis**
- **PCA Visulisation**

### Machine Learning Models

#### Supervised Learning Models

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting**
- **Neural Network**

#### Unsupervised Learning Model

- **K-Means Clustering**

### Model Evaluation

#### Supervised Learning Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC**

#### Unsupervised Learning Metrics

- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

## Results

### Supervised Learning Results

The SVM model demonstrated the highest performance with an accuracy of 98.25%, precision of 100%, recall of 95.35%, F1 score of 97.62%, and ROC AUC of 0.997. These metrics indicate that SVM is the most effective model for this classification task.

### Unsupervised Learning Results

The K-Means clustering results showed a Silhouette Score of 0.2806, a Davies-Bouldin Index of 1.4894, and a Calinski-Harabasz Index of 158.8235, suggesting moderate clustering performance.

## Deployment

The SVM model was deployed using Streamlit to provide an accessible and user-friendly interface for making predictions. Users can input parameters via a sidebar and instantly receive a diagnosis result. A PCA visualization is included to show the position of the new data point relative to the dataset.
![WhatsApp Image 2024-06-26 at 11 31 03_3332b76e](https://github.com/Roua91/Wisconsin_Breast_Cancer/assets/165356652/0a1441af-cd60-4dad-9346-8ce95b38adeb)


## Conclusion

This project demonstrates the effectiveness of machine learning in enhancing breast cancer diagnosis. The SVM model, with its superior performance metrics, was selected for deployment. The interactive web application developed using Streamlit allows for real-time predictions and visualization, aiding healthcare professionals in early detection and personalized treatment strategies.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-diagnosis
   cd breast-cancer-diagnosis
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run Wisconsin_Streamlit.py
   ```

4. **Interact with the app**:
   Open your web browser and go to `http://localhost:8501` to use the Breast Cancer Diagnosis Classifier.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Streamlit

## Files In this Directory

- `wisconsin_unsupervised.ipynb`: Notebook for unsupervised learning analysis.
- `wisconsin_supervised.ipynb`: Notebook for supervised learning analysis.
- `Wisconsin_Streamlit.py`: Streamlit app for deploying the SVM model.
- `Data.csv`: Wisconsin diagnostic breast cancer dataset
- `requirments.txt`

## Acknowledgements

- The dataset was provided by Wolberg et al. (1995).
- Streamlit was used for deploying the model.

