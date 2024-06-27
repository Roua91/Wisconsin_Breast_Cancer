import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('data.csv')
    # Drop the 'Unnamed: 32' column
    df = df.drop(columns=['Unnamed: 32'])
    # Encode diagnosis (M -> 1, B -> 0)
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    return df

df = load_data()

# Split data into features and target
X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_scaled, y)

# Sidebar inputs
st.sidebar.title('Input Parameters')
input_data = {}
for feature in X.columns:
    input_data[feature] = st.sidebar.slider(f'{feature} (mean)', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()), step=0.1)

input_df = pd.DataFrame([input_data])

# Standardize user input
input_scaled = scaler.transform(input_df)

# Make predictions
prediction = svm_model.predict(input_scaled)
prediction_label = 'Malignant' if prediction[0] == 1 else 'Benign'

# Display result on main page
st.title('Breast Cancer Diagnosis Classifier')
st.write("""
         ## Diagnosis Result:
         """)

st.write(f'The diagnosis is **{prediction_label}**.')

# PCA Visualization
st.write("""
         ## PCA Visualization:
         """)

# Plotting PCA
fig, ax = plt.subplots(figsize=(10, 6))

# Plot original data points
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['diagnosis'], palette='tab10', legend='full', ax=ax)

# Highlight user input data point
if prediction_label == 'Malignant':
    color = 'orange'
else:
    color = 'blue'

ax.scatter(input_scaled[:, 0], input_scaled[:, 1], color=color, marker='*', s=200, label='New Data Point')

plt.title('PCA Visualization of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
st.pyplot(fig)


# Predictions on the entire dataset for evaluation
y_pred = svm_model.predict(X_scaled)

