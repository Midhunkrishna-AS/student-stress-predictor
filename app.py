import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Title
st.title("🎓 Student Stress Level Predictor")

# Load dataset
data = pd.read_csv("stress_data.csv")

# Prepare data
X = data[['study_hours', 'sleep_hours', 'screen_time', 'assignments', 'social_interaction']]
y = data['stress']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# User input UI
st.header("Enter Student Details")

study_hours = st.slider("Study Hours", 0, 12, 5)
sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
screen_time = st.slider("Screen Time", 0, 12, 4)
assignments = st.slider("Assignments", 0, 10, 3)
social_interaction = st.slider("Social Interaction (1-5)", 1, 5, 3)

# Predict button
if st.button("Predict Stress Level"):
    new_data = pd.DataFrame([[study_hours, sleep_hours, screen_time, assignments, social_interaction]],
                            columns=['study_hours','sleep_hours','screen_time','assignments','social_interaction'])
    
    prediction = model.predict(new_data)
    
    st.success(f"Predicted Stress Level: {prediction[0]}")