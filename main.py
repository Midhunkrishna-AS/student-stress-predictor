import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("stress_data.csv")

# Inputs and output
X = data[['study_hours', 'sleep_hours', 'screen_time', 'assignments', 'social_interaction']]
y = data['stress']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 🔥 USER INPUT
print("\nEnter Student Details:")

study_hours = float(input("Study hours: "))
sleep_hours = float(input("Sleep hours: "))
screen_time = float(input("Screen time: "))
assignments = int(input("Assignments: "))
social_interaction = int(input("Social interaction (1-5): "))

# Convert to DataFrame (no warning)
new_data = pd.DataFrame([[study_hours, sleep_hours, screen_time, assignments, social_interaction]],
                        columns=['study_hours','sleep_hours','screen_time','assignments','social_interaction'])

# Prediction
prediction = model.predict(new_data)

print("\nPredicted Stress Level:", prediction[0])