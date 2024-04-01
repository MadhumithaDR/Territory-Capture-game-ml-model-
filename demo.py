import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Sample dataset
data = {
    "Player_Endurance": [2, 2, 5, 1, 6, 2, 8, 10, 4, 1, 5, 9, 5, 5, 8, 10, 1, 10, 10, 6],
    "Team_Cohesion": [8, 10, 8, 7, 7, 3, 6, 2, 6, 10, 4, 5, 1, 9, 4, 7, 2, 8, 3, 5],
    "Team_Adaptability": [1, 2, 3, 5, 9, 7, 4, 1, 6, 8, 5, 2, 10, 4, 3, 6, 9, 7, 5, 8],
    "Win": [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Select features
selected_features = ["Player_Endurance", "Team_Cohesion", "Team_Adaptability"]
X = df[selected_features]
y = df["Win"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1-score for "Win"
accuracy_win = accuracy_score(y_test, y_pred)
precision_win = precision_score(y_test, y_pred, pos_label=1)
recall_win = recall_score(y_test, y_pred, pos_label=1)
f1_win = f1_score(y_test, y_pred, pos_label=1)

# Calculate accuracy, precision, recall, and F1-score for "Lose"
accuracy_lose = accuracy_score(1 - y_test, 1 - y_pred)  # 1 - y for "Lose" class
precision_lose = precision_score(1 - y_test, 1 - y_pred, pos_label=1)
recall_lose = recall_score(1 - y_test, 1 - y_pred, pos_label=1)
f1_lose = f1_score(1 - y_test, 1 - y_pred, pos_label=1)

# Print the results
print("Metrics for 'Win' class:")
print(f"Accuracy: {accuracy_win:.2f}")
print(f"Precision: {precision_win:.2f}")
print(f"Recall: {recall_win:.2f}")
print(f"F1 Score: {f1_win:.2f}")

print("\nMetrics for 'Lose' class:")
print(f"Accuracy: {accuracy_lose:.2f}")
print(f"Precision: {precision_lose:.2f}")
print(f"Recall: {recall_lose:.2f}")
print(f"F1 Score: {f1_lose:.2f}")
