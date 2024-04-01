import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import graphviz
from sklearn.model_selection import train_test_split

# Load the data
file_path = r'Territory capture-Sample dataset - Sheet1.csv'
df = pd.read_csv(file_path)

# Select features
selected_features = ["Player_Endurance", "Team_Cohesion", "Team_Adaptability"]
X = df[selected_features]
y = df["Win"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Visualize the decision tree using graphviz
dot_data = export_graphviz(dt_model, out_file=None, 
                           feature_names=selected_features,  
                           class_names=["Lose", "Win"],  
                           filled=True, rounded=True,  
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)
graph.view("decision_tree")
# Print the decision tree in text format
tree_text = export_text(dt_model, feature_names=selected_features)
print(tree_text)