import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset
data = pd.read_csv("placement_data.csv")

X = data[['CGPA', 'Internship', 'Communication', 'Technical', 'Certifications', 'Projects', 'ExtraActivities']]
y = data['Placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("placement_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")
