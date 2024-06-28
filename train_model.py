import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('software_clones.csv')
data['Clone_Type'] = data['Clone_Type'].astype('category').cat.codes
X = data[['Clone_Type', 'Code_Complexity']]
y = data['Refactoring_Applied']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to 'model.pkl'.")
