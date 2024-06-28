import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate synthetic data
np.random.seed(42)
num_samples = 1000
clone_types = np.random.choice(['Type1', 'Type2', 'Type3'], size=num_samples)
code_complexity = np.random.randint(1, 11, size=num_samples)
refactoring_applied = np.random.choice([0, 1], size=num_samples)

data = pd.DataFrame({
    'Clone_Type': clone_types,
    'Code_Complexity': code_complexity,
    'Refactoring_Applied': refactoring_applied
})

data.to_csv('software_clones.csv', index=False)

# Load the dataset and train the model
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
