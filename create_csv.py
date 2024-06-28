import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000
clone_types = np.random.choice(['Type1', 'Type2', 'Type3'], size=num_samples)
code_complexity = np.random.randint(1, 11, size=num_samples)  # Complexity between 1 and 10
refactoring_applied = np.random.choice([0, 1], size=num_samples)  # 0 or 1

# Create a DataFrame
data = pd.DataFrame({
    'Clone_Type': clone_types,
    'Code_Complexity': code_complexity,
    'Refactoring_Applied': refactoring_applied
})

# Save to CSV
data.to_csv('software_clones.csv', index=False)

print("CSV file 'software_clones.csv' created successfully.")
