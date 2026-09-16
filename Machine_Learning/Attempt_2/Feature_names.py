import numpy as np
import joblib

# Load the saved preprocessing pipeline
preprocessor = joblib.load("Machine_Learning\Attempt_2\lthc_preprocessor.pkl")

# Get the names of the processed features
feature_names = preprocessor.get_feature_names_out()

# Display them
print("Number of processed features:", len(feature_names))

print("\nFirst 20 feature names:")
for feature in feature_names[:20]:
    print(feature)

# Save them
np.save("feature_names.npy", feature_names)

print("\nfeature_names.npy saved successfully.")