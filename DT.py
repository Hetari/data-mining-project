from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# Load cleaned dataset
data = pd.read_csv('tornados_clear.csv')

# Define relevant features
selected_features = [
    'year', 'track_width_yards', 'track_length_miles',
    'start_latitude', 'start_longitude', 'injuries',
    'fatalities', 'property_loss', 'state_fips'
]

X = data[selected_features]  # Feature matrix
y = data['severe']  # Target variable

# Initialize and train a deep Decision Tree Classifier
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=15,            # Increase depth for more levels
    min_samples_split=4,     # Minimum samples to split an internal node
    min_samples_leaf=1,      # Minimum samples required at a leaf node
    random_state=42
)
clf.fit(X, y)

# Plot the improved Decision Tree
plt.figure(figsize=(20, 30))  # Increase figure size for better readability
tree.plot_tree(
    clf,
    feature_names=selected_features,
    class_names=['Not Severe', 'Severe'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
