from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = './tornados.csv'
data = pd.read_csv(file_path)

#! Step 1: Handle Missing Values
numeric_columns = [
    'year',
    'month',
    'day',

    'state_fips',
    'magnitude',

    'track_width_yards',
    'states_affected',
    'county_fips_1',
    'county_fips_2',
    'county_fips_3',
    'county_fips_4',

    'injuries',
    'fatalities',
    'property_loss',
    'track_width_yards',

    # float
    'start_latitude',
    'start_longitude',
    'end_latitude',
    'end_longitude',
    'track_length_miles'
]
categorical_columns = ['timezone', 'state_abbreviation']

# Impute missing values in numeric columns with the median
# * `SimpleImputer` is a scikit-learn class which is helpful in handling the missing data in the predictive model dataset
# ? print(data.isnull().sum())
num_imputer = SimpleImputer(strategy='median')
data[numeric_columns] = num_imputer.fit_transform(data[numeric_columns])
# ? print(data.isnull().sum())


# Check and drop categorical columns with excessive missingness
cat_missing = data[categorical_columns].isnull().sum()
threshold = 0.5 * len(data)
categorical_to_drop = cat_missing[cat_missing > threshold].index.tolist()
data = data.drop(columns=categorical_to_drop)

# Update categorical columns to reflect dropped ones
categorical_columns = [col for col in categorical_columns if col not in categorical_to_drop]

# Impute remaining categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

# # Step 2: Encode Categorical Variables
# label_encoder = LabelEncoder()
# for col in categorical_columns:
#     data[col] = label_encoder.fit_transform(data[col])

# # Step 3: Feature Scaling
# scaler = StandardScaler()
# data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# # Step 4: Extract Date and Time Features
# data['datetime_utc'] = pd.to_datetime(data['datetime_utc'], errors='coerce')
# data['year'] = data['datetime_utc'].dt.year
# data['month'] = data['datetime_utc'].dt.month
# data['day'] = data['datetime_utc'].dt.day
# data['hour'] = data['datetime_utc'].dt.hour
# data = data.drop(columns=['datetime_utc'])

# # Step 5: Handle Outliers
# for col in numeric_columns:
#     Q1 = data[col].quantile(0.25)
#     Q3 = data[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# # Step 6: Define Binary Outcome for Analysis
# data['severe'] = (data['injuries'] > 0) | (data['fatalities'] > 0)

# # Step 7: Feature Selection (Correlation Analysis)
# correlation_matrix = data[numeric_columns].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Feature Correlation Matrix')
# plt.show()

# # Drop features with low correlation to the target variable
# selected_features = ['injuries', 'fatalities', 'track_length_miles', 'track_width_yards']

# # Step 8: Apriori Algorithm (Association Rule Mining)
# basket_data = data[['state_abbreviation', 'timezone']]
# basket_data = pd.get_dummies(basket_data, columns=['state_abbreviation', 'timezone'])

# apriori_results = apriori(basket_data, min_support=0.1, use_colnames=True)
# association_rules_results = association_rules(apriori_results, metric='lift', min_threshold=1.0)

# # Visualize association rules
# print("Apriori Association Rules:\n", association_rules_results.head())
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=association_rules_results, x='support', y='confidence', size='lift', hue='lift', legend=False)
# plt.title('Association Rules Visualization')
# plt.xlabel('Support')
# plt.ylabel('Confidence')
# plt.show()

# # Step 9: Naive Bayes Classification
# X = data[selected_features + ['timezone', 'state_abbreviation']]
# y = data['severe']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# naive_bayes_model = GaussianNB()
# naive_bayes_model.fit(X_train, y_train)
# nb_predictions = naive_bayes_model.predict(X_test)
# print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
# print(classification_report(y_test, nb_predictions))

# # Step 10: Decision Tree (ID3 Algorithm)
# decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
# decision_tree.fit(X_train, y_train)
# id3_predictions = decision_tree.predict(X_test)
# print("ID3 Decision Tree Accuracy:", accuracy_score(y_test, id3_predictions))
# print(classification_report(y_test, id3_predictions))

# # Step 11: K-Means Clustering
# kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
# kmeans.fit(data[selected_features])
# data['cluster'] = kmeans.labels_

# # Visualize clusters
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=data, x='track_length_miles', y='track_width_yards', hue='cluster', palette='viridis')
# plt.title('K-Means Clustering')
# plt.xlabel('Length (len)')
# plt.ylabel('Width (wid)')
# plt.show()

# # Evaluate clustering quality
# silhouette_avg = silhouette_score(data[selected_features], kmeans.labels_)
# print("Silhouette Score for K-Means:", silhouette_avg)

# # Save Preprocessed Data for Future Use
# processed_file_path = './processed_tornados.csv'
# data.to_csv(processed_file_path, index=False)
# print(f"Preprocessed data saved to {processed_file_path}")
