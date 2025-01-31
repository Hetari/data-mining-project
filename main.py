from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler

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
#
file_path = './tornados.csv'
data = pd.read_csv(file_path)
#! Step 1: Handle Missing Values
numeric_columns = [
    'year', 'month', 'day',
    'state_fips', 'magnitude',
    'track_width_yards', 'states_affected', 'county_fips_1',
    'county_fips_2', 'county_fips_3', 'county_fips_4',
    'injuries', 'fatalities', 'property_loss', 'track_width_yards',
    'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'track_length_miles'
]
categorical_columns = [
    'timezone', 'state_abbreviation', 'magnitude_estimated'
]

# Impute missing values in numeric columns with the median
# * `SimpleImputer` is a scikit-learn class which is helpful in handling the missing data in the predictive model dataset
num_imputer = SimpleImputer(strategy='median')
data[numeric_columns] = num_imputer.fit_transform(data[numeric_columns])


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

# ! Step 2: Handle Outliers
numeric_columns_to_scale = [
    'magnitude', 'track_width_yards', 'injuries', 'fatalities',
    'property_loss', 'start_latitude', 'start_longitude',
    'end_latitude', 'end_longitude', 'track_length_miles'
]
z_scores = data[numeric_columns].apply(zscore)

outlier_counts = (z_scores.abs() > 3).sum()
high_outlier_columns = outlier_counts[outlier_counts > 500].index.tolist()


# Clip outliers to 1.5 * IQR range
for col in high_outlier_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)


# ! Define Binary Outcome for Analysis
# Why it's important:
# For Classification Models:
# If you are building a classification model to predict whether an event is severe or not based on the tornado data, creating a binary outcome like this (severe) is crucial. It allows you to classify each tornado event as either "severe" or "not severe" based on whether there were any injuries or fatalities.

# For Predictive Analysis:
# A binary outcome is often used in predictive models (like Logistic Regression, Decision Trees, etc.) when you're interested in identifying or classifying events based on certain criteria (like injury or fatality occurrence). It transforms a complex multi-value problem (number of injuries/fatalities) into a simpler yes/no problem, making predictions easier.
data['severe'] = (data['injuries'] > 0) | (data['fatalities'] > 0)
data.to_csv('./tornados_clear.csv', index=False)
# ! Step 3: Feature Selection (Correlation Analysis)
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
# plt.show()

# Drop features with low correlation to the target variable
selected_features = [
    'year',  # Keep for temporal trends
    'track_width_yards',
    'track_length_miles',
    'start_latitude',
    'start_longitude',
    'injuries',
    'fatalities',
    'property_loss',
    'state_fips',  # Keep for regional analysis
    'severe'  # Keep for classification
]

# # Data Splitting (Train-Test Split)
# X = data[selected_features].drop(columns=['severe'])  # Features
# y = data['severe']  # Target (Binary Outcome)

# # Split into 80% training and 20% testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Train Random Forest Classifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)

# # Feature Importance (Random Forest)
# importance = rf_model.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(10, 5))
# sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
# plt.title("Feature Importance (Random Forest)")
# # plt.show()

# ! Step 5: Apriori Algorithm (Association Rule Mining)
# save the feature_importance_df to a CSV file
data[
    categorical_columns + ['severe']
].to_csv('./Apriori.csv', index=False)

#! Step 9: Naive Bayes Classification
data[
    numeric_columns + ['severe']
].sample(frac=0.5).to_csv('./NV.csv', index=False)
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
