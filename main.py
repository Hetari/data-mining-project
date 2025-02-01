from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    'state_abbreviation', 'magnitude_estimated'
]

# Impute missing values in numeric columns with the median
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
# for col in high_outlier_columns:
#     Q1 = data[col].quantile(0.25)
#     Q3 = data[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)


# ! Define Binary Outcome for Analysis
data['severe'] = (data['injuries'] > 0) | (data['fatalities'] > 0)
data.sample(frac=0.1).to_csv('./tornados_clear.csv', index=False)

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

# ! Step 5: Apriori Algorithm (Association Rule Mining)
# save the feature_importance_df to a CSV file
final_categorical_selected_features = categorical_columns + [
    'severe'
]

# Ensure only valid numeric columns are selected
final_numeric_selected_features = [
    col for col in numeric_columns if col in selected_features
] + ['severe', 'state_fips']


data[
    final_categorical_selected_features
].to_csv('./Apriori.csv', index=False)

#! Step 6: Naive Bayes Classification
data[
    final_numeric_selected_features
].sample(frac=0.01).to_csv('./NV.csv', index=False)
