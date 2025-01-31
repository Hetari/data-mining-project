import pandas as pd

# Step 1: Load the dataset


def load_data(file_path):
    data_set = pd.read_csv(file_path)
    return data_set


# Step 2: Split the dataset into training and test sets
def split_data(data_set, train_size=300, shuffle=True):
    if shuffle:
        data_set = data_set.sample(frac=1).reset_index(drop=True)
    train_data = data_set.iloc[:train_size]
    test_data = data_set.iloc[train_size:]
    return train_data, test_data


# Step 3: Train the Naive Bayes model
def train(train_data, target_column='injuries'):
    attribute_counts = {}  # Dictionary to store counts of attribute values per target class
    attributes = train_data.columns.drop(target_column)  # All columns except the target

    for attribute in attributes:
        # Count the occurrences of each attribute value for each target class
        attribute_counts[attribute] = train_data.groupby(target_column)[attribute].value_counts()

    return attribute_counts


# Step 4: Predict the target class for instances in the test set
def predict(test_data, attribute_counts, target_column='injuries'):
    """
    Predict the target class for instances in the test set using the trained model.
    """
    target_classes = test_data[target_column].unique()  # Unique target classes (e.g., injury levels)
    predictions = []  # List to store predictions

    for _, instance in test_data.iterrows():  # Iterate over each instance in the test set
        class_probabilities = {}  # Dictionary to store probabilities for each target class

        for target_class in target_classes:
            probability = 1.0  # Initialize probability for the target class

            for attribute, value in instance.items():  # Iterate over each attribute-value pair
                if attribute == target_column:
                    continue  # Skip the target column

                # Calculate the conditional probability P(attribute=value | target_class)
                if (target_class, value) in attribute_counts[attribute]:
                    probability *= attribute_counts[attribute][(target_class, value)
                                                               ] / attribute_counts[attribute].groupby(level=0).sum()[target_class]
                else:
                    probability *= 0.0001  # Small epsilon for unseen attribute values

            # Multiply by the prior probability P(target_class)
            prior_probability = (test_data[target_column] == target_class).mean()
            class_probabilities[target_class] = probability * prior_probability

        # Predict the target class with the highest probability
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        predictions.append(predicted_class)

    return predictions


# Step 5: Evaluate the predictions
def evaluate(test_data, predictions, target_column='injuries'):
    """
    Evaluate the accuracy of the predictions.
    """
    correct = (test_data[target_column] == predictions).sum()  # Count correct predictions
    accuracy = correct / len(test_data)  # Calculate accuracy
    return accuracy


# Main workflow
if __name__ == "__main__":
    # Load the dataset
    file_path = "NV.csv"
    data_set = load_data(file_path)

    # Split the dataset into training and test sets
    train_data, test_data = split_data(data_set, train_size=300)

    # Train the Naive Bayes model
    attribute_counts = train(train_data, target_column='injuries')

    # Predict the target class for the test set
    predictions = predict(test_data, attribute_counts, target_column='injuries')

    # Evaluate the predictions
    accuracy = evaluate(test_data, predictions, target_column='injuries')
    print(f"Accuracy: {accuracy:.2f}")
    print()
