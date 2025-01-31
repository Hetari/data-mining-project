import numpy as np
import pandas as pd


# Step 1: Load the dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file and transform it into a usable format.
    """
    data_set = pd.read_csv(file_path)
    return data_set


# Step 2: Split the dataset into training and test sets
def split_data(data_set, train_size=0.7):
    """
    Split the dataset into a training set and a hold-out test set.
    """
    test_size = 1 - train_size
    train_data = data_set.sample(frac=train_size, random_state=42)
    test_data = data_set.drop(train_data.index)  # Test data (30%)
    return train_data, test_data


# Step 3: Train the Naive Bayes model
def train(train_data, target_column='injuries'):
    """
    Build a supervised Naive Bayes model from the training data.
    """
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
    predictions = np.array(predictions)
    test_data[target_column] = test_data[target_column].astype(int)

    # Convert predictions to binary values
    predictions = (predictions > 0).astype(int)

    true_positives = ((test_data[target_column] > 0) & (predictions == 1)).sum()
    true_negatives = ((test_data[target_column] == 0) & (predictions == 0)).sum()
    false_positives = ((test_data[target_column] == 0) & (predictions == 1)).sum()
    false_negatives = ((test_data[target_column] > 0) & (predictions == 0)).sum()

    accuracy = (true_positives + true_negatives) / len(test_data)

    # Calculate Precision, Recall, and F1-score safely
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


# Main workflow
if __name__ == "__main__":
    # Load the dataset
    file_path = "NV.csv"
    # fatalities
    target_column = "injuries"
    data_set = load_data(file_path)

    # Split the dataset into training and test sets
    train_data, test_data = split_data(data_set, train_size=0.7)

    # Train the Naive Bayes model
    attribute_counts = train(train_data, target_column=target_column)

    # Predict the target class for the test set
    predictions = predict(test_data, attribute_counts, target_column=target_column)

    # Evaluate the predictions
    accuracy, precision, recall, f1 = evaluate(test_data, predictions, target_column=target_column)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print()
