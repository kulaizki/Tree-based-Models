# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Define a random seed for reproducibility
SEED = 42

# Generate mock dataset (features and labels)
# Features: 100 samples, each with 4 features
# Labels: Binary classification (0 or 1)
np.random.seed(SEED)
X = np.random.rand(100, 4)  # 100 samples, 4 features
y = np.random.choice([0, 1], size=100)  # 100 labels (0 or 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# Fit the model to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))