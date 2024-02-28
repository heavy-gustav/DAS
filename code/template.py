# Template for Coursework of GINT
import pandas as pd

#paths to the file containing the samples with the processed features
feature_of_counts = "../processed_data/feature_vectors_counts.csv"

# Importing the dataset
dataset = pd.read_csv(feature_of_counts, index_col=0)
X = dataset.iloc[:,1:9].values
y = dataset.iloc[:, 9].values

# Splitting (randomly) the dataset into the Training set and the (unseen) Test set
# Note this is only for the first task of the coursework. You'll need a different approach for the other tasks, as they also need a validation stage in addition to the test with unseen data.
# Also note the split is training 80% and test 20%) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.2)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the model to the Training set
# TODO

# Using the model to predict the Test set
# TODO

# Computing and printing evaluation metrics (confusion matrix, accuracy, precision, recall, f1) 
# TODO
