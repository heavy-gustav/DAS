import pandas as pd

#paths to the file containing the samples with the processed features
feature_of_counts = "../processed_data/feature_vectors_counts.csv"

# Importing the dataset
dataset = pd.read_csv(feature_of_counts, index_col=0)
X = dataset.iloc[:,1:9].values
y = dataset.iloc[:, 9].values

# Splitting (randomly) the dataset into the Training set and the (unseen) Test set
# Training-80% and test-20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.2)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the model to the Training set
from sklearn.ensemble import RandomForestClassifier

# creating a Random Forest classifier
clf = RandomForestClassifier() 

# Training the model on the training dataset
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)


# Computing and printing evaluation metrics (confusion matrix, accuracy, precision, recall, f1) 

# confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title('Confusion Matrix')
plt.ylabel('True Values')
plt.xlabel('Predicted Values')
plt.show()

#Importing matrices to get different statistics
from sklearn import metrics

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

# precision
print("PRECISION OF MODEL", metrics.precision_score(y_test, y_pred))

# recall
print("RECALL OF MODEL", metrics.recall_score(y_test, y_pred))

# f1-score
print("F1-SCORE OF MODEL", metrics.f1_score(y_test, y_pred))