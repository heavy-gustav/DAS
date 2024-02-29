# Template for Coursework of GINT
import pandas as pd
from matplotlib import pyplot
from sklearn import metrics 

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
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), random_state=42, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the model to the Training set
from sklearn.ensemble import RandomForestClassifier

# creating a RF classifier
clf = RandomForestClassifier() 

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing probability predictions on the test dataset
scores_clf = clf.predict_proba(X_test)[:,1]

fpr_clf, tpr_clf, thresholds = metrics.roc_curve(y_test, scores_clf)
AUC_clf = metrics.auc(fpr_clf, tpr_clf)
pyplot.plot(fpr_clf,tpr_clf ,'r-')
pyplot.xlabel("detector false positive rate")
pyplot.ylabel("detector true positive rate")
pyplot.title(f"Detector ROC curve for Random Forest Classifier, the AUC is: {AUC_clf}")
pyplot.show()

# confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title('Confusion Matrix')
plt.ylabel('True Values')
plt.xlabel('Predicted Values')
plt.show()


# Log Reg
AUC_log_min = 0
from sklearn.linear_model import LogisticRegression
for algorithm in ['lbfgs', 'newton-cg']:
  for toleration in (1e-4, 1e-3,1e-2):
    for regulation in (1e-1, 1e1):
      for iter in (100, 1000):
        log = LogisticRegression(solver = algorithm, tol = toleration, C= regulation, random_state = 23,  max_iter=iter).fit(X_train, y_train)
        log.fit(X_train, y_train)
        scores_log = log.predict_proba(X_test)[:,1]
        AUC_log = metrics.roc_auc_score(y_test, scores_log)
        print(AUC_log)
        if AUC_log > AUC_log_min:
          AUC_log_min = AUC_log
          print(f"Solver {algorithm!s}, toleration {toleration:g}, regulation {regulation:g}, itteration{iter:g} gives this AUC {AUC_log_min:g}")
 

# Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier
gb= GradientBoostingClassifier()

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
gb.fit(X_train, y_train)

scores_gb = gb.predict_proba(X_test)[:,1]

fpr_gb, tpr_gb, thresholds = metrics.roc_curve(y_test, scores_gb)
AUC_gb = metrics.auc(fpr_gb, tpr_gb)
pyplot.plot(fpr_gb,tpr_gb ,'r-')
pyplot.xlabel("detector false positive rate")
pyplot.ylabel("detector true positive rate")
pyplot.title(f"Detector ROC curve for Gradient Boosting Classifier, The AUC is: {AUC_gb}")
pyplot.show()


from sklearn.ensemble import AdaBoostClassifier
abc= AdaBoostClassifier()

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
abc.fit(X_train, y_train)

scores_abc = abc.predict_proba(X_test)[:,1]

fpr_abc, tpr_abc, thresholds = metrics.roc_curve(y_test, scores_abc)
AUC_abc = metrics.auc(fpr_abc, tpr_abc)
pyplot.plot(fpr_abc,tpr_abc ,'r-')
pyplot.xlabel("detector false positive rate")
pyplot.ylabel("detector true positive rate")
pyplot.title(f"Detector ROC curve for Ada Boost Classifier, The AUC is: {AUC_abc}")
pyplot.show()