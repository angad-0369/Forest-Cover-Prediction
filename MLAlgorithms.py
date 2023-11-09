import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

# Load dataset
df = pd.read_csv("D:\\EDUCATION\\OTHERS\\MACHINE LEARNING\\Forest cover prediction\\covtype.csv")

# Preprocessing
le = LabelEncoder()
df['Cover_Type'] = le.fit_transform(list(df['Cover_Type']))
X = df[["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
"Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
"Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2",
"Wilderness_Area3", "Wilderness_Area4"]]
y = df['Cover_Type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_score = gnb.score(X_test, y_test)
print("Accuracy using Gaussian Naive Bayes: ", round(gnb_score * 100, 3), "%", sep="")

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
print("Accuracy using KNN: ", round(knn_score * 100, 3), "%", sep="")

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print("Accuracy using Random Forest: ", round(rf_score * 100, 3), "%", sep="")

# Logistic Regression
lr = LogisticRegression(solver="saga", max_iter=100)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print("Accuracy using Logistic Regression: ", round(lr_score * 100, 3), "%", sep="")

# SVM Classifier
svm_clf = svm.SVC(kernel="rbf", C=1)
svm_clf.fit(X_train, y_train)
svm_score = svm_clf.score(X_test, y_test)
print("Accuracy using SVM: ", round(svm_score * 100, 3), "%", sep="")