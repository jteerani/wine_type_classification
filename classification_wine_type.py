import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
features = ['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols', 'Flavanoids' ,
                'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
data.columns = ['Label'] + features 
data.sample(5)


# #### Normalisation
from sklearn.preprocessing import StandardScaler
data_std = pd.DataFrame(StandardScaler().fit_transform(data[data.columns[1:]]))
data_std.columns = features
data_std['Label'] = data['Label']

# #### Train-Test-Split 
from sklearn.model_selection import train_test_split
X = data_std.drop('Label',axis=1)
y = data_std['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# # Support Vector Machines with Python
# #### Train a Model
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
# ####SVM Model Evaluation
predictions_svc = svc_model.predict(X_test)
matrix_svc = confusion_matrix(y_test,predictions_svc)
report_svc = classification_report(y_test,predictions_svc)
acc_svc = accuracy_score(y_test, predictions_svc)
print("SVM Model Evaluation")
print(matrix_svc)
print(report_svc)
print("-"*80)

# # Decision Trees with Python
# #### Train a Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
# #### Decision Tree Model Evaluation
predictions_dtree = dtree.predict(X_test)
matrix_dtree = confusion_matrix(y_test,predictions_dtree)
report_dtree = classification_report(y_test,predictions_dtree)
acc_dtree = accuracy_score(y_test, predictions_dtree)
print("Decision Tree Model Evaluation")
print(matrix_dtree)
print(report_dtree)
print("-"*80)

# # Artificial Neural Network with python
# #### Train a Model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
predictions_ann = mlp.predict(X_test)
# #### Model Evaluation
matrix_ann = confusion_matrix(y_test,predictions_ann)
report_ann = classification_report(y_test,predictions_ann)
acc_ann = accuracy_score(y_test, predictions_ann)
print("ANN Model Evaluation")
print(matrix_ann)
print(report_ann)
print("-"*80)

#Summary accuracy of supervised learning
print("Accuracy of SVM model",acc_svc)
print("Accuracy of Decision tree model",acc_dtree)
print("Accuracy of ANN model",acc_ann)
print("-"*80)


# # K-Mean Clustering with python
# #### Train a Model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state= 12)
kmeans.fit(data_std.drop('Label',axis=1))
# kmeans.cluster_centers_
# kmeans.labels_
# #### K-Mean Clustering Model Evaluation
sse = kmeans.inertia_
print("K-Means Clustering Model Evaluation")
print('SEE of 3 Clusters =  ',sse)
data['cluster']=kmeans.labels_
print("Cross check cluster and label")
print(pd.crosstab(data_std['Label'], data['cluster']))
print("-"*80)
