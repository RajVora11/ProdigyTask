Cleaning

import pandas as pd
dataset = pd.read_csv('E:\\Latha\\LathaSKPIMCS\\Machine Learning\\Class\\Practical\\Preprocessing\\Data1.csv')
X = dataset.iloc[:,:-1].values #Takes all rows of all columns except the last column
Y = dataset.iloc[:,-1].values # Takes all rows of the last column
X
Y

print(dataset.columns)
dataset

dataset.info()
dataset.head()
dataset.tail()

#Row and column count
dataset.shape


#Count missing values
dataset.isnull().sum().sort_values(ascending=False)
dataset.isnull().sum()


#Removing insufficient column
dataset_new = dataset.drop(['Age',], axis = 1)
dataset_new

#To measure the central tendency of variables
dataset_new.describe()

#To change column name
dataset.rename(index=str, columns={'Country' : 'Countries','Age' : 'age', 'Salary' : 'Sal','Purchased' : 'Purchased'}, inplace = True)

dataset
#Count missing values
dataset.isnull().sum().sort_values(ascending=False)

#Print the missing value column
dataset[dataset.isnull().any(axis=1)].head()

#Remove missing value rows
ds_new = dataset.dropna()
ds_new
ds_new.shape
ds_new.isnull().sum()

#To check datatype
ds_new.dtypes

#To convert as integer
ds_new['age'] = ds_new['Age'].astype('int64')

ds_new.dtypes
ds_new
Imputing Mean, Median and Most_frequent
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

==========
Placment Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('E://Latha//JG_MCA//MachineLearning//Placement//Placement.csv')
dataset

dataset = dataset.drop('sl_no', axis=1)


# catgorising col for further labelling
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes


# labelling the columns
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
 
# display dataset
dataset

columns = ['gender','ssc_p','ssc_b','hsc_p','etest_p','specialisation','Masters','status']
import csv
with open('E://Latha//JG_MCA//MachineLearning//Placement//placement_record.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(columns)
    writer.writerows(dataset)

# selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
 
# display dependent variables
Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
 
# display dataset
dataset.head()

# creating a classifier using sklearn
from sklearn.linear_model import LogisticRegression
 
clf = LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000).fit(X_train, Y_train)
# printing the acc
clf.score(X_test, Y_test)


clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])


# creating a Y_pred for test data
Y_pred = clf.predict(X_test)
 
# display predicted values
Y_pred

# evaluation of the classifier
from sklearn.metrics import confusion_matrix, accuracy_score
 
# display confusion matrix
print(confusion_matrix(Y_test, Y_pred))
 
# display accuracy
print(accuracy_score(Y_test, Y_pred))
======
Naivebytes

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


play_tennis = pd.read_csv("E:\\Latha\\LathaSKPIMCS\\Machine Learning\\Class\\Practical\\Algorithms\\All_Algorithms\PlayTennis.csv")
play_tennis.head()

number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])
play_tennis

#define the features and the target variables
#features = play_tennis.iloc[:, :-1].values
#target = play_tennis.iloc[:, -1].values
#features
#target


#define the features and the target variables
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"

features_train, features_test, target_train, target_test = train_test_split(play_tennis[features],play_tennis[target],test_size = 0.33,random_state = 54)


model = GaussianNB()
model.fit(features_train, target_train) 
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(accuracy)

print (model.predict([[1,2,0,1]]))
print(model.predict([[2,0,0,0]]))
print(model.predict([[0,0,0,1]]))

======
Diabitic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('E:\\Latha\\JG_MCA\\MachineLearning\\Diabetic\\diabetes.csv', encoding='latin-1')
data

sns.heatmap(data.isnull())

correlation = data.corr()
print(correlation)

X = data.drop("Outcome", axis = 1)
Y = data["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size = 0.2,random_state = 54)
X_train

# model fit
model = LogisticRegression(C=1)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
predictions


accuracy = accuracy_score(predictions, Y_test)
print(accuracy)
=========
algorithaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('E:\\Latha\\JG_MCA\\MachineLearning\\KNN\\shirtsize.csv')
print(df.columns)
df



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['T Shirt Size'] = le.fit_transform(df['T Shirt Size'])
df

X = df.iloc[:, :-1].values
Y = df.iloc[:, 2].values
X
Y

#Split traing and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

#K nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
#accuracies['KNN'] = acc
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))
print(knn.score(X_test,Y_test))

#Predict for single
print(knn.predict([[158,58]]))
print(knn.predict([[170,68]]))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, prediction)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction)
print("Accuracy:",result2)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, Y_train)
prediction1 = clf.predict(X_test)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, Y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, Y_test)))
result = confusion_matrix(Y_test, prediction1)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction1)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction1)
print("Accuracy:",result2)


#SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
prediction2 = svm.predict(X_test)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, Y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, Y_test)))
result = confusion_matrix(Y_test, prediction2)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction2)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction2)
print("Accuracy:",result2)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
prediction3 = gnb.predict(X_test)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, Y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, Y_test)))
result = confusion_matrix(Y_test, prediction3)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, prediction3)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,prediction3)
print("Accuracy:",result2)
======================================

