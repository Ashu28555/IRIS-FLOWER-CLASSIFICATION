# IRIS-FLOWER-CLASSIFICATION

# Task 1  : IRIS FLOWER CLASSIFICATION


### Iris flower has three species; setosa, versicolor, and virginica, which differs according to their measurements. Now assume that you have the measurements of the iris flowers according to their species, and here your task is to train a machine learning model that can learn from the measurements of the iris species and classify them.
### Although the Scikit-learn library provides a dataset for iris flower classification, you can also download the same dataset from here for the task of iris flower classification with Machine Learning. 

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

#### Load the dataset

data = pd.read_csv("Iris.csv")

data

data.describe()

data.info()

#### Data cleaning and pre processing

data.isna().sum()

data["Species"] = data["Species"].str.replace("Iris-","")

data

#### EDA

sns.pairplot(data, hue = 'Species')

x= data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

x

y = data['Species']

y

y.value_counts()

#### Splitting the dataset into testing and training

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train

x_test

### Train with Logistic Regression model

model = LogisticRegression()

model.fit(x_train,y_train)

predict = model.predict(x_test)

predict

cm = confusion_matrix(predict,y_test)
cm

cm = ConfusionMatrixDisplay(cm)
cm.plot()
plt.title("Confusion Matrix by Logistic Regression")

report = classification_report(predict,y_test)
print(report)

#### Model Evaluation

new_data = x.iloc[[20]]
pred = model.predict(new_data)

pred

result = y.iloc[73]
result

#### Run the Model

sl = float(input("Enter the Sepal length :"))
sw = float(input("Enter the Sepal width :"))
pl = float(input("Enter the Petal length :"))
pw = float(input("Enter the Petal width :"))
new_data = [[sl,sw,pl,pw]]
pred = model.predict(new_data)
print("The species of flower with repect of follwing measurements is :",pred[0])

## The End

