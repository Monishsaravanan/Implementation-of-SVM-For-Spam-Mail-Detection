# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.Start the program.

step 2.Import the necessary python packages using import statements.

step 3.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

step 4.Split the dataset using train_test_split.

step 5.Calculate Y_Pred and accuracy.

step 6.Print all the outputs.

step 7.End the Program.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: MONISH S
RegisterNumber:  212223040115
```
```
import pandas as pd
data = pd.read_csv("C:/Users/ANANDAN S/Documents/ML labs/spam.csv",encoding = 'Windows-1252')
data.head()
data.info()
data.isnull().sum()
x= data["v1"].values
y= data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer is a method to convert text into numerical data.the text is transformed to a sparse matrix
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:
# Head:
![Screenshot 2024-10-24 030744](https://github.com/user-attachments/assets/9a244c05-b1f9-4e95-8c2d-0ebd23ca5e99)
# info:
![Screenshot 2024-10-24 030752](https://github.com/user-attachments/assets/9558487e-97f0-480e-a16b-e7a58f38715e)
# accuracy
![Screenshot 2024-10-24 030759](https://github.com/user-attachments/assets/94396dab-562c-4b20-b3e8-821e2c261059)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
