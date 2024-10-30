# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vishwa K
RegisterNumber:212223080061  
*/
from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

uploaded = files.upload()  
df = pd.read_csv("spam.csv", encoding='latin-1')  

print(df.head())

df = df[['v1', 'v2']]  
df.columns = ['label', 'text']  

df['label'] = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm = SVC(kernel='linear')
svm.fit(X_train_tfidf, y_train)

y_pred = svm.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(svm, 'svm_spam_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

```

## Output:
![SVM For Spam Mail Detection](sam.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
