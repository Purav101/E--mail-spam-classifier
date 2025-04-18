import numpy as np
import pandas as pd 
import re,string
import nltk 
from nltk.corpus import stopwords
nltk.download("stopwords")
df=pd.read_csv(r"C:\Users\apurv\Downloads\all ml dataset\Spam_SMS.csv")
#print(df.info())
#data preprocessing for model
stop_word=stopwords.words("english")
def preprocess(text):
    text= text.lower()
    text=''.join([char for char in text if char  not in string.punctuation])
    text=' '.join([word for word in text.split() if word not in stop_word])
    return text
#print(preprocess("!!!!!  , ;=;a an the hello"
df['prepro_data']= df['Message'].apply(preprocess)
#libraries for tokenizer and model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline 
#model
X=df['prepro_data']
y=df['Class'].map({'ham':0,'spam':1})
#print(X)
cv = CountVectorizer(max_features=5000)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)
X_train=cv.fit_transform(X_train)
X_test=cv.transform(X_test)
#model
cl=LogisticRegression(class_weight='balanced')
#pipe=make_pipeline(preprocess,cv,classifier)
#pipe.fit(X_train,y_train)
cl.fit(X_train,y_train)
y_pred=cl.predict(X_test)
#evaluation
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))
import pickle
pickle.dump(cl,open('ss.pkl','wb'))
pickle.dump(cv,open('cv.pkl','wb'))

