import pandas as p
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split

import numpy as nm
from sklearn.linear_model import LogisticRegression

df = p.read_csv("heart.csv")
#print(df.head())

age = df["age"]
heartAttack = df["target"]

age_train,age_test,heartAttack_train,heartAttack_test = train_test_split(age , heartAttack , test_size = 0.25,random_state = 0)

X = nm.reshape(age_train.ravel(), (len(age_train), 1))
Y = nm.reshape(heartAttack_train.ravel(), (len(heartAttack_train), 1))

classifier = LogisticRegression(random_state = 0) 
classifier.fit(X, Y)

