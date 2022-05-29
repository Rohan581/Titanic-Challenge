import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("train.csv")
embarked_mapping_1 = {label: idx for idx, label in enumerate(np.unique(df['Embarked'].astype(str)))}
embarked_mapping_2 = {label: idx for idx, label in enumerate(np.unique(df['Sex'].astype(str)))}
df['Embarked'] = df['Embarked'].map(embarked_mapping_1)
df['Sex'] = df['Sex'].map(embarked_mapping_2)
df = df.fillna(df['Embarked'].mode())
df = df.drop(['Cabin', 'Name', 'Ticket', 'Parch'], axis=1)
df = df.fillna(df['Age'].mean())

X = df.iloc[:, 2:]
y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=42, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=100, random_state=1, solver='lbfgs', max_iter=100, multi_class='auto')
classifier = lr
classifier.fit(X_train_std, y_train)
y_pred = classifier.predict(X_test_std)
print(y_pred)