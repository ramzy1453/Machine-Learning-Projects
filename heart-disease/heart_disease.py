import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('datasets/heart.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(
  penalty='elasticnet', solver='saga',
  max_iter=10000, multi_class='multinomial',
  C=428.1332398719391, l1_ratio=0.2222222222222222,
)

log_model.fit(X_train, y_train)

def predict_heart_disease(patient_parametrs):
  pred_is_heart_sick = log_model.predict(np.array(patient_parametrs))
  return "Harmful" if pred_is_heart_sick == 1 else "Harmless"

print(predict_heart_disease(
  [[54, 1, 0, 122, 286, 0, 0, 116, 1, 3.2, 1, 2, 2]]
))
