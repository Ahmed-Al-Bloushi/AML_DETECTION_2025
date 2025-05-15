import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import pickle

df = pd.read_csv("data.csv")
df['type'] = df['type'].astype('category')
df['type_code'] = df['type'].cat.codes
df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']

X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'type_code', 'balance_diff']]
y = df['isFraud']

X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1]
}

grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved as best_model.pkl")
