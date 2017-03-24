import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode

df = pd.read_csv('train.csv')

df = df.drop(['Name','Ticket','Cabin'], axis = 1)

age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
train_data = df.values
#print(df.values)
#print(type(df.values))

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:], train_data[0:,0])
df_test = pd.read_csv('test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values

print(df_test.info())

output = model.predict(test_data[:,1:])
print(output)

result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('titanic_1-2.csv', index=False)


