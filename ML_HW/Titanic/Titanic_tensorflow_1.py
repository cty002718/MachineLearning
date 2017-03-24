import pandas as pd
import numpy as np
import tensorflow as tf
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

#print(train_data.shape)

#print(train_data[:,1:], train_data[:,0:1])
x = tf.placeholder(tf.float32, [None, 9]) 
y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([9, 1]))
b = tf.Variable(tf.zeros([1,1]))
Wx_plus_b = tf.matmul(x, W) + b


predict = tf.sigmoid(Wx_plus_b)
predict = tf.floor(predict + 0.5)

wrong = tf.reduce_sum(tf.abs(predict - y))


entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = Wx_plus_b, labels = y)
s = tf.reduce_sum(entropy,reduction_indices=[1])
loss = tf.reduce_mean(s)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
	sess.run(train_step,feed_dict={x:train_data[:,2:],y:train_data[:,0:1]})
	#if i % 100 == 0:
		#print(sess.run(loss, feed_dict={x:train_data[:,2:],y:train_data[:,0:1]}))
		#print(sess.run(ans, feed_dict={x:train_data[:,2:],y:train_data[:,0:1]}))

df_test = pd.read_csv('test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values
#print(df_test)
#print(df_test.info())

output = sess.run(predict,feed_dict={x:test_data[:,1:]})
output = np.squeeze(output)
print(output)
#print(result)

result = np.c_[test_data[:,0].astype(int), output.astype(int)]
#print(result)
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('titanic_1-3.csv', index=False)



