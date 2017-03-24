import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data and package into an object
x_data = np.empty([15*240, 18*9])
y_data = np.empty([15*240,1])
counter = 0

for row in pd.read_csv('train.csv',iterator = True, chunksize = 18):
	row = row.set_index('item', inplace = False, drop = True)
	row = row.T
	#print(row)
	for i in range(15):
		mtx = np.array(row[i:i+9])
		mtx = mtx.ravel()
		x_data[counter] = mtx
		y_data[counter] = row.iloc[i+9,9]
		counter = counter + 1

#print(y_data)

x = tf.placeholder(tf.float32, [None, 18*9]) 
y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([18*9, 1]))
b = tf.Variable(tf.zeros([1,1]))
Wx_plus_b = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - Wx_plus_b),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.000000001).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(240*15):
	sess.run(train_step,feed_dict={x:x_data,y:y_data})
	if i % 1 == 0:
		print(sess.run(loss, feed_dict={x:x_data,y:y_data}))

#xtmp = np.arange(10*18*9).reshape(10,18*9)
#Wtmp = np.arange(18*9).reshape(18*9,1)
#btmp = np.arange(1).reshape(1,1)

#print(xtmp.dot(Wtmp) + btmp)
#print(xtmp)

#m = 100
#diff_sum = np.zeros(16*9)
#learn_rate = 0.1

