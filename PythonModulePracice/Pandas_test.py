import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#s = pd.Series([1,3,6,np.nan,44,1])
#print(s)

#dates = pd.date_range('20160101',periods=6)
#dates = list(range(6))
#dates = np.arange(6)
#df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
#print(type(dates))
#print(df)
#print(df['b'])

#df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
#print(df1)

#df2 = pd.DataFrame({'A' : 1.,
#					'B' : pd.Timestamp('20130102'),
#					'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
#					'D' : np.array([3] * 4,dtype='int32'),
#					'E' : pd.Categorical(["test","train","test","train"]),
#					'F' : 'foo'})

#print(df2)
#print(df2.dtypes)
#print(df2.index)
#print(df2.columns)
#print(df2.values)
#print(df2.describe()) #統計數據
#print(df2.T)
#print(df2.sort_index(axis=1, ascending=False))
#print(df2.sort_index(axis=0, ascending=False))
#print(df2.sort_values(by='B'))

#選擇數據

"""
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
print(df)
print(df['A'])
print(df.A)
print(df[0:3])
print(df['20130102':'20130104'])
"""

"""
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
print(df)
print(df.loc['20130102'])
print(df.loc[:,['A','B']])
print(df.loc['20130102',['A','B']])

print(df.iloc[3,1])
print(df.iloc[3:5,1:3])
print(df.iloc[[1,3,5],1:3])

print(df.ix[:3,['A','C']]) #混合loc, iloc

print(df[df.A>8])
"""

#設置數值

#dates = pd.date_range('20130101', periods=6)
#df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
#print(df)

#df.iloc[2,2] = 1111
#df.loc['20130101','B'] = 2222
#print(df)

#df.B[df.A>4] = 0
#print(df)

#df['F'] = np.nan
#print(df)

#df['E'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods=6))
#print(df)

#處理丟失數據

"""
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)

print(df.dropna(axis=0,how='any'))
print(df.fillna(value=0))
print(df.isnull())
print(np.any(df.isnull()))
"""

#導入導出

#data = pd.read_csv('test.csv')
#print(data)
#data.to_pickle('test_out.pickle')

#合併concate

"""
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

res = pd.concat([df1,df2,df3], axis=0)
print(res)
res = pd.concat([df1,df2,df3], axis=0, ignore_index=True) #重置index
print(res)
"""

"""
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

res = pd.concat([df1,df2], axis=0, join='outer')
print(res)
res = pd.concat([df1,df2], axis=0, join='inner')
print(res)
res = pd.concat([df1,df2], axis=1, join_axes=[df1.index])
print(res)
res = pd.concat([df1,df2], axis=1)
print(res)
"""

"""
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])

res = df1.append(df2, ignore_index=True)
print(res)
res = df1.append([df2,df3], ignore_index=True)
print(res)
res = df1.append(s1,ignore_index=True)
print(res)
"""

#合併merge

#尚
#待
#研
#究

#出圖

"""
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()
"""

"""
data = pd.DataFrame(
	np.random.randn(1000,4),
	index=np.arange(1000),
	columns=list("ABCD")
	)
data = data.cumsum()
data.plot()
plt.show()
"""

"""
data = pd.DataFrame(
	np.random.randn(1000,4),
	index=np.arange(1000),
	columns=list("ABCD")
	)
ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()
"""









