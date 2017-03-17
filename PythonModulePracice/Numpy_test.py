import numpy as np

#array = np.array([[1,2,3],[4,5,6]]);
#print(array)
#print('number of dim',array.ndim)
#print('shape',array.shape)
#print('size',array.size)

#a = np.array([2,23,4], dtype=np.float32)
#print(a.dtype)

#a = np.array([[2,23,4],
#			  [2,32,4]])
#print(a)

#a = np.zeros((3,4),dtype=int)
#print(a)

#a = np.ones((3,4), dtype=int)
#print(a)

#a = np.empty((3,4))
#print(a)

#a = np.arange(12).reshape((3,4))
#print(a)

#a = np.linspace(1,10,6).reshape((2,3))
#print(a)

#運算處理

"""
a = np.array([10,20,30,40])
b = np.arange(4)
print(a,b)
c=a-b
d=a+b
e=a*b
f=b**2
g=10*np.sin(a)
print(c,d,e,f,g)
print(b==3)
"""

"""
a = np.array([[1,1],
			  [0,1]])
b = np.arange(4).reshape((2,2))
print(a)
print(b)

c=a*b #逐項相乘
c_dot=np.dot(a,b) #矩陣相乘
c_dot_2=a.dot(b)
print(c)
print(c_dot)
print(c_dot_2)
"""

"""
a = np.random.random((2,4))
print(a)

print(np.sum(a))
print(np.min(a))
print(np.max(a))

print(np.sum(a,axis=1)) #x軸
print(np.min(a,axis=0)) #y軸
print(np.max(a,axis=1))
"""

#運算處理2

#A = np.arange(2,14).reshape((3,4))
#print(np.argmin(A)) #最小值索引
#print(np.argmax(A)) #最大值索引
#print(np.mean(A))
#print(np.average(A))
#print(np.median(A)) #中位數
#print(np.cumsum(A))
#print(np.diff(A)) #每一列中後一項與前一項之差
#print(np.nonzero(A)) #自行體會

#A = np.arange(14,2,-1).reshape((3,4))
#print(A)
#print(np.sort(A)) #針對每一列排序
#print(np.transpose(A))
#print(A.T)
#print(np.clip(A,5,9)) #有點像把數據侷限在這兩個數字中


#索引

"""
A = np.arange(3,15)
print(A[3])
"""

"""
A = np.arange(3,15).reshape((3,4))
print(A)
print(A[2])
print(A[1][1])
print(A[1,1])
print(A[1,1:3])

for row in A:
	print(row)

for column in A.T:
	print(column)
"""

"""
A = np.arange(3,15).reshape((3,4))
print(A)
print(A.flatten()) #多維矩陣轉成一維陣列

for item in A.flat: #多維矩陣轉乘一維陣列的迭代器
	print(item)
"""

#array合併

#A = np.array([1,1,1])
#B = np.array([2,2,2])
#print(np.vstack((A,B))) #vertical stack，上下合併
#C = np.vstack((A,B))
#print(A.shape,C.shape)
#D = np.hstack((A,B))
#print(D)
#print(A.shape,D.shape)

#A = np.array([1,1,1])
#print(A[np.newaxis,:])
#print(A[np.newaxis,:].shape)
#print(A[:,np.newaxis])
#print(A[:,np.newaxis].shape)

#A = np.array([1,1,1])[:,np.newaxis]
#B = np.array([2,2,2])[:,np.newaxis]
#C = np.vstack((A,B))
#D = np.hstack((A,B))
#print(D)
#print(A.shape,D.shape)
#C = np.concatenate((A,B,B,A),axis=0)
#print(C)
#D = np.concatenate((A,B,B,A),axis=1)
#print(D)

#array分割

"""
A = np.arange(12).reshape((3,4))
print(A)
print(np.split(A, 2, axis=1))
print(np.split(A, 3, axis=0))
#以上只能等量切割
"""

"""
A = np.arange(12).reshape((3,4))
print(np.array_split(A, 3, axis=1))
"""

"""
A = np.arange(12).reshape((3,4))
print(np.vsplit(A,3)) #相當於print(np.split(A,3,axis=0))
print(np.hsplit(A,2)) #相當於print(np.split(A,2,axis=1))
"""

#copy and deep copy

#a = np.arange(4)
#b = a
#c = a
#d = b
#a[0] = 1
#print(a)
#print(b is a)
#print(c is a)
#print(d is a)
#會同時改變
#b = a.copy
#print(b)
#a[3] = 44
#print(a)
#print(b)
#不會同時改變，已沒有關聯

