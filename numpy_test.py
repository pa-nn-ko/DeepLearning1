import numpy as np
a = np.array([0,1,2,3,4,5])
print(a)
print("")

b = np.array([[0,1,2],[3,4,5]])
print(b)
print("")

c= np.array([[[0,1,2],[3,4,5]],[[5,4,3,],[2,1,0]]])
print(c)
print("")

print(np.shape(c))
print(np.size(c))
print("")

d = [[1,2],[3,4],[5,6]]
print(len(d))
print(len(np.array(d)))
print("")

print(np.zeros(10))
print(np.ones(10))
print(np.random.rand(10))

print(np.zeros((2,3)))
print(np.ones((2,3)))
print("")

print(np.arange(0,1,0.1))
print("")

print(np.arange(10))
print("")

print(np.linspace(0,1,11))
print("")

print(np.linspace(0,1))
a = np.array([0,1,2,3,4,5,6,7])
b = a.reshape(2,4)
print(b)
print("")

c = b.reshape(2,2,2)
print(c)
print("")

d = c.reshape(4,2)
print(d)
print("")

e = d.reshape(-1)
print(e)
print("")

f = e.reshape(2, -1)
print(f)
print("")

a = np.array([0,1,2,3,4,5]).reshape(2,3)
print(a)
print(a + 3)
print(a * 3)
print("")

b = np.array([5,4,3,2,1,0]).reshape(2,3)
print(b)
print(a + b)
print(a * b)
print("")

