import numpy as np

a = np.ones((2,3,4))
print(a)

mask = np.array([[True,False,False],[False,True,True]])
a[~mask] = np.zeros((4))
print(a)
