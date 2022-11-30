import numpy as np
np.seterr(divide='ignore',invalid='ignore')

print(np.array([0,0])/np.array([0,0]))