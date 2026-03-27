import pandas as pd 
import numpy as np
from sklearn.ensemble import IsolationForest as IF

arr = np.array([1,2,3])
print(arr)

data = [1]

clf = IF(
    random_state=0
).fit(data)