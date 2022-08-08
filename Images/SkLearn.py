import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import os
import sys

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1,  0, 0, 1, 1])
# skf = StratifiedKFold(n_splits=3)
# train_index, test_index = skf.split(X, y)

print(X,y)
# print(train_index)

args = argparser.parse_args(sys.argv[1:])
print(args)