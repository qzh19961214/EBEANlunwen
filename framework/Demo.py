import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc
import copy
import csv

slipRange = np.arange(0,1,0.1)
guessRange = np.arange(0,1,0.1)
for i in slipRange:
    for j in guessRange:
        print(i, " ", j)