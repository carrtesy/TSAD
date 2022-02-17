import pandas as pd
import numpy as np
from random import random
TRAIN_STEPS = 10000
TEST_STEPS = 10000

p = 100
f = 1/100

ANOMALY_THRESHOLD = 0.9
train_t = np.arange(TRAIN_STEPS)
test_t = np.arange(TEST_STEPS)

train_X = np.sin(f*2*np.pi*train_t)
train_y = np.array([0] * TRAIN_STEPS).astype(int)

test_X = []
test_y = []
for i in range(TEST_STEPS // p):
    if random() > ANOMALY_THRESHOLD:
        test_X += [0] * p
        test_y += [1] * p
    else:
        sub_t = np.arange(p)
        sub_x = list(np.sin(f*2*np.pi*sub_t))
        test_X += sub_x
        test_y += [0] * p
test_X = np.array(test_X)
test_y = np.array(test_y).astype(int)


np.savetxt("train_X.csv", train_X)
np.savetxt("train_y.csv", train_y, fmt = "%i")
np.savetxt("test_X.csv", test_X)
np.savetxt("test_y.csv", test_y, fmt = "%i")

