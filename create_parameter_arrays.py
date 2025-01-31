# THe goal of this file is to create the arrays N_param and H
# N_param contains the number of parameters of all the neural networks to obtain figure 4
# H contains the hidden layer size corresponding to the number of parameters in the array N_param
# Remark: H does not exactly produce a number of parameters equal to the ones in N_param
# because there is a rounding step after applying the formula to go from number of parameters to hidden size

import numpy as np
from pathlib import Path
import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) # change working directory to the directory where the file is executed

Path('parameters_arrays').mkdir(parents=True, exist_ok=True)

N_param = np.array([3, 4, 7, 10, 20, 25, 27, 29, 31, 32, 33, 34, 35, 37, 39, 40, 42, 45, 50, 55, 70, 200, 250, 800]) * 1e3
np.save('parameters_arrays/N_param.npy', N_param)

H = (N_param - 10)/795
H = np.rint(H).astype(int)
np.save('parameters_arrays/H.npy', H)

