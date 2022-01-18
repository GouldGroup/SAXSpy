import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import saxspy
import matplotlib.pyplot as plt
import numpy as np

verbose = True
lm = saxspy.LamellarModel()


#----------------------- generate synthetic data -----------------------#
# ranges of: lattice parameter, bilayer (lipid length), sigma tail, sigma bilayer

generated_distribution = np.load('generated_prob_dist_lam.npy')
params = np.array([[40,104], [25, 35], [2, 4], [1, 3]])
print('Generating Synthetic lamellar data...')
store_it = lm.generateSynthLamellar(params, generated_distribution)
#----------------------- single example synthetic data -----------------------#
if verbose == True:
    temp = store_it[np.random.choice(len(store_it[:,0]),1)[0]]
    plt.stem(temp[0,:], temp[1,:].real, use_line_collection=True)
    plt.show()


# save data
np.save('../Synthetic_raw/lamellar.npy', store_it)