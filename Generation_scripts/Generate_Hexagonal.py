import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import saxspy
import matplotlib.pyplot as plt
import numpy as np

verbose = True
hm = saxspy.HexagonalModel()

#----------------------- generate synthetic data -----------------------#
# ranges of: lattice parameter, head position, sigma head, sigmal tail
params = np.array([[20, 78], [5, 30], [0.5, 3], [0.5, 5]])
print('Generating Synthetic hexagonal data...')
store_it = hm.generateSynthHexagonal(params)

#----------------------- single example synthetic data -----------------------#
if verbose == True:
    temp = store_it[np.random.choice(len(store_it[:,0]),1)[0]]
    plt.stem(temp[0,:], temp[1,:].real, use_line_collection=True)
    plt.show()


    # #save data
np.save('../Synthetic_raw/hexagonal.npy', store_it)




