import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import saxspy
import matplotlib.pyplot as plt
import numpy as np
# Instantiate the synthetic model: 'P', 'G', or 'D'
phase = 'G'
print(f"running cubic {phase} model...")
cm = saxspy.CubicModel(phase)
#----------------------- generate synthetic data -----------------------#
# ranges of: lattice parameter, length of lipid, lipid head sigma
params = np.array([[5, 30], [1, 5], [.0001, .01]])
store_it = cm.generateSynthCubic(params)

np.save(f'../Synthetic_raw/{phase}_cubic.npy', store_it)