import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import saxspy
from saxspy import debyeWaller as dwf
from scipy.interpolate import CubicSpline

# Normalization Functions
def norm_log(data):
    data=np.log(data)
    return (data-np.min(data))/(np.max(data)-np.min(data))
def log(data):
    return np.log(data)
def minmax(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
# Generate 2Dx3 Product matrix 
def gen_product_matrix(I_values):
    return np.repeat(np.expand_dims(np.outer(I_values, I_values), -1), 3, axis=-1)

# Processing function - performs cubic spline and outer product on raw samples
def Process_lamellar(data, blanks):
    q=np.linspace(0,0.452,1566)
    q2=np.linspace(0.01,0.43,200)
    random_blank = random.randint(0,169)
    blanky=blanks[random_blank]
    I_data = []
    for k in tqdm(range(len(data[:,0,0]))):
        random_voigt_gaussalpha = random.uniform(0.0001,0.005) #alpha and gamma params varied for each sample - parameterise this with a distribution reflecting test set
        random_voigt_lorenzgamma = random.uniform(0.0001,0.005)
        dw_param = random.uniform(0.02,0.1)# randomise debye waller factor
        zero_array = np.zeros(1566)
        for i in range(len(data[0,0,:])):
            if not data[k,0,i] == 0:
                temp = data[k,1,i]*dwf(data[k,0,i],dw_param)*saxspy.voigtSignal(q, random_voigt_gaussalpha, random_voigt_lorenzgamma, data[k,0,i])
                zero_array+=temp
            else:
                temp = 0*saxspy.voigtSignal(q, random_voigt_gaussalpha, random_voigt_lorenzgamma, data[k,0,i])
                zero_array+=temp
      
        zero_array = log(zero_array+1)
        I = (zero_array+log(blanky))
        
        I2 = []
        
        for i in range(0,len(I)):
            if i < 1100:
                I2.append(I[i])
            else:
                if I[i]>log(blanky[i]):
                    temp = I[i]-log(blanky[i])
                    I2.append(log(blanky[i])+temp)
                    
                else:
                    I2.append((I[i]))
                    
        I = minmax(np.array(I2))
        cubics = CubicSpline(q,I, bc_type = 'natural')
        I = cubics(q2)
        I_mat = gen_product_matrix(I)
        I_data.append(I_mat)
    I_data = np.array(I_data)
    return I_data, q2


if __name__=='__main__':
    raw_data = np.load('../Synthetic_raw/lamellar.npy')
    blanks = np.load('blanks_raw.npy')
    processed, q = Process_lamellar(raw_data, blanks)
    np.save('../Synthetic_Processed/lamellar.npy', processed)
    np.save('../Synthetic_Processed/lamellar_q.npy', q)
    