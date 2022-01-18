import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.special import wofz
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

# global blanks
blanks = np.load('blanks_raw.npy')

#Norm functions
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

def voigtSignal(x, alpha, gamma, shift):
    """
    Return the Voigt line shape at x with x component HWHM gamma
    and Gaussian component HWHM alpha.
    Voigt profile is the convolution of the Gaussian Profile and the Lorentzian Profile
    Gamma is HWHM of Lorentzian
    Sigma is SD of Gaussian
    Alpha is HWHM of Gaussian
    https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
    
    """

    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x-shift + 1j*gamma)/sigma/np.sqrt(2))) / sigma/np.sqrt(2*np.pi)

def gaussianSignal(x, sigma, shift):
    """
    Return Gaussian signal at of array x with standard deviation sigma that has been shifted by shift
    """
    return np.exp((-1/sigma)*((x-shift))**2)

def debyeWaller(q, decay_parameter):
    """
    Return debeye waller factor which is just a gaussian decay of parameter decay_parameter
    Wikipedia - describe the attenuation of x-ray scattering or coherent neutron scattering caused by thermal motion.
    """
    return gaussianSignal(q, decay_parameter, 0)

dwf = debyeWaller
#Function for generating DWF / Voigts / Lorenztians and Interpolating data in q2 range
def Process_cubic(data):
    random_blank = random.randint(0,169)
    blanky=blanks[random_blank]
    q=np.linspace(0,0.452,1566)
    q2=np.linspace(0.01,0.43,200)
    random_voigt_gaussalpha = random.uniform(0.0001,0.001)
    random_voigt_lorenzgamma = random.uniform(0.0001,0.005)
    dw_param = random.uniform(0.005,0.02)
    zero_array = np.zeros(1566)
    for i in range(len(data[0,:])):
        if not data[0,i] == 0:
            temp = data[1,i]*dwf(data[0,i],dw_param)*voigtSignal(q, random_voigt_gaussalpha, random_voigt_lorenzgamma, data[0,i])
            zero_array+=temp
        else:
            temp = data[1,i]*voigtSignal(q, random_voigt_gaussalpha, random_voigt_lorenzgamma, data[0,i])
            zero_array+=temp
            
    zero_array = log(zero_array+1)
    I = minmax(zero_array+log(blanky))
    cubics = CubicSpline(q,I, bc_type = 'natural')
    I = cubics(q2)
    I_mat = gen_product_matrix(I)
    I_data = np.array(I_mat)
    return I_data


if __name__ == '__main__':

    for raw_data in ['../Synthetic_raw/P_cubic.npy','../Synthetic_raw/G_cubic.npy','../Synthetic_raw/D_cubic.npy']:
        if os.path.isfile(raw_data):
            if 'P' in raw_data:
                # process p
                rawdata = np.load(raw_data)
                rawdat = [i for i in rawdata]
                print('Processing P cubic...')
                processed_p = parallel_process(rawdat, Process_cubic)
                processed_p = np.array(processed_p)
                np.save('../Synthetic_Processed/P_cubic.npy', processed_p)
            if 'G' in raw_data:
                # process g
                rawdata = np.load(raw_data)
                rawdat = [i for i in rawdata]
                print('Processing G cubic...')
                processed_g = parallel_process(rawdat, Process_cubic)
                processed_g = np.array(processed_g)
                np.save('../Synthetic_Processed/G_cubic.npy', processed_g)
            if 'D' in raw_data:
                # process d
                rawdata = np.load(raw_data)
                rawdat = [i for i in rawdata]
                print('Processing D cubic...')
                processed_d = parallel_process(rawdat, Process_cubic)
                processed_d = np.array(processed_d)
                np.save('../Synthetic_Processed/D_cubic.npy', processed_d)
            np.save('../Synthetic_Processed/cubic_q.npy', np.linspace(0.01,0.43,200))