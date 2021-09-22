import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv
from scipy.special import wofz
from skimage import measure
from tqdm import tqdm


#-------------GENERAL FUNCTIONS-------------#
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

#-------------LAMELLAR MODEL FUNCTIONS AND CLASS-------------#
class LamellarModel():


    def electronDensity(self, x, sigma_bilayer, sigma_tail, bilayer, rel_tail_intensity=0.2):
        """
        generate electron density distribution
            x - x coordinates in real space
            sigma_bilayer - head group thicknessr
            sigma_tail - tail thickness
            bilayer - position of rhead group
        return 
            rho - real space electron density distribution
        """
        rho = gaussianSignal(x, sigma_bilayer, bilayer) + gaussianSignal(x, sigma_bilayer, -bilayer) - gaussianSignal(x, sigma_tail, 0)*rel_tail_intensity
        return rho

    def lamellarLattice(self, lat_param, n_dim):
        """
        generate reciprocal lamellar lattice - does this not generate normal lamellar lattice -- input in generateSynthLamellar is the recipricol lattice
            lat_param - lattice parameter
            n_dim - number of lattice points to generate
        return
            q - reciprocal space scattering  
            np.arange creates an array starting with first input to second input in intervals of final input.
            i.e. np.arange(0,2,0.5) returns 0, 0.5, 1, 1.5 - evenly spaced points
        """
        q = np.linspace(0,lat_param*n_dim,n_dim)
        return q

    def lamellarCell(self, q, sigma_bilayer, sigma_tail,  bilayer):
        """
        generate reciprocal cell of lamellar model
            q - reciprocal space scattering
            sigma_bilayer - head group thickness
            sigma_tail - tail thickness
            bilayer - position of head group

        """
        return (np.sqrt(np.pi*sigma_bilayer))*np.exp(-(np.pi**2)*sigma_bilayer*(q**2))*(np.exp((-1j*q*bilayer)/2)+np.exp((1j*q*bilayer)/2))+\
        (np.sqrt(np.pi*sigma_tail))*np.exp(-(np.pi**2)*sigma_tail*(q**2))

    def generateSynthLamellar(self, params, generated_prob_list):
        """
        generate array of SAXS patterns given the parameters (min to max)
        ind     param
        0       lattice parameter
        1       head position
        2       thickness of tail
        3       thicnkess of headgroup
        """
        n_dim = 10
        store_it = []
        lat_paramets = []
   
        
        for lat_param in tqdm(generated_prob_list):
            # convert to reciprocal space lattice parameter
            lat_param_recip = 2*np.pi/lat_param
            q = self.lamellarLattice(lat_param_recip, n_dim)
            for bilayer in np.linspace(params[1,0],params[1,1],5):
                for sigma_tail in np.linspace(params[2,0],params[2,1],5):
                    for sigma_bilayer in np.linspace(params[3,0],params[3,1],4):
                        # ensure that the lattice parameter is larger than the lipid bilayer
                        if lat_param > 2*bilayer: #limits number of loops by this statement
                            recip_cell = self.lamellarCell(q, sigma_bilayer, sigma_tail,  bilayer)
                            store_it.append(np.array([q, (recip_cell*np.conj(recip_cell)).real]))
        return np.array(store_it)

#-------------HEXAGONAL MODEL FUNCTIONS AND CLASS-------------#
class HexagonalModel():

    def electronDensityRadial(self, x, sigma_head, sigma_tail, head, tail, rel_tail_intensity = 0.2):
        """
        generate electron density distribution
            x - x coordinates in real space
            sigma_head - head group thickness
            sigma_tail - tail thickness
            head - position of head
            tail - length of tail
        return 
            rho(x) - function: real space electron density distribution
        """
        def rho(x):
            return gaussianSignal(x, sigma_head, head)-gaussianSignal(x, sigma_tail, tail)*rel_tail_intensity
        return rho

    def hexagonalLattice(self, lat_param, max_q):
        """
        generate reciprocal hexagonal lattice
            lat_param - lattice parameter
            max_q - maximum q to generate
        return
            P - 2D array of hexagonal lattice points
            k_list - miller indices
        """
        P = []
        for k1 in range(-50,50):
            for k2 in range(-50,50):
                P.append([lat_param*(k1+k2/2), lat_param*k2*np.sqrt(3)/2])
        k_list = np.array(list(set([np.round(np.sqrt(p[0]**2+p[1]**2),5) for p in P if np.sqrt(p[0]**2+p[1]**2)<=max_q])))
        return np.array(P), k_list


    def hankelTransform(self, k_list, r_list, electron_density):
        """
        Hankel Transformation : Natalie Baddour in Adv. Imaging and Electron Physics, Dec 2011
        2 pi integral(0,inf)[f(r) J0(kr) r dr]
            k_list - miller indices
            r_list - point in real radial space
            electron_denisty - function of real electron density distribution
        return
            2D array of miller indices with power spectra of intensity
        """
        Fvk = []
        for k in k_list:
            temp = []
            for r in r_list:
                temp.append(electron_density(r)*jv(0,k*r)*r)
                Fv = np.sum(temp)
            Fvk.append([k,Fv])
        return np.array(Fvk)

    def generateSynthHexagonal(self, params):
        """
        generate array of SAXS patterns given the parameters (min to max)
        ind     param
        0       lattice parameter
        1       head position
        2       tail position
        
        """
        sigma_tail = 1
        sigma_head = 1
        r_list = np.linspace(0,50,100)
        max_q = 0.5
        store_it = []
        #pbar = tqdm(total=10**8+1)
        count = 0
        for lat_param in tqdm(np.linspace(params[0,0], params[0,1], 2000)):
            tail = lat_param/2
            lat_param_recip = 2*np.pi/lat_param
            _, k_list = self.hexagonalLattice(lat_param_recip, max_q)
            for head in np.linspace(params[1,0], params[1,1], 5):
                tail = tail + head
                if lat_param > 2*head:
                    for sigma_head in np.linspace(params[2,0], params[2,1], 5):
                         for sigma_tail in np.linspace(params[3,0], params[3,1], 4):
                            count +=1
                            #pbar.update()
                            rho = self.electronDensityRadial(r_list, sigma_head, sigma_tail, head, tail)
                            struc_fact = self.hankelTransform(k_list, r_list, rho)
                            dummy_struc_fact = np.zeros((2,100))
                            if len(struc_fact) <= 100:
                                dummy_struc_fact[:,:len(struc_fact)] = struc_fact.T
                            else:
                                dummy_struc_fact[:,:100] = struc_fact[:100,:].T
                            dummy_struc_fact[1,:] = dummy_struc_fact[1,:] * np.conj(dummy_struc_fact[1,:])
                            store_it.append(dummy_struc_fact)
        #pbar.close()
        print(f'generated {count} of {10**8} possible')

        return np.array(store_it)


#-------------CUBIC MODEL FUNCTIONS AND CLASS-------------#
# Scattering Vectors
def primitiveScatteringVectors():
    """
    Return the lookup table of scattering vectors for Primitive surface
    """
    return np.array([[0,0,0],[1,1,0], [2,0,0], [2,1,1],[2,2,0],[2,2,2],[3,1,0],[3,2,1],[3,3,0],[3,3,2],[4,0,0],[4,1,1],[4,2,0],[4,2,2],[4,3,1],[4,3,3],[4,4,0],[4,4,2],[4,4,4]])
def gyroidScatteringVectors():
    """
    Return the lookup table of scattering vectors for gyroid surface
    """
    return np.array([[0,0,0],[2,1,1],[2,2,0],[3,2,1],[3,3,2],[4,0,0],[4,2,0],[3,2,2],[4,2,2],[4,3,1],[5,2,1],[4,4,0],[5,3,2],[6,1,1],[4,4,2],[4,4,4]])
def diamondScatteringVectors():
    """
    Return the lookup table of scattering vectors for diamond surface
    """
    return np.array([[0,0,0],[1,1,0],[1,1,1],[2,0,0],[2,1,1],[2,2,0],[2,2,1],[2,2,2],[3,2,1]])

# Surface Generators
def gyroid(X,Y,Z):
    return np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)
def schwarz_p(X,Y,Z):
    return np.cos(X) + np.cos(Y) + np.cos(Z)
def schwarz_d(X,Y,Z):
    return np.sin(X)*np.sin(Y)*np.sin(Z) + np.sin(X)*np.cos(Y)*np.cos(Z) + np.cos(X)*np.sin(Y)*np.cos(Z) + np.cos(X)*np.cos(Y)*np.sin(Z)

class CubicModel():
    """
    Cubic model class
    ------------
    This takes 'P', 'D', 'G' to designate which phase to work with.
    """
    def __init__(self, surf):
        self.surf = surf

    def getScatVec(self):
        if self.surf == 'P':
            k_list = primitiveScatteringVectors()
            print('primitive scattering vectors')
        elif self.surf == 'G':
            k_list = gyroidScatteringVectors()
            print('Gyroid scattering vectors')
        elif self.surf == 'D':
            k_list = diamondScatteringVectors()
            print('Diamond scattering vectors')
        return k_list


    def getSurface(self, steps, lat_param):
        """
        input:
            steps - increment in mgrid mesh
            lat_param - lattice parameter
        """
        if self.surf == 'P':
            x, y, z = np.pi*np.mgrid[-1:1:steps*1j, -1:1:steps*1j, -1:1:steps*1j] * 1
            vol = schwarz_p(x,y,z)
        elif self.surf == 'D':
            # must be divided by 2 in order to obey the reflection condition
            x, y, z = np.pi*np.mgrid[-1:1:steps*1j, -1:1:steps*1j, -1:1:steps*1j] * 0.5
            vol = schwarz_d(x,y,z)
        elif self.surf == 'G':
            # must be divided by 2 in order to obey the reflection condition
            x, y, z = np.pi*np.mgrid[-1:1:steps*1j, -1:1:steps*1j, -1:1:steps*1j] * 1
            vol = gyroid(x,y,z)
        # convert 3D volume to triangulated surface
        verts, faces, normals, values = measure.marching_cubes_lewiner(vol, 0, spacing=(0.1, 0.1, 0.1))
        # scale vertices by lattice parameter
        verts = verts/np.max(verts)*lat_param

        return verts, faces, normals, values

    def garsteckiIntensity(self, verts, faces, k_list, lat_param):
        """
        simple model (J. Chem. Phys. Vol 113, No. 9, Garstecki 2000)
            verts - surface vertices
            faces - surface faces
            k_list - list of miller indices
            lat_param - lattice parameter
        returns:
            q - scattering position
            I - intensity
        """
        # initialize intenisity and q
        I = []
        q = []
        # loop scattering vectors
        for k in k_list/lat_param:
            qk = np.sqrt(k[0]**2+k[1]**2+k[2]**2)
            Ikc = []
            Iks =[]
            # loop faces of cubic surface
            for f in faces:
                A,B,C = verts[f[0]]*lat_param, verts[f[1]]*lat_param, verts[f[2]]*lat_param
                vecAB = B-A
                vecAC = C-A
                area = 0.5*np.linalg.norm(np.cross(vecAB, vecAC))
                mpv = np.array([(A[0]+B[0]+C[0])/3,(A[1]+B[1]+C[1])/3,(A[2]+B[2]+C[2])/3])
                Ikc.append(area*np.cos(2*np.pi*np.dot(k,mpv)))
                Iks.append(area*np.sin(2*np.pi*np.dot(k,mpv)))
            I.append(np.sum(Ikc)**2+np.sum(Iks)**2)
            q.append(qk)
            #print(k,np.sum(Ikc)**2)
        return np.array(q), np.array(I)

    def getIntensity(self, verts, faces, k_list, lat_param, L, sigma):
        """
        Modified model:
            verts - surface vertices
            faces - surface faces
            k_list - list of miller indices
            lat_param - lattice parameter
            L - bilayer length
            sigma - bilayer
        returns:
            q - scattering position
            I - intensity
        """
        # initialize intenisity and q
        I = []
        q = []
        # loop scattering vectors
        for k in k_list/lat_param:
            qk = np.sqrt(k[0]**2+k[1]**2+k[2]**2)
            Ikc = []
            Iks =[]
            # loop faces of cubic surface
            for f in faces:
                A,B,C = verts[f[0]]*lat_param, verts[f[1]]*lat_param, verts[f[2]]*lat_param
                vecAB = B-A
                vecAC = C-A
                # face area
                area = 0.5*np.linalg.norm(np.cross(vecAB, vecAC))
                # face unit normal
                n = np.cross(vecAB, vecAC)/np.sqrt(np.linalg.norm(vecAB)**2*np.linalg.norm(vecAC)**2-np.linalg.norm(np.dot(vecAB,vecAC))**2)
                mpv = np.array([(A[0]+B[0]+C[0])/3,(A[1]+B[1]+C[1])/3,(A[2]+B[2]+C[2])/3])
                Ikc.append(area*np.cos(2*np.pi*np.dot(k,mpv))*np.cos(2*np.pi*np.dot(k,n*L/2))*np.exp(-2*np.pi**2*sigma**2*(2*np.pi*np.dot(k,n))**2))
                Iks.append(area*np.sin(2*np.pi*np.dot(k,mpv))*np.cos(2*np.pi*np.dot(k,n*L/2))*np.exp(-2*np.pi**2*sigma**2*(2*np.pi*np.dot(k,n))**2))
            I.append(np.sum(Ikc)**2+np.sum(Iks)**2)
            q.append(qk)
        return np.array(q),np.array(I)

    def generateSynthCubic(self, params):
        store_it = []
        k_list = self.getScatVec()
        pbar = tqdm(total=11**4+1)
        count = 0
        for lat_param in np.linspace(params[0,0], params[0,1],22):
            verts, faces, _, _ = self.getSurface(12,lat_param)
            for L in np.linspace(params[1,0], params[1,1], 22):
                for sigma in np.linspace(params[2,0], params[2,1], 22):
                    count +=1
                    pbar.update()
                    q,I = self.getIntensity(verts, faces, k_list, lat_param, L, sigma)
                    store_it.append(np.array([q, I]))

        pbar.close()
        print(f'generated {count} of {11**4} possible')

        return np.array(store_it)