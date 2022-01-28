from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import random
from saxspy import debyeWaller as dwf
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import saxspy
import umap


def load_data(phase):
    data_3d = np.load(f'Synthetic_Processed/{phase.lower()}.npy')[:,:,:,0]
    data_1d = []
    for i in data_3d:
        # get matrix diagonal
        data = np.sqrt(np.diag(i))
        data_1d.append(data)
    data_1d = np.array(data_1d)
    q = np.load(f'Synthetic_Processed/{phase.lower()}_q.npy')
    # load experimental data
    exp_data = np.load(f'Experimental_data/{phase.lower()}.npy')
    return data_1d, data_3d, exp_data, q

def plot_saxs(pattern,q):
    plt.figure()
    plt.plot(q,pattern)
    plt.xlabel('q')
    plt.ylabel('Intensity')
    plt.show()

def plot_saxs_tsne(data_synth,data_exp):
    data = np.concatenate((data_synth,data_exp),axis=0)
    data_embedded = TSNE(n_components=2).fit_transform(data)
    plt.figure()
    plt.scatter(data_embedded[:len(data_synth),0],data_embedded[:len(data_synth),1], c='r', label='Synthetic')
    plt.scatter(data_embedded[len(data_synth):,0],data_embedded[len(data_synth):,1], c='b', label='Experimental')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('tSNE plot of SAXS data')
    plt.legend()
    plt.show()

def plot_saxs_pca(data_synth,data_exp):
    data = np.concatenate((data_synth,data_exp),axis=0)
    data_embedded = PCA(n_components=2).fit_transform(data)
    plt.figure()
    plt.scatter(data_embedded[:len(data_synth),0],data_embedded[:len(data_synth),1], c='r', label='Synthetic')
    plt.scatter(data_embedded[len(data_synth):,0],data_embedded[len(data_synth):,1], c='b', label='Experimental')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA plot of SAXS data')
    plt.legend()
    plt.show()

def plot_saxs_umap(data_synth,data_exp):
    data = np.concatenate((data_synth,data_exp),axis=0)
    data_embedded = umap.UMAP().fit_transform(data)
    plt.figure()
    plt.scatter(data_embedded[:len(data_synth),0],data_embedded[:len(data_synth),1], c='r', label='Synthetic')
    plt.scatter(data_embedded[len(data_synth):,0],data_embedded[len(data_synth):,1], c='b', label='Experimental')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP plot of SAXS data')
    plt.legend()
    plt.show()

def plot_saxs_featuremap(data,q):
    plt.figure()
    plt.imshow(data,cmap='hot')
    plt.xlabel('q')
    plt.ylabel('q')
    # change x and y labels to q
    plt.xticks(np.arange(0,data.shape[0],50), ["{:.2f}".format(i) for i in q[::50]])
    plt.yticks(np.arange(0,data.shape[0],50), ["{:.2f}".format(i) for i in q[::50]])
    plt.title('Feature map of SAXS data')
    plt.show()

if __name__ == '__main__':
    Phase = 'lamellar'
    d1, d3, exp_data, q = load_data('lamellar')
    plot_saxs_umap(d1,exp_data)
    plot_saxs_tsne(d1,exp_data)
    plot_saxs_pca(d1,exp_data)
    plot_saxs(d1[0],q)
    plot_saxs_featuremap(d3[0],q)
    
    
    

