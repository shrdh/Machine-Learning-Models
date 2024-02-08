import numpy as np
from scipy.linalg import eigh


def LDA(data, labels, alpha=0, nvars=10):
    """
    Linear Discriminant Analysis projection of a labeled data set
    
    Parameters:
    data: n-by-N matrix representing a set of N points in R^n
    labels: 1-by-N vector of categorical labels containing C unique values
    alpha: regularization parameter for covariance matrix (default: 0)
    nvars: number of dimensions to project onto (default: 10)
    
    Returns:
    U: projection matrix
    L: eigenvalues of the covariance matrix
    """
    n, N = data.shape
    
    if alpha is None:
        alpha = 0
    if nvars is None:
        nvars = 10
    nvars = min(nvars, n-1)
    
    C = len(np.unique(labels))
    
    categoryMeans = np.zeros((n, C))
    for i in range(C):
     
        iinds = np.where(labels == i)[1] #PCA
        #iinds = np.where(labels == i)[0] # LDA
       
        categoryMeans[:, i] = np.mean(data[:, iinds], axis=1)
       
        data[:, iinds] = data[:, iinds] - np.tile(categoryMeans[:, i], (len(iinds), 1)).T
    
    sigmab = np.cov(categoryMeans)  
    sigma = np.cov(data)  
    
    sigma = (1-alpha)*sigma+alpha*np.mean(np.diag(sigma))*np.eye(sigma.shape[0])
   
    
    
    L, U =eigh(sigmab, sigma, subset_by_index=[n-3,n-1])  
    sinds = np.argsort(np.abs(L))[::-1]  
    L = L[sinds]
    U = U[:, sinds]
    
    return U[:, :nvars], L[:nvars]
def PCA(Y, p):
    """
    Principal Component Analysis (PCA)
    
    Parameters:
    Y: n-by-N matrix with columns y_i in R^n for i=1,...,N
    p: percentage of variance to maintain (or dimension if p>1)
    
    Returns:
    X: m-by-N matrix with columns x_i in R^m for i=1,...,N
    s: singular values of the covariance matrix
    U: principal components
    mu: mean of the data
    """
    mu = np.mean(Y, axis=1)
    Y = Y - np.tile(mu, (Y.shape[1], 1)).T  
    C = Y @ Y.T
    U, s, _ = np.linalg.svd(C)  
    X = U[:, :p].T @ Y  
    
    return X, s, U, mu
def sLDA(data, target, slices=30, alpha=0, nvar=10):
    """
    Supervised linear Discriminant Analysis projection of a labeled data set
    
    Parameters:
    data: n-by-N matrix representing a set of N points in R^n
    target: 1-by-N vector containing the target function values
    slices: number of slices to cut the target into
    alpha: regularization for the covariance matrix, required when n>N
    nvar: number of variables to project down to
    
    Returns:
    U: projection matrix
    L: eigenvalues of the between-class scatter matrix
    projecteddata: projected data
    """
    if slices is None:
        slices = 30
    if alpha is None:
        alpha = 0
    if nvar is None:
        nvar = 10
    
    N = len(target)
    
    labels = np.zeros([1, N])  
    l = int(np.ceil(N / slices))
    for i in range(slices - 1):
        labels[0, i * l:(i + 1) * l] = i
    labels[0,(slices - 1) * l:] = slices-1
    
    sinds = np.argsort(target, axis=0)  
    sinv = np.zeros_like(sinds)
    sinv[sinds[:,0]] = np.arange(len(sinds)).reshape(-1,1)  #For PCA
    #sinv[sinds[:]] = np.arange(len(sinds)) #For LDA
    labels = labels[0,sinv[:,0:1].astype(int)].T  # PCA
    #labels = labels[0,sinv[:].astype(int)].T  # LDA
    U, L = LDA(data, labels, alpha, nvar)
    
    for i in range(U.shape[1]):  
        U[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    
    projecteddata = U.T @ (data - np.tile(np.mean(data, axis=1), (N, 1)).T)
    
    return U, L, projecteddata



    
