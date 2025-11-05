#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst

def ex_m_solution():
    alpha = 0.95
    # Chargement du nuage
    data = np.loadtxt('nuage'); xi = data[:,0]; yi = data[:,1]; zi = data[:,2];
    # Valeur a priori du FUV
    sigma02=1;
    # Matrice modèle
    A = np.vstack((np.sqrt(xi**2+yi**2), xi, yi)).T
    # Vecteur des observations et matrice de covariance apriori associée
    Y = zi; Sigma = np.eye(Y.shape[0]); SigmaY = sigma02*Sigma;
    # Moindres carrés
    n, p, Xhat, _, sigma02hat, SigmaXhat, _, _ = calc_ls(A,Y,SigmaY)
    ddl = n-p
    print(sigma02hat)
    fuv_chi2_test(sigma02hat, ddl, alpha)
        # Normalisation
    n, p, Xhat, _, sigma02hat, SigmaXhat, _, _ = calc_ls(A,Y,sigma02hat*SigmaY)
    print(sigma02hat)
    fuv_chi2_test(sigma02hat, ddl, alpha)
    conf_X(Xhat, SigmaXhat, ddl, alpha)
    # Affichage de la solution
    xg,yg=  np.meshgrid(np.arange(min(xi),max(xi),1),np.arange(min(yi),max(yi),1))
    zg = Xhat[0]*np.sqrt(xg**2*yg**2)+Xhat[1]*xg+Xhat[2]*yg
    # Librairie d'affichage 3D
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xi, yi, zi,'o', c='k')
    ax.plot_surface(xg, yg, zg, rstride=1, cstride=1, cmap=cm.terrain,linewidth=0)
    
def calc_ls(A, Y, SigmaY):
    # nombre d'observations
    n = A.shape[0]
    # nombre de paramètres inconnus
    p = A.shape[1]
    # Matrice de poids
    iSigmaY = np.linalg.inv(SigmaY)
    # Matrice normale
    N = (np.dot(np.dot(A.T,iSigmaY),A))
    iN = np.linalg.inv(N)
    # Calcul de la solution des moindres carrés
    Xhat=np.dot(np.dot(np.dot(iN,A.T),iSigmaY),Y)
    # Vecteur des résidus
    Vhat = np.dot(A,Xhat)-Y
    # FUV a posteriori
    sigma02hat = (np.dot(np.dot(Vhat.T,iSigmaY),Vhat)/(n-p)).item(0)
    # Covariance a posteriori de la solution
    SigmaXhat = sigma02hat*iN
    # Covariance a posteriori des observations
    SigmaYhat = sigma02hat*SigmaY;
    # Covariance a posteriori des résidus
    SigmaVhat = SigmaYhat-np.dot(np.dot(A,SigmaXhat),A.T)
    
    return n,p,Xhat,Vhat,sigma02hat,SigmaXhat,SigmaYhat,SigmaVhat

def conf_X(Xhat, SigmaXhat, ddl, alpha):
    z0=sst.t.ppf((1+alpha)/2,ddl)
    print(z0)
    for i in np.arange(Xhat.shape[0]): print(Xhat[i],"+/-",z0*np.sqrt(SigmaXhat[i,i]))
    
def fuv_chi2_test(sigma02hat, ddl, alpha):
    # Calcul des centiles correspondant aux bornes de l'intervalle @95%
    theta1=sst.chi2.ppf((1-alpha)/2,ddl)
    theta2=sst.chi2.ppf((1+alpha)/2,ddl)
    # Valeurs min et max validant le test du chi2
    chi2_min = sigma02hat*ddl/theta2
    chi2_max = sigma02hat*ddl/theta1
    res = 0
    if (chi2_min <= 1 and chi2_max >= 1): res = 1
    print(chi2_min, chi2_max)
    return res
    
def main():
    ex_m_solution();
    plt.show()
    
if __name__ == "__main__":
    main();

