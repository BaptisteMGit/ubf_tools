#!/usr/bin/env python3

########################## BE: MENETRIER Baptiste 05/10/2020 ##########################

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as sst


######################### Least-squares solution #######################
def calc_ls(A, Y, SigmaY):
    # nombre d'observations
    n = A.shape[0]
    # nombre de paramètres inconnus
    p = A.shape[1]
    # Matrice de poids
    iSigmaY = np.linalg.inv(SigmaY)
    # Matrice normale
    N = np.dot(np.dot(A.T, iSigmaY), A)
    iN = np.linalg.inv(N)
    # Calcul de la solution des moindres carrés
    Xhat = np.dot(np.dot(np.dot(iN, A.T), iSigmaY), Y)
    #  Vecteur des résidus
    Vhat = np.dot(A, Xhat) - Y
    # FUV a posteriori
    sigma02hat = (np.dot(np.dot(Vhat.T, iSigmaY), Vhat) / (n - p)).item(0)
    #  Covariance a posteriori de la solution
    SigmaXhat = sigma02hat * iN
    #  Covariance a posteriori des observations
    SigmaYhat = sigma02hat * SigmaY
    #  Covariance a posteriori des résidus
    SigmaVhat = SigmaYhat - np.dot(np.dot(A, SigmaXhat), A.T)

    return n, p, Xhat, Vhat, sigma02hat, SigmaXhat, SigmaYhat, SigmaVhat


########################### Chi2 test for FUV ##########################
def fuv_chi2_test(sigma02hat, ddl, alpha):
    res = 0

    #  Calcul des centiles correspondant aux bornes de l'intervalle @95%
    theta1 = sst.chi2.ppf((1 - alpha) / 2, ddl)
    theta2 = sst.chi2.ppf((1 + alpha) / 2, ddl)

    #  Valeurs min et max validant le test du chi2
    chi2_min = sigma02hat * ddl / theta2
    chi2_max = sigma02hat * ddl / theta1
    print("FUV chi2 test minmax", chi2_min, chi2_max)
    if chi2_min <= 1 <= chi2_max:
        res = 1
    return res


#################### Confident interval for solution ###################
def conf_X(Xhat, SigmaXhat, ddl, alpha):
    lambdai = sst.t.ppf((1 + alpha) / 2, ddl)
    """
    Si H0 est vérifiée, sous entendu le test du Chi2 est validé (éventuellement après normalisation de la 
    matrice de covariance des observations), les lambdai suivent une loi de Student à ddl degrés de liberté 
    et on utilise sst.t.ppf pour calculer la valeur de lambdai.
    
    Si H0 n'est pas vérifiée mais que la matrice de covariance des observations est supposée correcte alors 
    les lambdai suivent une loi normale centrée réduite et on utilise sst.norm.ppf pour calculer la valeur 
    de lambdai.
    
    """
    lambdai = sst.t.ppf((1 + alpha) / 2, ddl)
    # lambdai = sst.norm.ppf((1 + alpha) / 2, ddl)

    print("lamdai = %s" % lambdai)
    print(
        "Intervalle de confiance des estimateurs au niveau alpha = {} :".format(alpha)
    )
    for i in range(Xhat.shape[0]):
        print(".     ", Xhat[i, 0], "+/-", lambdai * np.sqrt(SigmaXhat[i, i]))
    return


###################### Ellipse error for solution ######################
def draw_ellipse(X, SigmaX, ddl, alpha, title, color="k", fac=1):
    from scipy.stats import chi2
    from matplotlib.patches import Ellipse

    D, R = np.linalg.eig(SigmaX)
    lambda_m = np.sqrt(sst.f.ppf(alpha, 2, ddl) * 2)
    a, b = lambda_m * np.sqrt(D[0]), lambda_m * np.sqrt(D[1])
    try:
        theta = np.arctan(R[1, 0] / R[0, 0]) * 180 / np.pi
    except:
        theta = np.sign(R[1, 0]) * 90
    # re-order semi axis of ellipse
    if a <= b:
        theta = theta + 90
        a, b = lambda_m * np.sqrt(D[1]), lambda_m * np.sqrt(D[0])
    theta = np.mod(theta, 360)

    ellipse = Ellipse(
        xy=(X[0], X[1]),
        width=a * 2 * fac,
        height=b * 2 * fac,
        angle=theta,
        edgecolor=color,
        lw=2,
        facecolor="none",
    )
    plt.plot(
        X[0],
        X[1],
        "o" + color,
        color=color,
        ms=6,
        lw=1,
        label="Instrument : {}".format(title),
    )
    # plt.title(title)
    plt.legend()
    plt.gca().add_patch(ellipse)
    return plt.gca().add_patch(ellipse)


################# Chi2 GOF test for normalized residuals ###############
def chi2_test(What, ddl, alpha):
    res = 0
    n = What.shape[0]
    #  nombre de classe
    m = 50
    p = 0
    d_chi2 = 0
    c = np.linspace(-0.004, 0.004, m)
    plt.figure()
    width = 0.3
    for k in np.arange(0, c.shape[0] - 1):
        # Number of values in class
        nk = What[np.logical_and(What >= c[k], What < c[k + 1])].shape[0]
        #  Probability to belong to this class
        pk = sst.t.cdf(c[k + 1], ddl) - sst.t.cdf(c[k], ddl)
        d_chi2 = d_chi2 + (nk - n * pk) ** 2 / (n * pk)
        p1 = plt.bar(1 + k, nk, width=width, color="r", alpha=0.8)
        p2 = plt.bar(1 + k + width, n * pk, width=width, color="b", alpha=0.8)
    plt.xlim([1 - 0.4, k + 0.8])
    plt.xlabel("Classe")
    plt.ylabel("Effectifs")
    plt.grid()
    plt.legend([p1[0], p2[0]], [r"$n_i$", r"$np_i$"], fontsize=10, fancybox=True, loc=1)
    theta = sst.chi2.ppf(0.95, m - 1)
    if d_chi2 <= theta:
        res = 1
    return res


################## KS GOF test for normalized residuals ################
def ks_test(What, ddl, alpha):
    res = 0

    n = What.shape[0]
    Whats = np.sort(What)

    Fhat = 1 / n * np.cumsum(np.ones(What.shape))

    F = sst.t.cdf(Whats, ddl)

    plt.figure()
    plt.plot(Whats, F, "r", lw=2)
    plt.plot(Whats, Fhat, "--b", lw=2)

    dFmax = np.max(F - Fhat)
    if np.abs(dFmax) <= np.sqrt(1 / n * (-0.5 * np.log((1 - alpha) / 2))):
        res = 1
    return res


############## Confident interval for normalized residuals #############
def conf_W(What, ddl, alpha):
    n = What.shape[0]

    plt.figure()
    plt.plot(What, "x-b")

    tmin = sst.t.ppf((1 - alpha) / 2, ddl)
    tmax = sst.t.ppf((1 + alpha) / 2, ddl)

    plt.plot([0, n], [tmin, tmin], "r", lw=2)
    plt.plot([0, n], [tmax, tmax], "r", lw=2)
    return


def main():
    data = np.loadtxt("mesures.txt")
    theta = data[:, 0]
    x = data[:, 1:2]
    y = data[:, 2:3]

    #  Valeur a priori du FUV
    sigma02 = 1
    # Matrice modèle
    A = []
    for i in range(theta.shape[0]):
        A.append([np.cos(theta[i]), np.cos(3 * theta[i]), np.cos(6 * theta[i])])
    for i in range(theta.shape[0]):
        A.append([np.sin(theta[i]), np.sin(3 * theta[i]), np.sin(6 * theta[i])])
    A = np.array(A)
    # print(A)
    # print(A.shape)

    # Vecteur des observations et matrice de covariance apriori associée
    Y = np.vstack((x, y))
    # print("Y : ", Y)
    Sigma = np.eye(np.shape(A)[0])
    SigmaY = sigma02 * Sigma

    # Moindres carrés
    n, p, Xhat, Vhat, sigma02hat, SigmaXhat, SigmaYhat, SigmaVhat = calc_ls(
        A, Y, SigmaY
    )
    print("Xhat:", Xhat, "\n")
    print("SigmaXhat:", SigmaXhat, "\n")
    print("sigma02hat:", sigma02hat, "\n")

    #  Covariance a posteriori des observations
    print("SigmaYhat:", SigmaYhat, "\n")
    """
    On avait supposé une variance de 1 
    Une estimation de la variance des mesures de x et y est donnée par le FUV 
    Cette variance est donc de l'ordre de sigma02hat: 0.02357
    
    """

    # Test du facteur unitaire de variance obtenue
    ddl = n - p
    alpha = 0.95
    res = fuv_chi2_test(sigma02hat, ddl, alpha)
    print("Résulat du test du chi 2 : ", res, "\n")

    """ 
    On a sigma02hat = 0.023565266305137677, on a donc sur-estimé la matrice 
    de pondération. Le test du chi2 n'est pas validé, on normalise donc cette matrice à l'aide du facteur sigma02hat.
    """

    # Vecteur des observations et matrice de covariance a posteriori
    Sigma = np.eye(np.shape(A)[0])
    SigmaY = sigma02hat * Sigma

    # Moindres carrés
    n, p, Xhat, Vhat, sigma02hat, SigmaXhat, SigmaYhat, SigmaVhat = calc_ls(
        A, Y, SigmaY
    )
    print("Résultat après normalisation ")
    print("Xhat:", Xhat, "\n")
    print("SigmaXhat:", SigmaXhat, "\n")
    print("sigma02hat:", sigma02hat, "\n")

    # On peut vérifier que cette fois ci le FUV vérifie bien le test du Chi2
    ddl = n - p
    alpha = 0.95
    res = fuv_chi2_test(sigma02hat, ddl, alpha)
    print("Résulat du test du chi 2 après normalisation: ", res)

    # Intervalle de confiance au niveau alpha de l'estimateur de X
    # H0 vérifiée après normalisation de la matrice de covariance des observations
    conf_X(Xhat, SigmaXhat, ddl, alpha)

    #  Affichage de la solution
    a, b, c = Xhat[0, 0], Xhat[1, 0], Xhat[2, 0]
    print("Paramètres a, b, c du modèle :", (a, b, c))

    x = []
    y = []
    for i in range(theta.shape[0]):
        x.append(
            a * np.cos(theta[i]) + b * np.cos(3 * theta[i]) + c * np.cos(6 * theta[i])
        )
    x = np.array(x)

    for i in range(theta.shape[0]):
        y.append(
            a * np.sin(theta[i]) + b * np.sin(3 * theta[i]) + c * np.sin(6 * theta[i])
        )
    y = np.array(y)
    plt.scatter(x, y)

    x = []
    y = []
    theta = np.arange(0, np.pi * 2, 1e-3)
    for i in range(theta.shape[0]):
        x.append(
            a * np.cos(theta[i]) + b * np.cos(3 * theta[i]) + c * np.cos(6 * theta[i])
        )
    x = np.array(x)
    for i in range(theta.shape[0]):
        y.append(
            a * np.sin(theta[i]) + b * np.sin(3 * theta[i]) + c * np.sin(6 * theta[i])
        )
    y = np.array(y)
    plt.plot(x, y, color="r")

    plt.title("Courbe paramétrée solution")
    plt.xlabel("x($theta$)")
    plt.ylabel("y($theta$)")
    plt.show()


if __name__ == "__main__":
    main()
