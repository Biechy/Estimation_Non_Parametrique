#!/usr/bin/env python
# coding: utf-8

# # Code

# ## Imports

# In[ ]:


#Bibliothèques
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
import scipy as sc
from sklearn.metrics import mean_squared_error


# ## Latex sur Jupyter

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')

#LaTex
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# ## Noyau

# In[ ]:


#Noyau gaussien
def K5(u) :
    return (1/np.sqrt(2*np.pi))*np.exp(-(u**2)/2)


# ## Création des lois pour les échantillons

# In[ ]:


def Gauss(n,mu,sigma):
    return np.random.normal(mu,sigma,n)

def Kurtotic(n):
    weights = [2/3,1/3]
    X = []
    multinom = np.random.multinomial(1, weights, n) 
    X.append(np.random.normal(0,1,n))
    X.append(np.random.normal(0,0.1,n))
    Y = multinom * np.transpose(X)
    return np.extract(Y!=0,Y)

def AsymmetricBimodal(n):
    weights = [3/4,1/4]
    X = []
    multinom = np.random.multinomial(1, weights, n) 
    X.append(np.random.normal(0,1,n))
    X.append(np.random.normal(3/2,1/3,n))
    Y = multinom * np.transpose(X)
    return np.extract(Y!=0,Y)

def Claw(n):
    weights = np.append(0.1 * np.ones(5), 0.5)
    multinom = np.random.multinomial(1, weights, n) 
    X1 = []
    for i in range (len(weights)-1):
        X1.append(np.random.normal(0.5*i - 1,0.1,n))  
    X1.append(np.random.normal(0,1,n))
    X2 = multinom * np.transpose(X1)
    return np.extract(X2!=0,X2)

def AsymmetricClaw(n):    
    X1 = []
    weights = []
    for i in range (-2,3):
        X1.append(np.random.normal(0.5+i,((2**-i)/10),n))
        weights.append((2**(1-i))/31)
    X1.append(np.random.normal(0,1,n))
    weights.append(0.5)
    multinom = np.random.multinomial(1, weights, n)
    X2 = multinom * np.transpose(X1)
    
    return np.extract(X2!=0,X2)


# ## Densités réelles correspondantes

# In[ ]:


def density_Gaussian(x,mu,sigma):
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

def density_Kurtotic(x):
    return (2/3)*(1/np.sqrt(2*np.pi)*np.exp(-x**2/2)) + (1/3)*(1/np.sqrt(2*np.pi*0.1**2)*np.exp(-x**2/(2*0.1**2)))

def density_ABi(x):
    return (3/4)*(1/np.sqrt(2*np.pi)*np.exp(-x**2/2)) + (1/4)*(1/np.sqrt(2*np.pi*(1/3)**2)*np.exp(-(x-3/2)**2/(2*(1/3)**2)))

def density_Claw(x):
    f = 0.5*(1/np.sqrt(2*np.pi)*np.exp(-x**2/2))
    for i in range(0,5):
        f+= 0.1*(1/np.sqrt(2*np.pi*(0.1)**2))*np.exp(-(x-(0.5*i - 1))**2/(2*(0.1)**2))
    return f

def density_AC(x):
    f = 0.5*(1/np.sqrt(2*np.pi)*np.exp(-x**2/2))
    for i in range(-2,3):
        f+= (2**(1-i)/31)*(1/np.sqrt(2*np.pi*(2**(-i)/10)**2))*np.exp(-(x-(0.5+i))**2/(2*(2**(-i)/10)**2))
    return f


# ## Implémentation de l'estimateur à noyau

# In[ ]:


#création de l'estimateur à noyau d'une densité en un point
def Density_EstimatorPoint(X, x, h, K,n):
    return (1/(h*n))*np.sum(K((X-x)/h))

#création de l'estimateur à noyau d'une densité en tout point
def Density_Estimator(X, x, h, K,n):
    density_estimator = np.zeros(len(x))
    for i in range(len(x)):
        density_estimator[i] = Density_EstimatorPoint(X, x[i], h, K,n)
    return density_estimator


# ## Risque quadratique

# In[ ]:


#On définit la MISE
def MISE (pn,p):
    return (1/pn.size)*np.sum((pn-p)**2)


# ## Minimisation du risque quadratique

# ### Silverman's Rule

# In[ ]:


def Silverman (X,n):
    sigma = np.std(X)
    return 1.06 * sigma * n**(-1/5) 


# ### Unbiased Cross Validation

# In[ ]:


def UCV(X,x,h,K,n):
    R = []
    
    Xi, Xj = np.meshgrid(X, X)
    XX = Xi - Xj
    
    for i in h:
        a1 = K_h(K, np.sqrt(2)*i, XX)
        a = np.sum(np.sum(a1,axis=0)) / (n**2)
        
        b1 = K(XX/i)
        b1[np.diag_indices(len(X))]=0
        b2 = np.sum(np.sum(b1, axis=0))/i
        b = 2/(n*(n-1))*b2

        R = np.append(R, a-b )
        
    return R


# ### Penalized Comparison to Overfitting

# In[ ]:


def K_h(K,h,u):
    return (1/h)*K(u/h)


# In[ ]:


def PCO(X,x,h,K,n,lam=1):
    hmin = min(h)
    K_hmin = K_h(K,hmin,x)
    Crit = []
    
    
    #création du tableau pour la double somme
    Xi, Xj = np.meshgrid(X, X)
    XX = Xi - Xj
    for i in h :
        a1 = K_h(K, np.sqrt(2)*i, XX)
        a2 = np.sum(np.sum(a1,axis=0)) / (n**2)
        a3 = K_h(K, np.sqrt(i**2+hmin**2), XX)
        a4 = 2 * np.sum( np.sum(a3,axis=0)) / (n**2)
        a = a2 - a4
        b = np.trapz((K_hmin-K_h(K,i,x))**2,x)/n
        c = lam*np.trapz(K_h(K,i,x)**2,x)/n
        newCrit = a - b + c
        Crit = np.append(Crit, newCrit)
    
    return Crit


# ### Algorithme de Recuit Simulé

# In[ ]:


#Premier algorithme de minimisation de l'erreur utilisant les chaînes de Markov
def Errmin(Ni,beta,Var,dom,K,p,h_init):
    pn_init = Density_Estimator(Var,dom,h_init,K,n)
    Err_init = MISE(pn_init,p)

    Err = [Err_init]
    h = [h_init]
    X1 = pn_init

    for i in range (Ni) :
        htest = np.random.rand()
        X_tmp = Density_Estimator(Var,dom,htest,K,n)
        Err1 = MISE(X1,p)
        Err2 = MISE(X_tmp,p)
        r = np.exp( -beta * (Err2 - Err1) ) 
        if r > 1 :
            X1 = X_tmp
            Err.append(Err2)
            h.append(htest)
        else : 
            u = np.random.random()
            if r > u :
                X1 = X_tmp
                Err.append(Err2)
                h.append(htest)
            else :
                Err.append(Err1)
    
    return Err,h


# # Génération des échantillons et des espaces associés

# ## Taille de l'échantillon et fenêtre initiale

# In[ ]:


#Nombre d'échantillons
N = 50

#Taille de l'échantillon
n = 1000

#Fenêtre initiale non-optimale
h_i = 0.4

#tableau des fenêtres
h = np.linspace(10**(-2),0.5,1000)


# ## Vecteurs aléatoires

# In[ ]:


#Générations des échantillons
Gaussian = np.zeros((N,n)) 
Kurt = np.zeros((N,n)) 
ABi = np.zeros((N,n))
Cl = np.zeros((N,n))
ACl = np.zeros((N,n))

for i in range (N) : 
    Gaussian[i] = Gauss(n,0,1)
    Kurt[i] = Kurtotic(n)
    ABi[i] = AsymmetricBimodal(n)
    Cl[i] = Claw(n)
    ACl[i] = AsymmetricClaw(n)


# ## Espaces associés

# In[ ]:


#On cherche à avoir un nombre de points sur lesquels l'estimateur est évalué très supérieur à la taille de l'échantillon
domaine = np.linspace(-3,3,10000)


# # Test du Code

# ## Tableaux des densités réelles

# In[ ]:


density_Gauss = density_Gaussian(domaine,0,1)
density_Kurtotic = density_Kurtotic(domaine)
density_ABi = density_ABi(domaine)
density_Claw = density_Claw(domaine)
density_AC = density_AC(domaine)


# ## Plot des densités réelles 

# In[ ]:


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,figsize=(9, 12), dpi=100)
ax1.set_title("Densités réelles")

ax1.plot(domaine,density_Gauss,label = "Densité Gaussienne")
ax2.plot(domaine,density_Kurtotic,label = "Densité Kurtotic")
ax3.plot(domaine,density_ABi,label = "Densité Bimodale asymétrique")
ax4.plot(domaine,density_Claw, label = "Densité Griffe")
ax5.plot(domaine,density_AC, label = "Densité Griffe asymétrique")


ax1.grid(True,which="both", linestyle='--')
ax2.grid(True,which="both", linestyle='--')
ax3.grid(True,which="both", linestyle='--')
ax4.grid(True,which="both", linestyle='--')
ax5.grid(True,which="both", linestyle='--')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

ax1.set_ylabel("y")
ax2.set_ylabel("y")
ax3.set_ylabel("y")
ax4.set_ylabel("y")
ax5.set_ylabel("y")

ax5.set_xlabel("x")


plt.legend()
plt.show()


# ## Premières estimations et risque quadratique avec une fenêtre non-optimale

# In[ ]:


#Tableaux des estimateurs pour tous les échantillons
Gaussian_estimation1 = np.zeros((N,10000))
Kurtotic_estimation1 = np.zeros((N,10000))
AsymmetricBimodal_estimation1 = np.zeros((N,10000))
Claw_estimation1 = np.zeros((N,10000))
AsymmetricClaw_estimation1 = np.zeros((N,10000))

Gauss_MISE1 = np.zeros((N,1))
Kurtotic_MISE1 = np.zeros((N,1))
AsymmetricBimodal_MISE1 = np.zeros((N,1))
Claw_MISE1 = np.zeros((N,1))
AsymmetricClaw_MISE1 = np.zeros((N,1))


for i in range (N):
    
    Gaussian_estimation1[i] = Density_Estimator(Gaussian[i],domaine,h_i,K5,n)
    Kurtotic_estimation1[i] = Density_Estimator(Kurt[i],domaine,h_i,K5,n)
    AsymmetricBimodal_estimation1[i] = Density_Estimator(ABi[i],domaine,h_i,K5,n)
    Claw_estimation1[i] = Density_Estimator(Cl[i],domaine,h_i,K5,n)
    AsymmetricClaw_estimation1[i] = Density_Estimator(ACl[i],domaine,h_i,K5,n)

#Risques quadratiques associés
    Gauss_MISE1[i] = MISE(Gaussian_estimation1[i], density_Gauss)
    Kurtotic_MISE1[i] = MISE(Kurtotic_estimation1[i], density_Kurtotic)
    AsymmetricBimodal_MISE1[i] = MISE(AsymmetricBimodal_estimation1[i], density_ABi)
    Claw_MISE1[i] = MISE(Claw_estimation1[i], density_Claw)
    AsymmetricClaw_MISE1[i] = MISE(AsymmetricClaw_estimation1[i], density_AC)


# In[ ]:


#Plots des premiers estimateurs
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,figsize=(9, 12), dpi=100)
ax1.set_title("Premières estimations pour une fenêtre arbitraire non-optimale")


ax1.plot(domaine,density_Gauss,label = "Densité Gaussienne")
ax2.plot(domaine,density_Kurtotic,label = "Densité Kurtotic")
ax3.plot(domaine,density_ABi,label = "Densité Bimodale asymétrique")
ax4.plot(domaine,density_Claw, label = "Densité Griffe")
ax5.plot(domaine,density_AC, label = "Densité Griffe asymétrique")


ax1.plot(domaine,Gaussian_estimation1[0],label = "Echantillon Gaussien")
ax2.plot(domaine,Kurtotic_estimation1[0],label = "Echantillon Kurtotic")
ax3.plot(domaine,AsymmetricBimodal_estimation1[0],label = "Echantillon Bimodal asymétrique")
ax4.plot(domaine,Claw_estimation1[0], label = "Echantillon Griffe")
ax5.plot(domaine,AsymmetricClaw_estimation1[0], label = "Echantillon Griffe asymétrique")


ax1.grid(True,which="both", linestyle='--')
ax2.grid(True,which="both", linestyle='--')
ax3.grid(True,which="both", linestyle='--')
ax4.grid(True,which="both", linestyle='--')
ax5.grid(True,which="both", linestyle='--')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

ax1.set_ylabel("y")
ax2.set_ylabel("y")
ax3.set_ylabel("y")
ax4.set_ylabel("y")
ax5.set_ylabel("y")

ax5.set_xlabel("x")


plt.legend()
plt.show()


# In[ ]:


#Affichage des risques quadratiques
print("Pour une fenêtre arbitraire, le risque quadratique moyen de 50 échantillons vaut : ")
print(" ")
print("R_INIT =",np.mean(Gauss_MISE1),"pour l'échantillon gaussien.")
print(" ")
print("R_INIT =",np.mean(Kurtotic_MISE1),"pour l'échantillon Kurtotic.")
print(" ")
print("R_INIT =",np.mean(AsymmetricBimodal_MISE1),"pour l'échantillon Bimodal asymétrique.")
print(" ")
print("R_INIT =",np.mean(Claw_MISE1),"pour l'échantillon Griffe.")
print(" ")
print("R_INIT =",np.mean(AsymmetricClaw_MISE1),"pour l'échantillon Griffe asymétrique.")


# ## Silverman's Rule

# In[ ]:


#Calcul du h_SILVER pour tous les échantillons
h_Gauss_SILVER = np.zeros((N,1))
h_Kurtotic_SILVER = np.zeros((N,1))
h_ABi_SILVER = np.zeros((N,1))
h_Claw_SILVER = np.zeros((N,1))
h_AC_SILVER = np.zeros((N,1))

Gaussian_estimation_SILVER = np.zeros((N,10000))
Kurtotic_estimation_SILVER = np.zeros((N,10000))
AsymmetricBimodal_estimation_SILVER = np.zeros((N,10000))
Claw_estimation_SILVER = np.zeros((N,10000))
AsymmetricClaw_estimation_SILVER = np.zeros((N,10000))

Gauss_MISE_SILVER = np.zeros((N,1))
Kurtotic_MISE_SILVER = np.zeros((N,1))
AsymmetricBimodal_MISE_SILVER = np.zeros((N,1))
Claw_MISE_SILVER = np.zeros((N,1))
AsymmetricClaw_MISE_SILVER = np.zeros((N,1))

for i in range (N):

    h_Gauss_SILVER[i] = Silverman(Gaussian[i],n)
    h_Kurtotic_SILVER[i] = Silverman(Kurt[i],n)
    h_ABi_SILVER[i] = Silverman(ABi[i],n)
    h_Claw_SILVER[i] = Silverman(Cl[i],n)
    h_AC_SILVER[i] = Silverman(ACl[i],n)

#Calcul de l'estimateur avec le h_Silver
    Gaussian_estimation_SILVER[i] = Density_Estimator(Gaussian[i],domaine,h_Gauss_SILVER[i],K5,n)
    Kurtotic_estimation_SILVER[i] = Density_Estimator(Kurt[i],domaine,h_Kurtotic_SILVER[i],K5,n)
    AsymmetricBimodal_estimation_SILVER[i] = Density_Estimator(ABi[i],domaine,h_ABi_SILVER[i],K5,n)
    Claw_estimation_SILVER[i] = Density_Estimator(Cl[i],domaine,h_Claw_SILVER[i],K5,n)
    AsymmetricClaw_estimation_SILVER[i] = Density_Estimator(ACl[i],domaine,h_AC_SILVER[i],K5,n)

#Risques quadratiques associés
    Gauss_MISE_SILVER[i] = MISE(Gaussian_estimation_SILVER[i], density_Gauss)
    Kurtotic_MISE_SILVER[i] = MISE(Kurtotic_estimation_SILVER[i], density_Kurtotic)
    AsymmetricBimodal_MISE_SILVER[i] = MISE(AsymmetricBimodal_estimation_SILVER[i], density_ABi)
    Claw_MISE_SILVER[i] = MISE(Claw_estimation_SILVER[i], density_Claw)
    AsymmetricClaw_MISE_SILVER[i] = MISE(AsymmetricClaw_estimation_SILVER[i], density_AC)


# In[ ]:


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,figsize=(9, 12), dpi=100)
ax1.set_title("Estimations pour une fenêtre optimisée avec la règle du pouce")


ax1.plot(domaine,density_Gauss,label = "Densité Gaussienne")
ax2.plot(domaine,density_Kurtotic,label = "Densité Kurtotic")
ax3.plot(domaine,density_ABi,label = "Densité Bimodale asymétrique")
ax4.plot(domaine,density_Claw, label = "Densité Griffe")
ax5.plot(domaine,density_AC, label = "Densité Griffe asymétrique")


ax1.plot(domaine,Gaussian_estimation_SILVER[0],label = "Echantillon Gaussien")
ax2.plot(domaine,Kurtotic_estimation_SILVER[0],label = "Echantillon Kurtotic")
ax3.plot(domaine,AsymmetricBimodal_estimation_SILVER[0],label = "Echantillon Bimodal asymétrique")
ax4.plot(domaine,Claw_estimation_SILVER[0], label = "Echantillon Griffe")
ax5.plot(domaine,AsymmetricClaw_estimation_SILVER[0], label = "Echantillon Griffe asymétrique")


ax1.grid(True,which="both", linestyle='--')
ax2.grid(True,which="both", linestyle='--')
ax3.grid(True,which="both", linestyle='--')
ax4.grid(True,which="both", linestyle='--')
ax5.grid(True,which="both", linestyle='--')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

ax1.set_ylabel("y")
ax2.set_ylabel("y")
ax3.set_ylabel("y")
ax4.set_ylabel("y")
ax5.set_ylabel("y")

ax5.set_xlabel("x")


plt.legend()
plt.show()


# In[ ]:


#Affichage des risques quadratiques pour la règle du pouce
print("Pour une fenêtre optimisée avec la règle du pouce, le risque quadratique moyen de 50 échantillons vaut : ")
print(" ")
print("R_SILVER =",np.mean(Gauss_MISE_SILVER),"pour l'échantillon gaussien.")
print(" ")
print("R_SILVER =",np.mean(Kurtotic_MISE_SILVER),"pour l'échantillon Kurtotic.")
print(" ")
print("R_SILVER =",np.mean(AsymmetricBimodal_MISE_SILVER),"pour l'échantillon Bimodal asymétrique.")
print(" ")
print("R_SILVER =",np.mean(Claw_MISE_SILVER),"pour l'échantillon Griffe.")
print(" ")
print("R_SILVER =",np.mean(AsymmetricClaw_MISE_SILVER),"pour l'échantillon Griffe asymétrique.")


# ## Unbiased Cross Validation

# In[ ]:


#Tableau du critère UCV
Gauss_ucv = np.zeros ((50,1000))
Kurtotic_ucv = np.zeros ((50,1000))
ABi_ucv = np.zeros ((50,1000))
Claw_ucv = np.zeros ((50,1000))
AC_ucv = np.zeros ((50,1000))

h_Gauss_UCV = np.zeros((50,1))
h_Kurtotic_UCV = np.zeros((50,1))
h_ABi_UCV = np.zeros((50,1))
h_Claw_UCV = np.zeros((50,1))
h_AC_UCV = np.zeros((50,1))

Gaussian_estimation_UCV = np.zeros((N,10000))
Kurtotic_estimation_UCV = np.zeros((N,10000))
AsymmetricBimodal_estimation_UCV = np.zeros((N,10000))
Claw_estimation_UCV = np.zeros((N,10000))
AsymmetricClaw_estimation_UCV = np.zeros((N,10000))

Gauss_MISE_UCV = np.zeros((N,1))
Kurtotic_MISE_UCV = np.zeros((N,1))
AsymmetricBimodal_MISE_UCV = np.zeros((N,1))
Claw_MISE_UCV = np.zeros((N,1))
AsymmetricClaw_MISE_UCV = np.zeros((N,1))

for i in range (N): 
    
    #recherche du critère pour un échantillon
    Gauss_ucv[i] = UCV(Gaussian[i],domaine,h,K5,n)
    Kurtotic_ucv[i] = UCV(Kurt[i],domaine,h,K5,n)
    ABi_ucv[i] = UCV(ABi[i],domaine,h,K5,n)
    Claw_ucv[i] = UCV(Cl[i],domaine,h,K5,n)
    AC_ucv[i] = UCV(ACl[i],domaine,h,K5,n)
    
    #prend le h optimal pour ce même échantillon
    h_Gauss_UCV[i] = h[np.argmin(Gauss_ucv[i])]
    h_Kurtotic_UCV[i] = h[np.argmin(Kurtotic_ucv[i])]
    h_ABi_UCV[i] = h[np.argmin(ABi_ucv[i])]
    h_Claw_UCV[i] = h[np.argmin(Claw_ucv[i])]
    h_AC_UCV[i] = h[np.argmin(AC_ucv[i])]
    
    #estimations des densitées avec ce h opti
    Gaussian_estimation_UCV[i] = Density_Estimator(Gaussian[i],domaine,h_Gauss_UCV[i],K5,n)
    Kurtotic_estimation_UCV[i] = Density_Estimator(Kurt[i],domaine,h_Kurtotic_UCV[i],K5,n)
    AsymmetricBimodal_estimation_UCV[i] = Density_Estimator(ABi[i],domaine,h_ABi_UCV[i],K5,n)
    Claw_estimation_UCV[i] = Density_Estimator(Cl[i],domaine,h_Claw_UCV[i],K5,n)
    AsymmetricClaw_estimation_UCV[i] = Density_Estimator(ACl[i],domaine,h_AC_UCV[i],K5,n)

    #Risques quadratiques associés
    Gauss_MISE_UCV[i] = MISE(Gaussian_estimation_UCV[i], density_Gauss)
    Kurtotic_MISE_UCV[i] = MISE(Kurtotic_estimation_UCV[i], density_Kurtotic)
    AsymmetricBimodal_MISE_UCV[i] = MISE(AsymmetricBimodal_estimation_UCV[i], density_ABi)
    Claw_MISE_UCV[i] = MISE(Claw_estimation_UCV[i], density_Claw)
    AsymmetricClaw_MISE_UCV[i] = MISE(AsymmetricClaw_estimation_UCV[i], density_AC)


# In[ ]:


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,figsize=(9, 12), dpi=100)
ax1.set_title("Estimations pour une fenêtre optimisée avec la méthode UCV")


ax1.plot(domaine,density_Gauss,label = "Densité Gaussienne")
ax2.plot(domaine,density_Kurtotic,label = "Densité Kurtotic")
ax3.plot(domaine,density_ABi,label = "Densité Bimodale asymétrique")
ax4.plot(domaine,density_Claw, label = "Densité Griffe")
ax5.plot(domaine,density_AC, label = "Densité Griffe asymétrique")


ax1.plot(domaine,Gaussian_estimation_UCV[0],label = "Echantillon Gaussien")
ax2.plot(domaine,Kurtotic_estimation_UCV[0],label = "Echantillon Kurtotic")
ax3.plot(domaine,AsymmetricBimodal_estimation_UCV[0],label = "Echantillon Bimodal asymétrique")
ax4.plot(domaine,Claw_estimation_UCV[0], label = "Echantillon Griffe")
ax5.plot(domaine,AsymmetricClaw_estimation_UCV[0], label = "Echantillon Griffe asymétrique")


ax1.grid(True,which="both", linestyle='--')
ax2.grid(True,which="both", linestyle='--')
ax3.grid(True,which="both", linestyle='--')
ax4.grid(True,which="both", linestyle='--')
ax5.grid(True,which="both", linestyle='--')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

ax1.set_ylabel("y")
ax2.set_ylabel("y")
ax3.set_ylabel("y")
ax4.set_ylabel("y")
ax5.set_ylabel("y")

ax5.set_xlabel("x")


plt.legend()
plt.show()


# In[ ]:


#Affichage des risques quadratiques pour la méthode UCV
print("Pour une fenêtre optimisée avec la méthode UCV, le risque quadratique moyen de 50 échantillons vaut : ")
print(" ")
print("R_UCV =",np.mean(Gauss_MISE_UCV),"pour l'échantillon gaussien.")
print(" ")
print("R_UCV =",np.mean(Kurtotic_MISE_UCV),"pour l'échantillon Kurtotic.")
print(" ")
print("R_UCV =",np.mean(AsymmetricBimodal_MISE_UCV),"pour l'échantillon Bimodal asymétrique.")
print(" ")
print("R_UCV =",np.mean(Claw_MISE_UCV),"pour l'échantillon Griffe.")
print(" ")
print("R_UCV =",np.mean(AsymmetricClaw_MISE_UCV),"pour l'échantillon Griffe asymétrique.")


# ## Penalized Comparison to Overfiting

# In[ ]:


#Tableau du critère PCO
Gauss_pco = np.zeros ((50,1000))
Kurtotic_pco = np.zeros ((50,1000))
ABi_pco = np.zeros ((50,1000))
Claw_pco = np.zeros ((50,1000))
AC_pco = np.zeros ((50,1000))

h_Gauss_PCO = np.zeros((50,1))
h_Kurtotic_PCO = np.zeros((50,1))
h_ABi_PCO = np.zeros((50,1))
h_Claw_PCO = np.zeros((50,1))
h_AC_PCO = np.zeros((50,1))

Gaussian_estimation_PCO = np.zeros((N,10000))
Kurtotic_estimation_PCO = np.zeros((N,10000))
AsymmetricBimodal_estimation_PCO = np.zeros((N,10000))
Claw_estimation_PCO = np.zeros((N,10000))
AsymmetricClaw_estimation_PCO = np.zeros((N,10000))

Gauss_MISE_PCO = np.zeros((N,1))
Kurtotic_MISE_PCO = np.zeros((N,1))
AsymmetricBimodal_MISE_PCO = np.zeros((N,1))
Claw_MISE_PCO = np.zeros((N,1))
AsymmetricClaw_MISE_PCO = np.zeros((N,1))

for i in range (N): 

    Gauss_pco[i] = PCO(Gaussian[i],domaine,h,K5,n)
    Kurtotic_pco[i] = PCO(Kurt[i],domaine,h,K5,n)
    ABi_pco[i] = PCO(ABi[i],domaine,h,K5,n)
    Claw_pco[i] = PCO(Cl[i],domaine,h,K5,n)
    AC_pco[i] = PCO(ACl[i],domaine,h,K5,n)
    
    #prend le h optimal pour ce même échantillon
    h_Gauss_PCO[i] = h[np.argmin(Gauss_pco[i])]
    h_Kurtotic_PCO[i] = h[np.argmin(Kurtotic_pco[i])]
    h_ABi_PCO[i] = h[np.argmin(ABi_pco[i])]
    h_Claw_PCO[i] = h[np.argmin(Claw_pco[i])]
    h_AC_PCO[i] = h[np.argmin(AC_pco[i])]
    
    #estimations des densitées avec ce h opti
    Gaussian_estimation_PCO[i] = Density_Estimator(Gaussian[i],domaine,h_Gauss_PCO[i],K5,n)
    Kurtotic_estimation_PCO[i] = Density_Estimator(Kurt[i],domaine,h_Kurtotic_PCO[i],K5,n)
    AsymmetricBimodal_estimation_PCO[i] = Density_Estimator(ABi[i],domaine,h_ABi_PCO[i],K5,n)
    Claw_estimation_PCO[i] = Density_Estimator(Cl[i],domaine,h_Claw_PCO[i],K5,n)
    AsymmetricClaw_estimation_PCO[i] = Density_Estimator(ACl[i],domaine,h_AC_PCO[i],K5,n)

    #Risques quadratiques associés
    Gauss_MISE_PCO[i] = MISE(Gaussian_estimation_PCO[i], density_Gauss)
    Kurtotic_MISE_PCO[i] = MISE(Kurtotic_estimation_PCO[i], density_Kurtotic)
    AsymmetricBimodal_MISE_PCO[i] = MISE(AsymmetricBimodal_estimation_PCO[i], density_ABi)
    Claw_MISE_PCO[i] = MISE(Claw_estimation_PCO[i], density_Claw)
    AsymmetricClaw_MISE_PCO[i] = MISE(AsymmetricClaw_estimation_PCO[i], density_AC)


# In[ ]:


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,figsize=(9, 12), dpi=100)
ax1.set_title("Estimations pour une fenêtre optimisée avec la méthode PCO")


ax1.plot(domaine,density_Gauss,label = "Densité Gaussienne")
ax2.plot(domaine,density_Kurtotic,label = "Densité Kurtotic")
ax3.plot(domaine,density_ABi,label = "Densité Bimodale asymétrique")
ax4.plot(domaine,density_Claw, label = "Densité Griffe")
ax5.plot(domaine,density_AC, label = "Densité Griffe asymétrique")


ax1.plot(domaine,Gaussian_estimation_PCO[0],label = "Echantillon Gaussien")
ax2.plot(domaine,Kurtotic_estimation_PCO[0],label = "Echantillon Kurtotic")
ax3.plot(domaine,AsymmetricBimodal_estimation_PCO[0],label = "Echantillon Bimodal asymétrique")
ax4.plot(domaine,Claw_estimation_PCO[0], label = "Echantillon Griffe")
ax5.plot(domaine,AsymmetricClaw_estimation_PCO[0], label = "Echantillon Griffe asymétrique")


ax1.grid(True,which="both", linestyle='--')
ax2.grid(True,which="both", linestyle='--')
ax3.grid(True,which="both", linestyle='--')
ax4.grid(True,which="both", linestyle='--')
ax5.grid(True,which="both", linestyle='--')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

ax1.set_ylabel("y")
ax2.set_ylabel("y")
ax3.set_ylabel("y")
ax4.set_ylabel("y")
ax5.set_ylabel("y")

ax5.set_xlabel("x")


plt.legend()
plt.show()


# In[ ]:


#Affichage des risques quadratiques pour la méthode PCO
print("Pour une fenêtre optimisée avec la méthode PCO, le risque quadratique moyen de 50 échantillons vaut : ")
print(" ")
print("R_PCO =",np.mean(Gauss_MISE_PCO),"pour l'échantillon gaussien.")
print(" ")
print("R_PCO =",np.mean(Kurtotic_MISE_PCO),"pour l'échantillon Kurtotic.")
print(" ")
print("R_PCO =",np.mean(AsymmetricBimodal_MISE_PCO),"pour l'échantillon Bimodal asymétrique.")
print(" ")
print("R_PCO =",np.mean(Claw_MISE_PCO),"pour l'échantillon Griffe.")
print(" ")
print("R_PCO =",np.mean(AsymmetricClaw_MISE_PCO),"pour l'échantillon Griffe asymétrique.")


# ## Synthèse minimisation

# In[ ]:


#Affichage des risques quadratiques

print("Pour l'échantillon gaussien :")

print(" ")

print("R_INIT =",np.mean(Gauss_MISE1))
print("R_SILVER =",np.mean(Gauss_MISE_SILVER))
print("R_UCV =",np.mean(Gauss_MISE_UCV))
print("R_PCO =",np.mean(Gauss_MISE_PCO))

print(" ")

print("Pour l'échantillon Kurtotic :")

print(" ")

print("R_INIT =",np.mean(Kurtotic_MISE1))
print("R_SILVER =",np.mean(Kurtotic_MISE_SILVER))
print("R_UCV =",np.mean(Kurtotic_MISE_UCV))
print("R_PCO =",np.mean(Kurtotic_MISE_PCO))

print(" ")

print("Pour l'échantillon Bimodal asymétrique : ")

print(" ")

print("R_INIT =",np.mean(AsymmetricBimodal_MISE1))
print("R_SILVER =",np.mean(AsymmetricBimodal_MISE_SILVER))
print("R_UCV =",np.mean(AsymmetricBimodal_MISE_UCV))
print("R_PCO =",np.mean(AsymmetricBimodal_MISE_PCO))

print(" ")

print("Pour l'échantillon Griffe :")

print(" ")

print("R_INIT =",np.mean(Claw_MISE1))
print("R_SILVER =",np.mean(Claw_MISE_SILVER))
print("R_UCV =",np.mean(Claw_MISE_UCV))
print("R_PCO =",np.mean(Claw_MISE_PCO))

print(" ")

print("Pour l'échantillon Griffe asymétrique :")

print(" ")

print("R_INIT =",np.mean(AsymmetricClaw_MISE1))
print("R_SILVER =",np.mean(AsymmetricClaw_MISE_SILVER))
print("R_UCV =",np.mean(AsymmetricClaw_MISE_UCV))
print("R_PCO =",np.mean(AsymmetricClaw_MISE_PCO))


# In[ ]:




