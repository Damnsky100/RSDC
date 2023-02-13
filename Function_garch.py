

## Import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#Importer les données sp500
from yahoo_fin import stock_info as si
import re
from datetime import datetime

#Pour graphique
from matplotlib import pyplot as plt


def load_Sp500():
    
    data = si.get_data("^GSPC", start_date = '01/01/2000')
    data.index = pd.to_datetime(data.index)

    n_tradingday = 252
    n_years = 4
    nb_bins = 5

    date =['2000', '2004', '2008', '2012', '2016']

    df_data = pd.DataFrame()


    for i in range(nb_bins):
        tmp =  data["close"][ i* (n_tradingday * n_years)  : (i+1)*(n_tradingday * n_years) ].values
        df_data.insert(i, date[i], value= tmp)
        
    return df_data



def load_nasdaq():
    data = si.get_data("^IXIC", start_date = '01/01/2000')
    data.index = pd.to_datetime(data.index)

    n_tradingday = 252
    n_years = 4
    nb_bins = 5

    date =['2000', '2004', '2008', '2012', '2016']

    df_data = pd.DataFrame()


    for i in range(nb_bins):
        tmp =  data["close"][ i* (n_tradingday * n_years)  : (i+1)*(n_tradingday * n_years) ].values
        df_data.insert(i, date[i], value= tmp)
    return df_data

############ Premiére différence * 100 pour le rendement ##############

def logf_diff(data):
            tmp = (np.log(data) - np.log(data.shift(1))) * 100
    
            return tmp.iloc[1:,]    

    
    

############# Define GARCH recursion ###################

def generer_garch(alpha_0, alpha_1, beta, U):
    
    T = len(U) 
    
    sigma_2 = np.zeros(T)
    
    sigma_2[0] = alpha_0 / (1 - alpha_1 - beta)
    
    for i in range(1, T):
        sigma_2[i] = alpha_0 + alpha_1 * U[i-1]**2 + beta * sigma_2[i-1]
        
    return sigma_2


############# Define GARCH log-Likelihood ###################
def garch_loglike(params, U):
    
    alpha_0 = params[0]
    
    alpha_1 = params[1]
    
    beta = params[2]
    
    
    
    sigma_2 = generer_garch(alpha_0, alpha_1, beta, U)
    
    LogL = -np.sum(-np.log(sigma_2) - U**2 / sigma_2 )
    
    return LogL
    
    
    
############# Fonction Maximisation ###################
#Contrainte à ajouter

def maximiser(U):
    #On déclare la contrainte
    #cons = {"type": "ineq", "fun": cons1}
    
    bounds = ((0.001,None),(0.001,1),(0.001,1))
    
    vP0 = (0.1, 0.1, 0.8)
    
    #On utilise la librairie pour optimiser (À voir l'algorithme sélectionné)
    
    res = opt.minimize(garch_loglike, vP0, args= U,
                    bounds = bounds,
                    method="SLSQP",
                    options = {"disp" : False})
    return res



#Utiliser l'optimisateur afin de voir le GARCH par MV

def generer_estime(ret):
    
    
    alpha_0, alpha_1, beta = maximiser(ret).x

    sigma_2 = generer_garch(alpha_0, alpha_1, beta, ret) #Variance conditionnelle
    
    x = np.arange(0, len(sigma_2))
    
    std_return = ret / np.sqrt(sigma_2) #return standardisé par l'écart-type conditionnel
    
    return   np.around([alpha_0, alpha_1, beta],4), np.around(sigma_2, 4), np.around(std_return, 4)






