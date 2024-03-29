import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as spo
from hmmlearn.hmm import GaussianHMM
import random
import pandas as pd
import numpy.random as rnd
import scipy.stats as stats
from tqdm import tqdm
np.random.seed(893)

# Generates the states
def generate_state(mat_transition, n):
    x=np.arange(n)
    x[0]=0
    dim=len(mat_transition[0])
    for i in np.arange(1,n):
        x[i]=rnd.choice(a=dim,p=mat_transition[x[i-1]])
    return x

# Generate thes AR
def generateNorm(params,x):

    u0 = params[0]
    u1 = params[1]
    sig = params[2]
    sig2=params[5]
    T= len(x)

    series0 = np.random.normal(u0, math.sqrt(sig),T)
    series1 = np.random.normal(u1,math.sqrt(sig2),T)
    st=[]

    for i in range(len(x)-1):
        if (x[i]==0):
            st.append(series0[i])
        else:
            st.append(series1[i])
    return st

#generates probility
def generate_p(params,y) :
    p1filt = np.zeros(len(y))
    p2filt = np.zeros(len(y))
    p1_prev = np.zeros(len(y))
    p2_prev = np.zeros(len(y))
    fyt_global = np.zeros(len(y))

    u = params[0]
    sig = params[1]
    p11 = params[2]
    p22=params[3]
    p21 = 1 - p22
    p12 = 1 - p11
    sig2 = params[4]


    #we initialize
    p1filt[0] = ((1 - p22) / (2 - p11 - p22))
    p2filt[0] = ((1 - p11) / (2 - p11 - p22))


    for t in range(1, len(y)) : 
        #etape 1
        p1_prev[t] = p1filt[t-1] * p11 + p2filt[t-1] * p21
        p2_prev[t] = p1filt[t-1] * p12 + p2filt[t-1] * p22
        
        
        fyt_global[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(y[t]-u)**2/(2*sig))*p1_prev[t] + 1/(math.sqrt(2*np.pi*sig2))*math.exp(-1*(y[t]-u)**2/(2*sig2))*p2_prev[t]
    
        p1filt[t] = (1/(math.sqrt(2*np.pi*sig)))*math.exp(-1*(y[t]-u)**2/(2*sig))* p1_prev[t] / fyt_global[t]
        p2filt[t] = (1/(math.sqrt(2*np.pi*sig2)))*math.exp(-1*(y[t]-u)**2/(2*sig2))* p2_prev[t] / fyt_global[t]

    return p1filt,p2filt

#Function to create HMM des probabilité fil trouvé préalablement
def hmm(p1filt, p2filt) :
    hmm = np.zeros(len(p1filt))
    for i in range(0, len(p1filt)-1):
        if p1filt[i] > p2filt[i]:
            hmm[i] = 0
        elif p1filt[i] == p2filt[i]:
            hmm[i]= 0.5
        else: hmm[i] = 1
    return hmm

def find_transition(hmm):
    mt11 = 0
    mt12 = 0
    mt21 = 0
    mt22 = 0
    for m in range(len(hmm)-1):
        
        
        if hmm[m] == 0 :
            if hmm[m+1]  == 0:
                mt11 = mt11 + 1
            else : 
                mt12 =  mt12 + 1
        else:
            if hmm[m+1]  == 0:
                mt21 = mt21 + 1
            else : 
                mt22 = mt22 + 1
    
    sum_all1 = mt11 + mt12 
    sum_all2 = mt21 + mt22  
    
    matrice_transition_estime = np.array([[(mt11/sum_all1),(mt12/sum_all1) ], 
            [(mt21/sum_all2), (mt22/sum_all2)]])
    return matrice_transition_estime

def hamilton(params,y):
    
    p1filt = np.zeros(len(y))
    p2filt = np.zeros(len(y))
    f= np.zeros(len(y))

    u = params[0]
    sig = params[1]
    p11=params[2]
    p22 = params[3]
    p12 = 1 - p11
    p21 = 1 - p22
    sig2= params[4]

    
    p1filt[0] = ((1 - p22) / (2 - p11 - p22))
    p2filt[0] = ((1 - p11) / (2 - p11 - p22))

    for t in range(1,len(y)):

        #etape 1
        p1pred = p11*p1filt[t-1] + p21*p2filt[t-1]
        p2pred = p12*p1filt[t-1] + p22*p2filt[t-1]
    
        #etape 2
        f[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(y[t]-u)**2/(2*sig))*p1pred + 1/(math.sqrt(2*np.pi*sig2))*math.exp(-1*(y[t]-u)**2/(2*sig2))*p2pred
            
        #etape 3
        p1filt[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(y[t]-u)**2/(2*sig))*p1pred / f[t]
        p2filt[t] = 1/(math.sqrt(2*np.pi*sig2))*math.exp(-1*(y[t]-u)**2/(2*sig2))*p2pred / f[t]

    res = f[1:]
    log = np.sum((np.log(res)))

    #plt.plot(range(len(y)),f)
    #plt.show()
    return -1*log

def optimize(y):
    #Constraints for the optimization of parameters

    def constraint1(params):
        return 1 - params[2] - (1-params[2])

    def constraint2(params):
        return 1 - params[3]-(1-params[3])

    def constraint3(params):
        return params[1]-0.01

    def constraint4(params):
        return params[4]-0.01


    #Optimization
    con1 = {"type": "eq", "fun": constraint1}
    con2 = {"type": "eq", "fun": constraint2}
    con3 = {"type": "ineq", "fun": constraint3}
    con4 = {"type": "ineq", "fun": constraint4}
    cons = [con1,con2,con3,con4]
    bounds = ((0,None),(0.1,2),(0.001,1),(0.001,1),(0.1,2))
    xo=[0.01,1,0.5,0.5,1]
    optparams = spo.minimize(hamilton,args=(y,), x0=xo, bounds = bounds, method = "SLSQP")
    return optparams

def AIC(optparams):

    #Since the loglikelyhood in the aic is a positive number, we multiply by -1 since we minimized the -log
    aic = -1*-2*optparams.fun + 2*len(optparams.x)

    return aic


def MonteCarloSimulate(count):

    df = pd.DataFrame(columns = ["u", "sige", "p11e", "p22e"])

    for i in tqdm(range(count)):

        #Step 1: We initiate the data
        p11= random.random()+0.0001
        p22 = random.random() + 0.0001

        p11= 0.9
        p22=0.9
        p12 = 1 - p11
        p21 = 1 - p22
        u = 1
        sig= 1
        sig2=2
        t=2000

        transmat = [[p11,p12],[p21,p22]]
        params=[u,sig,p11,p22,sig2]

        #Step 2, we generate states according to the matrix
        x = generate_state(transmat, t)

        #Step 3, we generate the ar according to the states
        y = generateNorm(params,x)

        #Step 4 we generate the probabilities with the filter
        p1filt,p2filt = generate_p(params, y)

        #Step 5: We obtain the states predicted by the filts
        hmmstates = hmm(p1filt,p2filt)

        #Step 6: We find the transition matrix accroding to the predicted states
        #hmmtransmat = find_transition(hmmstates)

        #Step 7: optimize the parameters
        optimizeroutput = optimize(y)

        #Step 8: calculate squared error
        #squarederrormat = np.subtract(hmmtransmat,transmat)
        #squaredmat = np.square(squarederrormat)
        #squarederror = np.sum(squaredmat)

        #Step 9: calculate squared error of params
        u = optimizeroutput.x[0]
        sig =optimizeroutput.x[1]
        p11 = optimizeroutput.x[2]
        p22 = optimizeroutput.x[3]
        sig2 = optimizeroutput.x[4]

        df.loc[len(df)]=[u,sig,p11,p22,sig2]


        print()
        print(f"Actual params are : {params}")
        print(f"actual transmat is : {transmat}")
        print(f"optimizer params are : {optimizeroutput.x}")
        #print(f"hmmtransmat is : {hmmtransmat}")
        #print(f"squared error is : {squarederror}")

    return df

#result = MonteCarloSimulate(100)
#print(result)
#result.to_csv("result22.csv")

# #Step 1, we initiate data
# delta1 = 3
# delta2 = 0.8
# phi = 0.6
# sig = 0.4
# p11 = 0.9
# p22 = 0.85
# p12 = 1 - p11
# p21 = 1 - p22
# params = [delta1,delta2,phi,sig,p11,p22]
# transmat = [[p11,p12],[p21,p22]]

# #Step 2 generate states according to the matrix
# realstates = generate_state(transmat,400)

# #Step 3: generate the ar according to the states
# y = generateAR(params, realstates)

# #step 4: We generate the probabilities with the filter
# p1filt, p2filt = generate_p(params,y)

# #Step 5: We obtain the states predicted by the filts
# hmmstates = hmm(p1filt, p2filt)

# #Step 6: We find the transition matrix according to the predicted states
# hmmtransmat = find_transition(hmmstates)

# #Step 7: Optimize the parameters using y
# optimizeroutput = optimize(y)
# print(optimizeroutput.x)

# x_axis = range(len(realstates))
# y_axis1 = y
# y_axis2 = realstates
# y_axis3= hmmstates

# print(transmat)
# print(hmmtransmat)

# plt.plot(x_axis,y_axis1)
# plt.plot(x_axis,y_axis2, color= "red", linewidth ="6")
# plt.plot(x_axis,y_axis3, color = "green")

# plt.show()