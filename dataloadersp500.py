import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import hmmnorm as hmm
from hmmlearn import hmm as hml
import Function_garch as fg

data = fg.load_Sp500()
logreturns = fg.logf_diff(data)

ret2000 = logreturns["2000"]
ret2004 = logreturns["2004"]
ret2008 = logreturns["2008"]
ret2012 = logreturns["2012"]
ret2016 = logreturns["2016"]



def ImplementOptim(logreturns):
    optim = hmm.optimize(logreturns)
    optimparam = optim.x
    p1filt, p2filt = hmm.generate_p(optimparam,logreturns)
    states = hmm.hmm(p1filt,p2filt)
    var1 = optimparam[2]
    sigma1 = math.sqrt(var1)
    var2 = optimparam[5]
    sigma2 = math.sqrt(var2)
    pfiltnorm = p1filt*var1 + p2filt*var2

    
    print(p1filt)
    print(p2filt)

    print(sigma1)
    print(sigma2)
    
    return states, pfiltnorm




def plot(returns,pfiltnorm):

    x= range(len(returns))
    y=returns

    plt.plot(x,y)
    #plt.plot(x,states)
    plt.plot(x,pfiltnorm)

    plt.show()


#states2000,pfilt2000 = ImplementOptim(ret2000)
#states2004,pfilt2004 = ImplementOptim(ret2004)
#states2008,pfilt2008 = ImplementOptim(ret2008)
#states2012,pfilt2012 = ImplementOptim(ret2012)
states2016,pfilt2016 = ImplementOptim(ret2016)

#plot(ret2000,pfilt2000)
#plot(ret2004,pfilt2004)
#plot(ret2008,pfilt2008)
#plot(ret2012,pfilt2012)
plot(ret2016,pfilt2016)



test = fg.generer_estime(ret2000)

