import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import hmmnorm as hmm
from hmmlearn import hmm as hml

insample = pd.read_excel("In_Sample_Pelletier.xlsx")

#Column names are GBP/USD DM/USD JPY/USD CHF/USD

# We find the first difference of the log
log = np.log(insample)
logdiff = log.diff()*100
logdiff = logdiff.iloc[1:]

#Create plots
figure, axis = plt.subplots(4, 1)
x= range(len(logdiff["GBP/USD"]))
axis[0].plot(x,logdiff["GBP/USD"])
axis[1].plot(x,logdiff["DM/USD"])
axis[2].plot(x,logdiff["JPY/USD"])
axis[3].plot(x,logdiff["CHF/USD"])


#Create dataframe with parmas estimated by the optimizer
df = pd.DataFrame(columns = ["U0", "U1", "sig", "p11", "p22","sig2"])

#df = pd.DataFrame(columns = ["U", "sig", "p11", "p22","sig2"]) # Comment and uncomment depending on the hmmnorm to be used



GBPoptim = hmm.optimize(logdiff["GBP/USD"])
GBPparams = GBPoptim.x

DMoptim = hmm.optimize(logdiff["DM/USD"])
DMparams = DMoptim.x

JPYoptim = hmm.optimize(logdiff["JPY/USD"])
JPYparams = JPYoptim.x

CHFoptim = hmm.optimize(logdiff["CHF/USD"])
CHFparams = CHFoptim.x

df.loc[len(df)]=GBPparams
df.loc[len(df)]=DMparams
df.loc[len(df)]=JPYparams
df.loc[len(df)]=CHFparams


p1filt, p2filt = hmm.generate_p(GBPparams,logdiff["GBP/USD"])
GBPstates = hmm.hmm(p1filt,p2filt)

p1filt, p2filt = hmm.generate_p(DMparams,logdiff["DM/USD"])
DMstates = hmm.hmm(p1filt,p2filt)

p1filt, p2filt = hmm.generate_p(JPYparams,logdiff["JPY/USD"])
JPYstates = hmm.hmm(p1filt,p2filt)

p1filt, p2filt = hmm.generate_p(CHFparams,logdiff["CHF/USD"])
CHFstates = hmm.hmm(p1filt,p2filt)


print(df)
print(hmm.AIC(GBPoptim))



axis[0].plot(x,GBPstates)
axis[1].plot(x,DMstates)
axis[2].plot(x,JPYstates)
axis[3].plot(x,CHFstates)

#Validate with HMM
#Trying to do it using the gaussianHMM model (seems to find two states (based on mean more than vol it seems))
#model = hml.GaussianHMM(n_components=2)
#data = np.array(logdiff["GBP/USD"]).reshape(-1,1)
#model.fit(data)
#states = model.predict(data)
#print(states)

plt.show()


