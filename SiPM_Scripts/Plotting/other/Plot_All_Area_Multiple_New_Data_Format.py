# for a single Temp that needs to be declared beforehand
# needs find Temp and create Data array function



from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import string
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
from scipy.stats import linregress as lreg
import pandas as pd
import statsmodels.api as sm
import numpy.ma as ma



def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '{:.3e}'.format(x)

def GetNames(destT,destV,Date):

    path = '/home/gebhardt/SiPM_MPIK/scripts/Data/Date'+Date+'/'+str(destT)+'deg/'+str(destV)+'V/'
    name = 'HAM_T_' + str(destT) + '_Vb' + str(destV) + '.trc'
    compl_name = os.path.join(path,name)
    return compl_name

def GetSavePath(Date):
    savepath = '/home/gebhardt/SiPM_MPIK/scripts/Data/Date'+Date+'/'
    return savepath



np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float':lambda x: format(x, '6.3E')})
formatter = FuncFormatter(log_10_product)

#date = ['1708166', '1708167','1708168']
date = ['1808162','1808163']

Temp = np.zeros(0)
Volt = np.zeros(0)
Gain = np.zeros(0)
Error = np.zeros(0)
dDCR = np.zeros(0)
cDCR = np.zeros(0)
dOCT =np.zeros(0)
cOCT = np.zeros(0)

for Date in date:
    savepath = GetSavePath(Date)
    datalist = np.load(savepath+'AreaTempVbGainFileNewFormat.npy')


    Temp = np.append(Temp,datalist[0]) #List of Temps
    Volt=np.append(Volt,datalist[1]) #List of Voltages
    Gain=np.append(Gain,datalist[2])#NP array of lists of Gains per Voltage, per Temperature
    Error=np.append(Error,datalist[3]) #List of Errors of the GainCalc
    dDCR=np.append(dDCR,datalist[4]) #List of dirty DarkCountRate
    cDCR=np.append(cDCR,datalist[5]) #List of clean DarkCountRate
    dOCT=np.append(dOCT,datalist[6]) #List of dirty OpticalCrossTalk
    cOCT=np.append(cOCT,datalist[7]) #List of clean OpticalCrossTalk




print 'Data :'
print 'Temp ',Temp#.shape
print 'Volt ',Volt#.shape
print '"Gain" ',Gain#.shape
print 'Error', Error#.shape
print 'dDCR', dDCR#.shape
print 'cDCR', cDCR#.shape
print 'dOCT',dOCT#.shape
print 'cOCT',cOCT#.shape



#colors=iter(cm.rainbow(np.linspace(0,1,n)))
colors = ['r','b','g','c','m','y','k','w']

#Temp = [0.0]
#for i in Temp:
	for index,i in enumerate(Temp):
	#get index of temp, every temp has his own dataset
		#wont work, needs to look for all Temps with T = o.o, to calc regression line for all gain corresponding to that temp!!


	if i == '0.0':c=color[0] 	
	if i == '5.0':c=color[1] 
	if i == '10.0':c=color[2] 
	if i == '15.0':c=color[3] 
	if i == '20.0':c=color[4] 
	if i == '25.0':c=color[5] 


	slope = []
	intercept = []
	Vbr = []
	OVlist = []

	#for i in range(len(Temp)):
	#regression line for Vb = 28V - 30V , 0.5V steps
	X = Volt#[2:]#4:1]
	X = sm.add_constant(X)
	Y = Gain#[2:]#4:1]
	y_err = Error#[2:]#4:1]
	weights = pd.Series(y_err)
	wls_model = sm.WLS(Y,X, weights=1 / weights)
	results = wls_model.fit()
	print 'results',results.params
	inter,slo = results.params
	slope.append(slo)
	intercept.append(inter)
	vbreak = -inter/slo
	#if (vbreak<60) or (vbreak>70):vbreak = 60
	Vbr.append(vbreak)
	OVlist.append(np.array(Volt) - np.array(Vbr))

	Vbr = np.asarray(Vbr)
	print 'Vbr ',Vbr
	print 'OVlist ',OVlist




	
	x = np.arange(60,80)
	null = np.zeros(len(Vbr))

	fig5 = plt.figure(5)
	ax5 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
	#ax5.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
	#ax5.xaxis.set_minor_locator(ticker.MultipleLocator(1))
	ax5.grid(True)
	#ax5.set_ylim([0.0,0.04])
	ax5.set_xlim(60,80)
	ax5.set_xlabel('Bias Voltage [V]')
	ax5.set_ylabel('"Gain" [V*bins]')
	ax5.scatter(Vbr,null,c='red',s=30)
	ax5.scatter(Volt,Gain,c=c,s=30,label=str(i))
	ax5.plot(x,slo*x+inter,c=c)
	ax5.errorbar(Volt,Gain,c=c,yerr=Error,fmt='none')
	plt.legend(loc=2)




	fig1 = plt.figure(1)
	ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
	#ax5.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))
	#ax5.xaxis.set_minor_locator(ticker.MultipleLocator(1))
	ax1.grid(True)
	#ax5.set_ylim([0.0,0.04])
	ax1.set_xlim(60,80)
	ax1.set_xlabel('Bias Voltage [V]')
	ax1.set_ylabel('"Gain" [V*bins]')
	ax1.scatter(Volt,dDCR,c=c,s=30,label=str(i))
	#c = next(colors)
	ax1.scatter(Volt,cDCR,c=c,s=30,label=str(i))
	plt.legend(loc=2)











	plt.show()