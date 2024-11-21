import PyCO2SYS as pyco2
from multiprocessing import Pool
pyco2.hello()
import numpy as np
from scipy.io import loadmat
import pickle


data = loadmat("OrgAlk_get_phil.mat")
orgalk_2100 = data["orgalk_2100_avg"]
orgalk_2010 = data["orgalk_2010_avg"]

TA_2010 = data["TA_2010_avg"]
DIC_2010 =data["DIC_2010_avg"]
sal_2010 = data["sal_2010_avg"]
Temp_2010 = data["Temp_2010_avg"]
sil_2010 =data["sil_2010_avg"]
po4_2010 = data["po4_2010_avg"]

numRows, numCols, numTimes = orgalk_2010.shape
Korg1 = 2.68675 * 1E-6
Korg2 = 6.82858 * 1E-9
rat = 0.4035
import time
par1type = 1
par2type = 2
presin = 0
presout = 0
pHscale = 1
k1k2c = 10
kso4c = 1
T_orgalk = 0

def f(ele):
    row, data = ele
    par1, par2, sal,tempin,tempout,sil,po4 = data

    numCols,numTimes = par1.shape

    _ret = []

    for j in range(numCols):
        _a = []
        for t in range(numTimes):
            results = pyco2.CO2SYS(par1, par2, par1type, par2type, SAL=sal, TEMPIN=tempin, TEMPOUT=tempout,
                                   PRESIN=presin, PRESOUT=presout, SI=sil, PO4=po4, pHSCALEIN=pHscale,
                                   K1K2CONSTANTS=k1k2c, KSO4CONSTANTS=kso4c)
            _a.append(results)
        _ret.append(_a)
    return row, kso4c

if __name__ == '__main__':
    with Pool(5) as pool:
        result = pool.map(f, enumerate(zip(TA_2010, DIC_2010, sal_2010, Temp_2010, Temp_2010, sil_2010, po4_2010)))

    file = open("my_dump.txt", "wb")
    pickle.dump(result, file)
    file.close()
    print(result)