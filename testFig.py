
from matplotlib import pyplot as plt
import os

strDataDir = 'modelfiles/'
strFig = strDataDir + 'figs/'

if not os.path.exists(strDataDir):
    os.makedirs(strDataDir)
if not os.path.exists(strFig):
    os.makedirs(strFig)
    
plt.plot(1,1)
plt.title('hellow world')

plt.savefig(strFig+'foo.png', bbox_inches='tight')

