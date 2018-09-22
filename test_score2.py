import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

colors = [0, 'r', 'b', 'g', 'm', 'y']

df = pd.read_csv('results/ceesa_crs.csv')
df.CEESA[df.CEESA > 15] = 15

xaxis = df['Cls'].copy()
xaxis[xaxis == 'non-habitable'] = 1
xaxis[xaxis == 'mesoplanet'] = 2
xaxis[xaxis == 'psychroplanet'] = 3
xaxis[xaxis == 'hypopsychroplanet'] = 4
xaxis[xaxis == 'thermoplanet'] = 5
print(xaxis)

plt.scatter(xaxis, df['CEESA'], s=1.5, color='b')
plt.plot([0, 6], [0.9202, 0.9202], 'k--')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()