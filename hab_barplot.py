import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc('xtick',labelsize=2)
plt.rc('ytick',labelsize=8)

df = pd.read_csv('results/ceesa_crs.csv')
df.CEESA[df.CEESA > 15] = 15

barlist = plt.bar(
    np.arange(len(df['Name'])), df['CEESA'])
for i, bar in enumerate(barlist):
    if df.Cls[i] == 'non-habitable':
        bar.set_color('r')
    elif df.Cls[i] == 'mesoplanet':
        bar.set_color('b')
    elif df.Cls[i] == 'psychroplanet':
        bar.set_color('g')
    elif df.Cls[i] == 'hypopsychroplanet':
        bar.set_color('m')
    elif df.Cls[i] == 'thermoplanet':
        bar.set_color('y')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()