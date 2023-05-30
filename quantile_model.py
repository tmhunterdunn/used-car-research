import pandas as pd
import numpy as np

from quantile_regression.linear_sqr import linear_sqr 
from quantile_regression.linear_snqr import linear_snqr
from quantile_regression.kernel_aqr import kernel_aqr 
from quantile_regression.kernel_snqr import kernel_snqr
from quantile_regression.kernel_sqr import kernel_sqr



quantile_colors = ['r', 'g','b']

df = pd.read_csv('data/subaru/outback/2023-05-04 19.36.36.898638.csv')
df.year = df.year.astype(float)
df['age'] = df.year.max() - df.year
df['age_sq'] = df.age**2
df['year_norm'] =  (df.year - df.year.min()) / 100
df['year_norm_sq'] = df.year_norm ** 2
df['kms_norm'] = (df.kms -df.kms.min()) / 1000
df['kms_norm_sq'] = df.kms_norm ** 2

df = df.drop(df[df.kms > 400000].index)
df = df.drop(df[df.year < 2000].index)
df = df.drop(df[df.price > 50000].index)



X_og = np.array(df[['kms', 'age']])

X = np.array(df[['kms_norm', 'kms_norm_sq', 'age', 'age_sq']])
y = np.array(df.price)

my_qr = linear_snqr()

taus = [0.1, 0.5, 0.9]
my_qr.fit(X, y, taus)
est_quantiles = my_qr.predict(X)

for i, tau in enumerate(taus):

    df["{:.0f}th Quantile".format(100*tau)] = est_quantiles[:,i]



df['cheap'] = df['price'] < df['10th Quantile']

df['expensive'] = df['price'] > df['90th Quantile']

df.to_csv('data-w-quantiles.csv')