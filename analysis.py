import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from quantile_regression.linear_snqr import linear_snqr



quantile_colors = ['r', 'g','b']

df = pd.read_csv('data/mitsubishi/rvr/2023-06-19 09.47.51.142040.csv')
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

##############################################################################3
# price_per_year_linear_snqr
##############################################################################3

X_og = np.expand_dims(np.array(df.age), 1)

X = np.array(df[['age', 'age_sq']])
y = np.array(df.price)

my_qr = linear_snqr()

taus = [0.1, 0.5,  0.9]
my_qr.fit(X, y, taus)

plt.scatter(X_og,y, label='sample')
est_quantiles = my_qr.predict(X)
ind = np.argsort(X[:,0])
for i, tau in enumerate(taus):

    plt.plot(X_og[ind,0], est_quantiles[ind,i], label="{:.0f}th Quantile".format(100*tau), c=quantile_colors[i])


plt.xlabel("age")
plt.ylabel("price")
plt.legend()
plt.title("Price per year linear_snqr")
plt.savefig("output/price_per_year_linear_snqr.png", dpi=300)
plt.clf()

# ##############################################################################3
# # price_per_kms_linear_snqr
# ##############################################################################3

X_og = np.expand_dims(np.array(df.kms), 1)
X_og_norm = np.expand_dims(np.array(df.kms_norm), 1)

X = np.array(df[['kms_norm', 'kms_norm_sq']])
y = np.array(df.price)

my_qr = linear_snqr()

taus = [0.1, 0.5,  0.9]
my_qr.fit(X, y, taus)

plt.scatter(X_og,y, label='sample')
est_quantiles = my_qr.predict(X)




ind = np.argsort(X[:,0])
for i, tau in enumerate(taus):

    plt.plot(X_og[ind,0], est_quantiles[ind,i], label="{:.0f}th Quantile".format(100*tau), c=quantile_colors[i])



plt.xlabel("kms")
plt.ylabel("price")
plt.legend()
plt.title("Price per kms linear_snqr")
plt.savefig("output/price_per_kms_linear_snqr.png", dpi=300)
plt.clf()
