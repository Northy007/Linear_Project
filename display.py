import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("TEmperature in Madrid - 2019-2021-Mean.csv")
x_col = 'number'
y_col = 'AVMean'
a = data[x_col].values
b = data[y_col].values

X = a
y = b
X = X.reshape(-1,1)
y = y.reshape(-1,1)
#print(type(y[0]))
poly_reg = PolynomialFeatures(degree=3)
x_train_poly = poly_reg.fit_transform(X)
x_test_poly = poly_reg.fit_transform(X)

lm = LinearRegression()
lm.fit(x_train_poly,y)
#print("R^2 = {0}".format(lm.score(x_train_poly, y)))
lm.coef_
lm.intercept_
lm.score(x_test_poly, y)
lm.predict([x_train_poly[0]])

fig = plt.figure(figsize=(10,6))
plt.scatter(X, data['HighMean'], color = 'red', alpha=.3)
plt.scatter(X, data['LowMean'], color = 'blue', alpha=.3)
plt.scatter(X, data['AVMean'], color = 'Green', alpha=.3)
#plt.scatter(X, y, color = 'Green', alpha=.3)
#plt.plot(X, lm.predict(poly_reg.fit_transform(X)),color = 'red', alpha = .5)
plt.title('Polynomial Regression')
plt.ylabel("Temperatuure in Celsius")
plt.xlabel("Number")
plt.show()

