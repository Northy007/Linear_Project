import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
day_predict = 1
res = []
x_axis = input("Enter x-axis [lowT:Low Temperature, avT:Average Temperature, highT:High Temperature]\n\t[lowT:Low Temperature, avT:Average Temperature, highT:High Temperature\] :")
y_axis = input("Enter x-axis [lowT:Low Temperature, avT:Average Temperature, highT:High Temperature]\n\t[lowT:Low Temperature, avT:Average Temperature, highT:High Temperature\] :")
while day_predict <= 8 :
    data = pd.read_csv("TEmperature in Madrid - 2019-2021-Mean.csv")
    x_col = 'AVMean'
    y_col = 'AVDMean'
    a = data[x_col].values
    b = data[y_col].values
    if day_predict == 1: 
        X = a[day_predict - 1:day_predict + 6]
        y = b[day_predict - 1:day_predict + 6]
    elif day_predict >= 358 :
        X = a[day_predict - 7 : day_predict]
        y = b[day_predict - 7 : day_predict]
    else :
        X = a[day_predict - 1 : day_predict + 7]
        y = b[day_predict - 1 : day_predict + 7]
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
    if day_predict >= 358 :
        res.append(str(lm.predict([x_train_poly[-1]])))
    else :
        res.append(str(lm.predict([x_train_poly[0]])))
    day_predict += 1

# l = []
# for i in res :
#     l.append(float(i[2:-2]))
# for j in l :
#     print(j)


fig = plt.figure(figsize=(10,6))
plt.scatter(X, y, color = 'Green', alpha=.3)
plt.scatter(X, lm.predict(poly_reg.fit_transform(X)), color = 'Blue', alpha=.3)
plt.plot(X, lm.predict(poly_reg.fit_transform(X)),color = 'red', alpha = .5)
plt.title('Polynomial Regression')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.show()
