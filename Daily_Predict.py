import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
state = ""
mode = 0
monthlyDict = {1:0, 2:31, 3:59, 4:90, 5:120, 6:151, 7:181, 8:212, 9:243, 10:273, 11:304, 12:334}
close = 'n'
while close == 'n' :
    mode = int(input("Select mode [1:Daily Predict, 2:Relation]: "))
    if mode == 1 :
        day_in_month = {1:"[1-31]", 2:"[1-28]", 3:"[1-31]", 4:"[1-30]", 5:"[1-31]", 6:"[1-30]", 7:"[1-31]", 8:"[1-31]",
        9:"[1-30]", 10:"[1-31]", 11:"[1-30]", 12:"[1-31]"}
        print("***** Madrid Temperature Prediction ******")
        print("Jan:1 Feb:2 Mar:3 Apr:4 May:5 Jun:6 Jul:7 Aug:8 Sep:9 Oct:10 Nov:11 Dec:12")
        monthy = int(input("Enter Month : "))
        
        day = int(input("Enter Date" + day_in_month[monthy] + " : "))
        winter = pd.read_csv("TEmperature in Madrid - Winter-Mean.csv")
        spring = pd.read_csv("TEmperature in Madrid - Spring-Mean.csv")
        summer = pd.read_csv("TEmperature in Madrid - Summer-Mean.csv")
        autumn = pd.read_csv("TEmperature in Madrid - Autumn-Mean.csv")
        day_predict = monthlyDict[monthy] +day
        if monthy == 12 :
            state = "winter"
            print("               ---------Begin to Winter---------")
            print("Covariant Matrix")
            winterCov = winter.cov()
            print(winterCov)
            print("\nCorrelation Matrix")
            winterCor = winter.corr(method="pearson")
            print(winter.corr(method="pearson"))
        elif monthy <= 2 :
            state = "winter"
            print("               ---------Winter Season-----------") 
            print("Covariant Matrix")
            winterCov = winter.cov()
            print(winterCov)
            print("\nCorrelation Matrix")
            winterCor = winter.corr(method="pearson")
            print(winter.corr(method="pearson"))
        elif monthy == 3 :
            state = "spring"
            print("               ---------Begin to Spring---------")
            print("Covariant Matrix")
            springCov = spring.cov()
            print(springCov)
            print("\nCorrelation Matrix")
            springCor = spring.corr(method="pearson")
            print(spring.corr(method="pearson"))
        elif monthy <= 5 :
            state = "spring"
            print("               ---------Spring Season-----------")
            print("Covariant Matrix")
            springCov = spring.cov()
            print(spring.cov())
            print("\nCorrelation Matrix")
            springCor = spring.corr(method="pearson")
            print(spring.corr(method="pearson"))
        elif monthy == 6 :
            state = "summer"
            print("               ---------Begin to Summer---------")
            print("Covariant Matrix")
            summerCov = summer.cov()
            print(summer.cov())
            print("\nCorrelation Matrix")
            summerCor = summer.corr(method="pearson")
            print(summer.corr(method="pearson"))
        elif monthy <= 8 :
            state = "summer"
            print("               ---------Summer Season---------")
            print("Covariant Matrix")
            summerCov = summer.cov()
            print(summer.cov())
            print("\nCorrelation Matrix")
            summerCor = summer.corr(method="pearson")
            print(summer.corr(method="pearson"))
        elif monthy == 9 :
            state = "autumn"
            print("               ---------Begin to Autumn----------")
            print("Covariant Matrix")
            autumnCov = autumn.cov()
            print(autumn.cov())
            print("\nCorrelation Matrix")
            autumnCor = autumn.corr(method="pearson")
            print(autumn.corr(method="pearson"))
        elif monthy <= 11 :
            state = "autumn"
            print("               ---------Autumn Season-----------")
            print("Covariant Matrix")
            autumnCov = autumn.cov()
            print(autumn.cov())
            print("\nCorrelation Matrix")
            autumnCor = autumn.corr(method="pearson")
            print(autumn.corr(method="pearson"))
        data = pd.read_csv("TEmperature in Madrid - 2019-2021-Mean.csv")

        x_col = 'number'
        High_col = 'HighMean'
        AV_col = 'AVMean'
        Low_col = 'LowMean'
        z_col = 'Date'
        a = data[x_col].values
        b1 = data[High_col].values
        b2 = data[AV_col].values
        b3 = data[Low_col].values
        c = data[z_col]
        if day_predict == 1: 
            X = a[day_predict - 1:day_predict + 6]
            y1 = b1[day_predict - 1:day_predict + 6]
            y2 = b2[day_predict - 1:day_predict + 6]
            y3 = b2[day_predict - 1:day_predict + 6]
            Z = c[day_predict - 1:day_predict + 6]
        elif day_predict >= 358 :
            X = a[day_predict - 7 : day_predict]
            y1 = b1[day_predict - 7 : day_predict]
            y2 = b2[day_predict - 7 : day_predict]
            y3 = b3[day_predict - 7 : day_predict]
            Z = c[day_predict - 7 : day_predict]
        else :
            X = a[day_predict - 1 : day_predict + 7]
            y1 = b1[day_predict - 1 : day_predict + 7]
            y2 = b2[day_predict - 1 : day_predict + 7]
            y3 = b3[day_predict - 1 : day_predict + 7]
            Z = c[day_predict - 1 : day_predict + 7]
        X = X.reshape(-1,1)
        y1 = y1.reshape(-1,1)
        y2 = y2.reshape(-1,1)
        y3 = y3.reshape(-1,1)
        #print(type(y[0]))
        poly_reg = PolynomialFeatures(degree=3)
        x_train_poly1 = poly_reg.fit_transform(X)
        x_train_poly2 = poly_reg.fit_transform(X)
        x_train_poly3 = poly_reg.fit_transform(X)

        lm1 = LinearRegression()
        lm1.fit(x_train_poly1,y1)
        lm2 = LinearRegression()
        lm2.fit(x_train_poly2,y2)
        lm3 = LinearRegression()
        lm3.fit(x_train_poly3,y3)
        print("                 --------- Prediction ---------")
        if day_predict >= 358 :
            print('High Temperature is ' + str(lm1.predict([x_train_poly1[-1]]))[2:-2] + ' Celsius ' + "R^2 = {0}".format(lm1.score(x_train_poly1, y1)))
            print('Average Temperature is ' + str(lm2.predict([x_train_poly2[-1]]))[2:-2] + ' Celsius ' + "R^2 = {0}".format(lm2.score(x_train_poly2, y2)))
            print('Low Temperature is ' + str(lm3.predict([x_train_poly3[-1]]))[2:-2] + ' Celsius' + "R^2 = {0}".format(lm3.score(x_train_poly3, y3)))
        else :
            print('High Temperature is ' + str(lm1.predict([x_train_poly1[0]]))[2:-2] + ' Celsius ' + "R^2 = {0}".format(lm1.score(x_train_poly1, y1)))
            print('Average Temperature is ' + str(lm2.predict([x_train_poly2[0]]))[2:-2] + ' Celsius ' + "R^2 = {0}".format(lm2.score(x_train_poly2, y2)))
            print('Low Temperature is ' + str(lm3.predict([x_train_poly3[0]]))[2:-2] + ' Celsius' + "R^2 = {0}".format(lm3.score(x_train_poly3, y3)))
        s = "" #คำบรรยายสภาพอากาศ
        if float(lm2.predict([x_train_poly2[0]])) > 25 :
            s += "Today the weather is hot to very hot."
        elif float(lm2.predict([x_train_poly2[0]])) > 15 :
            s += "Today the weather is pretty good.!"
        elif float(lm2.predict([x_train_poly2[0]])) >= 0 :
            s += "Today the weather is pretty cool"
        elif float(lm2.predict([x_train_poly2[0]])) < 0 :
            s += "Today's weather is very cold."
        #แสดงสภาพอากาศของวันนั้น
        print("->>>>             {0}                  ".format(s))

        #จุดเข้มคืออุณหภูมิจริง - จุดอ่อนคืออุณหภูมิ predict
        fig = plt.figure(figsize=(10,6))
        plt.scatter(Z, lm1.predict(x_train_poly1), color = 'red', alpha=.3)
        plt.scatter(Z, lm2.predict(x_train_poly2), color = 'Green', alpha=.3)
        plt.scatter(Z, lm3.predict(x_train_poly3), color = 'blue', alpha=.3)
        plt.scatter(Z, y1,color = 'red', alpha = 1)
        plt.scatter(Z, y2,color = 'green', alpha = 1)
        plt.scatter(Z, y3,color = 'blue', alpha = 1)
        plt.plot(Z, y1,color = 'red', alpha = 1)
        plt.plot(Z, y2,color = 'green', alpha = 1)
        plt.plot(Z, y3,color = 'blue', alpha = 1)
        plt.title('Polynomial Regression')
        plt.ylabel("Temperature in Celsius")
        plt.xlabel("Date")
        plt.show()
        close = input("Exit?[Y/n] : ")
    elif mode == 2 :
        season = int(input("Select Season[1:winter, 2:spring, 3:summer, 4:autumn] : "))
        seasonDict = {1:'Winter' , 2:'Spring', 3:'Summer', 4:'Autumn'}
        path = ""
        if season == 1 :
            path = "TEmperature in Madrid - Winter-Mean.csv"
        elif season == 2 :
            path = "TEmperature in Madrid - Spring-Mean.csv"
        elif season == 3 :
            path = "TEmperature in Madrid - Summer-Mean.csv"
        elif season == 4 :
            path = "TEmperature in Madrid - Autumn-Mean.csv"
        data = pd.read_csv(path)
        x_axis = input("Enter x-axis [lowT:Low Temperature, avT:Average Temperature, highT:High Temperature]\n\t     [lowD:Low Dew Point, avD:Average Dew Point, highD:High Dew Point] :")
        y_axis = input("Enter y-axis [lowT:Low Temperature, avT:Average Temperature, highT:High Temperature]\n\t     [lowD:Low Dew Point, avD:Average Dew Point, highD:High Dew Point] :")
        print("                 ------- {0} -------".format(seasonDict[season]))
        print("Covariant Matrix")
        print(data.cov())
        print("\nCorrelation Matrix")
        print(data.corr(method="pearson"))

        x_col = x_axis
        y_col = y_axis
        X = data[x_col].values
        y = data[y_col].values
        X = X.reshape(-1,1)
        y = y.reshape(-1,1)
        #print(type(y[0]))
        poly_reg = PolynomialFeatures(degree=1)
        x_train_poly = poly_reg.fit_transform(X)
        cova = data.cov()
        corre = data.corr(method="pearson")
        lm = LinearRegression()
        lm.fit(x_train_poly,y)
        #print("R^2 = {0}".format(lm.score(x_train_poly, y)))
        fig = plt.figure(figsize=(10,6))
        plt.scatter(X, y, color = 'Green', alpha=.3)
        #.scatter(X, lm.predict(poly_reg.fit_transform(X)), color = 'Blue', alpha=.3)
        plt.plot(X, lm.predict(poly_reg.fit_transform(X)),color = 'red', alpha = .5)
        plt.title('Linear Regression\n '+ 'Cov({0},{1}) = '.format(x_axis,y_axis) + str(cova[x_axis][y_axis]) + '\nCorr({0},{1}) = '.format(x_axis,y_axis) + str(corre[x_axis][y_axis]))
        #plt.text(3.5, 0 , 'Cov({0},{1}) = '.format(x_axis,y_axis) + str(cova[x_axis][y_axis]), fontsize = 10)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
        close = input("Exit?[Y/n] : ")
    if close == 'n':
        print()


