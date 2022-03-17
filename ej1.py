from contextlib import redirect_stderr
from turtle import color, pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv ('Walmart.csv')
df = df.sort_values(by=['Weekly_Sales'])
df = df.sample(n = 500, random_state= 10)

x = df[["Unemployment"]]
y = df[["Weekly_Sales"]]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

linreg = LinearRegression()
linreg.fit(x_train,y_train)
print(linreg.score(x_test, y_test))

y_pred = linreg.predict(x_test)
plt.scatter(x, y)
plt.plot(x_test,y_pred, color = 'red')
plt.show()
