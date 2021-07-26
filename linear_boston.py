from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model._glm.glm import _y_pred_deviance_derivative
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics

boston = load_boston() # load dataset

boston_df = pd.DataFrame(boston.data , columns=boston.feature_names)

boston_df['target'] = boston.target # add target data to dataframe



x = boston_df.drop('target' , axis=1 )
y = boston_df.target

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state=42) # separate train and test

reg = LinearRegression() # call Linear classifire

reg.fit(x_train,y_train) # fit train data to reg
y_pred = reg.predict(x_test) # make predict with test data

plt.scatter(y_test,y_pred)
plt.plot()
plt.xlabel('price')
plt.ylabel('predicted prices')
plt.show() # real data vs predicted data , it must be a line with slope = 1 => mse = 0

mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
print('mse result :' ,mse) # mean squared error result . as you saw in chart the result of prediction wasnt in line with 1 slope and mse result is > 0