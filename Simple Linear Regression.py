import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('main.csv')
X = dataset.iloc[:109, 2:3].values    # Rating
y = dataset.iloc[:109, 7:8].values   # Price
z = dataset.iloc[:109, 5:6].values  # Number of installations

#encoding the data if necessary
#from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
X_train,X_test ,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting price
y_pred = regressor.predict(X_test)


#plotting
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Rating vs Price (training set)')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.show()