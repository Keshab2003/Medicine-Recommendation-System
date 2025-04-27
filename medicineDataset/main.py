import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model

diabetes = datasets.load_diabetes()
print(diabetes.keys())
# now this says take all rows from 3rd column and return in a 2D array having one column
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_x_train = diabetes_X[:-30]
diabetes_x_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_pridicted = model.predict(diabetes_x_test)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_pridicted, color='blue')
plt.show()
