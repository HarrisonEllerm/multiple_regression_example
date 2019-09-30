"""
Start.py: Demonstration of how machine learning can be applied to simple data set.
"""

__author__ = "Harry Ellerm"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sb
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Read in data set
data_set = pd.read_csv('C:/Users/vx146jc/Desktop/Listings_Input.csv')

# 2. Clean up junk values
data_set = data_set.fillna(0)

# 3. Divide data into attributes and labels

x = data_set[['Host_Rating_0_to_5', 'Minimum_Nights', 'Number_Of_Reviews', 'Reviews_Per_Month',
              'Calculated_Host_Listings_Count',	'Availability_365']]

y = data_set['Price']


# 4. Check the average of the price column
# Draws a histogram and fits a kernel
# density estimate. Here we can see a
# significant positive skew.
plt.figure(figsize=(15, 10))
plt.tight_layout()
sb.distplot(data_set['Price'])
plt.show()

# 5. Split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=0)

# 6. Train model
rgr = LinearRegression()
rgr.fit(x_train, y_train)

# 7. Analyse Co-efficients

# Host_Rating_0_to_5                 1.234980
# Minimum_Nights                     0.261615
# Number_Of_Reviews                 -0.192490
# Reviews_Per_Month                 -7.574965
# Calculated_Host_Listings_Count     0.212486
# Availability_365                   0.121364

# Therefore for a one unit increase in host rating
# we expect to see an increase of 1.234980 in price,
# for a one unit increase in reviews per month we expect
# to see a decrease of 7.574965 in price etc.

cf_df = pd.DataFrame(data=rgr.coef_, index=x.columns, columns=['Coefficient'])
print(cf_df)

# 8. Perform prediction on test data
price_prediction = rgr.predict(x_test)

# 9. Check difference between actual and predicted
# Obviously due to the quality of the data the model is
# not very accurate
actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': price_prediction})
actual_vs_predicted = actual_vs_predicted.head(25)
actual_vs_predicted.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# 10. Evaluate performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, price_prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, price_prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, price_prediction)))

