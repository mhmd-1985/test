import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diamonds.csv")
d=data.head()
data.drop(columns = 'Unnamed: 0', axis = 1, inplace = True )
data.head()
data.info()
d=data.describe()
data = data.loc[(data[['x','y','z']]!=0).all(axis=1)]
d=data.describe()
data.hist(bins=50,figsize=(20,15))
plt.show()
sns.pairplot(data , diag_kind = 'kde')
plt.figure(figsize = (10,5))
sns.heatmap(data.corr(),annot = True , cmap = 'coolwarm' );
corr_mat = data.corr()
plt.figure(figsize = (10,5))
corr_mat['price'].sort_values(ascending = False).plot(kind = 'bar');
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories = [['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 
                                               ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                              ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
diamond_cat_encoded = ordinal_encoder.fit_transform(data[['cut', 'color', 'clarity']])
data[['cut','color','clarity']]=diamond_cat_encoded[:,0:3]
X=data.drop(['price'],axis=1).values

y=data['price'].values
###############################################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import the class from Sci-kit learn
from sklearn.linear_model import LinearRegression
# Create an object of LinearRegression model
reg_all = LinearRegression()
# Fit the model to X_train and y_train
reg_all.fit(X_train,y_train) 
# Make predictions
y_pred=reg_all.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,c='r')