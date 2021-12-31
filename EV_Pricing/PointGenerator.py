import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import random



#https://www.analyticsvidhya.com/blog/2021/09/data-analysis-and-price-prediction-of-electric-vehicles/

#Open the file
df= pd.read_csv("EVDatabase.csv")
print(df.info())
print(df.describe())

#-----------------------------------------------------Cleaning, encoding, converting

#Drop the rows with Nan value
df = df.dropna()
print(len(df))
#clean some columns and convert them to numbers
df['PriceinUK'] = df['PriceinUK'].str.replace('£','').str.replace(',','').astype(int)
df['PriceinGermany'] = df['PriceinGermany'].str.replace('€','').str.replace(',','').astype(int)
df['FastChargeSpeed'] = df['FastChargeSpeed'].str.replace(' km/h','').str.replace('-','0').astype(int)
df['Efficiency'] = df['Efficiency'].str.replace(' Wh/km','').astype(int)
df['Range'] = df['Range'].str.replace(' km','').astype(int)
df['TopSpeed'] = df['TopSpeed'].str.replace(' km/h','').astype(int)
df['Acceleration'] = df['Acceleration'].str.replace(' sec','').astype(float)
df['BatteryPower'] = df['BatteryPower'].str.replace('Battery Electric Vehicle \| ','').str.replace(' kWh','').str.replace('      ','').astype(float)
df['Brand'] = df['Name'].str.split(' ', 1, expand=True)[0]

#conver the currencies
PoundToEuro = 1.17
EuroToCHF = 1.04
df.loc[df['PriceinGermany']==0,'PriceinGermany'] = df.loc[df['PriceinGermany']==0,'PriceinUK']*PoundToEuro
df['PriceinSwitzerland'] = df['PriceinGermany']*EuroToCHF

#convert the text 
df['Drive_1'] = df['Drive'].map({'Front Wheel Drive':1,'Rear Wheel Drive':2,'All Wheel Drive':3})              


print(df.info())
print(df.describe())
print(df.columns)


#--------------------------------------------------Visualization

#Correlation plot
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)

#count the number of products per brand 
plt.figure(figsize=(8,6))
sns.countplot(x = 'Brand', data = df)
plt.xticks(rotation=90)

plt.figure(figsize=(8,6))
sns.countplot(x = 'Drive', data = df)

sns.relplot(x="TopSpeed", y="Range",height=6, hue="Drive",data=df)

#plt.show()


#--------------------------------------------------X,y

X = np.array(df.drop(['Name', 'PriceinGermany','PriceinUK','PriceinSwitzerland','Drive','Brand'], axis=1))
y = np.array(df['PriceinUK'])



#--------------------------------------------------train and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


#--------------------------------------------------Model training
regressor = RandomForestRegressor(n_estimators = 400, random_state = 0)
regressor.fit(X_train, y_train)


#--------------------------------------------------Model test
y_pred= regressor.predict(X_test)

print('Model R2 is: {}'.format(r2_score(y_test, y_pred)))

#--------------------------------------------------Kia motors approximate price
KiaFeatures = [110,8,190,700,208,900,5,1]
price = regressor.predict([KiaFeatures])[0]

print('The approximate price of the ne model is: {}'.format(price))

NumPoints = 10000

df2 = pd.DataFrame()
df2['BatteryPower'] = abs(np.random.normal(68, 30, NumPoints)).astype(int)
df2['Acceleration'] = abs(np.random.normal(7, 3, NumPoints)).round(1)
df2['TopSpeed'] = abs(np.random.normal(184, 44, NumPoints)).astype(int)
df2['Range'] = abs(np.random.normal(354, 125, NumPoints)).astype(int)
df2['Efficiency'] = abs(np.random.normal(191, 29, NumPoints)).astype(int)
df2['FastChargeSpeed'] = abs(np.random.normal(519, 265, NumPoints)).astype(int)
df2['NumberofSeats'] = [random.choice([1,2,3,4,5]) for i in range(NumPoints)]
df2['Drive'] = [random.choice([1,2,3]) for i in range(NumPoints)]
df2['PriceinSwitzerland'] = regressor.predict(df2.values.tolist())

print(df2)
df2.to_csv('EV_10000points.csv')
 


