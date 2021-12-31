import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


#-----------------------------------------------------Open the file
df = pd.read_csv("EVDatabase.csv")



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
df = pd.get_dummies(df, columns=['Drive_1'], prefix='', prefix_sep='')

print(df.info())
print(df.describe())





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

plt.show()






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

KiaFeatures = [110,8,190,700,208,900,5,1,0,0]
price = regressor.predict([KiaFeatures])[0]

print('The approximate price of the ne model is: {}'.format(price))



