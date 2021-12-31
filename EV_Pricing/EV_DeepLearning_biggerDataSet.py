import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns   #to have pair plot
from sklearn.metrics import r2_score
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)



# -----------------------------------------------------------Open the file
df= pd.read_csv("EV_10000points.csv")
print(df.info())
print(df.describe())

# -----------------------------------------------------------Cleaning, encoding, converting

#Drop the rows with Nan value
df = df.dropna()

#conver the Drive column to dummies
df['Drive'] = df['Drive'].map({1:'Front Wheel Drive',2:'Rear Wheel Drive',3:'All Wheel Drive'})              
df = pd.get_dummies(df, columns=['Drive'], prefix='', prefix_sep='')
print(df.tail())   

print(df.info())
print(df.describe())



# -----------------------------------------------------------ploting all the variable againseach other! That's just amazing!
sns.pairplot(df[['PriceinSwitzerland', 'TopSpeed', 'Range', 'Acceleration','BatteryPower']], diag_kind='kde')
plt.show()




# -----------------------------------------------------------Split the data into train and test

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

sns.pairplot(train_dataset[['PriceinSwitzerland', 'TopSpeed', 'Range', 'Acceleration','BatteryPower']], diag_kind='kde')
#-----Split features from labels,

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('PriceinSwitzerland')  #'PriceinSwitzerland' is removed from train_features and added to train_labels
test_labels = test_features.pop('PriceinSwitzerland')

print(train_features.head())
print(train_labels.head())


# -----------------------------------------------------------Normalize

a = train_dataset.describe().transpose() #To see the variable statistics
print(a)
print(a[['mean', 'std']])

normalizer = preprocessing.Normalization(axis=-1)  #definition of normalization
normalizer.adapt(np.array(train_features))   #.adapt() it to the data:
print(normalizer.mean.numpy())


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
  
 


# -----------------------------------------------------------a function to create and compile model
def build_and_compile_model(norm):  
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ]) 
  model.compile(loss='MeanAbsolutePercentageError',
                optimizer=tf.keras.optimizers.Adam(0.1))
  return model




# -----------------------------------------------------------a function to plot model residual history
def plot_loss(history):
  f = plt.figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [PriceinSwitzerland]')
  plt.legend()
  plt.grid(True)
  return f

  
  
  
  
# -----------------------------------------------------------Fitting the model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=2, epochs=100)

f2 = plot_loss(history)





# -----------------------------------------------------------model evaluation and test
test_results = {}

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [PriceinSwitzerland]']).T)



test_predictions = dnn_model.predict(test_features).flatten()
print('Model R2 is: {}'.format(r2_score(test_labels, test_predictions)))

f3 = plt.figure(figsize=(8,6))
a = plt.axes(aspect='equal') #Plot data versus predictions
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [PriceinSwitzerland]')
plt.ylabel('Predictions [PriceinSwitzerland]')



f4 = plt.figure()
error = test_predictions - test_labels #Error histogram
plt.hist(error, bins=100)
plt.xlabel('Prediction Error [PriceinSwitzerland]')
_ = plt.ylabel('Count')


plt.show()

#--------------------------------------------------Kia motors approximate price
'''
KiaFeatures = [110,8,190,700,208,900,5,1,0,0]
price = dnn_model.predict([KiaFeatures]).flatten()

print('The approximate price of the ne model is: {}'.format(price))
'''

