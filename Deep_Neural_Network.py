import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns   #to have pair plot


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)



# -----Get the data from the archive
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

   
   
# The unknown values

print(dataset.isna().sum())
dataset = dataset.dropna()


# -----Convert the origin column into US, EUROPE and JAPAN columns
                    
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})              
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())              


# -----Split the data into train and test

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# -----ploting all the variable againseach other! That's just amazing!

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')



#-----Split features from labels,

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')  #'MPG' is removed from train_features and added to train_labels
test_labels = test_features.pop('MPG')

print(train_features.head())
print(train_labels.head())


# -----Normalize

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
  
  
# -----Predicting MPG out of Horsepower

horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)



def build_and_compile_model(norm):  # a function to create layers and compile model
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ]) 
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

print(dnn_horsepower_model.summary())


    
history = dnn_horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)



hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())



def plot_loss(history):
  f = plt.figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  return f

  
f = plot_loss(history) #Plot the convergance history




# -----Plot the data and fitted curve

test_results = {}

test_results['horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

def plot_horsepower(x, y):
  f = plt.figure()
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
 
#print(dnn_horsepower_model.layers[1].weights)
f = plot_horsepower(x,y)



# -----Fitting based on full data

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

f2 = plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)


test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal') #Plot data versus predictions
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)



f3 = plt.figure()
error = test_predictions - test_labels #Error histogram
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


# -----Save and reload the model, it should give identical results
dnn_model.save('dnn_model')


reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

plt.show()



