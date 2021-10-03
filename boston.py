#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def deal_file(file):
    data_list_text = file.readlines()
    data_list = []
    for i in data_list_text:
        data_list.append(list(map(float, i.split(','))))
    data_list = np.asarray(data_list, float)
    return data_list


# In[3]:


data=deal_file(open('in.txt'))
label=deal_file(open('out.txt'))
test=deal_file(open('test.txt'))

testsplit=np.split(test,[0,13],axis=1)
testx=testsplit[1]
testy=testsplit[2]
print('prompt: data load finished')


# In[14]:


def scale(x):
    return preprocessing.StandardScaler().fit_transform(x)


# In[15]:


data=scale(data)
label=scale(label)
testx=scale(testx)
testy=scale(testy)


# In[16]:


model=tf.keras.Sequential([
    keras.layers.Dense(32,input_shape=([13]),activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(1)
])


# In[17]:


model.summary()


# In[18]:


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


# In[19]:


model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),loss='mse',metrics=['mae','mse'])
history=model.fit(
    data,label,epochs=200,
    validation_split = 0.2,
    verbose=0,
    callbacks=[early_stop,PrintDot()]
)


# In[20]:


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()
  plt.show()

plot_history(history)


# In[21]:


predict=model.predict(testx).flatten()


# In[31]:


plt.scatter(testy, predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[26]:


error = predict - testy
plt.hist(error)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


# In[1]:


get_ipython().system('which python3')


# In[ ]:




