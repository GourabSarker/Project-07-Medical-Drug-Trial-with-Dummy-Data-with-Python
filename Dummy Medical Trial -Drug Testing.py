#!/usr/bin/env python
# coding: utf-8

# ### Preprocess the Data

# In[106]:


import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


# In[107]:


train_labels =  []
train_samples = []


# ###  Example data:
# 
# * An experiemental drug was tested on individuals from ages 13 to 100.
# * The trial had 2100 participants. Half were under 65 years old, half were over 65 years old.
# * 95% of patientes 65 or older experienced side effects.
# * 95% of patients under 65 experienced no side effects.

# In[108]:


for i in range(50):
    #5% of younger with side effects
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    #5% of older with no side effects
    random_older=randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)


for i in range(1000):
    #95% of younger with no side effects
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    #95% of older with side effects
    random_older=randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


# In[109]:


for i in train_samples:
    print(i)


# In[110]:


for i in train_labels:
    print(i)


# In[111]:



train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
train_labels, train_samples=shuffle(train_labels, train_samples)


# In[112]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_samples=scaler.fit_transform(train_samples.reshape(-1,1))


# In[113]:


#Print Scaled Data

for i in scaled_train_samples:
    print(i)


# ### Simple Sequential Model

# In[114]:


import tensorflow
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# In[115]:


model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])


# In[116]:


model.summary()


# In[117]:


model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[118]:


model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.2, batch_size=10, epochs=30, shuffle=True, verbose=2)


# ### Preprocess Test- Data & Predict

# In[119]:



test_labels =  []
test_samples = []


# In[120]:


for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


# In[121]:



test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples= shuffle(test_labels, test_samples)


# In[122]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))


# In[123]:


predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)


# In[124]:


for i in predictions:
    print(i)


# In[125]:


rounded_predictions = np.argmax(predictions,axis=1)


# In[126]:


for i in rounded_predictions:
    print(i)


# In[127]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[128]:


cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)


# In[131]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[132]:


cm_plot_labels = ['no_side_effects','had_side_effects']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[134]:


396/420


# ### 1. Save Model & Load Model 

# ### Save Model

# In[135]:


import os.path
if os.path.isfile('medical_trial.h5') is False:
    model.save('medical_trial.h5')


# ### Load Model

# In[136]:


from tensorflow.keras.models import load_model
new_model = load_model('medical_trial.h5')


# In[137]:


new_model.summary()


# In[138]:


new_model.get_weights()


# In[139]:


new_model.optimizer


# In[ ]:




