#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth




df= pd.read_csv('tidieddf.csv')

# Exclude stopwords 
df['description_cleaned'] =  df['description'].str.lower().astype(str)
## tokenize words (turn into list)

df["description_cleaned"] = df["description_cleaned"].str.replace('[^\w\s]','')

#df['description_cleaned'] = df['description_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df['test'] = df.apply(lambda row: nltk.word_tokenize(row['description_cleaned']), axis=1)

df['tokenized_sents'] = df['description_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# In[80]:


##train on 10000 samples to speed up
df= df.sample(n=10000, replace=False).reset_index()


# In[81]:


df['tokenized_sets'] = df['test']
df.head()


# In[82]:


##convert tokens to list and then numbers
processed_inputs = df['tokenized_sents']
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))


# In[83]:


input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)


# In[84]:


seq_length = 250
x_data = []
y_data = []


# In[85]:


# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])


# In[86]:


n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)


# In[87]:


X = np.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)


# In[88]:


y = np_utils.to_categorical(y_data)


# In[94]:


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

model.add(LSTM(256, return_sequences=True))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(LSTM(128))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dense(y.shape[1], activation='softmax'))


# In[95]:


model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[96]:


#It takes the model quite a while to train,
#and for this reason we'll save the weights and reload them when the training is finished.
#We'll set a checkpoint to save the weights to, and then make them the callbacks for our future model.

filepath = "model_weights_saved_valid.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]


# In[92]:


model.fit(X, y, epochs=30, batch_size=128, callbacks=desired_callbacks, validation_split=0.05)


# In[97]:


filename = "model_weights_saved_valid.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[98]:


#convert model output back to chars
num_to_char = dict((i, c) for i, c in enumerate(chars))


# In[100]:


num_to_char[3000]


# In[60]:


##random seed to set off model
num_to_char[2000]


# In[52]:


pattern = x_data[start]
pattern
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


# In[197]:


for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = num_to_char[index]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]



