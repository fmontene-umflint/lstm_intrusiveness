# Prep for the Docker container
# Update all packages
!pip install --upgrade pip
!pip install pandas
!pip install sklearn

# Load extensions and imports
# Load all external packages using pip before executing this in a Docker container
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Activation
from tensorflow.keras.layers import Embedding, Bidirectional, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn import preprocessing

# Retrieve preprocessed file
df_new = pd.read_csv('Clean-ContextualDataNormalized.csv', sep = '\t')

# Read preprocessed Descriptions data and merge
df_desc = pd.read_csv('CategoryStemming2Enhanced.txt', sep = '\t')

# Merge the two 
df = pd.merge(df_new,df_desc,left_on=['pkgname'], right_on=['pkgname'])

# Free memory up and keep only one dataframe
df_new = None
df_desc = None

# Turn Genr into numbers and normalize diving by 50

numGenre = {'educ' : 1,
            'person' : 2,
            'entertain' : 3,
            'lifestyl' : 4,
            'tool' : 5,
            'busi' : 6,
            'puzzl' : 7,
            'arcad' : 8,
            'casual' : 9,
            'music audio' : 10,
            'book refer' : 11,
            'travel local' : 12,
            'photographi' : 13,
            'product' : 14,
            'health fit' : 15,
            'sport' : 16,
            'action' : 17,
            'news magazin' : 18,
            'communic' : 19,
            'social' : 20,
            'financ' : 21,
            'simul' : 22,
            'adventur' : 23,
            'shop' : 24,
            'race' : 25,
            'medic' : 26,
            'map navig' : 27,
            'video player editor' : 28,
            'trivia' : 29,
            'casino' : 30,
            'board' : 31,
            'card' : 32,
            'strategi' : 33,
            'food drink' : 34,
            'weather' : 35,
            'word' : 36,
            'art design' : 37,
            'role play' : 38,
            'librari demo' : 39,
            'music' : 40,
            'comic' : 41,
            'beauti' : 42,
            'hous home' : 43,
            'auto vehicl' : 44,
            'event' : 45,
            'date' : 46,
            'parent' : 47}

df['n_genre'] = df['Genr'].map(lambda x: numGenre[x]/50)

# Data sample and hyperparameter settings
# Make choices for each run

# Change this to the VM folder where the data is stored
# and where TensorBoard logs will be created
data_folder='/tf/'
# Number of bins for histograms
bin_num = 20
# Set timestamp for logs
log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

#
# Tokenization settings
#
# If set to True, this paramater reads a tokenizer object from file
use_saved_tokenizer = False
# Name of the file to use to save/retrieve the tokenizer object
tokenizer_filename = 'tokenizer_hybrid'
# The following tokenizer parameters are only considered if 
# use_saved_tokenizer is set to False
# Max number of words considered for the NN. This parameter determines the size
# of the embedings layer, along with the dimensions below
vocab_size = 32768
# According to 
# https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
# the embedding dimensions should be vocab_size^(1/4), 
# but they say you can use anything. Most experts use a number
# between 50 and 1000. We started with 16 and went to 64.
# 64 provided good results, but showed overfitting
# Trial and error gave us 4 as the best number of dimensions for this problem
embedding_dim = 4

#
# Input sequence settings
#
# Minimum and maximum length of app descriptions fed to the NN
# Too few words may not provide enough information to make a decision
minimum_length = 70
# Notes: looking at the app data, 120 max words covers 43 percentile of apps
#        250 covers 87.5 percentile
max_length = 250

#
# Sampling settings
#
batch_size = 32768
# Percentile to use
# Only samples below this threshold will be used as 0's and
# samples above 1 - threshold will be used as 1's
percentile = 0.35

#
# Hyperparameters
#
# Recurrent Dropout is the Keras equivalent to Variational Dropout
# Setting this value to > 0 will disable cuDNN architecture
rec_dropout = 0.25 # Only used for LSTM branch
dropout = 0.3 # Only used for MLP branch
# Adam optimizer settings
# Adjust this accordingly to use a customized optimizer
use_custom_adam = False
# Customize learning rate
# Default rate for Adam is 1e-3
# Only used if use_custom_adam is True
custom_adam = Adam(learning_rate=1e-2)
# Number of epochs to run for each test
num_epochs = 160
# Percent of sample data to use for validation
val_ratio = .25
# Number of cells for MLP Dense layers
num_units = 16

# Tokenization and Sequencing
# ---

trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words= vocab_size, oov_token=oov_tok)
# Obtain tokenizer object
if use_saved_tokenizer:
    # Get object from previously saved file
    token_file = open(tokenizer_filename, 'rb')
    tokenizer = pickle.load(token_file)
    token_file.close()
else:
    # Create a new tokenizer
    tokenizer.fit_on_texts(df['Description'])
    # Save the tokenizer to a file for later use
    token_file = open(tokenizer_filename, 'wb')
    pickle.dump(tokenizer, token_file, protocol=pickle.HIGHEST_PROTOCOL)
    token_file.close()
    
# Before creating sequences, we are removing descriptions that are too short
# and could be causing problems with the NN
df = df.loc[df['Description'].map(
    lambda x: len(x.split())) >= minimum_length]

# Add Label column, to be used as target for training
df = df.assign(Label = 2)

# Find the percentile values to be used
# By default, this function skips null values
p2_threshold1 = df['permission_2'].quantile(percentile)
p2_threshold2 = df['permission_2'].quantile(1 - percentile)

print('Thresholds to use are ' +
    str(p2_threshold1) +
    ' and ' +
    str(p2_threshold2)
    )

# if the score is below the threshold1, label = 0, 
# if the score is equal or higher than threshold2, set the label = 1

#Setting the class label
column_score = 'Label'
df.loc[df['permission_2'] <= p2_threshold1, ['Label']] = 0
df.loc[df['permission_2'] >= p2_threshold2, ['Label']] = 1

# Remove items that we are not going to use
df = df.loc[df['Label'] < 2]
# Drop any NAs
df = df.dropna()
# Reset the index to avoid blanks
df = df.reset_index(drop=True)
# df

# MLP Branch
# This branch analyses the other characteristics of an app
# Rank of Dimensions importance, based on training accuracy:
# 
# Genre
# Android Minimum Version
# Download Count
# Age Rating
# Review Average
# Free <- Probably unimportant because there are so many free apps

# Assign different inputs to different variables and prepare
# category inputs as one-hot matrices
lb = preprocessing.LabelBinarizer()
# RevAvg is a normalized version of the review averages, goes in as-is
x_1 = df['RevAvg']
# free is a booloean indicator showing if the app is free, used as-is
x_2 = df['free']
# AgeRating needs one-hot encoding
lb.fit(np.array(list(str(x) for x in df.AgeRating)))
x_3 = lb.transform(np.array(list(str(x) for x in df.AgeRating)))
# d_count is a categorized measure of downloads. Needs one-hot encoding
lb.fit(np.array(list(str(x) for x in df.d_count)))
x_4 = lb.transform(np.array(list(str(x) for x in df.d_count)))
# NormVer also category, also one-hot
lb.fit(np.array(list(str(x) for x in df.NormVer)))
x_5 = lb.transform(np.array(list(str(x) for x in df.NormVer)))
# n_genre needs one-hot encoding
lb.fit(np.array(list(str(x) for x in df.n_genre)))
x_6 = lb.transform(np.array(list(str(x) for x in df.n_genre)))
# Putting it all together
x = np.hstack([np.swapaxes(np.vstack([x_1, x_2]),0,1), x_3, x_4, x_5, x_6])

# x = df[['RevAvg','free','AgeRating','d_count','NormVer','n_genre']]
y = np.array(list(x for x in df.Label))

mlp = Sequential()
mlp.add(Dense(num_units, activation = 'relu', input_shape = (x.shape[-1],)))
mlp.add(Dense(num_units, activation = 'relu'))
mlp.add(Dropout(dropout))

# LSTM Branch
# This branch does App description analysis

sequences = tokenizer.texts_to_sequences(df['Description'])
data = pad_sequences(sequences,
    maxlen=max_length,
    truncating=trunc_type,
    padding = 'post'
    )

# We tokenize the data on the full dataset to make it more robust,
# especially the word_index

lstm = Sequential()
lstm.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length = max_length,
        mask_zero = True
        )
    )
lstm.add(
    Bidirectional(
        LSTM(
            embedding_dim,
            recurrent_dropout = rec_dropout,
            dropout = dropout,
            return_sequences = True
            )
        )
    )
lstm.add(
    Bidirectional(
        LSTM(
            int(embedding_dim/2),
            recurrent_dropout = rec_dropout,
            dropout = dropout,
            return_sequences = True
            )
        )
    )
lstm.add(
    Bidirectional(
        LSTM(
            int(embedding_dim/4),
            recurrent_dropout = rec_dropout,
            dropout = dropout
            )
        )
    )


# Merging branches
# Performing regression on the combined output of both branches

# All the metrics to use for our network
metrics = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

# We create a new Model which will receive its inputs from the
# concatenated outputs of our branches above
merged_inputs = concatenate([mlp.output, lstm.output])

# We don't use the sequential model for this section

merged_output = Dense(4, activation="relu")(merged_inputs)
merged_output = Dense(1, activation="sigmoid")(merged_output)

model = tf.keras.Model(inputs=[mlp.input, lstm.input], outputs=merged_output)

if use_custom_adam:
    model.compile(loss='binary_crossentropy', optimizer=custom_adam, metrics=metrics)
else:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

model.summary()

# Data features
Some histograms to see what our data looks like

df.hist(column='Label',bins=bin_num)
df.hist(column='n_genre',bins=bin_num)
df.hist(column='NormVer',bins=bin_num)
df.hist(column='d_count',bins=bin_num)
df.hist(column='AgeRating',bins=bin_num)
df.hist(column='free',bins=bin_num)
df.hist(column='RevAvg',bins=bin_num)
df['Label'].value_counts()

logdir = os.path.join(data_folder + "logs", log_timestamp)
print('Issuing callback for TensorBoard to log directory: '+logdir)
tb_cbk = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                        histogram_freq = 1,
                                        update_freq = 'batch',
                                        profile_batch = 0)

## Fit the model
history= model.fit([x, data],
                   y,
                   validation_split = val_ratio,
                   epochs = num_epochs,
                   callbacks = [tb_cbk],
                   verbose = 2,
                   batch_size = batch_size)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'loss')
