# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)

# All general imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer 

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Bidirectional, GlobalAveragePooling1D, GRU, GlobalMaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.layers import LSTM, GRU, Conv1D, MaxPool1D, Activation

from keras.models import Model, Sequential
from keras.layers.core import SpatialDropout1D

from keras.engine.topology import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Softmax
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K

from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import io, os, gc

#################### Importing ByteDance Datasets ####################
# Train set
train_df = pd.read_csv('cstance_train_full.csv')
print(train_df.columns)
train_df.head()

# Test set
test_df = pd.read_csv('cstance_test_full.csv')
print(test_df.columns)
test_df.head()

premise = ["chloroquine hydroxychloroquine are cure for the novel coronavirus"]
train_lst_1 = train_df['text'].tolist()
print(len(train_lst_1))
train_lst_1[:5]
uq_tr_1 = list(set(train_lst_1))
print(len(uq_tr_1))
train_merged = uq_tr_1 + premise
print('Train Length is', len(train_merged))
train_merged[:5]
test_lst_1 = test_df['text'].tolist()
uq_ts_1 = list(set(test_lst_1))
test_merged = uq_ts_1
print('Test merged', len(test_merged))
total_dataset = train_merged + test_merged
print('Dataset length is', len(total_dataset))

# Defining the tokenizer
def get_tokenizer(vocabulary_size):
  print('Training tokenizer...')
  tokenizer = Tokenizer(num_words= vocabulary_size)
  tweet_text = []
  print('Read {} Sentences'.format(len(total_dataset)))
  tokenizer.fit_on_texts(total_dataset)
  return tokenizer

# For getting the embedding matrix
def get_embeddings():
  print('Generating embeddings matrix...')
  embeddings_file = 'glove.6B.300d.txt'
  embeddings_index = dict()
  with open(embeddings_file, 'r', encoding="utf-8") as infile:
    for line in infile:
      values = line.split()
      word = values[0]
      vector = np.asarray(values[1:], "float32")
      embeddings_index[word] = vector
	# create a weight matrix for words in training docs
  vocabulary_size = len(embeddings_index)
  embeddinds_size = list(embeddings_index.values())[0].shape[0]
  print('Vocabulary = {}, embeddings = {}'.format(vocabulary_size, embeddinds_size))
  tokenizer = get_tokenizer(vocabulary_size)
  embedding_matrix = np.zeros((vocabulary_size, embeddinds_size))
  considered = 0
  total = len(tokenizer.word_index.items())
  for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
      print(word, index)
      continue
    else:
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        considered += 1
  print('Considered ', considered, 'Left ', total - considered)			
  return embedding_matrix, tokenizer

def get_data(tokenizer, MAX_LENGTH, input_df):
  print('Loading data')
  X1, X2, Y = [], [], []
  X2 = input_df['text'].tolist()
  length = len(X2)
  premise = "chloroquine hydroxychloroquine are cure for the novel coronavirus"
  X1 = [premise for i in range(length)]
  Y = input_df['stance'].tolist()
  
  len(X1) == len(X2) == len(Y)
  sequences_1 = tokenizer.texts_to_sequences(X1)
  sequences_2 = tokenizer.texts_to_sequences(X2)
	# for i, s in enumerate(sequences):
	# 	sequences[i] = sequences[i][-250:]
  X1 = pad_sequences(sequences_1, maxlen=MAX_LENGTH)
  X2 = pad_sequences(sequences_2, maxlen=MAX_LENGTH)
  new_Y = np.array(Y)
  return X1, X2, new_Y

embedding_matrix, tokenizer = get_embeddings()

MAX_LENGTH = 50
# read ml data
X1, X2, Y = get_data(tokenizer, MAX_LENGTH, train_df)

X1_test, X2_test, Y_test = get_data(tokenizer, MAX_LENGTH, test_df)

print(Y.shape)

print(type(X1))
X1.shape

y_train = keras.utils.to_categorical(Y)
print(y_train)
y_test = keras.utils.to_categorical(Y_test)
print(y_test)

from sklearn.model_selection import train_test_split
VALIDATION_RATIO = 0.1
RANDOM_STATE = 9527
x1_train, x1_val, \
x2_train, x2_val, \
y_train, y_val = \
    train_test_split(
        X1, X2, y_train, 
        test_size=VALIDATION_RATIO, 
        random_state=RANDOM_STATE
)

print("Training Set")
print("-" * 10)
print(f"x1_train: {x1_train.shape}")
print(f"x2_train: {x2_train.shape}")
print(f"y_train : {y_train.shape}")

print("-" * 10)
print(f"x1_val:   {x1_val.shape}")
print(f"x2_val:   {x2_val.shape}")
print(f"y_val :   {y_val.shape}")
print("-" * 10)
print("Test Set")

NUM_CLASSES = 3

MAX_SEQUENCE_LENGTH = 50

NUM_LSTM_UNITS = 128

MAX_NUM_WORDS = embedding_matrix.shape[0]

NUM_EMBEDDING_DIM = embedding_matrix.shape[1]

print('Getting Text CNN model...')
filter_sizes = [2, 3, 5]
num_filters = 128	#Hyperparameters 32,64,128; 0.2,0.3,0.4
drop = 0.4
top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')

embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM, weights = [embedding_matrix])
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)
reshape = Reshape((MAX_SEQUENCE_LENGTH, NUM_EMBEDDING_DIM, 1))(top_embedded)
reshape_1 = Reshape((MAX_SEQUENCE_LENGTH, NUM_EMBEDDING_DIM, 1))(bm_embedded)
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], NUM_EMBEDDING_DIM),  padding='valid', kernel_initializer='normal',  activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], NUM_EMBEDDING_DIM),  padding='valid', kernel_initializer='normal',  activation='relu')(reshape_1)
#conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),  padding='valid', kernel_initializer='normal', activation='relu')(reshape)
maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
#maxpool_2 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
predictions = Dense(units=NUM_CLASSES, activation='softmax')(dropout)

model = Model(
    inputs=[top_input, bm_input], 
    outputs=predictions)

from keras.optimizers import Adam
lr = 1e-3
opt = Adam(lr=lr, decay=lr/50)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

BATCH_SIZE = 512
NUM_EPOCHS = 50
stop = [EarlyStopping(monitor='val_loss', patience=0.001)]
history = model.fit(x=[x1_train, x2_train],
                    y=y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(
                      [x1_val, x2_val], 
                      y_val
                    ),
                    shuffle=True,
                    callbacks=stop,
          )

from sklearn import metrics
from sklearn.metrics import classification_report
predictions = model.predict(
    [X1_test, X2_test])

y_pred = [idx for idx in np.argmax(predictions, axis=1)]
print('CS classification accuracy is')
print(metrics.accuracy_score(Y_test, y_pred)*100)
print(classification_report(Y_test, y_pred, target_names = ['neutral', 'against', 'for']))

top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')

embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM, weights = [embedding_matrix])
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)

shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
bm_output = shared_lstm(bm_embedded)

merged = concatenate(
    [top_output, bm_output], 
    axis=-1)

dense =  Dense(
    units=NUM_CLASSES, 
    activation='softmax')
predictions = dense(merged)

model = Model(
    inputs=[top_input, bm_input], 
    outputs=predictions)

model.summary()

from keras.optimizers import Adam
lr = 1e-3
opt = Adam(lr=lr, decay=lr/50)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

BATCH_SIZE = 512
NUM_EPOCHS = 50
stop = [EarlyStopping(monitor='val_loss', patience=0.001)]
history = model.fit(x=[x1_train, x2_train],
                    y=y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(
                      [x1_val, x2_val], 
                      y_val
                    ),
                    shuffle=True,
                    callbacks=stop,
          )

from sklearn import metrics
from sklearn.metrics import classification_report
predictions = model.predict(
    [X1_test, X2_test])

y_pred = [idx for idx in np.argmax(predictions, axis=1)]
print('CS classification accuracy is')
print(metrics.accuracy_score(Y_test, y_pred)*100)
print(classification_report(Y_test, y_pred, target_names = ['neutral', 'against', 'for']))