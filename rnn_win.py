#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import MetaTrader5 as mt5
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


# In[ ]:


tf.test.is_built_with_cuda()


# In[ ]:


if(mt5.initialize()):
    print("Pai ta on!")
else:
    print("Cod de erro:", mt5.last_error())


# In[ ]:


winn = mt5.copy_rates_from("WIN$N", mt5.TIMEFRAME_M1,datetime.now(), 99999)
winn = pd.DataFrame(winn)
winn["time"] = pd.to_datetime(winn["time"], unit='s')
winn.drop('spread', axis=1, inplace=True)


# In[ ]:


wing = mt5.copy_rates_from("WING23", mt5.TIMEFRAME_M1,datetime.now(), 99999)
wing = pd.DataFrame(winz)
wing["time"] = pd.to_datetime(wing["time"], unit='s')
wing.drop('spread', axis=1, inplace=True)


# In[ ]:


winj = mt5.copy_rates_from("WINJ23", mt5.TIMEFRAME_M1,datetime.now(), 99999)
winj = pd.DataFrame(winj)
winj["time"] = pd.to_datetime(winj["time"], unit='s')
winj.drop('spread', axis=1, inplace=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
lag = 2


# In[ ]:


winn['hl'] = (winn['high'] + winn['low']) / 2
winn['mvhl'] = winn['hl'].rolling(lag).mean()
train = winn['mvhl'].dropna().values
train = train.reshape(-1, 1)
train = scaler.fit_transform(train)
train = train.flatten()


# In[ ]:


winn['hl'] = (winn['high'] + winn['low']) / 2
wing['mvhl'] = wing['hl'].rolling(lag).mean()
valid = wing['mvhl'].dropna().values
valid = valid.reshape(-1, 1)
valid = scaler.transform(valid)
valid = valid.flatten()


# In[ ]:


winj['hl'] = (winj['high'] + winj['low']) / 2
winj['mvhl'] = winj['hl'].rolling(lag).mean()
test = winj['mvhl'].dropna().values
test = test.reshape(-1, 1)
test = scaler.transform(test)
test = teste.flatten()


# In[ ]:


def seq2seq_window_dataset(series, window_size=5, batch_size=512):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


# In[ ]:


train_set = seq2seq_window_dataset(train, window_size, batch_size)
valid_set = seq2seq_window_dataset(valid, window_size, batch_size)


# In[ ]:


from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Dropout, Input, concatenate
from tensorflow.keras.models import Model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'keras.backend.clear_session()\ntf.random.set_seed(42)\nnp.random.seed(42)\n\nwindow_size = 32\nbatch_size = 128\nlearning_rate = 0.001\nl2_reg = 0.0001\n\ninput_lstm = Input(shape=[X_train_lstm.shape[1], X_train_lstm.shape[2]])\nlstm = LSTM(units=32, return_sequences=True, kernel_regularizer=keras.regularizers.L2(l2_reg))(input_lstm)\nlstm = Dropout(rate=dropout_rate)(lstm)\nlstm = LSTM(units=62, return_sequences=True, kernel_regularizer=keras.regularizers.L2(l2_reg))(lstm)\nlstm = Dropout(rate=dropout_rate)(lstm)\nlstm = LSTM(units=32, kernel_regularizer=keras.regularizers.L2(l2_reg))(lstm)\nlstm = Dropout(rate=dropout_rate)(lstm)\noutput_lstm = Dense(units=1)(lstm)\ninput_conv = Input(shape=[X_train_conv.shape[1], X_train_conv.shape[2]])\nconv = Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu")(input_conv)\nconv = Dropout(rate=dropout_rate)(conv)\nconv = Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu")(conv)\nconv = Dropout(rate=dropout_rate)(conv)\nconv = Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu")(conv)\nconv = Dropout(rate=dropout_rate)(conv)\noutput_conv = Conv1D(filters=1, kernel_size=1, strides=1)(conv)\ninput_gru = Input(shape=[X_train_gru.shape[1], X_train_gru.shape[2]])\ngru = GRU(units=32, return_sequences=True, kernel_regularizer=keras.regularizers.L2(l2_reg))(input_gru)\ngru = Dropout(rate=dropout_rate)(gru)\ngru = GRU(units=64, return_sequences=True, kernel_regularizer=keras.regularizers.L2(l2_reg))(gru)\ngru = Dropout(rate=dropout_rate)(gru)\ngru = GRU(units=32, kernel_regularizer=keras.regularizers.L2(l2_reg))(gru)\ngru = Dropout(rate=dropout_rate)(gru)\noutput_gru = Dense(units=1)(gru)\nconcat = concatenate([output_lstm, output_conv, output_gru])\noutput = Dense(units=1)(concat)\n\n\nmodel = Model(inputs=[input_lstm, input_conv, input_gru], outputs=output)\n\n\noptimizer = keras.optimizers.Adam(learning_rate=learning_rate)\n\n\nmodel.compile(loss=keras.losses.Huber(),\n              optimizer=optimizer,\n              metrics=["mse"])\n\nmodel_checkpoint = keras.callbacks.ModelCheckpoint(\n    "my_checkpoint.h5", save_best_only=True)\nearly_stopping = keras.callbacks.EarlyStopping(patience=50)\n\nmodel.fit(train_set, epochs=500,\n          validation_data=valid_set,\n          callbacks=[early_stopping, model_checkpoint])\n')


# In[ ]:


model = keras.models.load_model("my_checkpoint.h5")


# In[ ]:


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# In[ ]:


forecast = model_forecast(model, test[..., np.newaxis], window_size)


# In[ ]:


forecast = forecast[window_size - window_size:-1, -1, 0]


# In[ ]:


plt.plot(test[:100])
plt.plot(forecast[:100])

