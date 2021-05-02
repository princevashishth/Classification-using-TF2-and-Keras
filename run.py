import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('cancer.csv',header=None)
df.info()
df = df[~df[6].isin(['?'])] 
df = df.astype(float)
df.iloc[:,10].replace(2, 0,inplace=True)                 #changing from 2 to 0 in column 10
df.iloc[:,10].replace(4, 1,inplace=True)                 #changing from 4 to 1

scaled_df=df
names = df.columns[0:10]
scaler = MinMaxScaler()                                     # applying normalization
scaled_df = scaler.fit_transform(df.iloc[:,0:10])          #when we train the network, we will pick that column from the original df dataframe
scaled_df = pd.DataFrame(scaled_df, columns=names)


X_train=scaled_df.iloc[0:500,1:10].values
y_train=df.iloc[0:500,10:].values

X_test=scaled_df.iloc[501:683,1:10].values
y_test=df.iloc[501:683,10:].values

X_train.shape
y_train.shape

model = Sequential()


model.add(Dense(units=9,activation='relu'))

model.add(Dense(units=15,activation='relu'))

model.add(Dense(units=7,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

predictions = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test,predictions))
