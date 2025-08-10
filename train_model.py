import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")
X = df.iloc[:, 3:13].drop(df.columns[4:7], axis=1)
y = df.iloc[:, 13]

# Train - Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Build model
classifier = Sequential()
classifier.add(Dense(units=11, activation='relu'))
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dense(units=5, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    verbose=1,
    restore_best_weights=True
)
# Train
classifier.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=50, verbose=1)

# Save model and scaler
classifier.save("model.h5")
pickle.dump(sc, open("scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully.")
