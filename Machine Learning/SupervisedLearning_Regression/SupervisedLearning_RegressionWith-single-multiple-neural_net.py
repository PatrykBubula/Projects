# %% [markdown]
#### **IMPORT RELEVANT LIBRARIES**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression


# %% [markdown]
#### **IMPORT/PREPARE DATASET**

df = pd.read_csv('SeoulBikeData.csv', encoding='cp949').drop(['Date', 'Holiday', 'Seasons'], axis =1)
dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
df.columns = dataset_cols
df["functional"] = (df["functional"] == "Yes").astype(int) # Yes/No -> 1/0 with astype(int)
df = df[df["hour"] == 12] # filtering hour == 12 df structure and rewrite df for only this data
df = df.drop(["hour"], axis=1) # drop unnecessary column like hour


# %% [markdown]
#### **SCATTER PLOT** -> LOOK FOR ASPECTS [DISTRIBUTION]

for label in df.columns[1:]: # without first column 
    plt.scatter(df[label], df['bike_count'])
    plt.title(label)
    plt.ylabel('Bike Count at Noon')
    plt.xlabel(label)
    plt.show()


# %% [markdown]
#### **DROP BAD DATA**

df = df.drop(["wind", "visibility", "functional"], axis=1)

# %% [markdown]
#### **TRAIN/VALID/TEST DATASET**

train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# %% [markdown]
#### **FUNCTION GET_XY** - CHOOSE FEATURES AND RESHAPED

def get_xy(dataframe, y_label, x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None: # if undeclared (None) take all features without ylabel
        X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels) == 1: # if declared and = 1 feature reshape to (n, 1)
            X = dataframe[x_labels[0]].values.reshape(-1, 1)
        else:
            X = dataframe[x_labels].values # if more features take all as X and reshape (n, number_of_features)

    y = dataframe[y_label].values.reshape(-1, 1)  # y = always named y_label with shape (n, 1)
    data = np.hstack((X, y)) # combine data vertically

    return data, X, y
    
    
# %% [markdown]
#### **ASSIGN DATA TO VARIABLES**

_, X_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels=["temp"]) # underscore mean -> I don't need it, I want only X, y for function
_, X_val_temp, y_val_temp = get_xy(val, "bike_count", x_labels=["temp"])
_, X_test_temp, y_test_temp = get_xy(test, "bike_count", x_labels=["temp"])


# %% [markdown]
#### **CREATE MODEL FOR SINGLE LINEAR REGRESSION**

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
print(f"This is R^2 for multiple linear regression: {temp_reg.score(X_test_temp, y_test_temp)}") # about 0.4

# %% [markdown]
#### **SCATTER PLOT FOR OUR TRAIN DATA WITH REGRESSION LINE**

plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()

# %% [markdown]
#### **MULTIPLE LINEAR REGRESSION**

train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
_, X_train_all, y_train_all = get_xy(train, "bike_count", x_labels=df.columns[1:])
_, X_val_all, y_val_all = get_xy(val, "bike_count", x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, "bike_count", x_labels=df.columns[1:])

# %% [markdown]
#### **CREATE MODEL FOR MULTIPLE LINEAR REGRESSION**

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)
print(f"This is R^2 for multiple linear regression: {all_reg.score(X_test_all, y_test_all)}") # 0.53 improve about 0.15 


# %% [markdown]
#### **SCATTER PLOT FOR OUR TRAIN DATA WITH REGRESSION LINE**

# **Predicted vs. Actual in Multiple Regression**
# - We’re not visualizing a single “best-fit line” in feature space, as in simple linear regression.
# - Each point in the scatter represents (Actual Value, Predicted Value) for one observation.
# - If the model were perfect, all points would lie on the diagonal line (y = x), meaning Predicted = Actual.
#
# **How to Interpret the Chart?**
# - **Points near the diagonal**: Predictions are close to the true values (good fit).
# - **Points above the diagonal**: The model under-predicts (actual > predicted).
# - **Points below the diagonal**: The model over-predicts (actual < predicted).
#
# **Why No Single Best-Fit Line?**
# - In single-variable linear regression, we plot (X, y) and can draw a 2D line.
# - In multiple regression with several features, the “line” is a hyperplane in higher dimensions.
# - We can’t directly plot that hyperplane in 2D.
# - Therefore, the **Predicted vs. Actual** plot is a simpler 2D summary:
#   - It doesn’t show the full regression plane,
#   - But it does show how closely your model’s predictions match reality.


y_pred_lr = all_reg.predict(X_test_all)

min_val = min(y_test_all.min(), y_pred_lr.min())
max_val = max(y_test_all.max(), y_pred_lr.max())

plt.plot([min_val, max_val], [min_val, max_val], color='red', label='y = x')
plt.scatter(y_test_all, y_pred_lr, alpha=0.5)
plt.xlabel("Actual bike_count")
plt.ylabel("Predicted bike_count")
plt.title("Predicted vs. Actual (Multiple Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
#### **REGRESSION WITH NEURAL NETWORK**

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.legend()
  plt.grid(True)
  plt.show()
  
  
# %%
temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))


# %%
temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])


# %%
temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')


# %%
history = temp_nn_model.fit(
    X_train_temp.reshape(-1), y_train_temp,
    verbose=0,
    epochs=1000,
    validation_data=(X_val_temp, y_val_temp)
)


# %%
plot_loss(history)


# %%
plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_nn_model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()


# %% [markdown]
#### **NEURAL NETWORK**

temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')



# %%
history = nn_model.fit(
    X_train_temp, y_train_temp,
    validation_data=(X_val_temp, y_val_temp),
    verbose=0, epochs=100
)



# %%
plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, nn_model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()



# %%
all_normalizer = tf.keras.layers.Normalization(input_shape=(6,), axis=-1)
all_normalizer.adapt(X_train_all)



# %%
nn_model = tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')


# %%
history = nn_model.fit(
    X_train_all, y_train_all,
    validation_data=(X_val_all, y_val_all),
    verbose=0, epochs=100
)



# %% # calculate the MSE for both linear reg and nn
y_pred_lr = all_reg.predict(X_test_all)
y_pred_nn = nn_model.predict(X_test_all)


# %%
def MSE(y_pred, y_real):
  return (np.square(y_pred - y_real)).mean()


# %%
MSE(y_pred_lr, y_test_all)

# %%
MSE(y_pred_nn, y_test_all)

# %%
ax = plt.axes(aspect="equal")
plt.scatter(y_test_all, y_pred_lr, label="Lin Reg Preds")
plt.scatter(y_test_all, y_pred_nn, label="NN Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 1800]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
_ = plt.plot(lims, lims, c="red")
# %%
