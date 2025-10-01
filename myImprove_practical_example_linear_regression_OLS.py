# LINEAR REGRESSION MODEL - y = B0 + B1x1 + e (beta0, beta1, epsilon)
# y - dependent variable - what we predicted
# y_hat - predictable variable of y 
# x - independent variable - predictors who is affecting on y
# B0 - constant (intercept from axis x)
# B1 - coefficient of regression (slope) - determines the impact  x of y
# e - error of estimation (difference of point from line)

# %%
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from colorama import Fore, Style
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
#pd.options.display.max_rows = 999 # all rows to display
#pd.set_option('display.float_format', lambda x: '%.2f' % x) # zaokraglenie do 0.00

#print(Fore.CYAN + "-" * 100 + Style.RESET_ALL) 

raw_data = pd.read_csv('C:\\Users\\pbubula\\Desktop\\Data_science\\all_data\\1.04. Real-life example.csv')

# %%
raw_data.head()
# %%
raw_data.describe(include='all')
# %%
data = raw_data.drop(['Model'], axis=1)
data.describe(include='all')
# %%
data.isnull().sum() #how many missing values
# %%
data_no_mv = data.dropna(axis=0) #delete missing values
data_no_mv.describe(include='all')
sns.displot(data_no_mv['Price']) #logistic distribution because outliers
# %%
# REMOVING OUTLIERS OF NUMERICAL COLUMNS [PRICE, MILEAGE, ENGINEV, YEAR, RESET_INDEX]
q = data_no_mv['Price'].quantile(0.99) # it returns a value 99% quantile (1% removed)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')
sns.displot(data_1['Price'])
# %%
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
data_2.describe(include='all')
sns.displot(data_2['Mileage'])
# %%
data_3 = data_2[data_2['EngineV']<6.5]
data_3.describe(include='all')
sns.displot(data_3['EngineV'])
# %%
q = data_3['Year'].quantile(0.01)
print(q)
data_4 = data_3[data_3['Year']>q]
#data_4.describe(include='all')
sns.displot(data_4['Year'])
# %%
data_cleaned = data_4.reset_index(drop=True) 
data_cleaned.describe(include='all')
# %%
# CHECKING THE OLS ASSUMPTIONS (ORDINARY LEAST SQUARES)
# 1. LINEARITY ASSUMPTIONS - MUST BE
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()
# jak widać na wykresach nie mamy tu liniowej zależności (linie są zaokrąglone)
# %%
sns.displot(data_cleaned['Price'])
# jest tak dlatego, bo nasza zmienna niezależna Price nie jest w rozkładzie normalnym (jest w wykładniczym)
# %%
# można sobie z tym poradzić stosując log transformation (szczególnie przydatne, bo u nas jest wykładniczy rozkład Price)
# naturalny log dzięki numpy (w nawiasach co chcemy)
log_price = np.log(data_cleaned['Price'])
data_cleaned['Log_price'] = log_price

# Filtering 'Log_price' based on IQR
q1 = data_cleaned['Log_price'].quantile(0.25)
q3 = data_cleaned['Log_price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data_cleaned = data_cleaned[(data_cleaned['Log_price'] >= lower_bound) & (data_cleaned['Log_price'] <= upper_bound)]

# %%
# Filtering 'Mileage' based on Z-score
data_cleaned['Mileage_zscore'] = zscore(data_cleaned['Mileage'])
data_cleaned = data_cleaned[(data_cleaned['Mileage_zscore'] >= -3) & (data_cleaned['Mileage_zscore'] <= 3)]

# Final dataset after filtering
data_cleaned = data_cleaned.drop(['Mileage_zscore', 'Price'], axis=1)
final_rows = data_cleaned.shape[0]

# Checking percentage of removed data
removed_percentage = ((initial_rows - final_rows) / initial_rows) * 100
# %%
# Ponowna wizualizacja na zlogarytmowanej zmiennej zazleżnej Price -> Log_price (widać liniową zależność!!!)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Log_price'])
ax1.set_title('Log_price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Log_price'])
ax2.set_title('Log_price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Log_price'])
ax3.set_title('Log_price and Mileage')
plt.show()
# %%
# 2. NO ENDOGENEITY ASSUMPTIONS (ERROR, INDEPENDENT VARIABLES CORRELATED) - NOT VIOLATED IN OUR EXAMPLE - DISCUSS AFTER

# 3. NORMALITY AND HOMOSCEDASTICITY (NORMALITY, ZERO MEAN, HOMOSCEDASCTICITY) - LOG_TRANSFORMATION HELP US WITH IT

# 4. NO AUTOCORRELATION - OUR OBSERVATIONS IS NOT TIME SERIES DATA OR PANEL DATA, IT'S SNAPSHOT SALES CAR - LOGICALLY IS NOT A REASON TO BE DEPENDENT FROM EACH OTHER
    
# 5. MULTICOLLINEARITY
    # IT IS LOGICAL THAT 'YEAR' AND 'MILEAGE' WILL BE CORRELATED
    # SKLEARN DOES NOT HAVE A DEDICATED METHOD TO CHECK THIS ASSUMPTION (BUT STATSMODEL HAS) - VHF (VARIANCE INFLATION FACTOR) = 1/(1-R2)
    # It’s called the variance inflation factor because it estimates how much the variance of a coefficient is “inflated” because of linear dependence with other predictors. Thus, a VIF of 1.8 tells us that the variance (the square of the standard error) of a particular coefficient is 80% larger than it would be if that predictor was completely uncorrelated with all the other predictors.
    # VIF = [1, +inf) - no multicollinearity (it's a minimum value), 1 < VIF < 5 (perfectly okay), (from 5 to 10) discuss question < VIF (unacceptable)
    
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
print(vif)
# IN OUR EXAMPLE YEAR HAS ABOVE 10 - REMOVE THAT
data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
# %%
# CREATE DUMMY VARIABLES (ON CATEGORICAL DATA - COLUMNS)
# if we include a separate dummy variable for each category, we will introduce multicollinearity to the regression!
# if we have n categories for a feature, we have to create n-1 dummies
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
data_with_dummies.head()
# REARRANGE A BIT (log_price -> first_place)
data_with_dummies.columns.values
cols = ['Log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
data_preprocessed
# %%
# LINEAR REGRESSION MODEL 
# declare the inputs and the targets
targets = data_preprocessed['Log_price']
inputs = data_preprocessed.drop(['Log_price'], axis=1)
#print(inputs)
# scale our data
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
#print(inputs_scaled)
# note: it is not usually recommended to standarize dummy variables, but in AML we often use it because: note: scaling has no effect on the predictive power of dummies, once scaled, though, they lose all their dummy meaning

# %%
# train test split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
# create the regression
# in fact this is a log linear regression as the dependent variable is the log of Price
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
poly_reg = LinearRegression()
poly_reg.fit(x_train_poly, y_train)
poly_r2_train = r2_score(y_train, poly_reg.predict(x_train_poly))
poly_r2_test = r2_score(y_test, poly_reg.predict(x_test_poly))
# %%
# the closer our scatter plot to this line, the better the model, if y_hat=7 -> y_train=7 itd. best predictions
# our scatter plot is a good check to our regression (line 45 degree can be fit to this data)
plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
# %%
# another check for the regression is residuas plot (wykres reszt) = differences between the targets and the predictions
# from 3. Assumptions (normality and homoscedasticity) we know that would be normal distribution of our residuals
sns.displot(y_train - y_hat)
plt.title('Residuals PDF', size=18) # normally, but its tail on the left, its means: there are certain ovservations for which y_train - y_hat is much lower than the mean (a much higher price is predicted than is observed) it's our wskazówka
# %%
poly_r2_train
poly_r2_test#0.745 - relatively good result (our model it's explaining 75% of variability our data)
# %%
# FINDING THE WEIGHTS AND BIAS
print("Punkt przecięcia (odległość od osi x): ", reg.intercept_)
print("Współczynniki zmiennych x (dummies variables): ", reg.coef_)
# %%
# WEIGHTS INTERPRETATION - the bigger the weight, the bigger the impact:
# dummies are only compared to their respective benchmark (we not compare with numerical)
# I. Continues variables:
# 1. Positive weight shows that as a feature increases in value, so do the log_price and 'Price" respectively
# 2. Negative weight shows that as a feature increases in value, so do the log_price and 'Price" decrease

# II. Categorical (dummy) variables:
# 1. A positive weight shows that the respective category (brand) is more expensive than the benchmark (audi)
# 2. A negative weight shows that the respective category (brand) is less expensive than the benchmark (audi)
# since the dummy was scaled we don't absolute certains that mercedes is expensive, but it SEEMS TO BE
data_cleaned['Brand'].unique() # audi dropped one (nie ma go) - audi is the benchmark
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary
# this model is far from interpretable, dependent variable is a log and all features are standarize, include the dummies, but let's work with what we got

# %%
# TESTING
y_hat_test = reg.predict(x_test)
# the chart looks quite good, dane są bardziej skoncentrowane u góry (lepiej przewiduje wyższe ceny) wokół linii 45 degree, a gorzej na dole - bardziej rozproszone (niższe ceny)
plt.scatter(y_test, y_hat_test, alpha=0.2) # alpha - transparency
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
# %%
# dateframe performance - these are the predictions for the log prices - why we revert (np.exp)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
df_pf.head()
# %%
df_pf['Target'] = np.exp(y_test)
df_pf # a lot of missing values - why? below
# %%
y_test # we randomize our indexes, we must fixed it
# %%
y_test = y_test.reset_index(drop=True)
y_test.head()
# %%
df_pf['Target'] = np.exp(y_test)
df_pf

# %%
# examining the residuals is the same as examining the heart of the algorithm
df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
# we want to present absolute difference (whether an observation is off by +1% or -1% is mostly irrelevant)
df_pf['Difference_%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
# %%
df_pf.describe()
# min in difference% is 0.06 - in that case the output was spot on
# max is 512
# quantile (25, 50, 75) - for most of our predictions we got relatively close
# %%
#pd.options.display.max_rows = 999 # all rows to display
#pd.set_option('display.float_format', lambda x: '%.2f' % x) # zaokraglenie do 0.00
df_pf.sort_values(by=['Difference_%']).head(20)
# values at bottom is extremely differences (nie tak wiele ich jest powyzej 100), the observed prices (targets) are extremely low
# on average our model is pretty decent at predicting the price - values is always (negative) their predictions are higher than the targets
# wyjaśnieniem może być to, że brakuje nam ważnego czynnika który obniża cenę używanego samochodu
# it may be the model of the car, which we removed or it may be that the car was damaged in some way

# %%
# HOW TO IMPROVE OUR MODEL?
# 1. Use a different set of variables
# 2. Remove a bigger part of the outliers
# 3. Use different kinds od transformations

# Great model has created months, or years - we need to prepare our model with different situation and look what is the best 
# Feel free to dive into improving this model on your own!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!