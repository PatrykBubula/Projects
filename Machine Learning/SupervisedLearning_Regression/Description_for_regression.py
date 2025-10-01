# %% [markdown]
#### **Jak ulepszyć wydajność modeli regresyjnych?**

# **Wspólne strategie (preprocessing, radzenie sobie z outliers, walidacja krzyżowa)**

# 1. **Przetwarzanie danych (preprocessing)**:
#    - **Skalowanie** (np. StandardScaler, MinMaxScaler) może poprawić stabilność uczenia w modelach wrażliwych na zakres cech.
#    - **Czyszczenie danych** – usuwanie lub imputacja wartości odstających (outliers) i braków danych (missing values) zazwyczaj poprawia trafność modelu.
#    - **Feature Engineering** – tworzenie nowych cech (np. wielomiany, interakcje między zmiennymi) może znacząco zwiększyć moc modelu.
#      - **Przykład (PolynomialFeatures)**:
#        ```python
#        from sklearn.preprocessing import PolynomialFeatures
#        poly = PolynomialFeatures(degree=2, include_bias=False)
#        X_poly = poly.fit_transform(X)
#        ```

# 2. **Walidacja krzyżowa (cross validation)**:
#    - Metoda k-fold (np. cv=5) uwiarygadnia ocenę jakości i pomaga w doborze hiperparametrów.
#      - **Przykład:**
#        ```python
#        from sklearn.model_selection import cross_val_score
#        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
#        print(scores.mean())
#        ```

# 3. **Dobór metryki regresji**:
#    - Poza R² (coef. determinacji) rozważ **MSE**, **RMSE**, **MAE** – dają one cenną perspektywę na rozrzut błędów.
#    - Wybór metryki ma znaczenie przy optymalizacjach i porównywaniu różnych modeli.

# 4. **Zwiększenie zbioru treningowego**:
#    - Większa liczba próbek lub dodatkowe cechy (feature engineering) może poprawić wyniki i stabilność modelu.



## **Jak ulepszyć Linear Regression?**

# 1. **Feature Selection i transformacje**:
#    - Usuń nieskorelowane lub redundantne cechy, stosuj przekształcenia zmiennych (np. logarytmiczne) w razie nieliniowych zależności.
#    - **Przykład (Log transform)**:
#      ```python
#      import numpy as np
#      X_log = np.log1p(X)  # jeśli w danych są wartości dodatnie
#      ```

# 2. **Regularyzacja**:
#    - **Ridge (L2)** i **Lasso (L1)** chronią przed przeuczeniem, zwłaszcza gdy liczba cech jest duża.
#      - **Przykład (Ridge)**:
#        ```python
#        from sklearn.linear_model import Ridge
#        ridge_model = Ridge(alpha=1.0)
#        ridge_model.fit(X_train, y_train)
#        ```
#      - **Przykład (Lasso)**:
#        ```python
#        from sklearn.linear_model import Lasso
#        lasso_model = Lasso(alpha=0.01)
#        lasso_model.fit(X_train, y_train)
#        ```

# 3. **Sprawdzanie reszt (residuals)**:
#    - W standardowej regresji liniowej zakładamy homoskedastyczność (równy rozkład błędów).
#    - Warto wykonać wykres reszt, aby zweryfikować, czy nie występują wzorce wskazujące na naruszenie założeń modelu.



## **Jak ulepszyć modele drzew decyzyjnych (np. Decision Tree Regressor)?**

# 1. **Kontrola głębokości**:
#    - Ustaw parametry typu `max_depth`, `min_samples_split` lub `min_samples_leaf`, aby uniknąć przeuczenia.
#      - **Przykład**:
#        ```python
#        from sklearn.tree import DecisionTreeRegressor
#        dt_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
#        dt_model.fit(X_train, y_train)
#        ```

# 2. **Walidacja krzyżowa**:
#    - Drzewa decyzyjne są podatne na przeuczenie, więc regularne sprawdzanie jakości na zbiorze walidacyjnym (lub cross validation) jest kluczowe.

# 3. **Złożone modele zespołowe**:
#    - **Random Forest** lub **Gradient Boosting** często mają lepszą uogólnialność (generalization) niż pojedyncze drzewo.



## **Jak ulepszyć modele zespołowe (Random Forest, Gradient Boosting)?**

# 1. **Liczba estymatorów** (n_estimators):
#    - Więcej drzew zwykle poprawia stabilność, ale zwiększa czas trenowania.
#      - **Przykład (RandomForest)**:
#        ```python
#        from sklearn.ensemble import RandomForestRegressor
#        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)
#        rf_model.fit(X_train, y_train)
#        ```

# 2. **Regularyzacja w boosting**:
#    - W modelach typu XGBoost, LightGBM czy CatBoost dostrajaj learning rate, max_depth, min_child_weight, subsample itp.
#      - **Przykład (XGBoost)**:
#        ```python
#        import xgboost as xgb
#        xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
#        xgb_model.fit(X_train, y_train)
#        ```

# 3. **Feature importance**:
#    - W RandomForest czy XGBoost możesz sprawdzić, które cechy najbardziej wpływają na wynik i ewentualnie dokonać selekcji.


## **Jak ulepszyć sieć neuronową do regresji?**

# 1. **Architektura**:
#    - Liczba warstw (np. 2-3 ukryte) i neuronów (16, 32, 64 itp.) zależy od złożoności zadania.
#    - **Przykład**:
#      ```python
#      import tensorflow as tf
#      nn_model = tf.keras.Sequential([
#          tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
#          tf.keras.layers.Dense(32, activation='relu'),
#          tf.keras.layers.Dense(1)  # regresja -> brak sigmoid
#      ])
#      ```

# 2. **Funkcja straty i metryki**:
#    - Dla regresji: `mean_squared_error (MSE)` lub `mean_absolute_error (MAE)` jako loss.
#      ```python
#      nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#      ```
# 3. **Regularizacja**:
#    - **Dropout** (choć rzadziej stosowany niż w klasyfikacji), **Early Stopping**, **Batch Normalization** – by zapobiegać przeuczeniu.
# 4. **Hipertuning**:
#    - Zmieniaj liczbę epok, batch_size, learning rate i architekturę, obserwując MSE/MAE na zbiorze walidacyjnym.


## **Podsumowanie (Regresja)**:
# - **Linear Regression**: Warto rozważyć regularyzację (Ridge, Lasso), analizę reszt oraz transformacje cech.
# - **Trees / Ensemble**: Odpowiednio ustaw głębokość drzewa, liczbę estymatorów i rozważ techniki boosting.
# - **Neural Networks**: Dobrana liczba warstw/neuronów i skuteczne zarządzanie overfittingiem (dropout, early stopping) mają kluczowe znaczenie.
# - **Ogólne zasady** (preprocessing, walidacja krzyżowa, outlier handling) obowiązują w większości modeli.

# **Najważniejsze**: Dobór odpowiedniej metody i hiperparametrów zawsze zależy od danych, więc testowanie (np. walidacja krzyżowa) i ocena metrykami (MSE, R², MAE) są niezbędne.

# %%
