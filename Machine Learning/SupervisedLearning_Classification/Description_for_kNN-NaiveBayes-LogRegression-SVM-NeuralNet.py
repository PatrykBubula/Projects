# %% [markdown]
#### **Jak ulepszyć wydajność modeli kNN, Naive Bayes, Logistic Regression i SVM?**

# **Wspólne strategie (preprocessing, balansowanie klas, walidacja krzyżowa)**

# 1. **Przetwarzanie danych (preprocessing)**:
#    - Skalowanie (np. StandardScaler, MinMaxScaler) i oczyszczanie danych pomaga uniknąć dominacji cech o dużych wartościach.
#      - **Przykład (StandardScaler):**
#        ```python
#        from sklearn.preprocessing import StandardScaler
#        scaler = StandardScaler()
#        X_scaled = scaler.fit_transform(X)
#        ```
#    - Redukcja wymiarów (np. PCA) lub selekcja cech (usunięcie skorelowanych atrybutów) może poprawić wyniki i przyspieszyć uczenie.
#      - **Przykład (PCA):**
#        ```python
#        from sklearn.decomposition import PCA
#        pca = PCA(n_components=5)
#        X_pca = pca.fit_transform(X_scaled)
#        ```

# 2. **Balansowanie klas**:
#    - Przy wyraźnej nierównowadze (np. klasa 1 jest w mniejszości) rozważ oversampling (RandomOverSampler) lub SMOTE.
#      - **Przykład (SMOTE):**
#        ```python
#        from imblearn.over_sampling import SMOTE
#        smote = SMOTE()
#        X_resampled, y_resampled = smote.fit_resample(X, y)
#        ```

# 3. **Walidacja krzyżowa (cross validation)**:
#    - Metoda k-fold (np. cv=5) uwiarygadnia ocenę jakości i pomaga w doborze hiperparametrów.
#      - **Przykład:**
#        ```python
#        from sklearn.model_selection import cross_val_score
#        scores = cross_val_score(model, X, y, cv=5)
#        print(scores.mean())
#        ```

# 4. **Zwiększenie zbioru treningowego**:
#    - Większa liczba próbek lub dodatkowe cechy (feature engineering) może znacznie poprawić wyniki.






# %% [markdown]
#### **Jak ulepszyć k-Nearest Neighbors (kNN)**

# 1. **Liczba sąsiadów (n_neighbors)**:
#    - Zbyt mała (np. k=1) -> ryzyko przeuczenia, zbyt duża (np. k=20) -> nadmierne uśrednianie.
#      - **Przykład:**
#        ```python
#        from sklearn.neighbors import KNeighborsClassifier
#        knn_model = KNeighborsClassifier(n_neighbors=5)
#        ```

# 2. **Metryka odległości (metric)**:
#    - Testuj różne metryki: 'euclidean', 'manhattan', 'minkowski' (z różnym p).
#      - **Przykład:**
#        ```python
#        knn_model = KNeighborsClassifier(metric='manhattan')
#        ```

# 3. **Waga sąsiadów (weights)**:
#    - Użycie `weights='distance'` pozwala bliższym sąsiadom mieć większy wpływ.
#      - **Przykład:**
#        ```python
#        knn_model = KNeighborsClassifier(weights='distance')
#        ```








# %% [markdown]
#### **Jak ulepszyć Naive Bayes**

# 1. **Dobór wariantu Naive Bayes**:
#    - **GaussianNB** dla ciągłych cech ~ normalnych
#    - **MultinomialNB** dla danych dyskretnych (np. tekst)
#    - **BernoulliNB** dla danych binarnych
#      - **Przykład:**
#        ```python
#        from sklearn.naive_bayes import MultinomialNB
#        nb_model = MultinomialNB()
#        ```

# 2. **Rozkład danych**:
#    - W GaussianNB ważne, aby cechy były zbliżone do normalnych (ew. transformacje, np. logarytmiczna).

# 3. **Selekcja cech**:
#    - Założenie warunkowej niezależności cech sprawia, że bardzo skorelowane cechy mogą obniżać trafność.











# %% [markdown]
#### **Jak ulepszyć Logistic Regression**

# 1. **Parametr `C` (regularyzacja)**:
#    - Niskie C -> silna regularyzacja, mniejsze ryzyko przeuczenia.
#      - **Przykład:**
#        ```python
#        from sklearn.linear_model import LogisticRegression
#        log_model = LogisticRegression(C=0.1)
#        ```

# 2. **Rodzaj regularizacji (`penalty`)**:
#    - 'l2' (domyślnie) vs 'l1' (może wyzerować współczynniki -> selekcja cech).
#      - **Przykład:**
#        ```python
#        log_model = LogisticRegression(penalty='l1', solver='liblinear')
#        ```

# 3. **Dobór solvera**:
#    - 'lbfgs', 'liblinear', 'saga' - różnią się obsługą typów regularizacji i skalą problemu.









# %% [markdown]
#### **Jak ulepszyć Support Vector Machine (SVM)**

# 1. **Wybór kernela (`kernel`)**:
#    - 'linear' (gdy dane są mniej więcej liniowo separowalne),
#    - 'rbf' (uniwersalny, domyślny),
#    - 'poly' (dla zależności wielomianowych).
#      - **Przykład:**
#        ```python
#        from sklearn.svm import SVC
#        svm_model = SVC(kernel='rbf')
#        ```

# 2. **Parametr `C`**:
#    - Kontroluje kompromis między szerokim marginesem a liczbą błędnych klasyfikacji (wyższe C -> węższy margines).

# 3. **Parametr `gamma`** (dla 'rbf'/'poly'):
#    - Reguluje wpływ pojedynczych obserwacji (overfitting przy zbyt wysokiej wartości).
#      - **Przykład (GridSearch):**
#        ```python
#        from sklearn.model_selection import GridSearchCV
#        param_grid = {
#          'C': [0.1, 1, 10],
#          'gamma': [0.01, 0.1, 1],
#          'kernel': ['rbf', 'poly']
#        }
#        grid = GridSearchCV(SVC(), param_grid, cv=5)
#        grid.fit(X_train, y_train)
#        print(grid.best_params_, grid.best_score_)
#        ```








# %% [markdown]
#### **Jak ulepszyć wydajność sieci neuronowej (Neural Network)?**

# 1. **Struktura sieci (architektura)**:
#    - Liczba warstw i liczba neuronów (np. 2–3 warstwy ukryte, po kilkadziesiąt/kilkaset neuronów) zależy od złożoności problemu.
#      - **Przykład**:
#        ```python
#        import tensorflow as tf
#        model = tf.keras.Sequential([
#            tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
#            tf.keras.layers.Dropout(0.2),
#            tf.keras.layers.Dense(32, activation='relu'),
#            tf.keras.layers.Dropout(0.2),
#            tf.keras.layers.Dense(1, activation='sigmoid')
#        ])
#        ```
#    - Większa liczba warstw/neuronów -> większa moc obliczeniowa, ale i wyższe ryzyko przeuczenia (overfitting).

# 2. **Parametry uczenia (hyperparametry)**:
#    - **Learning rate (lr)**: Zbyt wysoki -> model może „skakać” po funkcji błędu, za niski -> powolna konwergencja.
#    - **Batch size**: Decyduje o tym, ile przykładów sieć przetwarza naraz. Mniejsze batch_size => bardziej hałaśliwe, ale szybsze aktualizacje.
#    - **Liczba epok (epochs)**: Za mało -> model niedouczony, za dużo -> ryzyko przeuczenia.
#      - **Przykład (Keras fit)**:
#        ```python
#        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
#                      loss='binary_crossentropy',
#                      metrics=['accuracy'])
#        history = model.fit(X_train, y_train,
#                            epochs=10,
#                            batch_size=64,
#                            validation_split=0.2,
#                            verbose=1)
#        ```

# 3. **Regularizacja**:
#    - **Dropout** (np. 0.2–0.5) losowo wyłącza część neuronów i zapobiega przeuczeniu.
#    - **Early Stopping**: Przerywa trening, gdy obserwujemy brak poprawy na zbiorze walidacyjnym.
#      - **Przykład (EarlyStopping)**:
#        ```python
#        early_stopping = tf.keras.callbacks.EarlyStopping(
#            patience=3, restore_best_weights=True)
#        history = model.fit(
#            X_train, y_train,
#            epochs=30,
#            batch_size=64,
#            validation_split=0.2,
#            callbacks=[early_stopping]
#        )
#        ```

# 4. **Architektury specjalizowane**:
#    - **CNN** (Convolutional Neural Networks) – dla obrazów.
#    - **RNN/LSTM** – dla sekwencji (np. dane czasowe, tekst).
#    - **Transformery** – dla zadań NLP (tłumaczenie, analiza tekstu), ale i coraz częściej innych typów danych.

# 5. **Monitorowanie i tuning**:
#    - Wizualizacja (np. tensorboard, wykresy loss/accuracy) umożliwia szybką detekcję przeuczenia lub zbyt wczesnej konwergencji.
#    - Testuj różne kombinacje liczby neuronów, warstw, dropoutów i learning rate w kontrolowany sposób (grid search lub narzędzia typu KerasTuner).

# **Podsumowanie (Neural Network)**:
# - Struktura sieci (liczba warstw/neuronów) i odpowiedni dobór dropoutu ma kluczowe znaczenie.
# - Hyperparametry (learning rate, batch size, liczba epok) dostrajamy, by uzyskać równowagę między niedouczeniem a przeuczeniem.
# - Stosuj regularizację (dropout, early stopping) i walidację krzyżową (jeśli możliwe) w celu pewniejszej oceny.



# %% [markdown]
#### **Podsumowanie**:
# - **kNN**: liczba sąsiadów, metryka odległości, waga sąsiadów.
# - **Naive Bayes**: wariant modelu (GaussianNB, MultinomialNB, BernoulliNB), transformacje cech, unikanie silnie skorelowanych atrybutów.
# - **Logistic Regression**: regularyzacja (parametr C, penalty) i właściwy solver.
# - **SVM**: dobór kernela (linear, rbf, poly), regulacja C i gamma.
# 
# Ostateczny wybór parametrów zawsze zależy od charakteru danych, więc regularne testy (np. k-fold cross validation) i ewaluacja na zbiorze walidacyjnym są niezbędne.