# %% [markdown] 
#### **IMPORT RELEVANT LIBRARIES**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# %% [markdown]
#### **IMPORT MAGIC DATASET**
# FROM URL: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"] # from URL (dataset haven't headers)
df = pd.read_csv('magic04.data', names=cols)
df.head()


# %% [markdown]
#### **PREPARE DATA**

df['class'] = (df['class'] == 'g').astype(int)
df.head()


# %% [markdown]
#### **Histograms for X features**

for label in cols[:-1]: # od 1 do przedostatniej kolumny, label iteruje po tych kolumnach
    plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df['class']==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()


# %% [markdown]
#### **Split dataset to train, validation, test**

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


# %% [markdown]
#### **Function scale_dataset -> 50/50 proportions**  
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values # wszystkie cechy dla modelu bez ostatniej kolumny czyli celu
    y = dataframe[dataframe.columns[-1]].values # kolumna celu
    
    scaler = StandardScaler() # tworzenie obiektu StandardScaler - przesuwa rozkład cech do średniej = 0 i skali = 1
    X = scaler.fit_transform(X) # doppasowanie scalera do danych (czyli cech)
    
    if oversample: # opcjonalnie oversampling (zwiększenie liczby przykładów mniejszościowej klasy)
        ros = RandomOverSampler() # losowe duplikuje próbki klasy mniejszościowej aby zrównoważyć proporcje
        X, y = ros.fit_resample(X, y) # dopasowanie do zmiennych cech i celu

    data = np.hstack((X, np.reshape(y, (-1,1)))) # horyzontalne łączenie danych w jedną macierz, reshape(-1,1) -> (n, 1) dla y bo X ma kształt (n, liczba_cech) -> dzięki temu uzyskujemy kształt (n, liczba_cech+1) gdzie ostatnia kolumna to etykieta 
    
    return data, X, y


train, X_train, y_train = scale_dataset(train, oversample=True) # podczas oversamplingu możliwe jest powiększenie danych (duplikowanie aby uzyskać dobre proporcje)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)


# %% [markdown]
#### **Key points for k-Nearest Neighbors (kNN) Model**
# **Kluczowe założenia:**
# - **Podobne obiekty znajdują się blisko siebie**: Próbki należące do tej samej klasy są zlokalizowane w niewielkiej odległości w przestrzeni cech.
# - **Metryka odległości** (np. Euklidesowa) poprawnie odzwierciedla podobieństwo między danymi, co ma kluczowy wpływ na jakość klasyfikacji.
#
# **Zalety:**
# - **Prostota i intuicyjność**: Łatwe do zrozumienia – szukamy „k” najbliższych sąsiadów i dokonujemy decyzji na podstawie ich klas.
# - **Brak fazy trenowania** (w tradycyjnym sensie): Praktycznie „zapamiętuje” dane, przez co kosztem jest większe zapotrzebowanie na pamięć w fazie predykcji.
# - **Elastyczność**: Można łatwo dostosować liczbę sąsiadów „k” czy rodzaj metryki, aby poprawić wyniki.
#
# **Wady:**
# - **Wysoki koszt predykcji**: Aby zaklasyfikować nową próbkę, trzeba obliczyć odległość do wszystkich przykładów w zbiorze, co jest kosztowne przy dużej liczbie danych.
# - **Wrażliwość na skalowanie cech**: Wszystkie cechy powinny być na podobnej skali (normalizacja lub standaryzacja danych jest praktycznie koniecznością).
# - **Efekt „klątw wielowymiarowości”**: Wysoka liczba cech sprawia, że odległości w przestrzeni przestają być w pełni reprezentatywne, co może znacząco obniżyć skuteczność.


# %% [markdown]
#### kNN - k Nearest Neighbors algorithm
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print(classification_report(y_test, y_pred_knn))

# %% [markdown] 
#### **Matrix for understand classification_report**
#
# |                | Predicted 0 | Predicted 1 |
# |----------------|-------------|-------------|
# | **Actual 0**   | TN          | FP          |
# | **Actual 1**   | FN          | TP          |
#
#
# **Interpretacja elementów macierzy:**
# - **TN (True Negative)** – przypadki, w których model poprawnie klasyfikuje obserwacje należące do klasy 0 jako 0.
# - **FP (False Positive)** – przypadki, w których model błędnie przypisuje klasę 1 mimo, że w rzeczywistości obserwacja jest z klasy 0.
# - **FN (False Negative)** – przypadki, w których model nie rozpoznaje klasy 1 (błędnie przypisuje klasę 0), mimo że w rzeczywistości obserwacja jest z klasy 1.
# - **TP (True Positive)** – przypadki, w których model poprawnie klasyfikuje obserwacje należące do klasy 1 jako 1.

# %% [markdown]
#### **Explanation of classification_report**
# - **Accuracy** (dokładność) – wynosi 0.81, co oznacza, że model z 3 sąsiadami prawidłowo rozpoznaje 81% przypadków w zbiorze testowym.
# - **Support** – liczba wystąpień danej klasy w zbiorze testowym.
# - **Precision** (precyzja) dla klasy 0 = 0.74 – z wszystkich przykładów sklasyfikowanych jako klasa 0, ok. 74% to prawdziwe „hadrony”. 
# - **Recall** (czułość) dla klasy 0 = 0.72 – spośród wszystkich przykładów klasy 0 w rzeczywistości, model poprawnie rozpoznał 72%.
# - **F1-score** = 0.73 – kompromis między precision i recall dla klasy 0.
# - **Precision** = 0.85 (dla klasy 1) – z wszystkich przykładów sklasyfikowanych jako klasa 1, ok. 85% to prawdziwe „gamm-y”.
# - **Recall** = 0.86 (dla klasy 1) – spośród wszystkich prawdziwych przypadków klasy 1, model poprawnie rozpoznał 86%.
# - **F1-score** dla klasy 1 wynosi 0.86 – jest wyższe niż w przypadku klasy 0.
# - **Macro avg** (0.79) – średnia z precision, recall i F1 obliczana dla obu klas z równą wagą.
# - **Weighted avg** (0.81) – średnia ważona, gdzie wpływ poszczególnych klas jest proporcjonalny do ich liczebności (większa liczebność klasy 1 podnosi ten wynik).


# %% [markdown]
#### **Wnioski dotyczące modelu kNN**
# - Model osiąga przyzwoity poziom 81% dokładności ogólnej.
# - Jest wyraźnie lepszy w rozpoznawaniu klasy 1 (gamma) niż klasy 0 (hadron).
# - Jeśli zależy nam na równym traktowaniu obu klas, można dostrajać parametry kNN (liczbę sąsiadów, metrykę), a także rozważyć dodatkowe zbalansowanie danych.
# - Jeżeli jednak priorytetem jest wysoka skuteczność w wykrywaniu klasy 1 (gamma), wynik F1 = 0.86 i recall = 0.86 może być już satysfakcjonujący.











# %% [markdown]
#### **Key Points for Naive Bayes Model**
# **Kluczowe założenia:**
# - **Niezależność cech**: Zakłada się, że cechy (atrybuty) są od siebie statystycznie niezależne (warunkowo względem klasy).
# - **Rozkład danych**: Najczęściej używa się rozkładu Gaussa (GaussianNB), Bernoulli lub Multinomial, w zależności od charakteru cech (ciągłe, binarne, kategorie zliczeniowe itp.).
#
# **Zalety:**
# - **Bardzo szybkie trenowanie** – model jest prosty, więc można go zastosować nawet przy dużej liczbie cech.
# - **Łatwa interpretacja** – wynik to prawdopodobieństwo przynależności do klasy (wyliczane na podstawie warunkowej niezależności cech).
# - **Skalowalność** – dobrze działa w środowiskach o ograniczonych zasobach i przy masywnych zbiorach danych.
#
# **Wady:**
# - **Założenie „naive”** (warunkowa niezależność cech) często nie jest spełnione w praktyce, co może zaniżać dokładność.
# - **Niektóre warianty** (np. GaussianNB) zakładają konkretny rozkład danych (normalny), co bywa nieadekwatne do rzeczywistej struktury zbioru.
# - **Mniejsza elastyczność** niż niektóre algorytmy, np. modele drzewa, które automatycznie wychwytują złożone zależności między cechami.


# %% [markdown]
#### Naive Bayes algorithm

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print(classification_report(y_test, y_pred_nb))


# %% [markdown]
#### Results for Naive Bayes model
# **Ogólna dokładność (accuracy)** wynosi 0.70, co jest odczuwalnie niższym wynikiem niż dla kNN (ok. 0.81). <br>
# **Klasa 0 (hadron)** ma dość niskie recall (0.41), co oznacza, że model często nie rozpoznaje poprawnie tej klasy (dużo przykładów klasy 0 jest błędnie klasyfikowanych jako klasa 1). <br>
# **Klasa 1 (gamma)** wypada wyraźnie lepiej: precision = 0.74 i recall = 0.89. W efekcie model ma tendencję do „faworyzowania” klasy 1. <br> <br>
# **Wnioski:** 
# Jeśli priorytetem jest wykrywanie klasy 1, to model Naive Bayes może być zadowalający (wysoki recall = 0.89). Jednak do zrównoważonej detekcji obu klas sprawdza się słabiej, zwłaszcza dla klasy 0. Można to próbować poprawić np. tuningiem parametrów, dodatkowymi cechami lub dalszym zbalansowaniem danych.














# %% [markdown]
#### Logistic Regression model

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
y_pred_log = lg_model.predict(X_test)
print(classification_report(y_test, y_pred_log))

# %% [markdown]
#### **Key points for Logistic Regression Model**

# **Kluczowe założenia:**
# - **Zależność liniowa** między zmiennymi objaśniającymi (cechami) a logarytmem szans (log-odds).  
# - **Brak (lub niewielka) współliniowości** między zmiennymi (multicollinearity może utrudnić interpretację i pogorszyć wyniki).  
# - **Niezależność obserwacji** – standardowe założenie statystyczne w modelach regresyjnych.  
#
# **Zalety:**
# - **Prostota i szybkość** uczenia – łatwość implementacji, a trenowanie jest zazwyczaj szybkie (również dla dużych zbiorów danych).
# - **Interpretowalność** – współczynniki regresji logistycznej można łatwo zinterpretować (log-odds, wpływ każdej cechy).
# - **Odpowiednia do problemów binarnych** i wieloklasowych (one-vs-rest), może też dostarczać prawdopodobieństwa przynależności do klas.
#
# **Wady:**
# - **Założenie liniowej granicy decyzyjnej** – w przypadku danych z wyraźną nieliniową strukturą klasyfikacja może być niedokładna.
# - **Wrażliwość na outliers** – skrajne wartości mogą silnie wpływać na estymację współczynników.
# - **Możliwe trudności z konwergencją** przy bardzo dużych współczynnikach lub źle dobranych cechach (ale zwykle łatwe do opanowania poprzez regularyzację).


# %% [markdown]
#### Results for Logistic Regression model
# **Ogólna dokładność (accuracy):** około 0.79, co plasuje się pomiędzy wynikami Naive Bayes (0.70) i kNN (0.81). <br>
# **Klasa 0 (hadron):** precision = 0.67, recall = 0.72, f1-score = 0.70. Model wciąż ma problemy z poprawnym rozpoznawaniem tej klasy (niższe wyniki niż dla klasy 1). <br>
# **Klasa 1 (gamma):** precision = 0.85, recall = 0.82, f1-score = 0.83. Znacznie lepiej niż w przypadku klasy 0, co wskazuje na tendencję do „faworyzowania” gammy. <br> <br>
#**Wnioski:**
# - Logistic Regression radzi sobie lepiej niż Naive Bayes (wyższa dokładność i lepsze wyniki dla klasy 1), ale ustępuje kNN (zwłaszcza w rozpoznawaniu klasy 0).
# - Jeśli zależy nam na bardziej zbalansowanej detekcji obu klas, można spróbować dostroić model (np. parametrem regularizacji C lub inną metryką) albo dalej zrównoważyć dane.













# %% [markdown]
#### **Key points for Support Vector Machine (SVM) Model**
# 1. **Główna idea**  
#    - SVM stara się znaleźć „optymalną” hiperpłaszczyznę (w 2D jest to linia, w 3D – płaszczyzna, itp.), która oddziela próbki obu klas z największym możliwym marginesem.
#
# 2. **Maksymalizacja marginesu**  
#    - Kluczowym założeniem jest, że szerszy margines między dwiema klasami przekłada się na lepsze uogólnienie i mniejszą wrażliwość na błędy.
#
# 3. **Wektory nośne** (support vectors)  
#    - Jedynie punkty najbliższe granicy decyzyjnej (tzw. wektory nośne) decydują o ostatecznym kształcie tej granicy.
#    - Pozostałe próbki, leżące dalej od marginesu, nie wpływają bezpośrednio na uczenie modelu.
#
# 4. **Jądra (kernels)**  
#    - Aby poradzić sobie z nieliniowymi danymi, SVM może używać różnych funkcji jądra (np. RBF, polynomial, sigmoid).
#    - Kernels pozwalają „zanurzyć” dane w wyższym wymiarze, gdzie stają się liniowo separowalne.
#
# 5. **Priorytety i parametry**  
#    - **Utrzymanie szerokiego marginesu**: minimalizowanie błędów przez skupienie na przypadkach granicznych.
#    - **Parametr `C`**: kontroluje kompromis między szerokością marginesu a liczbą błędnych klasyfikacji (zwiększenie `C` może prowadzić do węższego marginesu, ale mniejszej liczby błędów na zbiorze treningowym).
#    - **Elastyczność**: dzięki różnym jądrom SVM może dostosowywać się do wielu typów danych i złożonych relacji. 
# 
# **Zalety:**
# - **Wysoka skuteczność** w wielu zadaniach (szczególnie przy odpowiednim doborze jądra).
# - **Odporność na overfitting** – dzięki maksymalizacji marginesu i skupianiu się tylko na wektorach nośnych.
# - **Skuteczne w wysokim wymiarze** – można stosować do danych z wieloma cechami (funkcje jądra dobrze sobie radzą z nieliniowymi zależnościami).
# 
# **Wady:**
# - **Czas trenowania** może być wysoki przy bardzo dużych zbiorach danych (złożoność algorytmu).
# - **Wybór jądra i parametrów** (C, gamma) bywa trudny – wymaga eksperymentów i walidacji.
# - **Mniejsza interpretowalność** w porównaniu z modelami liniowymi – granica decyzyjna nie jest tak intuicyjna dla człowieka (zwłaszcza przy nieliniowych jądrach).

# %% 
#### SVC - Support Vector Classifier
from sklearn.svm import SVC 
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print(classification_report(y_test, y_pred_svm))

# %% [markdown]
#### Results for Support Vector Machine Model
# **Ogólna dokładność (accuracy):** około 0.87, co jest najwyższą wartością spośród dotychczasowych metod (Naive Bayes, kNN, Logistic Regression). <br>
# **Klasa 0 (hadron):** precision = 0.82, recall = 0.79, f1-score = 0.80. Model dość dobrze radzi sobie z tą klasą, choć niższy recall wskazuje, że część rzeczywistych hadronów bywa mylnie klasyfikowana jako gamma. <br>
# **Klasa 1 (gamma):** precision = 0.89, recall = 0.91, f1-score = 0.90. Wyraźnie wyższe wyniki, co oznacza, że model „faworyzuje” klasy gamma (rozpoznaje je z dużą skutecznością). <br> <br>
# **Wnioski:**
# - SVM osiąga najlepszy wynik ogólny (0.87) spośród porównywanych modeli.
# - Choć nadal widać tendencję do nieco lepszego rozpoznawania klasy 1 (gamma), dysproporcja między klasami jest mniejsza niż w słabszych modelach.
# - Można rozważyć dalszy tuning hiperparametrów (np. C, gamma, rodzaj kernela) lub dodatkowe zbalansowanie danych, aby jeszcze bardziej zbliżyć skuteczność rozpoznawania obu klas.




# %% [markdown]
#### **Jak ulepszyć dany model? - podsumowanie**
# - **kNN**: liczba sąsiadów, metryka odległości, waga sąsiadów.
# - **Naive Bayes**: wariant modelu (GaussianNB, MultinomialNB, BernoulliNB), transformacje cech, unikanie silnie skorelowanych atrybutów.
# - **Logistic Regression**: regularyzacja (parametr C, penalty) i właściwy solver.
# - **SVM**: dobór kernela (linear, rbf, poly), regulacja C i gamma.
# 
# Ostateczny wybór parametrów zawsze zależy od charakteru danych, więc regularne testy (np. k-fold cross validation) i ewaluacja na zbiorze walidacyjnym są niezbędne.

# %% [markdown]
#### **Neural Net Model**

import tensorflow as tf
from sklearn.metrics import classification_report
# %% [markdown]
#### **Plot function to great visualisation Neural Network**

def plot_history(history): # plot all training and validation loss/accuracy for every epochs (training cycles)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()
  

# %% [markdown]
#### **Create/Compile/Fit Neural Network Model**
# you can modify hyperparameters like: **num_nodes, dropout, learning_rate, batch_size, epochs, choose optimizer like Adam, choose loss algorithm and metrics**

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
      tf.keras.layers.Dropout(dropout_prob), #randomly choose at this rate certain nodes and don't train them in a certain iteration - prevent overfitting (argument probability)
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout_prob), 
      tf.keras.layers.Dense(1, activation='sigmoid') # 0 or 1 -> sigmoid 
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy']) # another optimizers like RMSprop, SGD with momentum itd. - test
  early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping] , verbose=0 # verbose no show/show training, split data = 0.2% -> validation dataset
  )

  return nn_model, history


# %% [markdown]
#### **Training with parameters/hyparameters Neural Network Model**

least_val_loss = float('inf') # infinity
least_loss_model = None
epochs= 50 
results_list = []

for num_nodes in [16, 32, 64]: # time consuming: 3*2*3*3 = 54 * epochs 
  for dropout_prob in[0, 0.2]:
    for lr in [0.01, 0.005, 0.001]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history)
        val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=0)

        
        results_dict = {'num_nodes': num_nodes, 'dropout_prob': dropout_prob, 'lr': lr, 'batch_size': batch_size,
                    'epochs': epochs, 'val_loss': val_loss, 'val_acc': val_acc}
        
        results_list.append(results_dict)
        
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model
          


# %%
# Convert results to DataFrame for analysis
df_results = pd.DataFrame(results_list)
# Optional: sort by validation loss (lowest first)
df_results.sort_values('val_loss', inplace=True)
print("\nAll runs sorted by validation loss (lowest first):")
print(df_results.head(10))


# %% [markdown]
#### **Create prediction for Neurak Network Model and analysis classification_report**

y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
print("\nClassification Report on Test Set for best (lowest val_loss) model:")
print(classification_report(y_test, y_pred))

# %% 
df_results.to_csv("./nn_results.csv", index=False)
# %%
