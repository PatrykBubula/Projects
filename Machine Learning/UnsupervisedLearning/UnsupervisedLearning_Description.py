# %% [markdown]
#### **Jak ulepszać wyniki klasteryzacji (Unsupervised Learning)?**

# **1. Wybór cech i preprocessing**
# - Im lepiej dobrane cechy, tym wyraźniejsze grupy (klastry) można otrzymać.
# - Skalowanie (np. StandardScaler, MinMaxScaler) często bywa kluczowe, gdy różne cechy są w różnych skalach.
# - Usuwanie outliers może pomóc, bo skrajne wartości zakłócają obliczanie centroidów (np. w K-Means).

# **2. Dobór liczby klastrów**
# - Użyj metod takich jak **Elbow Method** czy **Silhouette Score**, by oszacować optymalną liczbę klastrów.
# - Nie zawsze liczba klastrów musi pokrywać się z liczbą prawdziwych klas (jeśli znamy je z etykiet).

# **3. Metody klasteryzacji**
# - **K-Means**: Szybki, ale zakłada kulisty kształt klastrów i wrażliwy na początkowe centroidy.
# - **Hierarchiczne**: Nie wymaga wcześniejszego ustalania liczby klastrów, ale może być kosztowna obliczeniowo.
# - **DBSCAN**: Wykrywa nieregularne klastry, obsługuje outliers, nie wymaga liczby klastrów, lecz parametrów eps i min_samples.

# **4. Klasteryzacja w wysokim wymiarze**
# - Wysoka liczba cech (np. 10, 100 lub więcej) utrudnia rozdzielenie klastrów („klątwa wymiarowości”).
# - **PCA (Principal Component Analysis)** lub inne metody redukcji wymiaru (np. t-SNE, UMAP) mogą pomóc:
#   - Umożliwiają sprowadzenie danych do 2–3 wymiarów, co ułatwia:
#     - **Wizualizację**: sprawdzenie, czy klastry są separowalne.
#     - **Klasteryzację**: K-Means bywa dokładniejszy w zredukowanej przestrzeni.

# **5. Interpretacja klastrów**
# - Po dokonaniu klasteryzacji przeanalizuj cechy, które mają największe znaczenie w odróżnianiu poszczególnych klastrów (np. mean, std, czy feature importance z modelu).
# - W razie potrzeby wprowadzaj nowe cechy, łącz istniejące lub stosuj transformacje (np. log) – to może wyraźniej rozdzielić dane.

# **Podsumowanie (Unsupervised / Clustering)**
# - Dostosuj wybór cech i dokonaj ich skalowania.
# - Przetestuj różne algorytmy (K-Means, DBSCAN, Hierarchiczne), bo każdy ma inne założenia.
# - Rozważ redukcję wymiaru (PCA, t-SNE, UMAP) przy dużej liczbie cech – wizualizacja i sama klasteryzacja będą skuteczniejsze.
# - Używaj metryk takich jak Silhouette Score, aby obiektywnie ocenić jakość klastrów.

# %%
