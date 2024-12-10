import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

# Functie pentru incarcare si preprocesare date dintr-un fisier CSV sau Excel
def load_and_preprocess_data(file_path, is_excel=False):  
    try:
        # Citim fisierul in functie de tipul acestuia (CSV sau Excel)
        if is_excel:
            data = pd.read_excel(file_path)  
        else:
            data = pd.read_csv(file_path, encoding='ISO-8859-1')  
            
        # Convertim coloana 'Data' in format datetime
        data['Data'] = pd.to_datetime(data['Data'], dayfirst=True)
        data = data.dropna()  # Eliminam valorile lipsa
        
        # Convertim toate coloanele de tip obiect in numere, acolo unde este posibil
        for column in data.select_dtypes(include=['object']).columns:  
            data[column] = pd.to_numeric(data[column], errors='coerce')
        
        data = data.dropna()  # Eliminam valorile lipsa

        # Cream o coloana 'Max_Consum' care contine maximul dintre 'Consum[MW]' si 'Medie Consum[MW]'
        if 'Consum[MW]' in data.columns and 'Medie Consum[MW]' in data.columns:
            data['Max_Consum'] = data[['Consum[MW]', 'Medie Consum[MW]']].max(axis=1)
            data.drop(columns=['Consum[MW]', 'Medie Consum[MW]'], inplace=True)

        return data
    except pd.errors.ParserError as e:
        print(f"Error reading the file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Functie pentru incarcare date de antrenament si testare
def load_train_test_data(train_file, test_file):
    # Incarcam si preprocesam datele de antrenament si testare
    train_data = load_and_preprocess_data(train_file, is_excel=False)  
    test_data = load_and_preprocess_data(test_file, is_excel=True)  

    # Calculam 'Max_Consum' si pentru datele de testare
    if test_data is not None and 'Consum[MW]' in test_data.columns and 'Medie Consum[MW]' in test_data.columns:
        test_data['Max_Consum'] = test_data[['Consum[MW]', 'Medie Consum[MW]']].max(axis=1)
        test_data.drop(columns=['Consum[MW]', 'Medie Consum[MW]'], inplace=True)

    # Verificam daca datele de antrenament sau testare nu au fost incarcate corect
    if train_data is None or test_data is None:
        print("There was an error loading the training or test data.")
        exit(1)

    return train_data, test_data

# Functie pentru evaluarea modelului folosind RMSE si MAE
def evaluate_model(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))  # Rata de eroare patratica medie
    mae = mean_absolute_error(y_true, y_pred)  # Eroarea absoluta medie
    return rmse, mae

# Functie pentru predictii folosind Naive Bayes pe datele de antrenament si testare
def naive_bayes_predict(train_data, test_data, target, bins=5):
    train_data['target_bin'] = pd.qcut(train_data[target], bins, labels=False)  # Discretizarea tintei in intervale
    probabilities = {}

    # Calculam probabilitatile conditionate pentru fiecare interval
    for target_bin in train_data['target_bin'].unique():
        subset = train_data[train_data['target_bin'] == target_bin]
        probabilities[target_bin] = {
            feature: subset[feature].value_counts(normalize=True).to_dict()
            for feature in train_data.columns if feature != 'target_bin'
        }

    # Calculam predictiile pentru datele de testare
    predictions = []
    for _, row in test_data.iterrows():
        bin_scores = {}
        for target_bin in probabilities:
            score = 1
            for feature in row.index:
                if feature in probabilities[target_bin]:
                    score *= probabilities[target_bin][feature].get(row[feature], 1e-6)  # Daca nu exista valoare, folosim o valoare foarte mica
            bin_scores[target_bin] = score

        best_bin = max(bin_scores, key=bin_scores.get)  # Alegem cel mai probabil interval
        bin_values = train_data[train_data['target_bin'] == best_bin][target]
        predictions.append(bin_values.mean())  # Predicitia este media valorilor din intervalul respectiv

    return np.array(predictions)

# Functie pentru Naive Bayes folosind intervale uniforme
def naive_bayes_uniform_bins(train_data, test_data, target, num_bins=5):
    train_data['target_bin'] = pd.cut(train_data[target], bins=num_bins, labels=False)  # Discretizarea uniforma
    probabilities = {}

    # Calculam probabilitatile conditionate pentru fiecare interval
    for target_bin in train_data['target_bin'].unique():
        subset = train_data[train_data['target_bin'] == target_bin]
        probabilities[target_bin] = {
            feature: subset[feature].value_counts(normalize=True).to_dict()
            for feature in train_data.columns if feature != 'target_bin'
        }

    # Calculam predictiile pentru datele de testare
    predictions = []
    for _, row in test_data.iterrows():
        bin_scores = {}
        for target_bin in probabilities:
            score = 1
            for feature in row.index:
                if feature in probabilities[target_bin]:
                    score *= probabilities[target_bin][feature].get(row[feature], 1e-6)
            bin_scores[target_bin] = score

        best_bin = max(bin_scores, key=bin_scores.get)  # Alegem cel mai probabil interval
        bin_values = train_data[train_data['target_bin'] == best_bin][target]
        predictions.append(bin_values.mean())  # Predicitia este media valorilor din intervalul respectiv

    return np.array(predictions)

# Functie pentru Naive Bayes folosind KMeans pentru discretizare
def naive_bayes_kmeans_bins(train_data, test_data, target, num_bins=5):
    kmeans = KMeans(n_clusters=num_bins, random_state=42)  # Aplicam KMeans pentru discretizare
    train_data['target_bin'] = kmeans.fit_predict(train_data[[target]])
    probabilities = {}

    # Calculam probabilitatile conditionate pentru fiecare interval
    for target_bin in np.unique(train_data['target_bin']):
        subset = train_data[train_data['target_bin'] == target_bin]
        probabilities[target_bin] = {
            feature: subset[feature].value_counts(normalize=True).to_dict()
            for feature in train_data.columns if feature != 'target_bin'
        }

    # Calculam predictiile pentru datele de testare
    predictions = []
    for _, row in test_data.iterrows():
        bin_scores = {}
        for target_bin in probabilities:
            score = 1
            for feature in row.index:
                if feature in probabilities[target_bin]:
                    score *= probabilities[target_bin][feature].get(row[feature], 1e-6)
            bin_scores[target_bin] = score

        best_bin = max(bin_scores, key=bin_scores.get)  # Alegem cel mai probabil interval
        bin_values = train_data[train_data['target_bin'] == best_bin][target]
        predictions.append(bin_values.mean())  # Predicitia este media valorilor din intervalul respectiv

    return np.array(predictions)

# Clasa pentru implementarea regresorului ID3
class ID3Regressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Adancimea maxima a arborelui
        self.tree = None

    def _variance(self, y):
        y = pd.to_numeric(y, errors='coerce') 
        return np.var(y.dropna())  # Calculam varianta (dispersia)

    def _split(self, X, y):
        best_feature, best_threshold, best_reduction = None, None, -np.inf
        current_variance = self._variance(y)

        # Cautam cel mai bun prag pentru impartirea datelor
        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left = y[X[feature] <= threshold]
                right = y[X[feature] > threshold]

                if len(left) > 0 and len(right) > 0:
                    reduction = current_variance - (
                        len(left) / len(y) * self._variance(left)
                        + len(right) / len(y) * self._variance(right)
                    )
                    if reduction > best_reduction:
                        best_feature = feature
                        best_threshold = threshold
                        best_reduction = reduction

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        # Conditii de oprire: daca nu mai sunt date sau daca am atins adancimea maxima
        if len(y) == 0:
            return None

        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y)

        if len(y.unique()) == 1:
            return y.iloc[0]

        feature, threshold = self._split(X, y)
        if feature is None:
            return np.mean(y)

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict(self, tree, row):
        if not isinstance(tree, dict):
            return tree

        if row[tree['feature']] <= tree['threshold']:
            return self._predict(tree['left'], row)
        else:
            return self._predict(tree['right'], row)

    def predict(self, X):
        return X.apply(lambda row: self._predict(self.tree, row), axis=1)


# Functie pentru afisarea matricei de corelatie
def plot_correlation_matrix(data, target_column):
    corr_matrix = data.corr()  # Selectam doar coloanele numerice

    print(f"Matricea de corelatie pentru '{target_column}':\n")
    print(corr_matrix[target_column].sort_values(ascending=False))

    # Vizualizam matricea de corelatie
    plt.figure(figsize=(10, 8))
    plt.title(f"Corelatia intre atributele {target_column}", fontsize=14)
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
    plt.show()



if __name__ == "__main__":
    train_file = "C:\\Users\\Delia\\Documents\\GitHub\\Proiect-Machine-Learning\\antrenare_modified.csv"  # fisierul de antrenament
    test_file = "C:\\Users\\Delia\\Documents\\GitHub\\Proiect-Machine-Learning\\decembrie.xlsx"  # fisierul de testare 

    train_data, test_data = load_train_test_data(train_file, test_file)

    features = ['Max_Consum', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 
                'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
    target = 'Sold[MW]'

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    plot_correlation_matrix(train_data, target)
  
     # testare pentru ID3
    id3 = ID3Regressor(max_depth=10)
    id3.fit(X_train, y_train)
    id3_predictions = id3.predict(X_test)
    id3_rmse, id3_mae = evaluate_model(y_test, id3_predictions)
    print(f"ID3 RMSE: {id3_rmse}, MAE: {id3_mae}")

    # testare pentru Bayes Naiv
    nb_predictions = naive_bayes_predict(train_data, test_data, target)
    nb_rmse, nb_mae = evaluate_model(y_test, nb_predictions)
    print(f"Naive Bayes RMSE: {nb_rmse}, MAE: {nb_mae}")
    
    # testare pentru Bayes Gaussian
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    bayes_predictions = bayes.predict(X_test)
    bayes_rmse, bayes_mae = evaluate_model(y_test, bayes_predictions)
    print(f"Gaussian Bayes RMSE: {bayes_rmse}, MAE: {bayes_mae}")
    
    # testare pentru discretizare uniformă
    uniform_predictions = naive_bayes_uniform_bins(train_data, test_data, target, num_bins=5)
    uniform_rmse, uniform_mae = evaluate_model(y_test, uniform_predictions)
    print(f"Naive Bayes (uniform bins) RMSE: {uniform_rmse}, MAE: {uniform_mae}")


    # testare pentru discretizare KMeans
    kmeans_predictions = naive_bayes_kmeans_bins(train_data, test_data, target, num_bins=5)
    kmeans_rmse, kmeans_mae = evaluate_model(y_test, kmeans_predictions)
    print(f"Naive Bayes (KMeans bins) RMSE: {kmeans_rmse}, MAE: {kmeans_mae}")
    

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="True Values", marker='o')
    plt.plot(id3_predictions, label="ID3 Predictions", marker='x')
    plt.plot(bayes_predictions, label="Bayes Predictions", marker='s')
    plt.legend()
    plt.show()
