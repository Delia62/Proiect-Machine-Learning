import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Incarca setul de date
url = "C:\\Users\\Delia\\Documents\\GitHub\\Proiect-Machine-Learning\\AP2\\tourism_dataset.csv"  
data = pd.read_csv(url)
print(data.head())
print(data.columns)

data.columns = data.columns.str.strip()

# Creeaza coloana pentru profit pe cap de vizitator
data['Profit_Per_Visitor'] = data['Revenue'] / data['Visitors']

# Verificam daca exista valori lipsa
if data.isnull().sum().any():
    print("Exista valori lipsa in setul de date!")
else:
    print("Nu exista valori lipsa.")

# Selecteaza caracteristicile
X = data[['Country', 'Category', 'Visitors', 'Rating']]
y_revenue = data['Revenue']
y_country = data['Country']

# Transforma caracteristicile categorice
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['Country', 'Category'])], remainder='passthrough')

# Functie pentru antrenare si predictie pe fiecare tara
def train_and_predict_per_country(data, models):
    results = {}  # Dictionary pentru a salva MSE pentru fiecare model per tara

    for country in data['Country'].unique():
        print(f"Antrenare si predictie pentru {country}...")

        # Filtrare pe tara
        country_data = data[data['Country'] == country]

        
        required_columns = ['Country', 'Category', 'Visitors', 'Rating', 'Revenue']
        for col in required_columns:
            if col not in country_data.columns:
                raise ValueError(f"Coloana {col} lipseste din datele pentru tara {country}.")

        # Separa caracteristicile si etichetele
        X = country_data[['Category', 'Visitors', 'Rating']]
        y = country_data['Revenue']

        # Imparte datele in seturi de antrenament si testare
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creeaza pipeline-ul
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), ['Category']),
                          ('num', StandardScaler(), ['Visitors', 'Rating'])])

        for name, model in models.items():
            print(f"Antrenare model {name} pentru {country}...")

            # Pipeline pentru fiecare model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            # Antrenare
            pipeline.fit(X_train, y_train)

            # Predictii
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Salveaza rezultatele cu tara si modelul
            results[(country, name)] = mse

    # Alege modelul cu cel mai mic MSE pentru fiecare tara
    best_models = {}
    for country in data['Country'].unique():
        country_results = {model: mse for (c, model), mse in results.items() if c == country}
        best_model = min(country_results.items(), key=lambda x: x[1])  # Modelul cu MSE minim
        best_models[country] = best_model

    # Afiseaza cel mai bun model pentru fiecare tara
    for country, (model_name, mse) in best_models.items():
        print(f"Cel mai bun model pentru {country} este {model_name} cu MSE: {mse}")

    return best_models, results

# Definirea modelelor
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Antrenare si predictie
best_models, all_results = train_and_predict_per_country(data, models)

# Vizualizarea rezultatelor
for country, (model_name, mse) in best_models.items():
    print(f"Top activitati pentru {country} folosind {model_name}:")
    country_data = data[data['Country'] == country]
    X_country = country_data[['Category', 'Visitors', 'Rating']]
    
    # Folosim pipeline-ul pentru a face predictiile
    best_pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), ['Category']),
                      ('num', StandardScaler(), ['Visitors', 'Rating'])])), 
                                    ('model', models[model_name])])

    best_pipeline.fit(X_country, country_data['Revenue'])
    country_data['Predicted_Revenue'] = best_pipeline.predict(X_country)
    ranking = country_data[['Category', 'Predicted_Revenue']].sort_values(by='Predicted_Revenue', ascending=False)
    print(ranking.head(10))

    # Vizualizare ranking pentru o tara
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ranking.head(10), x='Category', y='Predicted_Revenue')
    plt.title(f'Top activitati pentru maximizarea profitului in {country}')
    plt.xticks(rotation=45)
    plt.show()

    # Pie chart pentru distributia activitatilor
    category_counts = ranking.groupby('Category').size()
    plt.figure(figsize=(8, 8))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distributia activitatilor in {country} - {model_name}')
    plt.show()

# Salvare in fisier CSV
for country, (model_name, mse) in best_models.items():
    country_data = data[data['Country'] == country].copy()
    X_country = country_data[['Category', 'Visitors', 'Rating']]
    country_data['Predicted_Revenue'] = models[model_name].predict(X_country)
    ranking = country_data[['Category', 'Predicted_Revenue']].sort_values(by='Predicted_Revenue', ascending=False)
    ranking.to_csv(f"ranking_{country}.csv", index=False)

# Compararea si vizualizarea performantei algoritmilor
# Impartirea datelor
X = data[['Country', 'Category', 'Visitors', 'Rating']]  # Caracteristicile
y = data['Revenue']  # Variabila de iesire

# Impartim datele in seturi de antrenament si testare (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea modelelor
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Evaluarea performantei
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

mse_scores = [mean_squared_error(y_test, pred) for pred in [rf_pred, gb_pred, xgb_pred]]
r2_scores = [r2_score(y_test, pred) for pred in [rf_pred, gb_pred, xgb_pred]]

# Rezultatele pentru comparatie
for model, mse, r2 in zip(['Random Forest', 'Gradient Boosting', 'XGBoost'], mse_scores, r2_scores):
    print(f"{model} - MSE: {mse:.2f}, R^2: {r2:.2f}")

# Graficul pie chart pentru distributia deciziilor favorabile
category_labels = ['Adventure', 'Cultural', 'Beach', 'Nature', 'Historical', 'Urban']
category_counts = {category: 0 for category in category_labels}

# Exemplu de mapare a predictiilor
for pred in gb_pred:
    predicted_category = category_labels[int(pred % len(category_labels))]  # Exemplu
    category_counts[predicted_category] += 1

plt.figure(figsize=(8, 8))
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Distributia deciziilor favorabile (Gradient Boosting)')
plt.show()
