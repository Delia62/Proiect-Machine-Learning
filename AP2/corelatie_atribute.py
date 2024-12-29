

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


url = "C:\\Users\\Delia\\Documents\\GitHub\\Proiect-Machine-Learning\\AP2\\tourism_dataset.csv"  # Înlocuiți cu calea corectă la fișierul CSV
# 1. Încarcă setul de date
data = pd.read_csv(url)
# Convertim variabilele categorice în numerice
encoded_data = data.copy()
le = LabelEncoder()

for col in data.select_dtypes(include=['object']).columns:
    encoded_data[col] = le.fit_transform(data[col])

# Calculăm matricea de corelații
correlation_matrix = encoded_data.corr()

# Vizualizare matrice de corelații
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Vizualizăm relația dintre Revenue și Visitors
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Visitors', y='Revenue', hue='Country', data=data, palette='viridis')
plt.title('Revenue vs Visitors')
plt.show()
