# Proiect-Machine-Learning
# Proiect: Analiză Predicție Consum Energetic

## Descrierea Proiectului
Acest proiect are scopul de a prezice consumul energetic pe baza unor date istorice folosind diferite tehnici de învățare automată. În cadrul proiectului, am implementat mai multe modele de predicție, inclusiv:
- Naive Bayes (cu discretizare uniformă și KMeans)
- ID3 Decision Tree
- Algoritmul Naive Bayes Gaussian

Scopul final este de a evalua performanța fiecărui model folosind metrici precum RMSE (Root Mean Squared Error) și MAE (Mean Absolute Error).

## Fișierele Proiectului
- `proiect.py`: Scriptul Python care implementează modelele și evaluările.
- `Raport.pdf`: Raportul detaliat cu explicații și analize.
- `data/`: Directorul care conține fișierele de date (dacă este cazul).

## Pașii Implementați în Proiect
1. **Încărcarea și Preprocesarea Datelor**  
   Fișierele de date sunt încărcate și procesate prin eliminarea valorilor lipsă și conversia corectă a coloanelor de tip `object` în format numeric.

2. **Antrenarea și Testarea Modelelor**  
   Am antrenat modelele de învățare automată folosind seturi de date de antrenament și testare. Modelele utilizate includ Naive Bayes și ID3, iar predicțiile sunt evaluate folosind metrici de performanță.

3. **Evaluarea Performanței**  
   Performanța fiecărui model a fost evaluată folosind RMSE și MAE. Aceste metrici au fost calculate pentru a compara modelele.

4. **Vizualizarea Rezultatelor**  
   Rezultatele predicțiilor au fost comparate cu valorile reale folosind grafice vizuale.

## Instalații Necesare
Pentru a rula acest proiect local, trebuie să instalați următoarele librării:
- pandas
- numpy
- matplotlib
- scikit-learn

