import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('CollegeDistance.csv')

# kodownie zmiennych kategorycznych
label_encoder = LabelEncoder()

# kodowanie, konwersja kolumn binarnych
df['gender'] = label_encoder.fit_transform(df['gender'])
df['fcollege'] = label_encoder.fit_transform(df['fcollege'])
df['mcollege'] = label_encoder.fit_transform(df['mcollege'])
df['home'] = label_encoder.fit_transform(df['home'])
df['urban'] = label_encoder.fit_transform(df['urban'])
df['income'] = label_encoder.fit_transform(df['income'])

# OneHot Encoding dla zmiennych z wiecej niz 2 kategoriami
df = pd.get_dummies(df, columns=['ethnicity', 'region'], drop_first=True)


X = df.drop('score', axis=1)  # cechy bez score
y = df['score']  # docelowa zmienna

# podzial na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standaryzacja zmiennych numerycznych
scaler = StandardScaler()
num_cols = ['rownames', 'unemp', 'wage', 'distance', 'tuition', 'education']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\nKsztalt zbioru treningowego:", X_train.shape)
print("\nKsztalt zbioru testowego:", X_test.shape)

model = LinearRegression() # regresja liniowa jako model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Regresja liniowa:")
print(f'MSE: {mse}')
print(f'R^2: {r2}')
print(f'MAE: {mae}')
# parametry dla grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None],
    'max_features': [None, 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42) # lasy losowe jako model
# uzycie grid search z walidacja krzyzowa
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_


# ocena najlepszego modelu
y_pred_best_rf = best_rf.predict(X_test)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)

print(f"Najbardziej istotne hiperparametry: {grid_search.best_params_}")
print(f"MSE: {mse_best_rf}")
print(f"R^2: {r2_best_rf}")
print(f"MAE: {mae_best_rf}")