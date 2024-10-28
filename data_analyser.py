import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('CollegeDistance.csv')

print("Ilosc danych (wiersze, kolumny):", df.shape)
print("\nPierwsze 5 wierszy:")
print(df.head())
print("\nOstatnie 5 wierszy:")
print(df.tail())

print("\nBrakujace wartosci w danych kolumnach:")
print(df.isnull().sum())

print("\nStatystyki dla zmiennych numerycznych:")
print(df.describe())

plt.figure(figsize=(8, 6))
sns.histplot(df['score'], bins=30, kde=True)
plt.title('Rozklad zmiennej score')
plt.xlabel('Score')
plt.ylabel('Czestotliwosc')
plt.show()

print("\nRozklad zmiennej gender:")
print(df['gender'].value_counts())

print("\nRozklad zmiennej ethnicity:")
print(df['ethnicity'].value_counts())

print("\nRozklad zmiennej income:")
print(df['income'].value_counts())

# Imputacja zmiennych numerycznych mediana
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Imputacja dla zmiennych kategorycznych najczesciej wystepujaca wartoscia
categorical_cols = df.select_dtypes(include=['string']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))  #

# OneHot Encoding dla zmiennych kategorycznych w celu wygenerowania heat mapy
df_encoded = pd.get_dummies(df, drop_first=True)

# Heat map
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='crest')
plt.title('Heat Mapa')
plt.show()


print("\nBrakujace wartosci po imputacji:")
print(df.isnull().sum())
