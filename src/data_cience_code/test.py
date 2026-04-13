import pandas as pd
from data.load import load_dataset

df = load_dataset("cleaned.csv", folder="processed")

print("Valores únicos en Sleep Disorder:")
print(df["Sleep Disorder"].value_counts(dropna=False))
print()

print("Filas donde Sleep Disorder es NaN:")
print(df[df["Sleep Disorder"].isna()].shape)

print("Filas donde Sleep Disorder es string vacío:")
print(df[df["Sleep Disorder"] == ""].shape)

print("Tipos de datos:")
print(df.dtypes)