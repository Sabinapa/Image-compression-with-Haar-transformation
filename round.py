import pandas as pd

input_file = "analysis_results.csv"
output_file = "analysis_results2.csv"

try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Napaka pri branju datoteke: {e}")
    exit()

df = df.round(3)

df.to_csv(output_file, index=False, sep=';')
print(f"Zaokroženi podatki so shranjeni v datoteko: {output_file}")