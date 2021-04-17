# Calculates McNemar’s test for evaluation results https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

original_results_path = "evaluation/original.xlsx"
modified_results_path = "evaluation/modified.xlsx"
sheet_name = "validation"

ALPHA = 0.05

original_results = pd.read_excel(
    original_results_path, sheet_name=sheet_name, engine="openpyxl", index_col=[0]
)

modified_results = pd.read_excel(
    modified_results_path, sheet_name=sheet_name, engine="openpyxl", index_col=[0]
)

results = pd.concat([original_results, modified_results], axis=1)
results.columns = ["original", "modified"]

cont_table = pd.crosstab(results.original == True, results.modified == True)
result = mcnemar(cont_table, exact=False, correction=False)

h0 = "There is no significant difference in error rate"
h1 = "There is a significant difference in error rate"
print(f"H0: {h0}\nH1: {h1}\n")
print(f"P-value: {result.pvalue}")
if result.pvalue > ALPHA:
    print(f"H0 not rejected. {h0}")
else:
    print(f"H0 rejected. {h1}")
