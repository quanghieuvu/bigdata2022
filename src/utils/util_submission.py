import os
import random
import pandas as pd

'''
mode 0: all preds are 0
mode 1: all preds are 1
mode 2: random predictions
'''
def generate_random_predictions(mode):
    df_template = pd.read_csv(os.path.join("input", "submission_template.csv"))
    if mode in [0, 1]:
        df_template["is_pair"] = mode
    else:
        random_values = [random.randint(0, 1) for i in range(len(df_template))]
        df_template["is_pair"] = random_values
        print(df_template["is_pair"].value_counts())

    df_template[["is_pair"]].to_csv(os.path.join("output", f"submission_mode_{mode}.txt"), index=False, header=False)

if __name__ == "__main__":
    for mode in [0, 1, 2]:
        generate_random_predictions(mode)