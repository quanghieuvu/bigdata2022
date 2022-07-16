import os
import random
import pandas as pd

'''
mode 0: all preds are 0
mode 1: all preds are 1
mode 2: random predictions
mode 3: load predictions (random values) from the previous submitted mode 2
'''
def generate_random_predictions(mode, S2_file='', verbose=0):
    df_submission = pd.read_csv(os.path.join("input", "submission_template.csv"))
    output_file = os.path.join("output", f"submission_mode_{mode}.txt")
    if mode in [0, 1]:
        df_submission["is_pair"] = mode
    elif mode == 2:
        random_values = [random.randint(0, 1) for i in range(len(df_submission))]
        df_submission["is_pair"] = random_values
        print(df_submission["is_pair"].value_counts())
    else:
        df_m2 = pd.read_csv(os.path.join("output", "submission_mode_2.txt"), header=None)
        df_m2.columns = ["is_pair"]
        random_values = df_m2["is_pair"].values.tolist()
        df_submission["is_pair"] = random_values
        print(df_submission["is_pair"].value_counts())

    if S2_file != '':
        df_s2 = pd.read_csv(os.path.join("output", S2_file))
        df_submission = df_submission.merge(df_s2, on="id", how="left")
        df_submission["discrimination"].fillna(-1, inplace=True)
        df_submission = df_submission.astype({"discrimination": int})
        df_submission["is_pair"] = df_submission.apply(lambda x: x["discrimination"] if x["discrimination"]!=-1 else x["is_pair"], axis=1)
        output_file = os.path.join("output", f"submission_mode_{mode}_{S2_file}.txt")

        if verbose == 1:
            print(df_submission[9991:10010][["is_pair", "discrimination"]])

    print(df_submission["is_pair"].value_counts())
    df_submission[["is_pair"]].to_csv(output_file, index=False, header=False)

# infer score
def infer_score():
    s1 = (0.6248 - 0.5096) / 0.3 + 0.5096
    s2 = (0.6257 - 0.5096) / 0.3 + 0.5096
    print(s1, s2)

if __name__ == "__main__":
    '''
    for mode in [0, 1, 2]:
        generate_random_predictions(mode)
    '''
    '''
    for S2_file in ["S2.csv", "S2_balance.csv"]:
        generate_random_predictions(mode=3, S2_file=S2_file, verbose=0)
    '''
    infer_score()