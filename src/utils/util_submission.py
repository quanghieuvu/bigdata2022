import os
import random
import numpy as np
import pandas as pd
import time

'''
mode 0: all preds are 0
mode 1: all preds are 1
mode 2: random predictions
mode 3: load predictions (random values) from the previously submitted mode 2
- if S1_file != '': replace random values from S1 with results from ML
- if S2_file != '': replace random values from S2 with results from ML
- if S3_file != '': replace random values from S3 with results from ML
'''
def generate_random_predictions(mode, verbose=0,
    S1_folder='', S1_file='', S1_percentage=100,
    S2_folder='', S2_file='', S2_percentage=100):

    df_submission = pd.read_csv(os.path.join("input", "submission_template.csv"))
    if mode in [0, 1]:
        df_submission["is_pair"] = mode
    elif mode == 2:
        random_values = [random.randint(0, 1) for i in range(len(df_submission))]
        df_submission["is_pair"] = random_values
        print(df_submission["is_pair"].value_counts())
    else: # mode 3
        df_m2 = pd.read_csv(os.path.join("output", "submission_mode_2.txt"), header=None)
        df_m2.columns = ["is_pair"]
        random_values = df_m2["is_pair"].values.tolist()
        df_submission["is_pair"] = random_values
        print(f"Target distribution from input file")
        print(df_submission["is_pair"].value_counts())

    output_file = os.path.join("output", f"submission_mode_{mode}")
    if S1_file != '':
        if S1_folder == '':
            pd.read_csv(os.path.join("output", S1_file))
        else:
            df_s1 = pd.read_csv(os.path.join("output", S1_folder, S1_file))
            if S1_percentage < 100:
                df_s1 = df_s1[:int(S1_percentage / 100 * len(df_s1))]

        df_submission = df_submission.merge(df_s1, on="id", how="left")
        df_submission["discrimination"].fillna(-1, inplace=True)
        df_submission = df_submission.astype({"discrimination": int})
        df_submission["is_pair"] = df_submission.apply(lambda x: x["discrimination"] if x["discrimination"] != -1 else x["is_pair"], axis=1)
        df_submission.drop("discrimination", axis=1, inplace=True)
        output_file += f"_{S1_file}"
        print(f"After adding S1, output file {output_file}")
        print(df_submission["is_pair"].value_counts())

    if S2_file != '':
        if S2_folder == '':
            df_s2 = pd.read_csv(os.path.join("output", S2_file))
        else:
            df_s2 = pd.read_csv(os.path.join("output", S2_folder, S2_file))
            if S1_percentage < 100:
                df_s2 = df_s2[:int(S2_percentage / 100 * len(df_s1))]
        df_submission = df_submission.merge(df_s2, on="id", how="left")
        df_submission["discrimination"].fillna(-1, inplace=True)
        df_submission = df_submission.astype({"discrimination": int})
        df_submission["is_pair"] = df_submission.apply(
            lambda x: x["discrimination"] if x["discrimination"]!=-1 else x["is_pair"], axis=1)
        df_submission.drop("discrimination", axis=1, inplace=True)
        output_file += f"_{S2_file}"
        print(f"After adding S2, output file {output_file}")
        print(df_submission["is_pair"].value_counts())

    print(f"Final target distribution, output file {output_file}")
    print(df_submission["is_pair"].value_counts())
    df_submission[["is_pair"]].to_csv(f"{output_file}.txt", index=False, header=False)

def analyze_S3():
    df_m2 = pd.read_csv(os.path.join("output", "submission_mode_2.txt"), header=None)
    df_m2.columns = ["is_pair"]
    random_values = df_m2["is_pair"].values.tolist()
    s3_1 = sum(random_values[20000:])
    print(f"Target distribution of S3 random values")
    print(len(random_values), s3_1, 10000 - s3_1)

def enhance_S3(file):
    df_submission = pd.read_csv(os.path.join("output", file), header=None)
    df_submission.columns = ["is_pair"]
    print("Before", df_submission["is_pair"].value_counts(),
          df_submission[20000:]["is_pair"].value_counts())

    new_values = df_submission["is_pair"].values.tolist()[:20000] + [random.randint(0, 1) for i in range(10000)]
    df_submission["is_pair"] = new_values

    value_counts = df_submission["is_pair"].value_counts().tolist()
    print("After", df_submission["is_pair"].value_counts(),
          df_submission[20000:]["is_pair"].value_counts())
    
    file_id = int(time.time())
    df_submission[["is_pair"]].to_csv(os.path.join("output", f"{file}_adjusted_s3_{value_counts[0]}_{file_id}"), index=False, header=False)


# infer accuracy and estimate score
def infer_score(s1_score=0.5, s1_percentage=0.1,
                s2_score=0.1, s2_percentage=1.0, base_score=0.5096):
    s1_acc = (s1_score - base_score) / 0.1 + base_score
    s2_acc = (s2_score - base_score) / 0.3 + base_score
    print(f"Accuracy of s1: {s1_acc}, s2: {s2_acc}")

    estimate_score = base_score * 0.6
    estimate_score += 0.1 * (s1_acc * s1_percentage + base_score / 10 * (1 - s1_percentage))
    estimate_score += 0.3 * (s2_acc * s1_percentage + base_score / 10 * (1 - s1_percentage))
    print(f"Estimate score: {estimate_score}")

def ensemble(files, is_submission=True):
    if is_submission:
        df_submission = pd.read_csv(os.path.join("input", "submission_template.csv"))
        for file, i in zip(files, range(len(files))):
            df_tmp = pd.read_csv(os.path.join("output", file), header=None)
            df_tmp.columns = [f"v_{i}"]
            df_submission[f"v_{i}"] = df_tmp[f"v_{i}"].values
        df_submission["is_pair"] = df_submission.apply(lambda x: 1 if np.average([x[f"v_{i}"] for i in range(len(files))]) >= 0.5 else 0, axis=1)
        df_submission[["is_pair"]].to_csv(os.path.join("output", "ensemble.txt"), index=False, header=False)
    else:
        df_ensemble = pd.DataFrame()
        for file, i in zip(files, range(len(files))):
            df_tmp = pd.read_csv(os.path.join("output", file))
            df_tmp = df_tmp[["id", "discrimination"]]
            if len(df_ensemble) == 0:
                df_ensemble = df_tmp
            else:
                df_ensemble = df_ensemble.merge(df_tmp, on="id", how="inner")
            df_ensemble[f"v_{i}"] = df_ensemble["discrimination"].values
            df_ensemble.drop("discrimination", axis=1, inplace=True)
        df_ensemble["discrimination"] = df_ensemble.apply(
            lambda x: 1 if np.average([x[f"v_{i}"] for i in range(len(files))]) >= 0.5 else 0, axis=1)
        df_ensemble.to_csv(os.path.join("output", "ensemble", "ensemble.csv"), index=False)

def get_value_counts(file):
    df_submission = pd.read_csv(os.path.join("output", file), header=None)
    df_submission.columns = ["is_pair"]
    print("Stats", df_submission["is_pair"].value_counts())

if __name__ == "__main__":
    running_params = {
        "generate_random_predictions": False,
        "update_predictions": False,
        "combination": False,
        "estimate_score": False,
        "analyze_s3": False,
        "enhance_s3": True,
        "ensemble": False,
        "ensemble_files": [
            os.path.join("five_fold_v1", "S2_balance_fold_by_distance.csv"),
            os.path.join("five_fold_v2", "S2_balance_fold_by_ranking.csv"),
            os.path.join("five_fold_v2", "S2_balance_fold_majority_vote.csv")
        ],
    }
    '''
    [
        "submission_mode_3_S2_balance_fold_by_distance.csv.txt",
        "submission_mode_3_S2_balance_fold_by_ranking.csv.txt",
        "submission_mode_3_S2_balance_fold_majority_vote.csv.txt"
    ]
    '''

    if running_params["generate_random_predictions"]:
        for mode in [0, 1, 2]:
            generate_random_predictions(mode)

    if running_params["update_predictions"]:
        folder = "five_fold_v2"
        for S1_file in ["S1_balance_fold_majority_vote.csv",
                        "S1_balance_fold_by_ranking.csv"]:
            generate_random_predictions(mode=3, verbose=0,
                S1_folder=folder, S1_file=S1_file)

        for S2_file in ["S2_balance_fold_majority_vote.csv",
                        "S2_balance_fold_by_ranking.csv"]:
            generate_random_predictions(mode=3, verbose=0,
                S2_folder=folder, S2_file=S2_file)

    if running_params["combination"]:
        generate_random_predictions(mode=3, verbose=0,
            S1_folder="five_fold_v1", S1_file="S1_balance_fold_by_ranking.csv", S1_percentage=100,
            S2_folder="ensemble", S2_file="ensemble.csv", S2_percentage=100)

    if running_params["estimate_score"]:
        infer_score(s1_score=0.5509, s1_percentage=1.0,
                    s2_score=0.6386, s2_percentage=1.0)

    if running_params["analyze_s3"]:
        analyze_S3()

    if running_params["enhance_s3"]:
        enhance_S3(file="submission_mode_3_S1_balance_fold_by_ranking.csv_ensemble.csv.txt")

    if running_params["ensemble"]:
        ensemble(running_params["ensemble_files"], is_submission=False)

    #get_value_counts(file="submission_mode_3_S1_balance_fold_by_ranking.csv_ensemble.csv.txt_adjusted_s3_15012")
    #get_value_counts(file="submission_mode_3_S1_balance_fold_by_ranking.csv_ensemble.csv.txt_adjusted_s3_15007")