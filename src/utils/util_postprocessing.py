import os
import glob
import pandas as pd

def post_processing(base_file):
    # compute scores of the last 10000 predictions of S3
    files = glob.glob(os.path.join("input", "submissions", "*.txt"))
    scores = [0] * 10000
    for file in files:
        score = int(file.split("_")[-1].split(".")[0])
        print(f"Processing {file}, score {score}")
        df_submission = pd.read_csv(file, header=None)
        df_submission.columns = ["is_pair"]
        values = df_submission["is_pair"].values.tolist()[20000:]
        for value, index in zip(values, range(10000)):
            if value == 1:
                scores[index] += score / 10000
            else:
                scores[index] += (1 - score / 10000)

    # sort the score of 10000 samples
    sorted_scores, sorted_indices = zip(*sorted(zip(scores, range(10000)), reverse=True))
    print(sorted_scores)
    s3_predictions = [0] * 10000
    for index in sorted_indices[:5000]:
        s3_predictions[index] = 1

    # update S3 predictions on the base predictions of S2 and S3
    df_submission = pd.read_csv(os.path.join("output", base_file), header=None)
    df_submission.columns = ["is_pair"]
    print("Before", df_submission["is_pair"].value_counts(),
          df_submission[20000:]["is_pair"].value_counts())

    new_values = df_submission["is_pair"].values.tolist()[:20000] + s3_predictions
    df_submission["is_pair"] = new_values
    print("After", df_submission["is_pair"].value_counts(),
          df_submission[20000:]["is_pair"].value_counts())

    df_submission[["is_pair"]].to_csv(os.path.join("output", f"{base_file}_poss_processing.txt"), index=False, header=False)

if __name__ == "__main__":
    post_processing(base_file="submission_mode_3_S1_balance_fold_by_ranking.csv_ensemble.csv.txt")