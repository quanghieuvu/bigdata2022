import os
import glob

'''
analyze the structure of enc binary files of S3
'''
def analyze_s3_enc_files():
    for filename in glob.glob(os.path.join("input", "S3", "train", "enc", "*")):
        print(filename)

if __name__ == "__main__":
    analyze_s3_enc_files()