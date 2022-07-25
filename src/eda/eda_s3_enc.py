import os
import glob
import time
import fasttext

import numpy as np
import pandas as pd

from collections import Counter

'''
convert all .bin files into a single text file for fasttext training
'''
def convert_bin_to_text(file_limit = -1):
    file_cnt = 1
    fout = open(os.path.join("input", "S3", "text.txt"), "w")

    for folder in ["train", "test"]:
        input_folder = os.path.join("input", "S3", folder, "enc")

        for filename_full in glob.glob(os.path.join(input_folder, "*")):
            filename = filename_full.split(os.path.sep)[-1].split(".")[0]
            print(f"{file_cnt}: processing {filename}")
            texts = []
            with open(filename_full, "rb") as fin:
                byte = fin.read(1)
                while byte:
                    if byte == '':
                        break
                    texts.append(byte.hex())
                    byte = fin.read(1)

            fout.write(" ".join(texts) + "\n")
            if file_cnt == file_limit:
                break
            file_cnt += 1

        if file_cnt == file_limit:
            break

    fout.close()

'''
train an nlp model from an input text file
'''
def train_nlp_model():
    model = fasttext.train_unsupervised(os.path.join("input", "S3", "text.txt"), dim=128)
    model.save_model(os.path.join("input", "S3", "fasttext.model"))

def generate_embeddings(file_limit = -1):
    model = fasttext.load_model(os.path.join("input", "S3", "fasttext.model"))
    for folder in ["train", "test"]:
        input_folder = os.path.join("input", "S3", folder, "enc")
        embedding_vectors = []

        file_cnt = 1
        for filename_full in glob.glob(os.path.join(input_folder, "*")):
            filename = filename_full.split(os.path.sep)[-1].split(".")[0]
            print(f"{file_cnt}: processing {filename}")
            texts = []
            with open(filename_full, "rb") as fin:
                byte = fin.read(1)
                while byte:
                    if byte == '':
                        break
                    texts.append(byte.hex())
                    byte = fin.read(1)

            embedding_vectors.append(model.get_sentence_vector(" ".join(texts)))
            if file_cnt == file_limit:
                break
            file_cnt += 1

        columns = [f"v_{i}" for i in range(128)]
        df = pd.DataFrame(data=embedding_vectors, columns=columns)
        print(df.head())
        df.to_csv(os.path.join("input", "S3", folder, "vectors.csv"), index=False)

        if file_cnt == file_limit:
            break

'''
analyze the structure of enc binary files of S3
ending file with b''
equal distribution of byte values from 0 to 255
'''
def analyze_enc_files(task='S3'):
    for filename in glob.glob(os.path.join("input", task, "train", "enc", "*")):
        print(filename)
        bytes = []
        with open(filename, "rb") as f:
            byte = f.read(1)
            while byte:
                if byte == '':
                    break
                bytes.append(byte.hex())
                byte = f.read(1)

        print(bytes[:16], bytes[-16:], len(bytes))
        c = Counter(bytes)
        print(len(c), c)
        break

'''
only for double check
'''
def analyze_enc_files_np(task='S3'):
    for filename in glob.glob(os.path.join("input", task, "train", "enc", "*")):
        print(filename)
        with open(filename, "rb") as f:
            numpy_data = np.fromfile(f, np.dtype('B'))
            print(numpy_data[:16], numpy_data[-16:], numpy_data.shape)
        break

if __name__ == "__main__":
    start = time.time()
    #analyze_enc_files(task='S3')
    #print(f"Elapsed time: {time.time() - start}")
    #analyze_enc_files_np(task='S3')
    #print(f"Elapsed time: {time.time() - start}")
    running_files = -1
    convert_bin_to_text(file_limit=running_files)
    train_nlp_model()
    generate_embeddings(file_limit=running_files)