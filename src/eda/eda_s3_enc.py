import os
import glob
import time
import fasttext

import numpy as np
import pandas as pd

from collections import Counter
from nlp_word2vec import word2vec

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
- mode=0: using fasttext
- mode=1: using word2vec class 
'''
def train_nlp_model(mode):
    if mode == 0:
        model = fasttext.train_unsupervised(os.path.join("input", "S3", "text.txt"), dim=128)
        model.save_model(os.path.join("input", "S3", "fasttext.model"))
    elif mode == 1:
        settings = {
            'n': 5,  # dimension of word embeddings
            'window_size': 2,  # context window +/- center word
            'min_count': 0,  # minimum word count
            'epochs': 5000,  # number of training epochs
            'neg_samp': 10,  # number of negative words to use during training
            'learning_rate': 0.01  # learning rate
        }
        np.random.seed(0)

        corpus = []
        fin = open(os.path.join("input", "S3", "text.txt"), "r")
        while True:
            line = fin.readline().strip()
            if line == '':
                break
            corpus.append(line.split())
        fin.close()
        print(f"Done data loading")

        # INITIALIZE W2V MODEL
        w2v = word2vec(settings)

        # generate training data
        start = time.time()
        training_data = w2v.generate_training_data(corpus)
        print(f"Done data preparation in {time.time() - start}")

        # train word2vec model
        start = time.time()
        w2v.train(training_data)
        print(f"Done training in {time.time() - start}")

        print(w2v.word_index)
        print(w2v.w1)

def generate_embeddings(mode = 0, file_limit = -1):
    model = fasttext.load_model(os.path.join("input", "S3", "fasttext.model"))
    for folder in ["train", "test"]:
        input_folder = os.path.join("input", "S3", folder, "enc")
        embedding_vectors = []
        files = []

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

            files.append(filename)
            embedding_vectors.append(model.get_sentence_vector(" ".join(texts)))
            if file_cnt == file_limit:
                break
            file_cnt += 1

        columns = [f"v_{i+1}" for i in range(128)]
        df = pd.DataFrame(data=embedding_vectors, columns=columns)
        df["file"] = files
        print(df.head())
        df.to_csv(os.path.join("input", "S3", folder, f"{folder}_vectors.csv"), index=False)

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
    running_files = 2 #-1
    convert_bin_to_text(file_limit=running_files)
    train_nlp_model(mode=1)
    #generate_embeddings(mode=1, file_limit=running_files)