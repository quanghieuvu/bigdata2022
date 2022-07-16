import os
import glob

from collections import Counter

'''
analyze the structure of enc binary files of S3
ending file with b''
equal distribution of byte values from 0 to 255
'''
def analyze_s3_enc_files():
    for filename in glob.glob(os.path.join("input", "S3", "train", "enc", "*")):
        print(filename)
        bytes = []
        with open(filename, "rb") as f:
            byte = f.read(1)
            while byte:
                byte = f.read(1)
                if byte != b'':
                    bytes.append(byte)
        c = Counter(bytes)
        print(len(bytes), len(c), c)

if __name__ == "__main__":
    analyze_s3_enc_files()