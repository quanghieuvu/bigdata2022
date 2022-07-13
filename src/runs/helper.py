from constants import *
from random import shuffle

def write_to_file(samples, file_path):
	output = open(file_path, 'w')
	for sample in samples:
		output.write(sample)
	output.close()

def generate_train_val(arch_id, model_id, task_name, training_percent=0.9):
	train_path = DATA_PATH + 'BigDataCup2022/{}/train/train.csv'.format(task_name)
	train_file = list(open(train_path, 'r'))
	train_samples = []
	for line in train_file[1:]:
		[_, image_path, encode_path] = line.strip().split(',')
		train_samples.append("{},{}\n".format(image_path, encode_path))

	shuffle(train_samples)
	training_size = int(training_percent * len(train_samples))

	timestamp = int(time.time())
	write_to_file(train_samples[:training_size], '{}{}_train_{}.txt'.format(TRAIN_VAL_PATH, task_name, timestamp))
	write_to_file(train_samples[training_size:], '{}{}_val_{}.txt'.format(TRAIN_VAL_PATH, task_name, timestamp))






