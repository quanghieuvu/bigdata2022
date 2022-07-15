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

def generate_train_val_shadow(arch_id, model_id, task_name, num_shadows=4):
	train_path = '{}{}_train.txt'.format(TRAIN_VAL_PATH, task_name)
	train_samples = list(open(train_path, 'r'))
	shuffle(train_samples)
	num_val = len(train_samples) // num_shadows
	for shadow_id in range(num_shadows):
		begin_val = int(shadow_id * len(train_samples) / num_shadows)
		finish_val = min(len(train_samples), int((shadow_id + 1) * len(train_samples) / num_shadows))
		shadow_val_samples = train_samples[begin_val: finish_val]
		shadow_train_samples = [s for s in train_samples if s not in shadow_val_samples]
		write_to_file(shadow_val_samples, '{}{}_shadow{}_val.txt'.format(TRAIN_VAL_PATH, task_name, shadow_id))
		write_to_file(shadow_train_samples, '{}{}_shadow{}_train.txt'.format(TRAIN_VAL_PATH, task_name, shadow_id))






