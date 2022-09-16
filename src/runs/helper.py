from constants import *
from random import shuffle
from utils import util

def write_to_file(samples, file_path):
	output = open(file_path, 'w')
	for sample in samples:
		output.write(sample)
	output.close()

def generate_train_val(arch_id, model_id, task_name, training_percent=0.8):
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

def generate_train_val_fold(arch_id, model_id, task_name):
	from shutil import copyfile
	#step 1: use the main training and validation sets as fold 0
	# copyfile('{}{}_train.txt'.format(TRAIN_VAL_PATH, task_name),
	# 		'{}{}_fold0_train.txt'.format(TRAIN_VAL_PATH, task_name))
	# copyfile('{}{}_val.txt'.format(TRAIN_VAL_PATH, task_name),
	# 		'{}{}_fold0_val.txt'.format(TRAIN_VAL_PATH, task_name))

	#step 2: generate the remaining 9 folds
	train_samples = list(open('{}{}_train_fold_1.txt'.format(TRAIN_VAL_PATH, task_name), 'r'))
	val_samples = list(open('{}{}_val_fold_1.txt'.format(TRAIN_VAL_PATH, task_name), 'r'))
	NUM_VAL_SAMPLES = 2000
	for train_id in range(4):
		val_samples_ = train_samples[(train_id * NUM_VAL_SAMPLES) : ((train_id + 1) * NUM_VAL_SAMPLES)]
		train_samples_ = train_samples[0 : (train_id * NUM_VAL_SAMPLES)] \
						+ train_samples[((train_id + 1) * NUM_VAL_SAMPLES) : ] \
						+ val_samples
		assert(len(train_samples_) % NUM_VAL_SAMPLES == 0)
		write_to_file(train_samples_, '{}{}_train_fold_{}.txt'.format(TRAIN_VAL_PATH, task_name, train_id+2))
		write_to_file(val_samples_, '{}{}_val_fold_{}.txt'.format(TRAIN_VAL_PATH, task_name, train_id+2))


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
		write_to_file(shadow_val_samples, '{}{}_shadow{}_val.txt'.format(TRAIN_VAL_PATH, task_name, shadow_id+1))
		write_to_file(shadow_train_samples, '{}{}_shadow{}_train.txt'.format(TRAIN_VAL_PATH, task_name, shadow_id+1))

def generate_discrimination_val(arch_id, model_id, task_name):
	val_path = '{}{}_val.txt'.format(TRAIN_VAL_PATH, task_name)
	image_paths, encode_paths = [], [] 
	val_samples = list(open(val_path, 'r'))
	for val_sample in val_samples:
		[image_path, encode_path] = val_sample.strip().split(',')
		image_paths.append(image_path)
		encode_paths.append(encode_path)

	val_file = open("{}{}_val_discrimination.txt".format(TRAIN_VAL_PATH, task_name), 'w')
	N = len(val_samples)
	for sample_id in range(N):
		val_file.write("{},{},{}\n".format(1, image_paths[sample_id], encode_paths[sample_id]))

		random_id = random.randint(sample_id + 1, sample_id + N - 1) % N
		val_file.write("{},{},{}\n".format(0, image_paths[sample_id], encode_paths[random_id]))

		random_id = random.randint(sample_id + 1, sample_id + N - 1) % N
		val_file.write("{},{},{}\n".format(0, image_paths[random_id], encode_paths[sample_id]))
	val_file.close()

def generate_discrimination_val_fold(arch_id, model_id, task_name):
	for fold_id in range(1, 6):
		val_path = '{}{}_val_fold_{}.txt'.format(TRAIN_VAL_PATH, task_name, fold_id)
		image_paths, encode_paths = [], [] 
		val_samples = list(open(val_path, 'r'))
		for val_sample in val_samples:
			[image_path, encode_path] = val_sample.strip().split(',')
			image_paths.append(image_path)
			encode_paths.append(encode_path)

		val_file = open("{}{}_val_discrimination_fold_{}.txt".format(TRAIN_VAL_PATH, task_name, fold_id), 'w')
		N = len(val_samples)
		for sample_id in range(N):
			val_file.write("{},{},{}\n".format(1, image_paths[sample_id], encode_paths[sample_id]))

			random_id = random.randint(sample_id + 1, sample_id + N - 1) % N
			val_file.write("{},{},{}\n".format(0, image_paths[sample_id], encode_paths[random_id]))

			random_id = random.randint(sample_id + 1, sample_id + N - 1) % N
			val_file.write("{},{},{}\n".format(0, image_paths[random_id], encode_paths[sample_id]))
		val_file.close()

def generate_pseudo_test(arch_id, model_id, task_name):
	val_path = '{}{}_val.txt'.format(TRAIN_VAL_PATH, task_name)
	pseudo_test_file = open('{}{}_pseudo_test.csv'.format(TRAIN_VAL_PATH, task_name), 'w')
	pseudo_test_file.write("id,input_path,encoded_path\n")

	val_samples = list(open(val_path, 'r'))
	for id, val_sample in enumerate(val_samples):
		pseudo_test_file.write("{},{}".format(id, val_sample))
	pseudo_test_file.close()




