from constants import *
from models import buffalo as model_
from loaders import loader as loader_
from archs import buffalo_0, buffalo_1

arch_list = [buffalo_0, buffalo_1]

"""
Remark:
model_id from 0 to 9: to generate RGB image
model_id >= 10: to generate grayscale image
model_id % 2 == 0: discrimination_loss = contrastive
model_id % 2 == 1: discrimination_loss = triplet
"""

def train(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	print ('training on {}'.format(task_name))	
	arch = arch_list[int(arch_id)]

	for fold_id in range(1, 6):
		# train_loader = loader_.Loader(
		# 				ann_file='{}_train_fold_{}.txt'.format(task_name, fold_id),
		# 				batch_size=6,
		# 				is_training=True)

		train_loader = loader_.Loader(
						ann_file='{}_train_fold_extra_{}.txt'.format(task_name, fold_id),
						batch_size=8,
						is_training=True)

		val_loader = loader_.Loader(
						ann_file='{}_val_fold_{}.txt'.format(task_name, fold_id),
						batch_size=4,
						is_training=False)
		
		model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth=output_depth))

		model.train(train_loader, val_loader, model_id)

def eval_discrimination(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	arch = arch_list[int(arch_id)]

	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_{}.ckpt".format(arch_id, task_name, model_id)
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.eval_discrimination(val_file='{}_val_discrimination.txt'.format(task_name))

#CUDA_VISIBLE_DEVICES=0 python -W ignore main.py buffalo eval_discrimination_fold 0 1 S1
def eval_discrimination_fold(arch_id, model_id, task_name):
	fold_id = int(model_id)
	output_depth = 1 #grayscale
	arch = arch_list[int(arch_id)]

	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_fold_extra_{}_11.ckpt".format(arch_id, task_name, fold_id)
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.eval_discrimination(val_file='{}_val_discrimination_fold_{}.txt'.format(task_name, fold_id))

def save_decoded_map(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	arch = arch_list[int(arch_id)]

	val_loader = loader_.Loader(
					ann_file='{}_val.txt'.format(task_name, model_id),
					batch_size=4,
					is_training=False)

	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_{}.ckpt".format(arch_id, task_name, model_id)
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.save_decoded_map(ckpt_path, val_loader)

def generate_test_result(arch_id, model_id, task_name):
	discrimination_distance_threshold = 1.63 # S2 1.63 S1 0.65
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	arch = arch_list[int(arch_id)]

	test_path = '{}BigDataCup2022/{}/test/test.csv'.format(DATA_PATH, task_name)
	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_{}.ckpt".format(arch_id, task_name, model_id)

	print ('ckpt_path:', ckpt_path)
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.generate_test_result(task_name, test_path, discrimination_distance_threshold)

def generate_test_result_fold(arch_id, model_id, task_name):
	threshold_dict = {'S1': [0.66, 0.58, 0.47, 0.53, 0.5],
					  'S2': [1.88, 1.47, 1.71, 2.1, 1.62]}
	fold_id = int(model_id)
	discrimination_distance_threshold = threshold_dict[task_name][fold_id - 1]
	output_depth = 1 #grayscale
	arch = arch_list[int(arch_id)]

	test_path = '{}BigDataCup2022/{}/test/test.csv'.format(DATA_PATH, task_name)
	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_fold_extra_{}_11.ckpt".format(arch_id, task_name, fold_id)

	print ('ckpt_path:', ckpt_path)
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.generate_test_result(task_name, test_path, discrimination_distance_threshold, fold_id)

def merge_fold_results(arch_id, model_id, task_name):
	result_dict = defaultdict(lambda : [], OrderedDict())
	for fold_id in range(1, 6):
		fold_file = list(open("{}extra/{}_balance_fold_{}.csv".format(RESULT_PATH, task_name, fold_id), 'r'))
		for line in fold_file[1:]:
			line = line.strip()
			result_dict[line[:-2]].append(int(line[-1]))

	out_file = open("{}extra/{}_balance_fold_majority_vote.csv".format(RESULT_PATH, task_name), 'w')
	out_file.write("id,input_path,encoded_path,discrimination\n")
	for key, value in result_dict.items():
		# print (value)
		final_decision = int(sum(value) > len(value) / 2)
		out_file.write("{},{}\n".format(key, final_decision))
	out_file.close()

def merge_fold_results_ranking(arch_id, model_id, task_name):
	result_dict = defaultdict(lambda : 0, OrderedDict())
	for fold_id in range(1, 6):
		fold_file = list(open("{}extra/{}_ranking_fold_{}.csv".format(RESULT_PATH, task_name, fold_id), 'r'))
		for line in fold_file[1:]:
			line = line.strip()
			components = line.split(',')
			key = ','.join(components[:-1])
			value = float(components[-1])
			result_dict[key] += value

	ranking_list = [(value, key) for key, value in result_dict.items()]
	ranking_list.sort()
	ranking_dict = OrderedDict()
	for i, (rank_, line_) in enumerate(ranking_list):
		ranking_dict[line_] = int(i < len(ranking_list) / 2)

	out_file = open("{}extra/{}_balance_fold_by_ranking.csv".format(RESULT_PATH, task_name), 'w')
	out_file.write("id,input_path,encoded_path,discrimination\n")
	for key, value in result_dict.items():
		# print (value)
		final_decision = ranking_dict[key]
		out_file.write("{},{}\n".format(key, final_decision))
	out_file.close()

def add_easy_test_to_fold(arch_id, model_id, task_name):
	#step 1: pick easy test
	result_dict = defaultdict(lambda : [], OrderedDict())
	for fold_id in range(1, 6):
		fold_file = list(open("{}{}_balance_fold_{}.csv".format(RESULT_PATH, task_name, fold_id), 'r'))
		for line in fold_file[1:]:
			line = line.strip()
			result_dict[line[:-2]].append(int(line[-1]))

	easy_test_samples = []
	for key, value in result_dict.items():
		consistancy = int(sum(value))
		if consistancy != 5: continue
		easy_test_samples.append(key)

	#step 2: get content in fold file
	for fold_id in range(1, 6):
		extra_file = open("{}{}_train_fold_extra_{}.txt".format(TRAIN_VAL_PATH, task_name, fold_id), 'w')
		fold_file = list(open("{}{}_train_fold_{}.txt".format(TRAIN_VAL_PATH, task_name, fold_id), 'r'))
		for line in fold_file:
			extra_file.write(line)

		for key in easy_test_samples:
			components = key.split(',')
			content = ','.join(components[1:])
			extra_file.write(content + '\n')
		extra_file.close()
