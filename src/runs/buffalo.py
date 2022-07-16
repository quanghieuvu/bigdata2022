from constants import *
from models import buffalo as model_
from loaders import loader as loader_
from archs import buffalo_0

arch_list = [buffalo_0]

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

	train_loader = loader_.Loader(
					ann_file='{}_train.txt'.format(task_name, model_id),
					batch_size=8,
					is_training=True)

	val_loader = loader_.Loader(
					ann_file='{}_val.txt'.format(task_name, model_id),
					batch_size=4,
					is_training=False)
	
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.train(train_loader, val_loader, model_id)


def eval_discrimination(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	arch = arch_list[int(arch_id)]

	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_{}.ckpt".format(arch_id, task_name, model_id)
	model = model_.Model(arch.Arch(output_depth), arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.eval_discrimination(val_file='{}_val_discrimination.txt'.format(task_name))

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
