from constants import *
from models import buffalo_s3 as model_
from loaders import loader_s3 as loader_
from archs import buffalo_1a, buffalo_1i


def train(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	print ('training on {}'.format(task_name))	

	train_loader = loader_.Loader(
					ann_file='{}_train.txt'.format(task_name),
					batch_size=128,
					is_training=True,
					input_size=SMALL_SIZE)

	val_loader = loader_.Loader(
					ann_file='{}_val.txt'.format(task_name),
					batch_size=8,
					is_training=False,
					input_size=SMALL_SIZE)
	
	# model = model_.Model(buffalo_1a.Arch(output_depth), buffalo_1b.Arch(output_depth))
	model = model_.Model(buffalo_1a.Arch(output_depth), buffalo_1i.Arch(output_depth))
	model.train(train_loader, val_loader, model_id)


def eval_discrimination(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale

	ckpt_path = CKPT_PATH + "BUFFALO_v{}_{}_train_{}.ckpt".format(arch_id, task_name, model_id)
	# model = model_.Model(buffalo_1a.Arch(output_depth), buffalo_1b.Arch(output_depth))
	model = model_.Model(buffalo_1a.Arch(output_depth), buffalo_1i.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	# model.eval_discrimination(val_file='{}_val_debug_discrimination.txt'.format(task_name), task_name=task_name)
	model.eval_discrimination(val_file='{}_val_discrimination.txt'.format(task_name), task_name=task_name)