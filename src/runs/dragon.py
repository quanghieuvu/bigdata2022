from constants import *
from models import dragon as model_
from loaders import loader as loader_
from archs import dragon_0, dragon_1

arch_list = [dragon_0, dragon_1]



def train(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	print ('training on {}'.format(task_name))
	
	arch = arch_list[int(arch_id)]

	train_loader = loader_.Loader(
					ann_file='{}_shadow{}_train.txt'.format(task_name, model_id),
					batch_size=12,
					is_training=True)

	val_loader = loader_.Loader(
					ann_file='{}_shadow{}_val.txt'.format(task_name, model_id),
					batch_size=4,
					is_training=False)
	
	model = model_.Model(arch.Arch(output_depth))
	model.train(train_loader, val_loader, model_id)


def save_decoded_map(arch_id, model_id, task_name):
	model_id = int(model_id)
	output_depth = 3 if model_id < 10 else 1 #RGB or grayscale
	arch = arch_list[int(arch_id)]
	val_loader = loader_.Loader(
					ann_file='{}_val.txt'.format(task_name),
					batch_size=2,
					is_training=False)
	ckpt_path = CKPT_PATH + "DRAGON_v{}_{}_train_{}.ckpt".format(arch_id, task_name, model_id)
	model = model_.Model(arch.Arch(output_depth))
	model.load_ckpt(ckpt_path)
	model.save_decoded_map(ckpt_path=ckpt_path, loader=val_loader)


