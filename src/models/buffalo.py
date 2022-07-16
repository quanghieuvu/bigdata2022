from constants import *
from utils import util, network, util_log, util_os, util_image



class Model():
	def __init__(self, arch_image, arch_encode):
		self.model_image = arch_image.to(device)
		self.model_encode = arch_encode.to(device)
		self.name = arch_image.name
		print ('Architecture Network: {}'.format(arch_image.name))
		self.output_depth = arch_image.output_depth   #add background

		self.opt_model_path = None
		self.last_model_path = None
		self.num_epoch = 512
		self.max_lr = 2e-4
		self.base_lr = 1e-6
		self.lr_step = 32
		self.lr = self.max_lr		

		self.criterion = nn.MSELoss().to(device)
		self.triplet_criterion = nn.MSELoss(reduction='none').to(device)
		self.optimizer_image = torch.optim.Adam(self.model_image.parameters(), lr=self.lr, weight_decay=1e-5)
		self.optimizer_encode = torch.optim.Adam(self.model_encode.parameters(), lr=self.lr, weight_decay=1e-5)

		self.loss = None
		self.image_decryption_loss = None
		self.encode_decryption_loss = None

		self.epoch_loss_ = 0  #for each of train or val loss
		self.epoch_loss = 0   #sum of train and val loss
		self.epoch_loss_image = 0
		self.epoch_loss_encode = 0
		self.epoch_loss_similarity = 0
		self.epoch_loss_difference = 0
		self.epoch_id = 0
		self.min_epoch_loss = 1e5
		self.current_time = time.time()

		self.log = None
		self.model_id = None


	def train(self, train_loader, val_loader, model_id=0):
		self.set_opt_model_path_and_log(train_loader, model_id)
		self.model_id = model_id
		self.set_log(self.opt_model_path)

		for epoch_id in range(1, self.num_epoch + 1):
			self.epoch_id = epoch_id

			self.show_epoch_info(epoch_id)
			self.epoch_loss = 0

			self.run_epoch(train_loader)
			self.run_epoch(val_loader)
			
			self.save_checkpoint(epoch_id)			
			self.update_lr(epoch_id)

	def test(self, test_loader):
		self.run_epoch(test_loader)


	def run_epoch(self, loader):
		if loader.is_training: 
			self.model_image.train()
			self.model_encode.train()
		else: 
			self.model_image.eval()
			self.model_encode.eval()
			
		self.refresh_eval_and_loss()

		for _, batch in enumerate(loader.loader):
			self.run_batch(loader, batch)
		
		self.epoch_loss_ /= loader.length
		self.epoch_loss += self.epoch_loss_
		self.epoch_loss_image /= loader.length
		self.epoch_loss_encode /= loader.length
		self.epoch_loss_similarity /= loader.length
		self.epoch_loss_difference /= loader.length
		self.summary_epoch(loader)


	def run_batch(self, loader, batch_data):
		encode = batch_data[3].to(device)
		image = batch_data[2].to(device)
		label = batch_data[4].to(device) if self.output_depth == 1 else image
		if image.size(0) != loader.batch_size: return

		pred_image, feature_image = self.model_image(image)
		pred_encode, feature_encode = self.model_encode(encode)

		self.image_decryption_loss = self.criterion(pred_image, label)
		self.encode_decryption_loss = self.criterion(pred_encode, label)
		
		assert(loader.batch_size % 2 == 0)
		feature_image_flip = torch.flip(feature_image, [0])
		feature_encode_flip = torch.flip(feature_encode, [0])
		margin = 1.
		D_1a = torch.mean(self.triplet_criterion(feature_image, feature_encode), 1)
		D_0a = torch.mean(self.triplet_criterion(feature_image, feature_encode_flip), 1)
		D_0b = torch.mean(self.triplet_criterion(feature_image_flip, feature_encode), 1)

		relu = torch.nn.ReLU()
		loss_similarity = torch.mean(D_1a)
		loss_difference = torch.mean(relu(margin - D_0a) + relu(margin - D_0b)) / 2

		triplet_loss = torch.mean(relu(margin + D_1a - D_0a) + relu(margin + D_1a - D_0b)) / 2
		if self.model_id % 2 == 0:
			self.loss = self.image_decryption_loss + self.encode_decryption_loss + loss_similarity + loss_difference
		else:
			self.loss = self.image_decryption_loss + self.encode_decryption_loss + triplet_loss

		self.epoch_loss_ += self.loss.item() #* loader.batch_size
		self.epoch_loss_image += self.image_decryption_loss.item()
		self.epoch_loss_encode += self.encode_decryption_loss.item()
		self.epoch_loss_similarity += loss_similarity.item()
		self.epoch_loss_difference += loss_difference.item() 

		
		self.backprop(loader, self.loss)

	def show_epoch_info(self, epoch_id):
		self.log.info ('\nEpoch [{}/{}], lr {:.6f}, runtime {:.3f}'.format(epoch_id, self.num_epoch, 
															self.lr, time.time()-self.current_time))
		self.current_time = time.time()

	def summary_epoch(self, loader):
		self.log.info("{}: loss={:.6f}, loss_image={:.6f}, loss_encode={:.6f}, loss_similarity={:.6f}, loss_difference={:.6f}"\
										.format(loader.name, self.epoch_loss_, self.epoch_loss_image, self.epoch_loss_encode,
												 self.epoch_loss_similarity, self.epoch_loss_difference))

	def save_checkpoint(self, epoch_id):
		if self.min_epoch_loss > self.epoch_loss:
			self.min_epoch_loss = self.epoch_loss
			torch.save(self.model_image.state_dict(), self.opt_model_path.replace('.ckpt', '_image.ckpt'))
			torch.save(self.model_encode.state_dict(), self.opt_model_path.replace('.ckpt', '_encode.ckpt'))
			self.log.info ("checkpoint saved")

		if epoch_id % self.lr_step == 0:
			torch.save(self.model_image.state_dict(), self.last_model_path.replace('.ckpt', '_image.ckpt'))
			torch.save(self.model_encode.state_dict(), self.last_model_path.replace('.ckpt', '_encode.ckpt'))

	def refresh_eval_and_loss(self):
		self.epoch_loss_ = 0
		self.epoch_loss_image = 0
		self.epoch_loss_encode = 0
		self.epoch_loss_similarity = 0
		self.epoch_loss_difference = 0

	def backprop(self, loader, loss):
		if loader.is_training:
			self.optimizer_image.zero_grad()
			self.optimizer_encode.zero_grad()
			self.loss.backward()
			self.optimizer_image.step()
			self.optimizer_encode.step()

	def set_opt_model_path_and_log(self, train_loader, model_id):
		model_name = self.name + '_' + train_loader.name + '_' + str(model_id)
		self.opt_model_path = CKPT_PATH + model_name + '.ckpt'
		self.last_model_path = CKPT_PATH + model_name + '_last.ckpt'

		
	def set_log(self, ckpt_path):
		model_name = ckpt_path.split(CKPT_PATH)[-1].split('.ckpt')[0]
		log_path = LOG_PATH + model_name + '.txt'
		self.log = util_log.Allog(log_path)
	
	
	def load_ckpt(self, ckpt_path=None):
		if ckpt_path == "default": ckpt_path = self.opt_model_path
		self.model_image.load_state_dict(torch.load(ckpt_path.replace('.ckpt', '_image.ckpt')))
		self.model_encode.load_state_dict(torch.load(ckpt_path.replace('.ckpt', '_encode.ckpt')))

	def update_lr(self, epoch_id):
		if epoch_id % self.lr_step == 0: self.max_lr *= 0.88
		self.lr = self.max_lr - (self.max_lr - self.base_lr) * (epoch_id % self.lr_step) / self.lr_step
		for param_group in self.optimizer_encode.param_groups:
			param_group['lr'] = self.lr
		for param_group in self.optimizer_image.param_groups:
			param_group['lr'] = self.lr

	def save_decoded_map(self, ckpt_path, loader):
		self.model_encode.eval()
		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		result_dir = util_os.gen_dir("{}{}_on_{}".format(RESULT_PATH, model_name, loader.name), True)
		print ("decoded maps are save at " + result_dir)

		for batch_data_id, batch_data in tqdm(enumerate(loader.loader)):
			image_paths = batch_data[0]
			encode_paths = batch_data[1]
			encode = batch_data[3].to(device)
			if encode.size(0) != loader.batch_size: return

			pred_image, _ = self.model_encode(encode)
			np_pred_image = pred_image.data.cpu().numpy() * 255.0
			np_pred_image[np_pred_image > 255] = 255
			np_pred_image[np_pred_image < 0] = 0
			np_pred_image = np_pred_image.transpose(0, 2, 3, 1)
			for batch_id in range(loader.batch_size):
				image = Image.open("{}{}".format(DATA_PATH, image_paths[batch_id])).resize((INPUT_SIZE, INPUT_SIZE), Image.ANTIALIAS)
				encode = Image.open("{}{}".format(DATA_PATH, encode_paths[batch_id])).resize((INPUT_SIZE, INPUT_SIZE), Image.ANTIALIAS)

				image_name = image_paths[batch_id].split('/')[-1].split('.')[0]
				pred_path = "{}{}.png".format(result_dir, image_name)

				if self.output_depth == 3:
					pil_pred_image = Image.fromarray(np_pred_image[batch_id].astype('uint8'), 'RGB')
				else:
					np_pred_image_ = np.squeeze(np_pred_image[batch_id], axis=2)
					pil_pred_image = Image.fromarray(np_pred_image_.astype('uint8'), 'L').convert('RGB')

				vis_image = util_image.stack_in_row([image, encode, pil_pred_image])
				vis_image.save(pred_path)

	def eval_discrimination(self, val_file):
		self.model_image.eval()
		self.model_encode.eval()

		val_samples = list(open(TRAIN_VAL_PATH + val_file, 'r'))
		true_diff, false_diff = [], []
		for val_sample in tqdm(val_samples):
			[similarity, image_path, encode_path] = val_sample.strip().split(',')
			similarity = int(similarity)
			image = util.preprocessing_image("{}{}".format(DATA_PATH, image_path), INPUT_SIZE).to(device)
			encode = util.preprocessing_image("{}{}".format(DATA_PATH, encode_path)).to(device)

			pred_image, feature_image = self.model_image(image)
			pred_encode, feature_encode = self.model_encode(encode)
			diff = self.criterion(feature_image, feature_encode).item()
			if similarity == 1: 
				true_diff.append(diff)
			else:
				false_diff.append(diff)
		true_diff.sort(reverse=True)
		false_diff.sort()

		join_diff = true_diff + false_diff
		np_true_diff = np.array(true_diff)
		np_false_diff = np.array(false_diff)
		join_diff.sort()

		join_diff = [0] + join_diff + [1e6]
		join_diff = [(join_diff[i] + join_diff[i+1]) / 2 for i in range(len(join_diff) - 1)]
		performances = []
		for threshold in join_diff:
			correct_1 = np.sum((np_true_diff <= threshold).astype('uint8')) / len(true_diff) * 100
			correct_0 = np.sum((np_false_diff > threshold).astype('uint8')) / len(false_diff) * 100
			performance = (correct_1 + correct_0) / 2
			performances.append((performance, correct_1, correct_0, threshold))
		performances.sort(reverse=True)

		peak_threshold = round(performances[0][-1], 2)
		num_bins, bin_size = 25, round(peak_threshold / 100, 3)
		print ("Impact of distance thresholds to discrimination performance:")
		print ("{:<12}{:>12}{:>12}{:>12}".format('threshold', '1', '0', 'average'))
		for step in range(-num_bins, num_bins+1):
			threshold = peak_threshold + step * bin_size
			correct_1 = np.sum((np_true_diff <= threshold).astype('uint8')) / len(true_diff) * 100
			correct_0 = np.sum((np_false_diff > threshold).astype('uint8')) / len(false_diff) * 100
			performance = (correct_1 + correct_0) / 2
			marking = '+++++' if step == 0 else ''
			print ("{:<12.3f}{:>11.2f}%{:>11.2f}%{:>11.2f}% {}"
									.format(threshold, correct_1, correct_0, performance, marking))








		


















	







