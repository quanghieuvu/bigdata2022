from constants import *
from utils import network, util_log, util_os, util_image



class Model():
	def __init__(self, arch):
		self.model = arch.to(device)
		self.name = arch.name
		print ('Architecture Network: {}'.format(self.name))
		self.output_depth = arch.output_depth   #add background

		self.opt_model_path = None
		self.last_model_path = None
		self.num_epoch = 512
		self.max_lr = 2e-4
		self.base_lr = 1e-6
		self.lr_step = 32
		self.lr = self.max_lr		

		self.criterion = nn.MSELoss().to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

		self.loss = None

		self.epoch_loss_ = 0  #for each of train or val loss
		self.epoch_loss = 0   #sum of train and val loss
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
		if loader.is_training: self.model.train()
		else: self.model.eval()
			
		self.refresh_eval_and_loss()

		for _, batch in enumerate(loader.loader):
			self.run_batch(loader, batch)
		
		self.epoch_loss_ /= loader.length
		self.epoch_loss += self.epoch_loss_
		self.summary_epoch(loader)


	def run_batch(self, loader, batch_data):
		encode = batch_data[3].to(device)
		image = batch_data[2].to(device) if self.output_depth == 3 else batch_data[4].to(device)
		if image.size(0) != loader.batch_size: return

		pred_image = self.model(encode)	
		self.loss = self.criterion(pred_image, image)
		self.epoch_loss_ += self.loss.item() #* loader.batch_size            
		
		self.backprop(loader, self.loss)

	def show_epoch_info(self, epoch_id):
		self.log.info ('\nEpoch [{}/{}], lr {:.6f}, runtime {:.3f}'.format(epoch_id, self.num_epoch, 
															self.lr, time.time()-self.current_time))
		self.current_time = time.time()

	def summary_epoch(self, loader):
		self.log.info("{}: loss={:.6f}".format(loader.name, self.epoch_loss_))

	def save_checkpoint(self, epoch_id):
		if self.min_epoch_loss > self.epoch_loss:
			self.min_epoch_loss = self.epoch_loss
			torch.save(self.model.state_dict(), self.opt_model_path)
			self.log.info ("checkpoint saved")

		if epoch_id % self.lr_step == 0:
			torch.save(self.model.state_dict(), self.last_model_path)

	def refresh_eval_and_loss(self):
		self.epoch_loss_ = 0

	def backprop(self, loader, loss):
		if loader.is_training:
			self.optimizer.zero_grad()
			self.loss.backward()
			self.optimizer.step()

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
		self.model.load_state_dict(torch.load(ckpt_path))

	def update_lr(self, epoch_id):
		if epoch_id % self.lr_step == 0: self.max_lr *= 0.88
		self.lr = self.max_lr - (self.max_lr - self.base_lr) * (epoch_id % self.lr_step) / self.lr_step
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def save_decoded_map(self, ckpt_path, loader):
		self.load_ckpt(ckpt_path)
		self.model.eval()
		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		result_dir = util_os.gen_dir("{}{}_on_{}".format(RESULT_PATH, model_name, loader.name), True)
		print ("decoded maps are save at " + result_dir)

		for batch_data_id, batch_data in tqdm(enumerate(loader.loader)):
			image_paths = batch_data[0]
			encode_paths = batch_data[1]
			encode = batch_data[3].to(device)
			if encode.size(0) != loader.batch_size: return

			pred_image = self.model(encode)
			# print ('pred_image', pred_image.size())
			np_pred_image = pred_image.data.cpu().numpy() * 255.0
			np_pred_image[np_pred_image > 255] = 255
			np_pred_image[np_pred_image < 0] = 0
			np_pred_image = np_pred_image.transpose(0, 2, 3, 1)
			for batch_id in range(loader.batch_size):
				image = Image.open("{}{}".format(DATA_PATH, image_paths[batch_id]))
				encode = Image.open("{}{}".format(DATA_PATH, encode_paths[batch_id]))

				image_name = image_paths[batch_id].split('/')[-1].split('.')[0]
				pred_path = "{}{}.png".format(result_dir, image_name)

				if self.output_depth == 3:
					pil_pred_image = Image.fromarray(np_pred_image[batch_id].astype('uint8'), 'RGB')
				else:
					# print ('np_pred_image[batch_id]', np_pred_image[batch_id].shape)
					np_pred_image_ = np.squeeze(np_pred_image[batch_id], axis=2)
					pil_pred_image = Image.fromarray(np_pred_image_.astype('uint8'), 'L').convert('RGB')

				vis_image = util_image.stack_in_row([image, encode, pil_pred_image])
				vis_image.save(pred_path)











	







