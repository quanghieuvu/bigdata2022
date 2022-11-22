from constants import *




class Dataset(Dataset):
	def __init__(self, ann_file, transform=None):
		self.ann_file = TRAIN_VAL_PATH + ann_file
		self.transform = transform
		stop = 32 if 'val' in ann_file else 512
		self.dataset = list(open(self.ann_file, 'r'))
		self.fasttext = self.load_fasttext()

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		sample = self.extract_content(self.dataset[idx])
		if self.transform:
			sample = self.transform(sample)
		
		sample_data = tuple([s_data for s_data in sample.values()])
		return sample_data

	def load_fasttext(self):
		fasttext = OrderedDict()
		file = list(open(PROJECT_PATH + 'fasttext/train_vectors_norm.csv', 'r'))[1:]
		for line in file:
			content = line.strip().split(',')
			fasttext[content[-1]] = content[:-1]
		return fasttext

	def extract_content(self, content_):
		[image_path, encode_path] = content_.strip().split(',')

		content = OrderedDict()
		content['image_path'] = image_path
		content['encode_path'] = encode_path
		
		content['image'] = Image.open("{}{}".format(DATA_PATH, image_path)).convert('RGB')
		content['encode'] = self.read_binary_file(DATA_PATH + encode_path)
		# content['encode'] = self.extract_fasttext(image_path)
		content['image_grayscale'] = content['image'].convert('L') #.convert('RGB')

		return content

	def read_binary_file(self, encode_path):
		with open(encode_path, "rb") as f:
			numpy_data = np.fromfile(f, np.dtype('B'))
			# print ('numpy_data', numpy_data.shape)
			data_length = numpy_data.shape[0]
			numpy_fill = np.zeros(524288 - data_length) #2 ^ 19
			# print ('numpy_fill', numpy_fill.shape)
			numpy_data_norm = np.concatenate((numpy_data, numpy_fill))
			numpy_data_flip = np.flip(numpy_data_norm).copy()
			# print ('numpy_data_flip begin', numpy_data_flip[:128])
			# print ('numpy_data_flip end', numpy_data_flip[-128:])
			# numpy_data_reshape = numpy_data_flip.reshape(-1, 1024)
			# numpy_data_reshape = numpy_data_flip.reshape(1, -1)
			numpy_data_reshape = numpy_data_flip.reshape(-1)
			numpy_data_reshape /= 255.
			# print ('numpy_data_reshape', numpy_data_reshape.shape)
			return numpy_data_reshape
	
	def extract_fasttext(self, image_path):
		image_id = image_path.split('.')[0].split('/')[-1]
		fasttext = self.fasttext[image_id]
		fasttext = [float(f) for f in fasttext]
		return fasttext


class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, tuple)
		self.output_size = output_size

	def __call__(self, sample):
		image = sample['image']
		image_L = sample['image_grayscale']
		assert isinstance(image, Image.Image)
		sample['image'] = image.resize(self.output_size, Image.ANTIALIAS)
		sample['image_grayscale'] = image_L.resize(self.output_size, Image.ANTIALIAS)
		return sample


class ToNumpy(object):

	def __call__(self, sample):
		for key in ['image', 'encode', 'image_grayscale']:
			sample[key] = np.array(sample[key])
		return sample


class ToTensor(object):
	def __call__(self, sample):
		np_image = sample['image']
		np_image = np_image.transpose((2, 0, 1))
		sample['image'] = torch.from_numpy(np_image).float()

		np_encode = sample['encode']
		sample['encode'] = torch.from_numpy(np_encode).float()

		np_gray = np.expand_dims(sample['image_grayscale'], axis=2)
		np_gray = np_gray.transpose((2, 0, 1))
		sample['image_grayscale'] = torch.from_numpy(np_gray).float()

		return sample


class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, sample):
		for key in ['image', 'image_grayscale']:
			sample[key] = sample[key] / 255.0
		
		return sample






