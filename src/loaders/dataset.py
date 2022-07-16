from constants import *




class Dataset(Dataset):
	def __init__(self, ann_file, transform=None):
		self.ann_file = TRAIN_VAL_PATH + ann_file
		self.transform = transform
		self.dataset = list(open(self.ann_file, 'r'))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		sample = self.extract_content(self.dataset[idx])
		if self.transform:
			sample = self.transform(sample)
		
		sample_data = tuple([s_data for s_data in sample.values()])
		return sample_data


	def extract_content(self, content_):
		[image_path, encode_path] = content_.strip().split(',')

		content = OrderedDict()
		content['image_path'] = image_path
		content['encode_path'] = encode_path
		
		content['image'] = Image.open("{}{}".format(DATA_PATH, image_path))
		content['encode'] = Image.open("{}{}".format(DATA_PATH, encode_path))
		content['image_grayscale'] = content['image'].convert('L') #.convert('RGB')

		return content


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

class Rotate(object):
	def __init__(self, is_training):
		self.is_training = is_training

	def __call__(self, sample):
		if self.is_training:
			image = sample['image']
			assert isinstance(image, Image.Image)
			rotate_degrees = [0, 90, 180, 270]
			shuffle(rotate_degrees)
			rotate_degree = rotate_degrees[0]
			if rotate_degree > 0:
				for key in ['image', 'encode', 'image_grayscale']:
					key_ = sample[key]			
					sample[key] = key_.transpose(
						Image.__getattribute__("ROTATE_{}".format(rotate_degree)))

		return sample


class ToNumpy(object):

	def __call__(self, sample):
		for key in ['image', 'encode', 'image_grayscale']:
			sample[key] = np.array(sample[key])
			# print (key, sample[key].shape)
		return sample

class AddGaussianNoise(object):
	def __init__(self, is_training):
		self.is_training = is_training
		self.gaussian_mean = 0.0
		self.gaussian_std = 5.0

	def __call__(self, sample):
		if self.is_training:
			image = sample['image']
			gaussian = np.random.normal(self.gaussian_mean, self.gaussian_std, image.shape).astype(np.float64)
			sample['image'] = image + gaussian

		return sample

class ToTensor(object):
	def __call__(self, sample):
		np_gray = np.expand_dims(sample['image_grayscale'], axis=2)
		sample['image_grayscale'] = np_gray
		for key in ['image', 'encode', 'image_grayscale']:
			image = sample[key].transpose((2, 0, 1))
			sample[key] = torch.from_numpy(image).float()
		return sample


class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, sample):
		for key in ['image', 'encode', 'image_grayscale']:
			sample[key] = sample[key] / 255.0
		
		# encode = sample['encode'] / 255.0
		# for t, m, s in zip(encode, self.mean, self.std):
		# 	t.sub_(m).div_(s)
		# sample['encode'] = encode
		return sample






