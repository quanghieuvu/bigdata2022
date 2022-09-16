from constants import *
from loaders.dataset_s3 import *

class Loader():
	def __init__(self, ann_file, batch_size, is_training, input_size=INPUT_SIZE):		
		self.batch_size = batch_size
		self.name = ann_file.split('.')[0]
		self.is_training = is_training
		dataset = Dataset(ann_file=ann_file,
								transform=transforms.Compose([
									Rescale((input_size, input_size)),
									ToNumpy(),
									ToTensor(),
									Normalize(IMAGENET_MEAN, IMAGENET_STD),
									]),
								)

		self.loader = DataLoader(dataset, batch_size=self.batch_size,
										shuffle=is_training, num_workers=4)
		self.length = len(self.loader)
		print ("Number of batches: ", self.length)



