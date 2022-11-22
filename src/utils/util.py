from constants import *
from skimage import transform, io



def preprocessing_image(image_path, input_size=-1):
	image = Image.open(image_path).convert('RGB')
	if input_size != -1:
		image = image.resize((input_size, input_size), Image.ANTIALIAS)
	image = np.asarray(image) / 255
	image = image.transpose(2, 0, 1)
	image = torch.from_numpy(image).float()
	image = torch.unsqueeze(image, 0)
	return image

# def preprocessing_binary_encode(encode_path):
# 	with open(encode_path, "rb") as f:
# 		numpy_data = np.fromfile(f, np.dtype('B'))
# 		data_length = numpy_data.shape[0]
# 		numpy_fill = np.zeros(524288 - data_length) #2 ^ 19
# 		numpy_data_norm = np.concatenate((numpy_data, numpy_fill))
# 		numpy_data_flip = np.flip(numpy_data_norm).copy()
# 		# numpy_data_reshape = numpy_data_flip.reshape(-1, 1024)
# 		numpy_data_reshape = numpy_data_flip.reshape(1, -1)
# 		# print (numpy_data_reshape.shape)

# 		torch_data = torch.from_numpy(numpy_data_reshape).float()
# 		torch_data = torch_data / 255.0
# 		torch_data = torch.unsqueeze(torch_data, 0)

# 		return torch_data

# fasttext = OrderedDict()
# file = list(open(PROJECT_PATH + 'fasttext/train_vectors_norm.csv', 'r'))[1:]
# for line in file:
# 	content = line.strip().split(',')
# 	fasttext[content[-1]] = content[:-1]

def preprocessing_binary_encode(encode_path):
	image_id = encode_path.split('.')[0].split('/')[-1]
	fasttext_ = fasttext[image_id]
	fasttext_ = [float(f) for f in fasttext_]
	fasttext_ = np.array(fasttext_)
	fasttext_ = torch.from_numpy(fasttext_).float()
	fasttext_ = torch.unsqueeze(fasttext_, 0)
	return fasttext_
