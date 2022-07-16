from constants import *
from skimage import transform, io



def preprocessing_image(image_path, input_size=-1):
    image = Image.open(image_path)
    if input_size != -1:
    	image = image.resize((input_size, input_size), Image.ANTIALIAS)
    image = np.asarray(image) / 255
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image, 0)
    return image