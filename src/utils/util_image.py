from PIL import Image
import numpy as np



def append_images(img_1, img_2, gap=16):
    if img_1 == None: return img_2
    (w_1, h_1) = img_1.size
    (w_2, h_2) = img_2.size
    assert (h_1 == h_2)
    img_new = Image.new('RGB', (w_1 + gap + w_2, h_1), (255, 255, 255))
    img_new.paste(img_1, (0, 0, w_1, h_1))
    img_new.paste(img_2, (w_1 + gap, 0, w_1 + gap + w_2, h_2))
    return img_new
    
def stack_in_row(images, gap=16):
    new_image = None
    for image in images:
        new_image = append_images(new_image, image, gap)
    return new_image