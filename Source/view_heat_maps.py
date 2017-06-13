from PIL import Image
import sys
from os import listdir
### el programa se ejecutra python view_heat_maps.py 2 48 1. Donde se incluye:
# layer = cantidad de capas
# cells = cantidad de neuronas
# fold = orden de files de entrenamiento y test

def get_images_path(path_dir,case,layer,cell,fold):
	path_view = path_dir + case + "/" + layer + "_layer/" + cell + "_units/" + fold + "_fold/" + "heat_images/"
	images = listdir(path_view)
	return images, path_view 

def open_images(images,path_view):
	for image in images:
		split_image = image.split(" ")
		#image_view = path_view + image
		#recurrent_kernel = Image.open(image_view).show()
		for i in range(0,len(split_image)-1):
			if split_image[i] == "_recurrent_kernel" and split_image[i+1] == "_layer2":
				image_view = path_view + image
				_kernel = Image.open(image_view).show()
#			if split_image[i] == "_kernel" and split_image[i+1] == "_layer2":
#				image_view = path_view + image
#				_kernel = Image.open(image_view).show()


def main():
	path_dir = "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/Stateless/25_time_steps/"

	#case = str(sys.argv[1])
	layer = str(sys.argv[1])
	cell = str(sys.argv[2])
	fold = str(sys.argv[3])

	#cases = ["19","55","91"]
	cases = ["11","99"]
	for case in cases:
		images, path_view = get_images_path(path_dir,case,layer,cell,fold)
		open_images(images,path_view)

if __name__ == "__main__":
	main()

