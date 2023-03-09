import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import cv2

def compare_image (original_img: np.array, new_img: np.array) -> float:
	original_img = cv2.resize(original_img, (1920, 1080))
	new_img = cv2.resize(new_img, (1920, 1080))
	original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	structural_similarity_measure = ssim(original_img, new_img)
	return (structural_similarity_measure)


if __name__ == '__main__':
	original_image = cv2.imread("grey.png")
	list_of_images = ["test.png", "grey.png", "apple.jpg", "pixabay_nature.jpg"]
	for img in list_of_images:
		new_image = cv2.imread(img)
		ssim_score = compare_image(original_image, new_image)
		print(ssim_score)
