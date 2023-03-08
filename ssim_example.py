import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import cv2

def compare_image (imageA, imageB) -> float:
	example = imageA  #cv2.imread(imageA)
	target =  imageB  #cv2.imread(imageB)
	example = cv2.resize(example, (1920, 1080))
	target = cv2.resize(target, (1920, 1080))
	example = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
	target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
	s = ssim(example, target)
	return (s)


#ssim_score = compare_image("examplar.jpg","720p.jpg")
#print (ssim_score)

