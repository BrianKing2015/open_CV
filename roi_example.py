import cv2
import numpy as np
import pytesseract

import ssim_example

## to get tesseract to work : https://stackoverflow.com/questions/50655738/tesseractnotfounderror/51662074#51662074


if __name__ == '__main__':
	with open ("file_locations.csv") as f:
		img_list = f.readlines()	
	
	fails = []

	#img_list = img_list[-5:]
	for i in img_list:
		img = cv2.imread(i.strip())   
		# Arrived at by trial and error
		## 1920x1080 values  AKA: full dimenions
		#y1 = 680
		#y2 = 710 
		#x1 =610
		#x2 =670
		## 576x324 values  AKA: mini-screenshot dimensions
		y1 = 280
		y2 = 310 
		x1 =250
		x2 =320


		# Select out only the framenumber area
		num_area = img[y1:y2,x1:x2]

		# Enlarge the image
		num_area = cv2.resize(num_area, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

		# Convert to gray scale
		num_area = cv2.cvtColor(num_area, cv2.COLOR_BGR2GRAY)

		# Dilate and erode the image to remove unimportant data
		kernel = np.ones((1, 1), np.uint8)
		num_area = cv2.dilate(num_area, kernel, iterations=1)
		num_area = cv2.erode(num_area, kernel, iterations=1)

		# Apply threshold in order to only have 0 or 255, maximize the contrast between white and black
		cv2.threshold(num_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		#cv2.threshold(num_area,127,255,cv2.THRESH_BINARY)

        # Uncomment in order to check what is showing up in the region of interest
		# cv2.imshow("my image", num_area)

		text = pytesseract.image_to_string(num_area, lang='eng')
		print (text)


		try:
			int(text)
			video_name = "buck_number.mp4"
			cap = cv2.VideoCapture(video_name)
			cap.set(1,int (text))
			ret, frame = cap.read()
			ssim_score = ssim_example.compare_image (frame, img)
			if ssim_score < 0.80:
				print (ssim_score)
				fails.append(i)

		except:
			fails.append(i)


		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	
	total_size = len(img_list)
	total_fails =  len(fails)
	print ("Total number of failures")
	print (total_fails)
	print ("Out of total images")
	print (total_size)
	print ("Or approx")
	print (total_fails / total_size )