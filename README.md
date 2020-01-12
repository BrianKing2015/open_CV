# open_CV
Collection of scripts using OpenCV (computer vision) library

There are two main files of interest: make_pickle and roi_example

The main difference is that make_pickle is using methods that will not be effecitve for videos but will check that multiple layers of checks pass for static images (keypoints, color histogram, and kmeans color check).
On the other hand, roi_example is using SSIM (Structural Similarity of IMages) algorithm and pytesseract to check that a specific image from a video matches an examplar.
