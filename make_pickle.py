import cv2
import numpy as np
import pickle
import datetime


def km_read_img(img: str) -> np.ndarray:
    image = cv2.imread(img).astype(np.float32) / 255.0
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image_lab


def apply_kmeans(img: str, num_clusters: int) -> tuple:
    data = img.reshape((-1, 3))
    num_classes = num_clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(
        data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return (labels, centers)


def create_kmeans_colors(img: str, num_clusters: int) -> list:
    color_center = []
    image_lab = km_read_img(img)
    labels, centers = apply_kmeans(img=image_lab, num_clusters=num_clusters)
    for color in range(len(centers)):
        color_center.append(centers[color])

    return color_center


def create_key_points(img: str) -> list:
    keyPoints = []
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    fast = cv2.FastFeatureDetector_create(
        160, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    )
    keys = fast.detect(img)
    for i in keys:
        keyPoints.append(i.pt)

    return keyPoints


def create_histogram(img: str) -> tuple:
    img1 = cv2.imread(img)
    histBlue = cv2.calcHist([img1], [0], None, [256], [0, 256])
    histRed = cv2.calcHist([img1], [1], None, [256], [0, 256])
    histGreen = cv2.calcHist([img1], [2], None, [256], [0, 256])
    return (histBlue, histRed, histGreen)


def make_pickles(img: str) -> None:
    colors = create_kmeans_colors(img, 5)
    keyPoints = create_key_points(img)
    histogram = create_histogram(img)
    pickle.dump((colors, keyPoints, histogram), open("examplar.p", "wb"))


if __name__ == "__main__":
    original_colors, original_keyPoints, original_histogram = pickle.load(
        open("examplar.p", "rb")
    )
    # make_pickles("examplar.jpg")
    with open("file_locations.csv") as f:
        img_list = f.readlines()

    # new_img = img_list[0].strip() #larger version
    new_img = img_list[1].strip()  # actual match

    color_start = datetime.datetime.now()
    new_colors = create_kmeans_colors(new_img, 5)
    color_end = datetime.datetime.now()

    key_start = datetime.datetime.now()
    new_keypoints = create_key_points(new_img)
    key_end = datetime.datetime.now()

    hist_start = datetime.datetime.now()
    new_histogram = create_histogram(new_img)
    hist_end = datetime.datetime.now()

    print("Kmeans color matching ----")
    print(color_end - color_start)
    for obj in range(len(new_colors)):
        for val in range(len(new_colors[obj])):
            print(new_colors[obj][val] == original_colors[obj][val])

    print("Keypoints matching ------")
    print(key_end - key_start)
    print(new_keypoints == original_keyPoints)

    print("Histogram percentage matching ---- ")
    print(hist_end - hist_start)
    new_old_blue = cv2.compareHist(new_histogram[0], original_histogram[0], 0)
    print(new_old_blue)
