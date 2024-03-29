import cv2
import skimage
from PIL import Image
import numpy as np
from timeit import timeit


class color_comparison:
    def __init__(self, img: str):
        self.img = img

    def color_histogram(self) -> tuple:
        image = cv2.imread(self.img)
        histBlue = cv2.calcHist([image], [0], None, [256], [0, 256])
        histRed = cv2.calcHist([image], [1], None, [256], [0, 256])
        histGreen = cv2.calcHist([image], [2], None, [256], [0, 256])
        return (histBlue, histRed, histGreen)

    def mean_color(self) -> list:
        img = skimage.io.imread(self.img)[:, :, :-1]
        mean = img.mean(axis=0).mean(axis=0)
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        return [mean, dominant]

    def average_color(self) -> list:
        img = cv2.imread(self.img)
        average_color_per_row = np.average(img, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        return average_color

    def extrema_color(self) -> tuple:
        extrema_color = Image.open(self.img).getextrema()
        return extrema_color

    def pallette_color(self) -> list:
        pallette_color = Image.open(self.img).getpalette()
        return pallette_color


def create_single_color_image(color: tuple) -> np.array:
    img = np.zeros((420, 680, 3), np.uint8)
    img[:] = color
    return img


if __name__ == "__main__":
    list_of_images = ["test.png", "red.png"]
    for img in list_of_images:
        color = color_comparison(img=img)
        print(img)
        print(
            "color_histogram: ",
            timeit("color.color_histogram()", globals=globals(), number=1000),
        )
        # print("mean_color: ", timeit ('color.mean_color()', globals=globals(), number=1000)) # takes too long
        print(
            "average_color: ",
            timeit("color.average_color()", globals=globals(), number=1000),
        )
        print(
            "extrema_color: ",
            timeit("color.extrema_color()", globals=globals(), number=1000),
        )
        print(
            "pallette_color: ",
            timeit("color.pallette_color()", globals=globals(), number=1000),
        )

    # red = create_single_color_image((0, 0, 255))
    # cv2.imwrite("red.png", red)
