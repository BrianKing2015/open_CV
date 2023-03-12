import cv2
import numpy as np


def contours_pca(contours):
    """
    Joins the contour points into array and removes unit values.
    Then calculates the PCA of the contour points.
    @param contours: list of contours
    @return: list of (center, angle, width, height)
    """
    contours_points = np.vstack(contours).squeeze().astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(contours_points, mean=None)

    center = mean.squeeze().astype(np.int32)
    delta = (150 * eigenvectors).squeeze().astype(np.int32)
    return center, delta


def draw_pca(img, contours, center, delta):
    """
    Draws the PCA on the image.
    @param img: image to draw on
    @param contours: list of contours
    @param center: center of the PCA
    @param delta: delta of the PCA
    """
    cv2.line(img, tuple(center + delta[0]), tuple(center - delta[0]), (0, 255, 0), 2)
    cv2.line(img, tuple(center + delta[1]), tuple(center + delta[1]), (0, 255, 0), 2)
    cv2.circle(img, tuple(center), 3, (0, 0, 255), 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")
    while True:
        status_cap, frame = cap.read()
        if status_cap == False:
            break
        frame = cv2.resize(frame, (0, 0), frame, 0.5, 0.5)
        edges = cv2.Canny(frame, 250, 150)
        print(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        _, contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            center, delta = contours_pca(contours)
            draw_pca(frame, contours, center, delta)

        cv2.imshow("PCA", frame)
        if cv2.waitKey(100) == 27:
            break

    cv2.destroyAllWindows()
