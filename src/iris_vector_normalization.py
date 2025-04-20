import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def show_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

def IrisVectorNormalization(img, verbose=False):
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))

    original = img.copy()

    filter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filter = cv2.medianBlur(filter, 21)
    detected_circles = cv2.HoughCircles(
        filter, cv2.HOUGH_GRADIENT, dp=3, minDist=20,
        param1=50, param2=20, minRadius=60, maxRadius=105
    )
    if detected_circles is None:
        print("Did not find iris")
        return None
    A, B, R = np.uint16(np.around(detected_circles))[0, 0]

    filter = cv2.medianBlur(filter, 21)
    detected_circles = cv2.HoughCircles(
    filter, cv2.HOUGH_GRADIENT, 1, 50, param1=80,
    param2=15, minRadius=1, maxRadius=50
    )
    if detected_circles is None:
        print("Did not find pupil")
        return None
    a, b, r = np.uint16(np.around(detected_circles))[0, 0]

    A, B = a, b
    H, W = original.shape[:2]
    left = max(0, int(A) - int(R))
    top = max(0, int(B) - int(R))
    right = min(W, int(A) + int(R))
    bottom = min(H, int(B) + int(R))

    cropped = original[top:bottom, left:right]
    if verbose:
        show_image(cropped)

    pupil_center = (a - left, b - top)
    pupil_radius = r
    iris_center = pupil_center
    iris_radius = R
    output_size = (400, 150)

    theta = np.linspace(0, 2 * np.pi, output_size[0])
    r = np.linspace(pupil_radius, iris_radius, output_size[1])
    R_grid, Theta = np.meshgrid(r, theta)

    Xs = iris_center[0] + R_grid * np.cos(Theta)
    Ys = iris_center[1] + R_grid * np.sin(Theta)

    img = cv2.remap(cropped, Xs.astype(np.float32), Ys.astype(np.float32), cv2.INTER_LINEAR)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    H, W = img.shape[:2]
    img = img[10:(H // 2) + 10, 0:W]

    if verbose:
        show_image(img)

    return img
