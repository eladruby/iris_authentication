import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import keras
from skimage.io import imread

def show_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

def IrisURLNormalization(path, verbose=False):

  filter = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  filter = cv2.medianBlur(filter, 21)

  detected_circles = cv2.HoughCircles(
  filter, cv2.HOUGH_GRADIENT, dp=3, minDist=20, param1=30,
  param2=10, minRadius=60, maxRadius=105
  )

  if detected_circles is not None:
      detected_circles = np.uint16(np.around(detected_circles))
  else:
      print("Breaked, Did not find Iris")
      return False

  A, B, R = detected_circles[0, 0]

  img = imread(path, cv2.IMREAD_GRAYSCALE)

  cv2.circle(img, (A, B), R, (255, 0, 0), 2)

  filter = cv2.imread(path, cv2.IMREAD_COLOR)
  filter = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
  filter = cv2.medianBlur(filter, 21)

  detected_circles = cv2.HoughCircles(
  filter, cv2.HOUGH_GRADIENT, 1, 50, param1=60,
  param2=23, minRadius=10, maxRadius=70
  )

  if detected_circles is not None:
      detected_circles = np.uint16(np.around(detected_circles))
  else:
      print("Breaked, Did not find Pupil")
      return False

  a, b, r = detected_circles[0, 0]

  img = cv2.imread(path, cv2.IMREAD_COLOR)

  cv2.circle(img, (a, b), r, (255, 0, 0), 2)
  cv2.circle(img, (a, b), R, (255, 0, 0), 2)

  A = a
  B = b

  H, W = img.shape[:2]
  left = max(0, int(A)-int(R))
  top = max(0, int(B)-int(R))
  right = min(W, int(A)+int(R))
  bottom = min(H, int(B)+int(R))

  img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  img = img.crop((left, top, right, bottom))

  img = np.array(img)

  if verbose:
    show_image(img)

  pupil_center = (a-left, b-top)
  pupil_radius = r
  iris_center = pupil_center
  iris_radius = R
  output_size = (400, 150)

  theta = np.linspace(0, 2 * np.pi, output_size[0])
  r = np.linspace(pupil_radius, iris_radius, output_size[1])

  R, Theta = np.meshgrid(r, theta)

  Xs = iris_center[0] + R * np.cos(Theta)
  Ys = iris_center[1] + R * np.sin(Theta)

  img = cv2.remap(img, Xs.astype(np.float32), Ys.astype(np.float32), cv2.INTER_LINEAR)
  img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

  H, W = img.shape[:2]

  img = img[10:(H//2)+10, 0:W]

  if verbose:
    show_image(img)

  return img
