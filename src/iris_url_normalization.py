import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import keras
from skimage.io import imread
from skimage.transform import resize

def show_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

def IrisURLNormalization(path, model=None, verbose=False):

  if (model is None):
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
    filter, cv2.HOUGH_GRADIENT, 1, 90, param1=200,
    param2=40, minRadius=10, maxRadius=50
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

  else:
    test_img = imread(path)
    dims = test_img.shape[:2]

    test_img = resize(test_img, (128, 128))

    if len(test_img.shape) == 2:
      test_img = np.stack([test_img] * 3, axis=-1)

    elif test_img.shape[-1] == 4:
      test_img = test_img[:, :, :3]

    test_img = np.expand_dims(test_img, axis=0)

    pred_mask = model.predict(test_img)[0]
    pred_mask = np.squeeze(pred_mask)

    threshold = 0.5
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    binary_mask = resize(binary_mask, dims, mode='constant', preserve_range=True)
    img = test_img[0]
    img = resize(img, dims, mode='constant', preserve_range=True)

    if(verbose):
      show_image(img)

    revert_binary_mask = 1 - binary_mask
    binary_mask = (binary_mask * 255).astype(np.uint8)
    revert_binary_mask = (revert_binary_mask * 255).astype(np.uint8)

    if binary_mask is None or binary_mask.size == 0 or np.sum(binary_mask) == 0:
      raise ValueError("Binary mask is empty or invalid. Check segmentation output.")

    if binary_mask is None or binary_mask.size == 0:
      raise ValueError("Binary mask is empty or invalid.")

    A, B, R = 0, 0, 0

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
      raise ValueError("Could not open or find the image at path")

    binary_mask = cv2.blur(binary_mask, (3, 3))


    if verbose:
      show_image(binary_mask)

      white_pixels = np.column_stack(np.where(binary_mask >= 150))

      if white_pixels.size > 0:
        y_min, x_min = white_pixels.min(axis=0)
        y_max, x_max = white_pixels.max(axis=0)

        box_width = x_max - x_min
        box_height = y_max - y_min

        boxed_img = cv2.merge([binary_mask, binary_mask, binary_mask])
        cv2.rectangle(boxed_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        square_sim = 1 - abs(box_height / box_width - 1)

        if verbose:
          show_image(boxed_img)
          print(square_sim)
          print(box_width, box_height)


        if square_sim > 0.75:
          square=True
      else:
        print("No white pixels detected above threshold.")

    #METHOD 1
    if not square:
      if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      height, width = img.shape[0:2]

      leftmost = width
      rightmost = 0
      middlepoint1 = 0
      middlepoint2 = 0

      for y in range(height):
        for x in range(width):
          if binary_mask[y, x] > 0:
            if x < leftmost:
              leftmost = x
              middlepoint = y
            if x > rightmost:
              rightmost = x
              middlepoint2 = y

      print(leftmost, rightmost)

      A = (leftmost + rightmost) // 2
      R = (rightmost - leftmost) // 2

      B = middlepoint2+middlepoint1 // 2

      adjusted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      cv2.circle(adjusted_img, (A, B), R, (0, 255, 0), 2)
      if verbose:
        show_image(adjusted_img)

  #METHOD 2

    else:
      detected_circles = cv2.HoughCircles(
        binary_mask, cv2.HOUGH_GRADIENT, 3, 20, param1=50,
        param2=20, minRadius=60, maxRadius=110
      )

      if detected_circles is None:
        print("No iris detected. Skipping frame")
        return None

      detected_circles = np.uint16(np.around(detected_circles))

      A, B, R = detected_circles[0, 0]


      cv2.circle(img, (A, B), R, (255, 0, 0), 2)

      if verbose:
        show_image(img)
      #-----------------------------------------------

    a, b, r = 0, 0, 0

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    edges = cv2.Canny(revert_binary_mask, 50, 150)
    edges = cv2.blur(edges, (5, 5))

    cutter = R/4

    if not square:
      cutter = R/2

    left = A - R + cutter
    top = B + R - cutter
    right = A + R - cutter
    bottom = B - R + cutter

    edges = edges[int(bottom):int(top), int(left):int(right)]

    if verbose:
      show_image(edges)

    detected_circles = cv2.HoughCircles(
    edges, cv2.HOUGH_GRADIENT, 1, 50, param1=80,
    param2=15, minRadius=1, maxRadius=60
    )

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
    else:
        print("Breaked, Did not find Pupil")
        return False

    a, b, r = detected_circles[0, 0]

    a = a + int(A - R + cutter)
    b = b + int(B - R + cutter)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.circle(img, (a, b), r, (255, 0, 0), 2)
    cv2.circle(img, (a, b), R, (255, 0, 0), 2)

    if verbose:
      show_image(img)

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

    show_image(img)

    return img