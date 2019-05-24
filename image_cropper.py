import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

img = cv2.imread('sample_image_3.png', 0)
color_img = cv2.imread('sample_image_3.png')
template = cv2.imread('sample_image_cropped.png', 0)
# mask = cv2.imread('sample_image_cropped_blacked.png', 0)

number_images = []

for x in range(10):
    filename = 'numbers/' + str(x) + ".png"
    number_images.append(cv2.imread(filename, 0))
    # reads them in as height, width

w, h = template.shape[::-1]

#################################################
# Finding photo in screenshot
#################################################

method = cv2.TM_CCOEFF

res = cv2.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

print("Image coordinates:", top_left, bottom_right)

print("Image width:", w, h)

crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
color_crop_img = color_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

####################################################
# Find marker in image
####################################################


g_img = color_crop_img.copy()
hsv_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2HSV)

# mask of green (36,50,50) ~ (70, 255,255)
green_mask = cv2.inRange(hsv_img, (36, 50, 50), (70, 255, 255))

# slice the green
imask = green_mask > 0
green = np.zeros_like(g_img, np.uint8)
green[imask] = g_img[imask]

green_grayscale = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(green_grayscale, 30, 255, 0)

kernel = np.ones((5, 5), np.uint8)
closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('closed img', closed_img)

contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))

markers = []

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(g_img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    if w > 10 and h > 10:
        markers.append((x + (w / 2), y + (h / 2)))

for item in markers:
    print(item)

# cv2.imshow('found markers', g_img)

#################################################
# Blacking out data in image
#################################################
blacked_out_img = crop_img.copy()

# Invert the image so that the "foreground" (the text) is white and teh background is black

blacked_out_img = cv2.bitwise_not(blacked_out_img)

# ret, blacked_out_img = cv2.threshold(whited_out_img, 5, 255, cv2.THRESH_BINARY)

cv2.rectangle(blacked_out_img, (20, 0), (532, 512), 0, -1)

# cv2.imshow('blacked out', blacked_out_img)

#################################################
# Find groups of text and recognize it
#################################################

left_rectangle_coordinates = ((2, 0), (12, 527))
bottom_rectangle_coordinates = ((0, 515), (532, 526))

left_rectangle_img = blacked_out_img[0:527, 2:12]
bottom_rectangle_img = blacked_out_img[515:526, 0:532]

left_rectangle_img = imutils.rotate_bound(left_rectangle_img, 90)

left_black_background = np.zeros_like(left_rectangle_img)
bottom_black_background = np.zeros_like(bottom_rectangle_img)

# Close up numbers
closed_bottom_img = cv2.morphologyEx(bottom_rectangle_img, cv2.MORPH_CLOSE, kernel)
closed_left_img = cv2.morphologyEx(left_rectangle_img, cv2.MORPH_CLOSE, kernel)

x_contours, x_heir = cv2.findContours(closed_bottom_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
y_contours, y_hier = cv2.findContours(closed_left_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('y img', closed_left_img)
cv2.imshow('x img', closed_bottom_img)

black_img = np.zeros_like(blacked_out_img)

x_text_roi = []
y_text_roi = []

for c in x_contours:
    (x, y, w, h) = cv2.boundingRect(c)

    # Move coordinates down to where they should be
    # Not doing it right now, instead showing small window
    # y += bottom_rectangle_coordinates[0][1]

    cv2.rectangle(bottom_black_background, (x, y), (x + w, y + h), (255, 255, 255), 1)

    x_text_roi.append((x, y, w, h))

    print(x, y, w, h)

for c in y_contours:
    (x, y, w, h) = cv2.boundingRect(c)

    # TODO rotate -90 degrees when showing

    cv2.rectangle(left_black_background, (x, y), (x + w, y + h), (255, 255, 255), 1)

    y_text_roi.append((x, y, w, h))

    print(x, y, w, h)

cv2.imshow('x boxes', bottom_black_background)
cv2.imshow('y boxes', left_black_background)
cv2.waitKey(0)

#################################################
# Recognizing text
# TODO: recognize text
#################################################

for x in range(closed_bottom_img.size[1]):
    # TODO iterate through 10 numbers, see if one matches
    pass