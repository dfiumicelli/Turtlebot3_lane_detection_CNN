import cv2
import numpy as np
img = cv2.imread("prova.jpg")
cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Image", img)
cv2.waitKey(100000)