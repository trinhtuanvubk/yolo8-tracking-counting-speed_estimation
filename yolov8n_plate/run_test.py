


from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

model = YOLO("weights/best.pt")
image = Image.open("plate_samples.jpg")
image = np.asarray(image)
results = model(image)
results = results[0].plot()
cv2.imwrite("test.jpg", results)