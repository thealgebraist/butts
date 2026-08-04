from transformers import pipeline
import cv2
from PIL import Image

print("Loading model...")
segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
print("Model loaded.")

img = Image.open("/Users/anders/projects/thrash/center1/can.png")
results = segmenter(img)
for r in results:
    print(f"Label: {r['label']}")
