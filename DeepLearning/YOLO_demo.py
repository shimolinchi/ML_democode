from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

Image1 = cv2.imread("Image1.jpg")
Image2 = cv2.imread("Image2.png")

image1 = cv2.resize(Image1, (640, 640))
image2 = cv2.resize(Image2, (640, 640))

cv2.imwrite("image1.jpg", image1)
cv2.imwrite("image2.jpg", image2)

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"result_{i}.jpg")  # save to disk