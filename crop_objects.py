import cv2
from ultralytics import YOLO
import os

model = YOLO("best_v2.pt")

def process_image(image_path: str, output_path: str) -> None:
    filename = image_path.split('/')[-1]
    frame = cv2.imread(image_path)

    # Perform detection
    detections = model(frame)
    extention = 0
    for detection in detections:
        boxes = detection.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            band_img = frame[y1:y2, x1:x2]
            print('old', band_img.shape)
            filname_ext = filename.replace('.jpg', f'_{extention}.jpg')
            saving_path = os.path.join(output_path, filname_ext)
            cv2.imwrite(saving_path, band_img)
            extention += 1

# give the path to the crop_dir and save_dir
output_dir = '/home/akshay/work/image_detect_identify_color/croped_bands'
os.makedirs(output_dir, exist_ok=True)
input_dir = '/home/akshay/work/image_detect_identify_color/bands/images'
for file in os.listdir(input_dir):
    print(file)
    try:
        if '.jpg' or '.png' or 'jpeg' in dir:
            image_path = os.path.join(input_dir, file)
            process_image(image_path, output_dir)
    except Exception as e:
        print(e)
