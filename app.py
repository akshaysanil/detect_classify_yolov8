from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
import streamlit as st

# added colors
color_dict = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (66, 236, 245),
    'violet': (245, 66, 111),
    'black': (0, 0, 0), 
    }

default_color = (255, 255, 255)
detection_model = YOLO("best_v2.pt")
classification_model = YOLO("best_clr_cls.pt")

def detect_bands(img, model,detections=False):
    results = model(img)  
    if len(results) == 0:
        print('no detections')
    else:
        print('detect happening')

    annotator = Annotator(img, line_width=3, font_size=4)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if detections is False and box.conf is not None:
                detections = True    
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            croped_img = img[y1:y2, x1:x2]
            color_name = find_color_of_bands(croped_img, classification_model)
            label = f'Band-({color_name})'  

            # if need to save cropped band
            # cv2.imwrite("croped.jpg", croped)

            if results is not None:
                annotator.box_label(
                    box.xyxy[0],
                    label=label, 
                    color=color_dict.get(color_name, default_color)
                )

                # save annotated_img to root dir
                # cv2.imwrite('annotated.jpg', img)
    if detections :
        return img, True
    else:
        return img, False


def find_color_of_bands(croped_band_img, model):
    classify_results = model(croped_band_img)
    for r in classify_results:
        id = r.probs.top1
        color_name = r.names[id]
        # print(f'detected band color is : {color_name}')
        return color_name


# streamlit app layout
st.title("Band detection and color classification")
st.write("Upload an image.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    print('true')
    # read the image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # ouputs
    annotated_img, is_detection = detect_bands(img, detection_model)

    # Convert BGR to RGB for displaying in Streamlit
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Display the original and annotated images
    if is_detection :
        st.image(annotated_img, caption='Annotated Image', use_column_width=True)
    else:
        st.image(annotated_img, caption='No Detection Happend', use_column_width=True)

