from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

color_dict = {'red': (0,0,255), 'green': (0, 255, 0), 'blue': (255,0, 0),
              'yellow': (66, 236,245), 'violet': ( 245,66,111), 'black': (0, 0, 0), }

default_color = (255, 255, 255)


def detect_bands(img_path):
    detec_model = YOLO("best_v2.pt")
    img = cv2.imread(img_path)
    results = detec_model(img)  # list of Results objects

    for r in results:
        boxes = r.boxes
        # print(box.xyxy)
        for box in boxes:
            # print(box.xyxy)
            print(box.xyxy[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # box.xyxy[0]
            croped = img[y1:y2, x1:x2]
            color_name = find_color_of_bands(croped)
            label = f'Band-({color_name})'
            cv2.imwrite("croped.jpg", croped)

            # print('croped image save in croped.jpg')
            if results is not None:
                anno = Annotator(img, line_width=2, font_size=2)
                # print(r.names)
                # color: Any = (128, 128, 128),
                # txt_color: Any = (255, 255, 255)
                # annotated_frame = results[0].plot(labels=False)
                anno.box_label(box.xyxy[0], label=label,color=color_dict.get(color_name,default_color))
                cv2.imwrite('annotated.jpg', img)
                # cv2.imwrite('annotated_2.jpg',anno.img)

        # cv2.imshow('annotated',annnotator)        # cv2.imwrite("annotated.jpg",annotated_frame)


def find_color_of_bands(band_croped_img):
    cls_model = YOLO("best_clr_cls.pt")
    classify_result = cls_model(band_croped_img)
    for result in classify_result:
        id = result.probs.top1
        name = result.names[id]
        print('detected band color is : ', name)
    return name


img_path = "bands/images/74d6a37f-984b-4ee7-813a-d6fcb50acace.jpg"
detect_bands(img_path)
