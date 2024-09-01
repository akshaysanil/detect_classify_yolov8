from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO("best_v2.pt")


# defe color ranges
# for red
lower_red1 = np.array([0, 100, 100])   # Red near 0° hue
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100]) # Red near 180° hue
upper_red2 = np.array([180, 255, 255])

# Lower and upper bounds for green
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

#lower and upper bounds for yellow
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# lower and upper bounds for blue
lower_blue = np.array([90, 100, 100])
upper_blue = np.array([120, 255, 255])




# Define path to the image file
img_path = "bands/images/fe14fbed-26e9-410c-b04e-c3093586db11.jpg"
img = cv2.imread(img_path)

source = img

# Run inference on the source
results = model(source,save=True,save_crop=True)  # list of Results objects
print(results,'dddddddddddddddddddddddddddd')
mask_sums = []

for r in results:
    boxes = r.boxes
    # print(box.xyxy)
    for box in boxes:
        # print(box.xyxy)
        print(box.xyxy[0])
        x1,y1,x2,y2 = map(int,box.xyxy[0]) # box.xyxy[0]
        croped = source[y1:y2,x1:x2]
        cv2.imwrite("croped.jpg",croped)

        img_hsv = cv2.cvtColor(croped, cv2.COLOR_BGR2HSV)
        cv2.imwrite("hsv.jpg",img_hsv)

        # create mask for each color

        mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        mask_red_s = mask_red.sum()
        # cv2.imwrite("red.jpg",mask_red)

        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        print(mask_green.sum())
        mask_green_s = mask_green.sum()
        # cv2.imwrite("green.jpg",mask_green)

        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        # cv2.imwrite("yellow.jpg",mask_yellow)
        mask_yellow_s = mask_yellow.sum()

        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        mask_blue_s = mask_blue.sum()
        mask_sums = [mask_red_s,mask_green_s,mask_yellow_s,mask_blue_s]
        # cv2.imwrite("blue.jpg",mask_blue)


        # apply mask to the image

        # mask_red = cv2.bitwise_and(img, img, mask=mask_red)
        if np.argmax(mask_sums) == 0:
            mask = mask_red
        elif np.argmax(mask_sums) == 1:
            mask = mask_green
        elif np.argmax(mask_sums) == 2:
            mask = mask_yellow
        elif np.argmax(mask_sums) == 3:
            mask = mask_blue

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (11,11), iterations = 2)
        cv2.imshow("opening",opening)

        



        cv2.imshow("mask_red",mask_red)
        cv2.imshow("mask_green",mask_green)
        cv2.imshow("mask_yellow",mask_yellow)
        cv2.imshow("mask_blue",mask_blue)
        cv2.imshow("croped",croped)
        cv2.waitKey(0)




# croped = img[:]









print(results)

