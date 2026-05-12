import cv2
import numpy as np
import pyttsx3
import winsound

engine = pyttsx3.init()

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

classes = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor"]

cap = cv2.VideoCapture(0)

last_alert = ""


def get_distance(box_width):

    if box_width > 350:
        return "Very Close"
    elif box_width > 250:
        return "Near"
    elif box_width > 150:
        return "Medium"
    else:
        return "Far"


def get_direction(x1, x2, frame_width):

    center = (x1 + x2) // 2

    if center < frame_width / 3:
        return "Left"
    elif center > frame_width * 2/3:
        return "Right"
    else:
        return "Center"


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame,0.007843,(300,300),127.5)

    net.setInput(blob)
    detections = net.forward()

    current_alert = ""

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.5:

            idx = int(detections[0,0,i,1])
            label = classes[idx]

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])

            (x1,y1,x2,y2) = box.astype("int")

            box_width = x2-x1

            distance = get_distance(box_width)
            direction = get_direction(x1,x2,w)

            text = f"{label} {distance} {direction}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,text,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

            # ✅ allow all objects
            if current_alert == "" and label != "background":

                current_alert = f"{label} {distance} {direction}"

                if distance == "Very Close":
                    winsound.Beep(1200,300)


    # ✅ speak only once
    if current_alert != "" and current_alert != last_alert:

        engine.say(current_alert)
        engine.runAndWait()

        last_alert = current_alert


    cv2.imshow("Camera Detection", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


