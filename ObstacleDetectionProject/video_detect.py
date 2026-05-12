import cv2
import numpy as np
import win32com.client
import winsound

# ✅ Voice setup
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# ✅ Load model
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ✅ Class labels
classes = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor"]

# ✅ Only important objects (reduce confusion)
important_objects = ["person", "car", "bicycle", "motorbike", "bus"]

# ✅ Video input
cap = cv2.VideoCapture("test.mp4")

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

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_alert = ""
    best_priority = 0   # 🔥 to prioritize person

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # ✅ higher confidence = better accuracy
        if confidence > 0.6:

            idx = int(detections[0, 0, i, 1])
            label = classes[idx]

            # ✅ filter unwanted objects
            if label not in important_objects:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            box_width = x2 - x1

            distance = get_distance(box_width)
            direction = get_direction(x1, x2, w)

            text = f"{label} {distance} {direction}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # 🔥 Priority system
            priority = 2 if label == "person" else 1

            if priority > best_priority:
                best_priority = priority
                current_alert = text

                if distance == "Very Close":
                    winsound.Beep(1200, 300)

    # ✅ Speak only once when alert changes
    if current_alert != "" and current_alert != last_alert:
        speaker.Speak(current_alert)
        last_alert = current_alert

    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()