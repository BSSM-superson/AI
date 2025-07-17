from flask import Flask, request
from flask_cors import CORS
import threading
import time
import socket
import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import serial
import json
import paho.mqtt.client as mqtt

# Flask setup
app = Flask(__name__)
CORS(app)

#Mqtt setup
mqtt_broker = "127.0.0.1"
mqtt_port = 1883
mqtt_topic = "superson/unity"

mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port, 60)

# Socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# Flags
tracking_active = False

# Hand detector and pose model
detector = HandDetector(detectionCon=0.8, maxHands=1)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Global log time
prev_log_time = time.time()

#camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def reverse_angle(angle):
    return 180 - angle

def camera_loop():
    global tracking_active, prev_log_time

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("⚠️ Failed to read from camera.")
            time.sleep(1)
            continue

        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Always show the image, but only process if tracking is active
        if tracking_active:
            hands, img = detector.findHands(img, draw=True)
            results = pose.process(imgRGB)

            joint_data, finger_angles = {}, {}
            arm_points = []

            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                right_arm_points = [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                ]
                for idx, point in enumerate(right_arm_points):
                    lm = pose_landmarks[point]
                    x, y, z = int(lm.x * w), int(lm.y * h), lm.z
                    joint_data[idx] = (x, y, z)
                    arm_points.append((x, y))

                if len(arm_points) == 3:
                    cv2.line(img, arm_points[0], arm_points[1], (0, 255, 0), 3)
                    cv2.line(img, arm_points[1], arm_points[2], (0, 255, 0), 3)

            if hands:
                hand = hands[0]
                lmList = hand["lmList"]

                def get_finger_angle(p1, p2, p3):
                    return calculate_angle(lmList[p1], lmList[p2], lmList[p3])

                finger_angles["Thumb"] = get_finger_angle(1, 2, 4)
                finger_angles["Index"] = get_finger_angle(5, 6, 8)
                finger_angles["Middle"] = get_finger_angle(9, 10, 12)
                finger_angles["Ring"] = get_finger_angle(13, 14, 16)
                finger_angles["Pinky"] = get_finger_angle(17, 18, 20)

                angles = [int(np.clip(reverse_angle(finger_angles[f]), 0, 180)) for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]]

                #mqtt
                mqtt_finger_data = {
                    "fingers": {
                        "Thumb": angles[0],
                        "Index": angles[1],
                        "Middle": angles[2],
                        "Ring": angles[3],
                        "Pinky": angles[4]
                    }
                }
                mqtt_client.publish(mqtt_topic, json.dumps(mqtt_finger_data))


            current_time = time.time()
            if current_time - prev_log_time >= 1 and len(joint_data) == 3:
                shoulder_angle = calculate_angle(joint_data[0], joint_data[1], joint_data[2])
                elbow_angle = calculate_angle(joint_data[1], joint_data[2], (joint_data[2][0], joint_data[2][1] + 100, joint_data[2][2]))

                print(f"Shoulder Angle : {shoulder_angle:.2f}")
                print(f"Elbow Angle    : {elbow_angle:.2f}")
                for finger, angle in finger_angles.items():
                    print(f"{finger} Angle    : {angle:.2f}")
                print("\n")

                angle_data = {
                    "shoulder": round(shoulder_angle, 2),
                    "elbow": round(elbow_angle, 2),
                    "fingers": {k: round(v, 2) for k, v in finger_angles.items()}
                }
                sock.sendto(str.encode(str(angle_data)), serverAddressPort)

                prev_log_time = current_time

        small_img = cv2.resize(img, (w // 2, h // 2))
        cv2.imshow("Image", small_img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask endpoints
@app.route("/start", methods=["POST"])
def start():
    global tracking_active
    tracking_active = True
    return "Started tracking", 200

@app.route("/end", methods=["POST"])
def end():
    global tracking_active
    tracking_active = False
    return "Ended tracking", 200

if __name__ == "__main__":
    # Start camera thread when server starts
    cam_thread = threading.Thread(target=camera_loop)
    cam_thread.daemon = True
    cam_thread.start()
   
    app.run(host="0.0.0.0", port=5000)
