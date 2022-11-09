# import os
# import face_recognition
# import numpy as np

from datetime import datetime
# from threading import Thread
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from utils import *

max_num_hands = 2
drone_gesture = {0: 'up', 1: 'left', 3: 'forward', 4: 'back', 5: 'down', 9: 'right', 10: 'camera'}

myDrone = initTello()
myDrone.takeoff()

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)


def img_name(image_File):
    image_name = []
    name = image_File.split('.')[0]
    image_name.append(name)
    return image_name


def videoRecorder():
    height, width, _ = frame_read.frame.shape

    today = datetime.today().strftime("%m%d-%H%M%S")  # 촬영하기 시작한 시간으로 파일 이름 셋팅
    video_name = today + '.avi'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        time.sleep(1 / 30)

    video.release()


while True:

    myDrone = initTello()
    # myDrone.takeoff()
    myDrone.streamon()
    cv2.namedWindow("drone")
    frame_read = myDrone.get_frame_read()

    img = frame_read.frame

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        drone_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in drone_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                # cv2.putText(img, text=drone_gesture[idx].upper(), org=(org[0], org[1] +l 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                drone_result.append({
                    'control': drone_gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(drone_result) >= 2:
                text = ''

                # if drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'up':
                #     myDrone.move_up(50)
                #     text = 'up'
                #
                # if (drone_result[0]['control'] == 'down' and drone_result[1]['control'] == 'up') or \
                #         (drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'down'):
                #     myDrone.move_down(50)
                #     text = 'down'
                #
                # if (drone_result[0]['control'] == 'forward' and drone_result[1]['control'] == 'up') or \
                #         (drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'forward'):
                #     myDrone.move_forward(50)
                #     text = 'forward'
                #
                # if (drone_result[0]['control'] == 'back' and drone_result[1]['control'] == 'up') or \
                #         (drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'back'):
                #     myDrone.move_back(50)
                #     text = 'back'
                #
                # if (drone_result[0]['control'] == 'right' and drone_result[1]['control'] == 'up') or \
                #         (drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'right'):
                #     myDrone.move_right(50)
                #     text = 'right'
                #
                # if (drone_result[0]['control'] == 'left' and drone_result[1]['control'] == 'up') or \
                #         (drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'left'):
                #     myDrone.move_left(50)
                #     text = 'left'
                #
                # if (drone_result[0]['control'] == 'camera' and drone_result[1]['control'] == 'up') or \
                #         (drone_result[0]['control'] == 'up' and drone_result[1]['control'] == 'camera'):
                #     text = 'camera'
                #     image = myDrone.get_frame_read().frame
                #     image = np.array(image)
                #     cv2.imwrite('camera.jpg', image)

                # 파노라마 사진 찍기
                if drone_result[0]['control'] == 'right' and drone_result[1]['control'] == 'right':
                    text = 'panorama'

                    myDrone.rotate_clockwise(30)
                    image = myDrone.get_frame_read().frame
                    image = np.array(image)
                    cv2.imwrite('panorama1.jpg', image)
                    time.sleep(0.25)

                    myDrone.rotate_clockwise(30)
                    image = myDrone.get_frame_read().frame
                    image = np.array(image)
                    cv2.imwrite('panorama2.jpg', image)
                    time.sleep(0.25)

                    myDrone.rotate_clockwise(30)
                    image = myDrone.get_frame_read().frame
                    image = np.array(image)
                    cv2.imwrite('panorama3.jpg', image)
                    time.sleep(0.25)

                    myDrone.rotate_clockwise(30)
                    image = myDrone.get_frame_read().frame
                    image = np.array(image)
                    cv2.imwrite('panorama4.jpg', image)
                    time.sleep(0.25)

                    myDrone.rotate_clockwise(30)
                    image = myDrone.get_frame_read().frame
                    image = np.array(image)
                    cv2.imwrite('panorama5.jpg', image)
                    time.sleep(0.25)

                    myDrone.rotate_clockwise(-150)
                    img_names = ['panorama1.jpg', 'panorama2.jpg', 'panorama3.jpg', 'panorama4.jpg', 'panorama5.jpg']

                    imgs = []
                    for name in img_names:
                        img = cv2.imread(name)

                        if img is None:
                            print('Image load failed!')
                            sys.exit()

                        imgs.append(img)

                    stitcher = cv2.Stitcher_create()
                    status, dst = stitcher.stitch(imgs)

                    if status != cv2.Stitcher_OK:
                        print('Stitch failed!')
                        sys.exit()

                    # Todo: 이미지 이쁘게 잘라내기
                    cv2.imwrite('output.jpg', dst)

        cv2.imshow('Drone_view', img)

        if cv2.waitKey(1) == ord('q'):
            break
