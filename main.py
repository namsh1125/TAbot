import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time
import os
import sys
import face_recognition
# from utils import initTello
from utils import *
from datetime import datetime
from threading import Thread

myDrone = initTello()

myDrone.takeoff()
keepRecording = True
myDrone.streamon()
frame_read = myDrone.get_frame_read()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

record_state = False

keyboard = cv2.waitKey(1)


def img_name(image_File):
    image_name = []
    name = image_File.split('.')[0]
    image_name.append(name)
    return image_name


def videoRecorder():
    height, width, _ = frame_read.frame.shape

    # today = datetime.today().strftime("%m%d-%H%M%S")  # 촬영하기 시작한 시간으로 파일 이름 셋팅
    # video_name = today + '.avi'
    video = cv2.VideoWriter('./video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        time.sleep(1 / 30)

    video.release()
#Ready moving



with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while True:
        if keyboard & 0xFF == ord('q'):
            break;
        image = myDrone.get_frame_read().frame
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        if not results.multi_hand_landmarks:
            # 동작 여부 확인 후 7초뒤에 내려가기
            print("No Gesture is detected")
            time.sleep(1)
        elif results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # 엄지를 제외한 나머지 4개 손가락의 마디 위치 관계를 확인하여 플래그 변수를 설정합니다. 손가락을 일자로 편 상태인지 확인합니다.
                thumb_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_MCP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
                            thumb_finger_state = 1

                index_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
                            index_finger_state = 1

                middle_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
                            middle_finger_state = 1

                ring_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > \
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
                            ring_finger_state = 1

                pinky_finger_state = 0
                if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_PIP].y * image_height:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > \
                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > \
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
                            pinky_finger_state = 1

                # 손가락 위치 확인한 값을 사용하여 가위,바위,보 중 하나를 출력 해줍니다.
                font = ImageFont.truetype("fonts/gulim.ttc", 80)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)


                #제스처 액션
                text = ""
                if record_state:
                    if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                        # 녹화 종료   엄지 + 검지 + 중지
                        time.sleep(1)
                        keepRecording = False
                        recorder.join()
                        record_state = False
                        print("Recoed Saved")
                        continue
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                    # text = "보"
                    text = 'panorama'
                    print("Panorama")
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
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    # text = "가위"  동영상 녹화
                    # 녹화 시작  엄지 + 검지
                    record_state = True
                    recorder = Thread(target=videoRecorder)
                    recorder.start()
                    print("Record Start!")
                    myDrone.move_up(50)
                    myDrone.rotate_counter_clockwise(180)
                    # myDrone.move_down(50)
                    # myDrone.rotate_counter_clockwise(90)
                    # myDrone.move_up(50)
                    # myDrone.rotate_counter_clockwise(90)
                    # myDrone.move_down(50)
                    # myDrone.rotate_counter_clockwise(90)
                if index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    # text = "주먹" 출석체크
                    print("Att")
                    images = os.listdir('images')  # 폴더에 저장된 이미지 파일의 이름을 리스트로 만듦
                    print(images)

                    # myDrone.rotate_counter_clockwise(360)
                    print("3초있다가 사진찍어욧")
                    time.sleep(3)

                    image = myDrone.get_frame_read().frame
                    class_image = np.array(image)
                    cv2.imwrite('class_image.jpg', class_image)

                    image_to_be_matched = face_recognition.load_image_file('class_image.jpg')  # 사진에서 얼굴추출하기
                    image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]  # 로드된 이미지에서 특징 추출하기   # encoded the loaded image into a feature vector

                    # 모든 학생들에 대해 찍힌 사진 비교
                    for image in images:
                        current_image = face_recognition.load_image_file("images/" + image)  # load the image
                        current_image_encoded = face_recognition.face_encodings(current_image)[0]  # 파일에서 가져온 이미지에서 얼굴 특징 추출하기     # encode the loaded image into a feature vector
                        result = face_recognition.compare_faces([image_to_be_matched_encoded],current_image_encoded)  # 저장된 학생이 교실 사진에 있는지 없는지 확인

                        if result[0] == True:  # 찍은 교실 사진에 저장된 학생이 있다면
                            print(img_name(image), 'is here')


        if cv2.waitKey(5) & 0xFF == 27:
            break
