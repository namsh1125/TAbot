# 201735812 김민수, 201835834 남승현
import os
import sys
import face_recognition
import numpy as np

from utils import *
from datetime import datetime
import time
import cv2
from threading import Thread

keepRecording = True
recorder = 0


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

if __name__ == "__main__":

    myDrone = initTello()
  #  myDrone.takeoff()
    time.sleep(1)
    myDrone.streamon()
    cv2.namedWindow("drone")
    frame_read = myDrone.get_frame_read()
    time.sleep(1)

    # Todo: 키보드 입력을 제스처로 변경
    while True:
        img = frame_read.frame
        cv2.imshow("drone", img)

        keyboard = cv2.waitKey(1)

        if keyboard & 0xFF == ord('q'):
            myDrone.land()
            frame_read.stop()
            myDrone.streamoff()
            keepRecording = False
            recorder.join()
            exit(0)
            break

        # Class recording
        if keyboard & 0xFF == ord('v'):

            # Todo: 비디오 추가 촬영이 가능하게

            recorder = Thread(target=videoRecorder)
            recorder.start()

            # Todo: 키보드 조작으로 녹화 종료하게

            myDrone.move_up(50)
            myDrone.rotate_counter_clockwise(90)
            myDrone.move_down(50)
            myDrone.rotate_counter_clockwise(90)
            myDrone.move_up(50)
            myDrone.rotate_counter_clockwise(90)
            myDrone.move_down(50)
            myDrone.rotate_counter_clockwise(90)

            keepRecording = False
            recorder.join()

        if keyboard & 0xFF == ord('a'):

            images = os.listdir('images') # 폴더에 저장된 학생들의 사진을 불러옴

            image = myDrone.get_frame_read().frame # 출석체크를 하기 위해 학생들이 있는 교실 촬영

            # FixMe: 아래 명령어가 어떤 명령인지 주석좀...ㅎ
            image_to_be_matched = face_recognition.load_image_file(img) # load your image - 'file type' in here - need to change

            # FixMe: 아래 명령어가 어떤 명령인지 주석좀...ㅎ
            image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0] # encoded the loaded image into a feature vector

            # 모든 학생들에 대해 찍힌 사진 비교
            for image in images:

                current_image = face_recognition.load_image_file("images/" + image) # load the image

                current_image_encoded = face_recognition.face_encodings(current_image)[0] # encode the loaded image into a feature vector

                result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded) # 저장된 학생이 교실 사진에 있는지 없는지 확인

                if result[0] == True: # 찍은 교실 사진에 저장된 학생이 있다면
                    print(img_name(image), 'is here')

        # 파노라마 사진 찍기
        if keyboard & 0xFF == ord('c'):
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
