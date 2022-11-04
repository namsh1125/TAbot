# 201735812 김민수, 201835834 남승현
import sys

import numpy as np

from utils import *
from datetime import datetime
import time
import cv2
from threading import Thread

keepRecording = True
recorder = 0

def videoRecorder():
    height, width, _ = frame_read.frame.shape
    # video_name = datetime.today().strftime("%Y/%m/%d %H:%M:%S")  # 촬영하기 시작한 시간으로 파일 이름 셋팅
    # video = cv2.VideoWriter('./' + video_name + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    video = cv2.VideoWriter('./video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        time.sleep(1 / 30)

    video.release()

if __name__ == "__main__":

    myDrone = initTello()
    # myDrone.takeoff()
    # time.sleep(1)
    myDrone.streamon()
    cv2.namedWindow("drone")
    frame_read = myDrone.get_frame_read()
    time.sleep(1)

    # Todo: keyboard to gesture
    while True:

        # Todo: 카메라를 함수가 실행될 때 키는건 어떨까
        img = frame_read.frame
        cv2.imshow("drone", img)

        keyboard = cv2.waitKey(1)

        if keyboard & 0xFF == ord('q'): # 키보드에서 'q' 입력 받았을 경우
            # myDrone.land()
            frame_read.stop()
            myDrone.streamoff()
            exit(0)
            break

        # Todo: fix video record
        if keyboard & 0xFF == ord('v'): # 키보드에서 'v' 입력 받았을 경우

            # keepRecording = True

            recorder = Thread(target=videoRecorder)
            recorder.start()

            if keyboard & 0xFF == ord('v'):  # 키보드에서 'v' 입력 받았을 경우
                keepRecording = False
                recorder.join()

        # Capture student
        if keyboard & 0xFF == ord('c'): # 키보드에서 'c' 입력 받았을 경우
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

# import time, cv2
# from threading import Thread
# from djitellopy import Tello
#
# tello = Tello()
#
# tello.connect()
#
# keepRecording = True
# tello.streamon()
# frame_read = tello.get_frame_read()
#
# def videoRecorder():
#     # create a VideoWrite object, recoring to ./video.avi
#     # 创建一个VideoWrite对象，存储画面至./video.avi
#     height, width, _ = frame_read.frame.shape
#     video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
#
#     while keepRecording:
#         video.write(frame_read.frame)
#         time.sleep(1 / 30)
#
#     video.release()
#
# # we need to run the recorder in a seperate thread, otherwise blocking options
# #  would prevent frames from getting added to the video
# # 我们需要在另一个线程中记录画面视频文件，否则其他的阻塞操作会阻止画面记录
# recorder = Thread(target=videoRecorder)
# recorder.start()
#
# tello.takeoff()
# tello.move_up(100)
# tello.rotate_counter_clockwise(360)
# tello.land()
#
# keepRecording = False
# recorder.join()