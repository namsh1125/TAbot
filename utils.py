#201735812 김민수, 201835834 남승현
from djitellopy import Tello
import cv2
import time

def initTello():
    myDrone = Tello()
    myDrone.connect()

    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0

    print("\n * Drone battery percentage : " + str(myDrone.get_battery()) + "%")
    myDrone.streamoff()

    return myDrone

# def moveTello(myDrone):
#
#     myDrone.takeoff()
#     time.sleep(5)
#
#     myDrone.move_back(50)
#     time.sleep(5)
#
#     myDrone.rotate_clockwise(360)
#     time.sleep(5)
#
#     myDrone.move_forward(50)
#     time.sleep(5)
#
#     myDrone.flip_right()
#     time.sleep(5)
#
#     myDrone.flip_left()
#     time.sleep(5)
#
#     myDrone.land()
#     time.sleep(5)
