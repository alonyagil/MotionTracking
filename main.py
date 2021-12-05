from CameraUsbCtr import Camera
import pickle

DUMP_PATH = '..\MotionTracking\data'

if __name__ == '__main__':
    cam = Camera()
    cam.grabScene(nFrames=50)
    #cam.showMoveDetect()
    targetData = cam.acquireTarget()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
