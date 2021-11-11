from CameraUsbCtr import Camera


if __name__ == '__main__':
    cam = Camera()
    cam.grabScene(nFrames=50)
    #cam.showMoveDetect()
    cam.acquireTarget()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
