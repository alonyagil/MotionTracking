import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import path
import argparse

DUMP_PATH = 'C:\gitRepos\cameraStuff\data'
N_CAM_CHANNEL = 3
MIN_FEATURE_SIZE = 10
TIMEOUT_FRAMES = 1000


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--connectivity", type=int, default=4,
	help="connectivity for connected componenqt analysis")
ARGS = vars(ap.parse_args())

class Camera:

    def __init__(self, camID=1):
        self.camID = camID
        self.cam = cv2.VideoCapture(camID)
        self.W = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reoccuringMovers = np.zeros((self.W, self.H), dtype=bool)
        self.bckImg = np.empty((self.H, self.W, N_CAM_CHANNEL))
        self.bckThresh = 0
        self.filterLength = np.uint(np.math.sqrt(MIN_FEATURE_SIZE))
        if self.filterLength % 2 == 0:
            self.filterLength = np.uint(np.math.sqrt(MIN_FEATURE_SIZE)) + 1
        print('camera successfully initiated')

    def __del__(self):
        cv2.destroyAllWindows()
        self.cam.release()

    def grabScene(self, nFrames=100):
       cv2.namedWindow('grabbing background')
       tempBuff = np.empty((self.H, self.W, N_CAM_CHANNEL, nFrames))
       for pp in range(nFrames):
           ret, frame = self.cam.read()
           if not ret:
               print("failed to grab frame")
               break
           cv2.imshow('grabbing background', frame)
           cv2.waitKey(1)
           tempBuff[:, :, :, pp] = frame
       scene = (tempBuff.mean(3)).astype(np.uint8)
       self.bckImg = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
       stdArr = tempBuff.std(3).astype(np.uint8)
       maxIdx = np.argmax(stdArr)
       idxSub = np.unravel_index(maxIdx, stdArr.shape)
       self.bckThresh = 3 * stdArr[idxSub]
       print('max std = {} GL'.format(self.bckThresh))
       cv2.destroyAllWindows()
       cv2.imwrite(path.join(DUMP_PATH, 'bckIm.tif'), scene)

       fig, ax = plt.subplots()
       ax.hist(tempBuff[idxSub[0], idxSub[1], idxSub[2], :], 100)
       fig.savefig(path.join(DUMP_PATH, 'hist.png'))

       stdIm = stdArr.max(2)
       fig, ax = plt.subplots()
       im = ax.imshow(stdIm)
       plt.colorbar(im)
       fig.savefig(path.join(DUMP_PATH, 'stdIm.png'))

    def detectMove(self):
        ret, frame = self.cam.read()
        if not ret:
            print("failed to grab frame")
            return np.empty((self.H, self.W, N_CAM_CHANNEL)), {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.filterLength, self.filterLength), 0)
        movedPixels = cv2.absdiff(blurred, self.bckImg)
        thresh = cv2.threshold(movedPixels.astype(np.uint8), self.bckThresh, 255,
                      cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        #thresh = cv2.erode(thresh, None, iterations=2)
        output = cv2.connectedComponentsWithStats(
            thresh, ARGS["connectivity"], cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        if numLabels == 1:
            return frame, [] ,np.zeros(frame.shape)
        foundFeatures = []
        for i in range(1, numLabels):
            if stats[i, cv2.CC_STAT_AREA] < MIN_FEATURE_SIZE: # if area smaller than min- ignore feature
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            (cX, cY) = centroids[i]
            leftCent, rightCent, topCent, bottomCent = self.calcBox(centroids[i], [MIN_FEATURE_SIZE, MIN_FEATURE_SIZE])
            isReocurrent = np.any(self.reoccuringMovers[leftCent:rightCent, topCent:bottomCent])
            if isReocurrent:# ignore reoccurent features
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(frame, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            foundFeatures.append({'label': i, 'box_xywh': [x, y, w, h], 'center_xy': [cX, cY]})
        return frame, foundFeatures, labels

    def calcBox(self, center_xy, size_wh):
        leftCent = max([round(center_xy[0] - size_wh[0] / 2), 0])
        rightCent = min([round(center_xy[0] + size_wh[0] / 2), self.W - 1])
        topCent = max([round(center_xy[1] - size_wh[1] / 2), 0])
        bottomCent = min([round(center_xy[1] + size_wh[1] / 2), self.H - 1])
        return leftCent, rightCent, topCent, bottomCent

    def updateReocurrentMovers(self, nIters=120, minOccur=10):
        nOccur = np.zeros((self.W, self.H), dtype=int)
        for iter in range(nIters):
            _, foundFeatures, _ = self.detectMove()
            for feature in foundFeatures:
                leftCent, rightCent, topCent, bottomCent = self.calcBox(feature['center_xy'],
                                                                        [MIN_FEATURE_SIZE, MIN_FEATURE_SIZE])
                nOccur[leftCent:rightCent, topCent:bottomCent] += 1
        self.reoccuringMovers =  nOccur > minOccur

    def showMoveDetect(self):
        self.updateReocurrentMovers()
        cv2.namedWindow('movement track')
        while True:
            frame, foundFeatures, _ = self.detectMove()
            cv2.imshow('movement track', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def acquireTarget(self, staticCounts=100):
        print('Updating reoccurent movers')
        self.updateReocurrentMovers()
        print('Place object in camera FOV')
        searchMove = True
        targetCenterOld = [0, 0]
        staticCounter = 0
        iterNum = 0
        while staticCounter < staticCounts:
            if iterNum > TIMEOUT_FRAMES:
                print('No movement detected- exiting function')
                break
            frame, foundFeatures, labels = self.detectMove()
            cv2.imshow('Place object in camera FOV', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if len(foundFeatures) < 1:
                continue
            maxFeature = 0
            maxFeatureIdx = 0
            for idx, feature in enumerate(foundFeatures):
                area = feature['box_xywh'][2] * feature['box_xywh'][3]
                if area > maxFeature:
                    maxFeatureIdx = idx
                    maxFeature = area
            targetCenterNew = foundFeatures[maxFeatureIdx]['center_xy']
            largeObjectDrift = np.math.sqrt((targetCenterNew[0] - targetCenterOld[0])**2
                                            + (targetCenterNew[1] - targetCenterOld[1])**2)
            targetCenterOld = targetCenterNew
            if largeObjectDrift < MIN_FEATURE_SIZE:
                staticCounter += 1
            iterNum += 1
        ret, newFrame = self.cam.read()
        if not ret:
            print("failed to grab frame")
            return
        targetBox = foundFeatures[maxFeatureIdx]['box_xywh']
        target = newFrame[targetBox[1] : targetBox[1] + targetBox[3], targetBox[0] : targetBox[0] + targetBox[2], :]
        cv2.imwrite(path.join(DUMP_PATH, 'target.tif'), target)

