from sklearn.cluster import MeanShift
import numpy as np
import pickle
from os import path
import cv2
import time

DUMP_PATH = '..\MotionTracking\data'

class MovingObject:
    # list of attributes:
    # clustering - MeanShift obj

    def __init__(self, targetData):
        self.__doColorMeanShift__(targetData)

    def __doColorMeanShift__(self, targetData, downsample=8):
        # extract pixels
        frame = targetData['frame']
        # do mean shift
        frame2cluster = cv2.resize(frame, (int(frame.shape[0]/downsample),
                                           int(frame.shape[0]/downsample)))
        nPix = frame2cluster.shape[0] * frame2cluster.shape[1]
        X = np.vstack([frame2cluster.reshape(nPix, frame2cluster.shape[2]), frame[targetData['targetMask']]])
        print('running mean shift on {} samples'.format(X.shape[0]))
        start = time.time()
        self.clustering = MeanShift(bandwidth=10, n_jobs=4, cluster_all=False).fit(X)
        end = time.time()
        elpased = end - start
        print('mean shift done, elaped time: {} m, {} s'.format(round(elpased/60), elpased % 60))
        # fit predict on box
        clusteredBox, boxLabels = self.__predictMeanShiftCluster__(targetData['targetFrame'])
        cv2.imwrite(path.join(DUMP_PATH, 'clusteredTarget.tif'), clusteredBox)
        # fit on frame
        clusteredFrame, frameLabels = self.__predictMeanShiftCluster__(frame)
        cv2.imwrite(path.join(DUMP_PATH, 'clusteredFrame.tif'), clusteredFrame)
        # get most useful colors
        optTargetMask, colors2use = self.__findColors4train__(frameLabels, targetData)
        frame2plot = np.zeros(frame.shape, dtype=np.uint8)
        frame2plot[optTargetMask] = frame[optTargetMask]
        cv2.imwrite(path.join(DUMP_PATH, 'optMaskPix.tif'), frame2plot)
        return optTargetMask, colors2use


    def __predictMeanShiftCluster__(self, frame):
        nPixFrame = frame.shape[0] * frame.shape[1]
        y2 = self.clustering.predict(frame.reshape(nPixFrame, frame.shape[2]))
        clusteredFrame = np.zeros((nPixFrame, frame.shape[2]), dtype=np.uint8)
        for label, center in enumerate(self.clustering.cluster_centers_):
            c1 = np.round(center)
            clusteredFrame[y2 == label, :] = c1
        return clusteredFrame.reshape(frame.shape), y2.reshape((frame.shape[0], frame.shape[1]))

    def __findColors4train__(self, frameLabels, targetData, accThresh=0.9):
        targetMask = targetData['targetMask']
        targetLabels = np.unique(frameLabels[targetMask])
        targetPix = frameLabels[targetMask]
        notTargetPix = frameLabels[np.logical_not(targetMask)]
        optTargetMask = np.zeros(targetMask.shape, dtype=bool)
        colors2use = []
        for label in targetLabels:
            correct = np.count_nonzero(targetPix == label) + \
                      np.count_nonzero(notTargetPix != label)
            accuracy = correct / frameLabels.size
            if accuracy > accThresh:
                idx = np.logical_and(frameLabels == label, targetMask)
                optTargetMask[idx] = True
                colors2use.append(self.clustering.cluster_centers_[label, :])
        return optTargetMask, colors2use

        print('cds')

if __name__ == '__main__':
    with open(path.join(DUMP_PATH, 'horseData.pkl'), 'rb') as file:
        horseData = pickle.load(file)
        horse = MovingObject(horseData)