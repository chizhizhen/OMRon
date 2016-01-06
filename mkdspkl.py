#!/usr/bin/python
#coding:utf-8
import argparse
import gzip, cPickle
import numpy as np
import cv2
import matplotlib.pyplot as pp

SAMPLING_STRIDE = 50.
MAX_COUNT_FRAME = 1024
#
OUTPUT_SHAPE = (98, 98)
OUTPUT_NUM_FRAME = 2
OUTPUT_INTERVAL = 10
FPS = 30.
#

class WDGT:
	def __init__(self, gtcsv):
		WDGT.readCsv(self, gtcsv)

	#class at the time
	def getClass(self, time):
		status = self.getStatus(time)["status"]
		if 0 <= status and status <= 4:
			return status
		return -1

	#status of the bed at the time
	def getReclining(self, time):
		status = self.getStatus(time)["reclining"]
		if 0 <= status and status <= 3:
			return status
		return -1

	#read ground truth
	def readCsv(self, gtcsv):
		self.events = []
		for gtline in open(gtcsv, 'r'):
			cols = gtline.rstrip().split(",")
			if len(cols) < 5:
				continue
			if WDGT.checkCsvLine(self, cols):
				self.events.append({"start":int(cols[0]), "end":int(cols[1]),
					"status":int(cols[2]), "reclining":int(cols[3]),
					"environment":int(cols[4])})
		self.unknownEvent = {"start":-1, "end":-1, "status":-1,
				"reclining":-1, "environment":-1}

	def getStatus(self, time):
		for event in self.events:
			if event["start"] <= time and time < event["end"]:
				return event
		return self.unknownEvent

	def length(self):
		maxTime = 0
		for event in self.events:
			if maxTime < event["end"]:
				maxTime = event["end"]
		return maxTime

	def checkCsvLine(self, cols):
		if not cols[0].isdigit():
			return False
		if not cols[1].isdigit():
			return False
		if not cols[2].isdigit():
			return False
		if not cols[3].isdigit():
			return False
		if not cols[4].isdigit():
			return False
		return True

def getFeature1clip(clipFile):
	emptyData = [np.zeros([0, OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
		* OUTPUT_NUM_FRAME], np.uint8), np.array([], np.uint8)]
	#open the clip and Ground Truth file.
	cap = cv2.VideoCapture(clipFile)
	if not cap.isOpened():
		print "cannot read " + clipFile + "."
		return emptyData
	try:
		wdgt = WDGT(clipFile[0:clipFile.rfind(".")] + ".csv")
	except:
		print "cannot read " + clipFile[0:clipFile.rfind(".")] + ".csv."
		return emptyData

	#read the image and GT
	mFeature = np.empty((OUTPUT_NUM_FRAME,) + OUTPUT_SHAPE, np.uint8)
	x = np.empty([MAX_COUNT_FRAME, OUTPUT_NUM_FRAME,
		OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]], np.uint8)
	y = np.empty([MAX_COUNT_FRAME], np.uint8)
	ret = cap.grab()
	fStep = float(SAMPLING_STRIDE)
	if MAX_COUNT_FRAME < wdgt.length() * FPS / SAMPLING_STRIDE:
		fStep = wdgt.length() * FPS / MAX_COUNT_FRAME
	fFrm = fStep / 2
	nF = 0
	while ret:
		nC = 0
		while nC < OUTPUT_NUM_FRAME:
			ret = cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, fFrm + OUTPUT_INTERVAL * nC)
			if not ret:
				fFrm = fFrm
			ret, mFrm = cap.retrieve()
			if not ret:
				break;
			mResized = cv2.resize(mFrm, OUTPUT_SHAPE)
			mFeature[nC] = cv2.cvtColor(mResized, cv2.COLOR_BGR2GRAY)
			nC += 1
		if not cap.grab() or not cap.grab():
			#grab twice for safe
			break
		nClass = wdgt.getClass((fFrm + OUTPUT_INTERVAL * (nC - 1)) / FPS)
		#
		if nClass != -1:
			x[nF] = mFeature
			y[nF] = nClass
			nF += 1
		if MAX_COUNT_FRAME <= nF or wdgt.length() * FPS * 2 < fFrm:
			break
		fFrm += fStep

	return [x[:nF], y[:nF]]

def getFeatures(listfile):
	XAll = np.zeros([0, OUTPUT_NUM_FRAME, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]],
			np.uint8)
	YAll = np.array([], np.uint8)
	for clipfile in open(listfile, 'r'):
		if(clipfile[0] == '#'):
			continue
		[x, y] = getFeature1clip(clipfile[:-1])
		print clipfile[:-1] + " " + str(y.shape[0])
		XAll = np.r_[XAll, x]
		YAll = np.r_[YAll, y]
	return [XAll, YAll]

parser = argparse.ArgumentParser(description='make dataset file from movie.')
parser.add_argument('--train', required=True, help='list of train set.')
parser.add_argument('--test', help='list of test set.')
parser.add_argument('--valid', help='list of valid set.')
parser.add_argument('dst', help='output. the type is .pkl.gz')
args=parser.parse_args()

#make training data
trainData = getFeatures(args.train)

#make test data
if args.test:
	testData = getFeatures(args.test)
else:
	testData = [np.zeros([0, trainData[0].shape[1]], trainData[0].dtype),
			np.array([], trainData[1].dtype)]
#make valid data
if args.valid:
	validData = getFeatures(args.valid)
else:
	validData = [np.zeros([0, trainData[0].shape[1]], trainData[0].dtype),
			np.array([], trainData[1].dtype)]

#output
f = gzip.open(args.dst, "wb")
cPickle.dump([trainData, testData, validData], f, protocol=-1)
f.close()
