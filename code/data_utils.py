'''
Code for Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data (MICCAI 2021)

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca & Megha Kalia - mkalia@ece.ubc.ca (equal contribution)
Robotics and Control Laboratory, University of British Columbia, Vancouver,
Canada
'''
import math
import numpy as np

def use_operating_points(operation_point, y_gt, y_pr):

	iou_all = []
	dice_all = []
	sen_all = []
	spe_all = []

	for i in range(np.shape(y_gt)[0]):
		y_temp = y_pr[i] >= operation_point
		TP, FP, TN, FN = perf_measure(y_gt[i] >= 0.5, y_temp)

		accuracy = (TP + TN)/(TP + FP + TN + FN)
		fpr_tempp = (FP / (FP + TN))
		tpr_tempp = (TP / (TP + FN))

		specificity = 1 - fpr_tempp
		sensitivity = tpr_tempp
		dice = (2*TP)/((2*TP)+FP+FN)
		iou = TP/(TP + FP + FN)

		if math.isnan(specificity):
			specificity = 0
		if math.isnan(sensitivity):
			sensitivity = 0
		if math.isnan(dice):
			dice = 0
		if math.isnan(iou):
			iou = 0

		iou_all.append(iou)
		dice_all.append(dice)
		sen_all.append(sensitivity)
		spe_all.append(specificity)

	return spe_all, sen_all, dice_all, iou_all

def perf_measure(y_actual, y_hat):

	TP = np.logical_and(y_actual,y_hat)
	FP = np.logical_and(y_hat,abs(y_actual-1))
	TN = np.logical_and(abs(y_hat-1),abs(y_actual-1))
	FN = np.logical_and(y_actual,abs(y_hat-1))

	return(np.sum(TP).astype('float64'), np.sum(FP).astype('float64'), np.sum(TN).astype('float64'), np.sum(FN).astype('float64'))





