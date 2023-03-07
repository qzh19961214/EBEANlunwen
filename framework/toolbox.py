import csv

import numpy as np
import pandas as pd
from copy import copy
from sklearn import metrics
from sklearn.metrics import auc


def loadTrueScoreAndPredictScore(trueFileName, predictFileName):
    trueScoreDf = pd.read_csv(trueFileName, header=None)
    predictScoreDf = pd.read_csv(predictFileName, header=None)

    trueScoreList = trueScoreDf[0].values.tolist()
    predictScoreList = predictScoreDf[0].values.tolist()

    return trueScoreList, predictScoreList


def calAccuracy(trueScoreList, predictScoreList):
    trueScoreListCopy = copy(trueScoreList)
    predictScoreListCopy = copy(predictScoreList)

    #学生做题总数
    accuracy = []
    answerNumber = len(trueScoreListCopy)
    for i in range(answerNumber):
        accuracy.append(1-abs(trueScoreListCopy[i]-predictScoreListCopy[i]))

    return np.average(accuracy)


def calRMSE(trueScoreList, predictScoreList):
    trueScoreListCopy = copy(trueScoreList)
    predictScoreListCopy = copy(predictScoreList)

    # 学生做题总数
    RMSE = []
    answerNumber = len(trueScoreListCopy)
    for i in range(answerNumber):
        RMSE.append((trueScoreListCopy[i] - predictScoreListCopy[i]) ** 2)

    return np.average(RMSE)

def calPrecisonRecallF1(trueScoreList, predictScoreList):
    trueScoreListCopy = copy(trueScoreList)
    predictScoreListCopy = copy(predictScoreList)
    answerNumber = len(trueScoreListCopy)

    for i in range(answerNumber):
        if predictScoreListCopy[i] >= 0.5:
            predictScoreListCopy[i] = 1
        else:
            predictScoreListCopy[i] = 0


    for i in range(answerNumber):
        if trueScoreListCopy[i] >= 0.5:
            trueScoreListCopy[i] = 1
        else:
            trueScoreListCopy[i] = 0

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(answerNumber):
        if trueScoreListCopy[i] == 1 and predictScoreListCopy[i] == 1:
            TP += 1
        if trueScoreListCopy[i] == 0 and predictScoreListCopy[i] == 1:
            FP += 1
        if trueScoreListCopy[i] == 1 and predictScoreListCopy[i] == 0:
            FN += 1
        if trueScoreListCopy[i] == 0 and predictScoreListCopy[i] == 0:
            TN += 1

    # 精确率Precision
    P = TP / (TP + FP)

    # 召回率Recall
    R = TP / (TP + FN)

    # F1
    F1 = 2 / (1 / P + 1 / R)

    #return Precision, Recall, F1
    return P, R, F1


def calAUC(trueScoreList, predictScoreList):
    act = np.array(trueScoreList)
    pre = np.array(predictScoreList)
    FPR, TPR, thresholds = metrics.roc_curve(act, pre)
    AUC = auc(FPR, TPR)
    return AUC


if __name__ == "__main__":
    # dataSetList = ["FrcSub", "Math1", "Math2"]
    # for dataName in dataSetList:
    #     trueFileName = r"../math2015/"+dataName+"/trueScore.csv"
    #     predictFileName = r"../math2015/" + dataName + "/predictScore.csv"
    #
    #     trueScoreList, predictScoreList = loadTrueScoreAndPredictScore(trueFileName, predictFileName)
    #
    #     accuracy = calAccuracy(trueScoreList, predictScoreList)
    #     RMSE = calRMSE(trueScoreList, predictScoreList)
    #     precision, recall, F1 = calPrecisonRecallF1(trueScoreList, predictScoreList)
    #     AUC = calAUC(trueScoreList, predictScoreList)


    #
    #
    # rateList = [3,5,7,9,11,13,15,17]
    # for rate in rateList:
    #     trueFileName = r"../rateResult/"  + "/trueScore" + str(rate) + ".csv"
    #     predictFileName = r"../rateResult/" + "/predictScore" + str(rate)+ ".csv"
    #
    #     trueScoreList, predictScoreList = loadTrueScoreAndPredictScore(trueFileName, predictFileName)
    #
    #     accuracy = calAccuracy(trueScoreList, predictScoreList)
    #     RMSE = calRMSE(trueScoreList, predictScoreList)
    #     precision, recall, F1 = calPrecisonRecallF1(trueScoreList, predictScoreList)
    #     AUC = calAUC(trueScoreList, predictScoreList)
    #
    #     print("Rate:",rate)
    #     print(accuracy)
    #     print(RMSE)
    #     print(precision)
    #     print(recall)
    #     print(F1)
    #     print(AUC)
    #     print("\n\n")
    #

    datasetList = ["FrcSub", "Math1", "Math2"]
    for dataset in datasetList:
        trueFileName = "C:\\Users\\ybz\\Desktop\\ResultOfBean1\\" + dataset + "\\trueScoreWhole.csv"
        predictFileName = "C:\\Users\\ybz\\Desktop\\ResultOfBean1\\" + dataset + "\\predictScoreWhole.csv"

        trueScoreList, predictScoreList = loadTrueScoreAndPredictScore(trueFileName, predictFileName)

        accuracy = calAccuracy(trueScoreList, predictScoreList)
        RMSE = calRMSE(trueScoreList, predictScoreList)
        precision, recall, F1 = calPrecisonRecallF1(trueScoreList, predictScoreList)
        AUC = calAUC(trueScoreList, predictScoreList)

        print("dataset:", dataset)
        print(accuracy)
        print(RMSE)
        print(precision)
        print(recall)
        print(F1)
        print(AUC)
        print("\n\n")

