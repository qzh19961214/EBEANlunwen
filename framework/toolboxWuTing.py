import csv

import numpy as np
import pandas as pd
from copy import copy
from sklearn import metrics
from sklearn.metrics import auc


def loadTrueScoreAndPredictScore(trueFileName, predictFileName):
    trueScoreDf = pd.read_csv(trueFileName)
    predictScoreDf = pd.read_csv(predictFileName)

    trueScoreList = trueScoreDf[trueScoreDf.columns[0]].values.tolist()
    predictScoreList = predictScoreDf[predictScoreDf.columns[0]].values.tolist()

    return trueScoreList, predictScoreList


def evaluate(trueScoreList, predictScoreList):
    acc = metrics.accuracy_score(trueScoreList, np.array(predictScoreList).round())
    try:
        auc = metrics.roc_auc_score(trueScoreList, predictScoreList)
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(trueScoreList, predictScoreList)
    rmse = metrics.mean_squared_error(trueScoreList, predictScoreList) ** 0.5

    # acc = metrics.accuracy_score(trueScoreList, np.array(predictScoreList).round())
    # try:
    #     auc = metrics.roc_auc_score(trueScoreList, np.array(predictScoreList).round())
    # except ValueError:
    #     auc = 0.5
    # mae = metrics.mean_absolute_error(trueScoreList, np.array(predictScoreList).round())
    # rmse = metrics.mean_squared_error(trueScoreList, np.array(predictScoreList).round()) ** 0.5
    print("acc: ", acc)
    print("auc: ", auc)
    print("rmse: ", rmse)
    print("mae: ", mae)
    _, _, F1 = calPrecisonRecallF1(trueScoreList, predictScoreList)
    print("f1: ", F1)
    print("-------------------------------------------")
    return acc, auc, rmse, mae


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
    try:
        P = TP / (TP + FP)
    except:
        print("TP: ", TP)
        print("FP: ", FP)
        P = 0

    # 召回率Recall
    R = TP / (TP + FN)

    # F1
    try:
        F1 = 2 / (1 / P + 1 / R)
    except:
        F1 = 0
        print(1 / P + 1 / R)
    #return Precision, Recall, F1
    return P, R, F1


def calAUC(trueScoreList, predictScoreList):
    act = np.array(trueScoreList)
    pre = np.array(predictScoreList)
    FPR, TPR, thresholds = metrics.roc_curve(act, pre)
    AUC = auc(FPR, TPR)
    # print('AUC:', AUC)
    return AUC


if __name__ == "__main__":
    dataSetList = ["FrcSub", "Math1", "Math2"]
    algorithmList = ["DINA","FuzzyCDF"]
    ALG = 'NCD'

    pathName = 'C:/Users/ybz/Desktop/KDD_compare_experiment/EduCDM-main/EduCDM-main/data/math2015Shuffled/'
    for dataName in dataSetList:
        for algorithm in algorithmList:
            print("algorithm: ", algorithm, " , dataName: ", dataName)
            trueFileName = pathName + dataName+ "/" + algorithm + "_trueScore_shuffle.csv"
            predictFileName = pathName + dataName + "/" + algorithm + "_predictScore_shuffle.csv"

            trueScoreList, predictScoreList = loadTrueScoreAndPredictScore(trueFileName, predictFileName)

            accuracy = calAccuracy(trueScoreList, predictScoreList)
            RMSE = calRMSE(trueScoreList, predictScoreList)
            precision, recall, F1 = calPrecisonRecallF1(trueScoreList, predictScoreList)
            AUC = calAUC(trueScoreList, predictScoreList)
            evaluate(trueScoreList, predictScoreList)
            clue = "accuracy: " + str(accuracy) + "\n"
            clue += "RMSE: " + str(RMSE) + "\n"
            clue += "precision: " + str(precision) + "\n"
            clue += "recall: " + str(recall) + "\n"
            clue += "F1: " + str(F1) + "\n"
            clue += "AUC: " + str(AUC) + "\n"
            print(clue)
            # file = open(r"data/" + pathName + dataName + 'statistic.txt', 'w')
            # file.write(r"data/" + pathName + clue)




    #
    # dataSetList = ["FrcSub", "Math1", "Math2"]
    # algorithmList = ["BEAN"]
    # ALG = 'NCD'
    #
    # pathName = "C:/Users/ybz/Desktop/ResultOfBean1/"
    # for dataName in dataSetList:
    #     for algorithm in algorithmList:
    #         print("algorithm: ", algorithm, " , dataName: ", dataName)
    #         trueFileName = pathName + dataName + "/"  + "trueScoreWhole.csv"
    #         predictFileName = pathName + dataName + "/" + "predictScoreWhole.csv"
    #
    #         trueScoreList, predictScoreList = loadTrueScoreAndPredictScore(trueFileName, predictFileName)
    #
    #         accuracy = calAccuracy(trueScoreList, predictScoreList)
    #         RMSE = calRMSE(trueScoreList, predictScoreList)
    #         precision, recall, F1 = calPrecisonRecallF1(trueScoreList, predictScoreList)
    #         AUC = calAUC(trueScoreList, predictScoreList)
    #         evaluate(trueScoreList, predictScoreList)
    #         clue = "accuracy: " + str(accuracy) + "\n"
    #         clue += "RMSE: " + str(RMSE) + "\n"
    #         clue += "precision: " + str(precision) + "\n"
    #         clue += "recall: " + str(recall) + "\n"
    #         clue += "F1: " + str(F1) + "\n"
    #         clue += "AUC: " + str(AUC) + "\n"
    #         print(clue)
    #         # file = open(r"data/" + pathName + dataName + 'statistic.txt', 'w')
    #         # file.write(r"data/" + pathName + clue)



