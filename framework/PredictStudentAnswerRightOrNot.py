import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc
import copy
import csv

def Demo(datasetName, testProblemNum):
    slipRange = np.arange(0,1,0.01)
    guessRange = np.arange(0,1,0.01)
    result = []
    #slip
    #guess
    # acc
    # auc
    # rmse
    # mae
    # f1
    for i in range(7):
        result.append([])
    for slip in slipRange:
        for guess in guessRange:
            predict_student_score = []
            true_student_score = []
            Q_matrix = np.loadtxt("../../../EduCDM-main/EduCDM-main/data/math2015Shuffled/" + datasetName + "/" + "shuffleQ_matrix.txt", dtype=np.int)
            raw_data = np.loadtxt("../../../EduCDM-main/EduCDM-main/data/math2015Shuffled/" + datasetName + "/" + "shuffleRaw_data.txt")

            student_knowledge_masterDF = pd.read_csv("C:\\Users\ybz\Desktop\ResultOfBean1\\" + datasetName + "\studentMasterWhole.csv", header=None)
            student_knowledge_master = student_knowledge_masterDF.values

            student_number = len(student_knowledge_master)

            # 当前题目所需知识点list
            thisProblemKnowledge = Q_matrix[testProblemNum]

            for studentIndex in range(student_number):
                cur_student_knowledge_master = student_knowledge_master[studentIndex]
                # 学生对应当前题目要求知识点掌握情况
                student_master_this_problem_list = []
                for know_index in range(len(thisProblemKnowledge)):
                    if thisProblemKnowledge[know_index] == 1:
                        student_master_this_problem_list.append(cur_student_knowledge_master[know_index])

                # 学生做
                # 对该题概率（取所有要求的知识点里面掌握度最小值）
                student_master_this_problem = min(student_master_this_problem_list)
                predictStudentAnswerRight = (1 - slip) * student_master_this_problem + student_master_this_problem * guess
                trueStudentAnswerRight = raw_data[studentIndex][testProblemNum]

                predict_student_score.append(predictStudentAnswerRight)
                true_student_score.append(trueStudentAnswerRight)

            acc = metrics.accuracy_score(true_student_score, np.array(predict_student_score).round())
            try:
                auc = metrics.roc_auc_score(true_student_score, predict_student_score)
            except ValueError:
                auc = 0.5
            mae = metrics.mean_absolute_error(true_student_score, predict_student_score)
            rmse = metrics.mean_squared_error(true_student_score, predict_student_score) ** 0.5
            _,_,F1 = calPrecisonRecallF1(true_student_score, predict_student_score)

            result[0].append(slip)
            result[1].append(guess)
            result[2].append(acc)
            result[3].append(auc)
            result[4].append(rmse)
            result[5].append(mae)
            result[6].append(F1)
    filePath = "./ParamTest/" + datasetName + "_Problem" + str(testProblemNum) + ".csv"
    with open(filePath, "w", newline='') as f:
        writer = csv.writer(f)
        for row in result:
            writer.writerow(row)

    print(datasetName, " ", testProblemNum, "\n")
    findMaxAccordingToEvaluateTarget(result, "acc")



def findMaxAccordingToEvaluateTarget(result, target):
    if target == "acc":
        index = result[2].index(max(result[2]))
    elif target == "auc":
        index = result[3].index(max(result[3]))
    elif target == "rmse":
        index = result[4].index(min(result[4]))
    elif target == "rmse":
        index = result[5].index(min(result[5]))
    else:
        index = result[6].index(max(result[6]))
    print("slip: ", result[0][index])
    print("guess: ", result[1][index])
    print("acc: ", result[2][index])
    print("auc: ", result[3][index])
    print("rmse: ", result[4][index])
    print("mae: ", result[5][index])
    print("f1: ", result[6][index])



def calPrecisonRecallF1(trueScoreList, predictScoreList):
    trueScoreListCopy = copy.copy(trueScoreList)
    predictScoreListCopy = copy.copy(predictScoreList)
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
        # print("TP: ", TP)
        # print("FP: ", FP)
        P = 0

    # 召回率Recall
    try:
        R = TP / (TP + FN)
    except:
        R = 0

    # F1
    try:
        F1 = 2 / (1 / P + 1 / R)
    except:
        F1 = 0
    #return Precision, Recall, F1
    return P, R, F1

if __name__ == "__main__":
    datasetList = ["FrcSub", "Math1", "Math2"]
    for dataSet in datasetList:
        if dataSet == "FrcSub":
            for problemIndex in range(16,20):
                Demo("FrcSub", problemIndex)
        elif dataSet == "Math1":
            for problemIndex in range(12,15):
                Demo("Math1", problemIndex)
        else:
            for problemIndex in range(13,16):
                Demo("Math2", problemIndex)

