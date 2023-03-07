import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc
import copy
import csv

def Demo(datasetName):
    if dataSet == "FrcSub":
        slipList = [0.08, 0.0, 0.03, 0.11]
        guessList = [0.0, 0.0, 0.0, 0.0]
    elif dataSet == "Math1":
        slipList = [0.09, 0.0, 0.02]
        guessList = [0.0, 0.0, 0.0]
    else:
        slipList = [0.0, 0.01, 0.0]
        guessList = [0.06, 0.0, 0.03]

    # slipList = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "\shuffleSlip.txt")
    # guessList = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "\shuffleGuess.txt")

    predict_student_score = []
    true_student_score = []
    if datasetName == "FrcSub":
        for problemIndex in range(16, 20):
            slip = slipList[problemIndex - 16]
            guess = guessList[problemIndex - 16]
            # slip = slipList[problemIndex]
            # guess = guessList[problemIndex]
            Q_matrix = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "/" + "shuffleQ_matrix.txt", dtype=np.int)
            raw_data = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "/" + "shuffleRaw_data.txt")

            student_knowledge_masterDF = pd.read_csv("C:\\Users\ybz\Desktop\ResultOfBean1\\" + datasetName + "\studentMasterWhole.csv", header=None)
            student_knowledge_master = student_knowledge_masterDF.values

            student_number = len(student_knowledge_master)

            # 当前题目所需知识点list
            thisProblemKnowledge = Q_matrix[problemIndex]

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
                trueStudentAnswerRight = raw_data[studentIndex][problemIndex]

                predict_student_score.append(predictStudentAnswerRight)
                true_student_score.append(trueStudentAnswerRight)

    if datasetName == "Math1":
        for problemIndex in range(12,15):
            slip = slipList[problemIndex - 12]
            guess = guessList[problemIndex - 12]
            # slip = slipList[problemIndex]
            # guess = guessList[problemIndex]
            Q_matrix = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "\\" + "shuffleQ_matrix.txt", dtype=np.int)
            raw_data = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "\\" + "shuffleRaw_data.txt")

            student_knowledge_masterDF = pd.read_csv("C:\\Users\ybz\Desktop\ResultOfBean1\\" + datasetName + "\studentMasterWhole.csv", header=None)
            student_knowledge_master = student_knowledge_masterDF.values

            student_number = len(student_knowledge_master)

            # 当前题目所需知识点list
            thisProblemKnowledge = Q_matrix[problemIndex]

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
                trueStudentAnswerRight = raw_data[studentIndex][problemIndex]

                predict_student_score.append(predictStudentAnswerRight)
                true_student_score.append(trueStudentAnswerRight)

    if datasetName == "Math2":
        for problemIndex in range(13,16):
            slip = slipList[problemIndex - 13]
            guess = guessList[problemIndex - 13]
            # slip = slipList[problemIndex]
            # guess = guessList[problemIndex]
            Q_matrix = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "/" + "shuffleQ_matrix.txt", dtype=np.int)
            raw_data = np.loadtxt("../../../EduCDM-main\EduCDM-main\data\math2015Shuffled/" + datasetName + "/" + "shuffleRaw_data.txt")

            student_knowledge_masterDF = pd.read_csv("C:\\Users\ybz\Desktop\ResultOfBean1\\" + datasetName + "\studentMasterWhole.csv", header=None)
            student_knowledge_master = student_knowledge_masterDF.values

            student_number = len(student_knowledge_master)

            # 当前题目所需知识点list
            thisProblemKnowledge = Q_matrix[problemIndex]

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
                trueStudentAnswerRight = raw_data[studentIndex][problemIndex]

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
    print(datasetName)
    print("acc: ", acc)
    print("auc: ", auc)
    print("rmse: ", rmse)
    print("mae: ", mae)
    print("f1: ", F1)




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
        Demo(dataSet)
