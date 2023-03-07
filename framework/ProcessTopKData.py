import csv

import numpy as np
import pandas as pd
from copy import copy
from sklearn import metrics
from sklearn.metrics import auc


def loadEbeanDataAndClean():
    datasetList = ["FrcSub", "Math1", "Math2"]
    sliceNumber = [2,9,8]

    for index in range(3):
        result = pd.DataFrame()
        dataset = datasetList[index]
        sliceNo = sliceNumber[index]
        for currentNo in range(sliceNo):
            if currentNo == 0:
                ourDf = pd.read_csv("C:\\Users\\ybz\\Desktop\\TopK2\\" + dataset + "\\" + dataset + "predictAndGroundtruth.csv")
                lastIndex = 0
                for indexs in ourDf.index:
                    if ourDf.loc[indexs].values[0] == "user_id":
                        lastIndex = indexs
                ourDf = ourDf[lastIndex+1:int(ourDf.index.values[-1]+1)]
                result = result.append(ourDf)
            else:
                ourDf = pd.read_csv(
                    "C:\\Users\\ybz\\Desktop\\TopK2\\" + dataset + "\\" + dataset +"predictAndGroundtruth" " (" + str(currentNo) + ").csv")
                lastIndex = 0
                for indexs in ourDf.index:
                    if ourDf.loc[indexs].values[0] == "user_id":
                        lastIndex = indexs
                ourDf = ourDf[lastIndex + 1:int(ourDf.index.values[-1] + 1)]
                result = result.append(ourDf)
        result.to_csv("C:\\Users\\ybz\\Desktop\\TopK2\\" + dataset + "\\" + dataset + "predictAndGroundtruthWhole.csv")



#vote 出 topK
def voteToGetGroundTruth(TopK):
    RETURNTOCAL = []
    datasetList = ["FrcSub", "Math1", "Math2"]
    problemNumberList = [20, 15, 16]
    for totalIndex in range(3):
        dataset = datasetList[totalIndex]
        studentMasterDINA = pd.read_csv("C:\\Users\\ybz\\Desktop\\KDD_compare_experiment\\EduCDM-main\\EduCDM-main\\data\\math2015Shuffled\\" + dataset +"\\DINA_studentMaster_shuffle.csv")
        studentMasterFuzzyCDF = pd.read_csv("C:\\Users\\ybz\\Desktop\\KDD_compare_experiment\\EduCDM-main\\EduCDM-main\\data\\math2015Shuffled\\" + dataset +"\\FuzzyCDF_studentMaster_shuffle.csv")
        studentMasterNCDM = pd.read_csv("C:\\Users\\ybz\\Desktop\\status\\" + dataset +"_studentSta.csv")

        studentMasterDINA = studentMasterDINA.values
        studentMasterFuzzyCDF = studentMasterFuzzyCDF.values
        studentMasterNCDM = studentMasterNCDM.values


        Q_matrix = pd.read_csv("C:\\Users\\ybz\\Desktop\\KDD_compare_experiment\\EduCDM-main\\EduCDM-main\\data\\math2015Shuffled\\" + dataset +"\\q_mShuffle.csv", header=None)
        Q_matrix = Q_matrix.values
        raw_data = np.loadtxt("C:\\Users\\ybz\\Desktop\\KDD_compare_experiment\\EduCDM-main\\EduCDM-main\\data\\math2015Shuffled\\" + dataset +"\\shuffleRaw_data.txt", dtype=int)
        print()
        studentNumber = raw_data.shape[0]
        questionNumber = problemNumberList[totalIndex]

        RESULT = []
        for studentNo in range(studentNumber):
            stuMasterDINA = studentMasterDINA[studentNo]
            stuMasterFuzzyCDF = studentMasterFuzzyCDF[studentNo]
            stuMasterNCDM = studentMasterNCDM[studentNo]
            for questionNo in range(questionNumber):
                if raw_data[studentNo][questionNo] == 0:
                    KCNeed = Q_matrix[questionNo]
                    KCNumber = np.sum(KCNeed)
                    if KCNumber > TopK:
                        KCNumber = TopK

                    DINATopKIndex = extractTopKBadKC(stuMasterDINA, KCNeed, KCNumber)
                    FuzzyCDFTopKIndex = extractTopKBadKC(stuMasterFuzzyCDF, KCNeed, KCNumber)
                    NeuralCDMTopKIndex = extractTopKBadKC(stuMasterNCDM, KCNeed, KCNumber)

                    topKDic = {}
                    for _ in DINATopKIndex:
                        if _ in topKDic:
                            topKDic[_] += 1
                        else:
                            topKDic[_] = 1

                    for _ in FuzzyCDFTopKIndex:
                        if _ in topKDic:
                            topKDic[_] += 1
                        else:
                            topKDic[_] = 1

                    for _ in NeuralCDMTopKIndex:
                        if _ in topKDic:
                            topKDic[_] += 1
                        else:
                            topKDic[_] = 1

                    voteResult = []
                    while(len(voteResult) < TopK):
                        max_absent = -1
                        max_key = -1
                        for key, value in topKDic.items():
                            if value > max_absent:
                                max_absent = value
                                max_key = key
                        if max_key == -1:
                            break
                        voteResult.append(max_key)
                        topKDic[max_key] = -1
                    RESULT.append([studentNo, questionNo, voteResult])
        RETURNTOCAL.append(RESULT)
        RESULT = np.array(RESULT)
        RESULT = pd.DataFrame(RESULT)
        RESULT.to_csv(dataset + "TopK_" + str(TopK) + ".csv")
    return RETURNTOCAL



def extractTopKBadKC(studentMaster, KCNeed, KCNumber):
    KCMaster = []
    result = []
    for i in range(len(KCNeed)):
        if KCNeed[i] == 1:
            KCMaster.append(studentMaster[i])
    for i in range(KCNumber):
        min = KCMaster[0]
        min_index = 0
        for index in range(len(KCMaster)):
            if KCMaster[index] < min:
                min = KCMaster[index]
                min_index = index
        result.append(min_index)
        KCMaster[min_index] = 2

    returnIndex = []
    for item in result:
        count = 0
        for i in range(len(KCNeed)):
            if KCNeed[i] == 1:
                count += 1
                if count-1 == item:
                    returnIndex.append(i)
                    break

    return returnIndex


def calTheTopKAccuracyOfEBEAN(topk):
    RETURNTOCAL = voteToGetGroundTruth(topk)
    datasetList = ["FrcSub", "Math1", "Math2"]
    problemNumberList = [20, 15, 16]
    testList = [[9,10,7,3,1], [8,2,9], [2,3,10]]
    kn_list_dict = {
        "FrcSub": ["Convert a whole number to a fraction", "Separate a whole number from a fraction",
                   "Simplify before subtracting",
                   "Find a common denominator", "Borrow from whole number part",
                   "Column borrow to subtract the second numerator from the first",
                   "Subtract numerators", "Reduce answers to simplest form"],

        "Math1": ["Set", "Inequality", "Trigonometric function", "Logarithm versus exponential", "Plane vector",
                  "Property of function",
                  "Image of function", "Spatial imagination", "Abstract summarization", "Reasoning and demonstration",
                  "Calculation"],

        "Math2": ["Property of inequality", "Methods of data sampling", "Geometric progression",
                  "Function versus equation",
                  "Solving triangle", "Principles of data analysis", "Classical probability theory",
                  "Linear programming", "Definitions of algorithm",
                  "Algorithm logic", "Arithmetic progression", "Spatial imagination", "Abstract summarization",
                  "Reasoning and demonstration",
                  "Calculation", "Data handling"]
    }
    for index in range(3):
        dataset = datasetList[index]
        testQuestion = testList[index]
        EBEANDf = pd.read_csv("C:\\Users\\ybz\\Desktop\\TopK2\\" + dataset+ "\\" + datasetList[index] + "predictAndGroundtruthWhole.csv")
        voteData = RETURNTOCAL[index]
        EBEANData = EBEANDf.values
        itemLength = EBEANData.shape[0]

        totalCount = []
        for i in range(itemLength):
            studentNo = EBEANData[i][1]
            questionNo = EBEANData[i][2]
            if questionNo not in testQuestion:
                continue
            pathList = EBEANData[i][3]
            questionList = kn_list_dict[dataset]

            ##找到vote中的部分
            for _ in voteData:
                if _[0] == studentNo and _[1] == questionNo:
                    votePathIndex = _[2]

            count = 0
            for _ in votePathIndex:
                if questionList[_] in pathList:
                    count += 1
            if count/len(votePathIndex) != 0:
                totalCount.append(count/len(votePathIndex))

        print(sum(totalCount)/len(totalCount))










        print("")
# loadEbeanDataAndClean()
# voteToGetGroundTruth(2)
# for i in range(1,10):

calTheTopKAccuracyOfEBEAN(2)




                
