import numpy as np
import random as rd
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import beta
from scipy.stats import binom


#用于模拟学生题目做对做错情况
def calculateStudentKnowledgeOneOrZero(stu, ques, studentMaster, studentAnswerQuestion, Q_matrix, slip_stu, guess_stu):
    AnswerCorrect = studentAnswerQuestion[stu][ques]
    knowledgeOfThisQuestion = Q_matrix[ques]
    currentStudentMaster = studentMaster[stu]
    KnowledgeConceptList = []
    if AnswerCorrect == 0:
        for i in range(len(knowledgeOfThisQuestion)):
            if knowledgeOfThisQuestion[i] == 0:
                KnowledgeConceptList.append("null")
            elif knowledgeOfThisQuestion[i] == 1:
                scoreCorrect = (1-slip_stu)*currentStudentMaster[i] + guess_stu*(1-currentStudentMaster[i])
                if rd.random() < scoreCorrect:
                    KnowledgeConceptList.append(1)
                else:
                    KnowledgeConceptList.append(0)
    else:
        for i in range(len(knowledgeOfThisQuestion)):
            if knowledgeOfThisQuestion[i] == 0:
                KnowledgeConceptList.append("null")
            elif knowledgeOfThisQuestion[i] == 1:
                KnowledgeConceptList.append(1)

    return KnowledgeConceptList


def theCandidateEdgeWithNormalizedMutualWeight(samples, nodeName):
    samples1 = pd.DataFrame(samples, columns =nodeName)
    nodeNumber = len(nodeName)
    CondidateEdgeMatrix = np.zeros((nodeNumber, nodeNumber))

    mutualInformationMatrix = np.zeros((nodeNumber, nodeNumber))
    # 上三角，避免了选的时候同一个值会计算两遍
    for i in range(nodeNumber):
        for j in range(i + 1, nodeNumber):
            mutualInformationMatrix[i][j] = normalized_mutual_info_score(samples1[nodeName[i]].values,
                                                                         samples1[nodeName[j]].values)
            mutualInformationMatrix[j][i] = mutualInformationMatrix[i][j]

    matrixRank = np.zeros((nodeNumber,nodeNumber))
    for i in range(nodeNumber):
        for j in range(nodeNumber):
            currentNumber = mutualInformationMatrix[i][j]
            count = 0
            for k in range(nodeNumber):
                for l in range(nodeNumber):
                    if mutualInformationMatrix[k][l] > currentNumber:
                        count += 1
            matrixRank[i][j] = count
    print(" ")


    return mutualInformationMatrix



if __name__ == '__main__':
    dataSetList = ["FrcSub", "Math1", "Math2"]
    problem_number_dict = {
        "FrcSub": 20,
        "Math1": 15,
        "Math2": 16
    }
    for dataName in dataSetList:
        if dataName == "FrcSub":
            masterFileName = "../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/"+ dataName + "/FrcSub_student_status.csv"
        if dataName == "Math1":
            masterFileName = "../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/" + dataName + "/Math12_student_status.csv"
        if dataName == "Math2":
            masterFileName = "../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/" + dataName + "/Math22_student_status.csv"
        masterFileName = "../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/"+ dataName + "/studentMaster.csv"
        studentMaster = pd.read_csv(masterFileName,dtype=np.float)
        studentMaster = studentMaster.drop(studentMaster.columns[0], axis=1)
        studentMaster = studentMaster.values
        # mutualMa = theCandidateEdgeWithNormalizedMutualWeight(studentMaster, ['sad0','1asd','dsg2','gj3','4kl','5asd','6sdf','7wer','8rty','9qwe','10rt'])

        studentAnswerQuestion = np.loadtxt("../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/"+ dataName + "/data.txt",dtype=np.float)
        studentAnswerQuestion = studentAnswerQuestion[:,range(problem_number_dict[dataName])]
        Q_matrix = np.loadtxt("../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/"+ dataName + "/q.txt",dtype=np.int)
        Q_matrix = Q_matrix[range(problem_number_dict[dataName]),:]

        kn_list_dict = {
            "FrcSub": ["Convert a whole number to a fraction", "Separate a whole number from a fraction",
                       "Simplify before subtracting",
                       "Find a common denominator", "Borrow from whole number part",
                       "Column borrow to subtract the second numerator from the first",
                       "Subtract numerators", "Reduce answers to simplest form"],

            "Math1": ["Set", "Inequality", "Trigonometric function", "Logarithm versus exponential", "Plane vector",
                      "Property of function",
                      "Image of function", "Spatial imagination", "Abstract summarization",
                      "Reasoning and demonstration", "Calculation"],

            "Math2": ["Property of inequality", "Methods of data sampling", "Geometric progression",
                      "Function versus equation",
                      "Solving triangle", "Principles of data analysis", "Classical probability theory",
                      "Linear programming", "Definitions of algorithm",
                      "Algorithm logic", "Arithmetic progression", "Spatial imagination", "Abstract summarization",
                      "Reasoning and demonstration",
                      "Calculation", "Data handling"]
        }


        Q_name = kn_list_dict[dataName]

        studentNumber = studentMaster.shape[0]
        knowledgeConceptNumber = studentMaster.shape[1]
        questionNumber = studentAnswerQuestion.shape[1]
        column = ['user_id', 'exercise_code']
        column.extend(Q_name)
        column.extend(["is_right_answer"])
        df = pd.DataFrame(columns=column)
        slip = np.loadtxt("../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/"+ dataName + "/slip.txt")
        guess = np.loadtxt("../../../EduCDM-main/EduCDM-main/data/math2015_FuzzyCDF/"+ dataName + "/guess.txt")
        for stu in range(studentNumber):
            for ques in range(questionNumber):
                instance = calculateStudentKnowledgeOneOrZero(stu, ques, studentMaster, studentAnswerQuestion, Q_matrix, slip[ques], guess[ques])
                thisData = [stu,ques]
                thisData.extend(instance)
                thisData.extend([studentAnswerQuestion[stu][ques]])
                df2 = pd.DataFrame([thisData], columns=column)
                df = df.append(df2,ignore_index=True)

        df.to_csv("../math2015/" + dataName + "/QA_Generate.csv")



