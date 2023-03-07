import pandas as pd
import numpy as np
from GetBN import *
# from framework.Plot_dag import *
from datetime import datetime
import os
import csv

def get_evd(data_name, value, file_name, kn_list, suyin_timu):
    df = pd.read_csv(file_name)
    list_name = [data_name, 'is_right_answer']
    for kn in kn_list:
        list_name.append(kn)
    data = df[list_name]
    #  得到某一学生/学校在所有题目的做题数据
    data = data.loc[df[data_name] == value]
    evidence_list =[]
    evidence_node_list = []
    suyin_list = []
    suyin_node_list = []
    exercise_code_list = []
    doAttributeOrnot = []
    for i in data.index:
        if df.loc[i, 'exercise_code'] not in suyin_timu:
            # print("evidence:",i)
            evidence = []
            evidence_node = []
            for kn in kn_list:
                value = df.loc[i, kn]
                if not pd.isnull(value):
                    evidence_node.append(kn)
                    evidence.append(int(value))
            evidence.append(df.loc[i, 'is_right_answer'])
            evidence_node.append("wrong")
            evidence_list.append(evidence)
            evidence_node_list.append(evidence_node)
            doAttributeOrnot.append(0)
        else:
            # print("suyin:", i)
            exercise_code_list.append(df.loc[i, 'exercise_code'])
            suyin_evidence = []
            suyin_evidence_node = []
            for kn in kn_list:
                value = df.loc[i, kn]
                if not pd.isnull(value):
                    suyin_evidence_node.append(kn)
                    suyin_evidence.append(int(value))
            suyin_evidence.append(df.loc[i, 'is_right_answer'])
            suyin_evidence_node.append("wrong")
            suyin_list.append(suyin_evidence)
            suyin_node_list.append(suyin_evidence_node)
            doAttributeOrnot.append(1)
    return evidence_list, evidence_node_list, suyin_list, suyin_node_list,exercise_code_list,doAttributeOrnot


def run(dataname, id, suyin_timu,csvFilePath, slip, guess, Q_matrix, train_scale, datasetName, our_studentMaster, predict_student_score, true_student_score, raw_data, rate):
    # file_name = "E:/ecnu/CASUAL IN EDU/causal/rankData/QA.csv"
    file_path = 'output/' + datasetName + "/" + dataname + "_" + str(id) + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    f_log = open(file_path + "log_.log", "a")
    f_evidence = open(file_path + "evidence.log", "a")
    f_suyin = open(file_path + "suyin.log", "a")

    file_name =  "../math2015/" + datasetName + "/" + datasetName + "QA_Generate.csv"

    kn_list_dict = {
        "FrcSub":["Convert a whole number to a fraction", "Separate a whole number from a fraction", "Simplify before subtracting",
     "Find a common denominator", "Borrow from whole number part","Column borrow to subtract the second numerator from the first",
     "Subtract numerators", "Reduce answers to simplest form"],

        "Math1":["Set", "Inequality", "Trigonometric function", "Logarithm versus exponential", "Plane vector","Property of function",
                 "Image of function", "Spatial imagination", "Abstract summarization", "Reasoning and demonstration","Calculation"],

        "Math2":["Property of inequality", "Methods of data sampling", "Geometric progression", "Function versus equation",
     "Solving triangle","Principles of data analysis", "Classical probability theory", "Linear programming", "Definitions of algorithm",
     "Algorithm logic","Arithmetic progression", "Spatial imagination", "Abstract summarization", "Reasoning and demonstration",
     "Calculation", "Data handling"]
    }


    kn_list = kn_list_dict[datasetName]
    matrix = []
    if dataname == "FrcSub":
        matrix =    [[0, 0, 1, 0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1],
                     [0 ,1 ,0 ,0 ,0 ,0 ,0, 0],
                     [0 ,0 ,0 ,0 ,1, 0, 0 ,1],
                     [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1],
                     [0 ,0 ,0 ,0, 0, 0 ,0 ,0]]
    if dataname == "Math1":
        matrix = [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]]
    if dataname == "Math2":
        matrix = [
                    [0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0],
                    [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ]

    evidence_list, evidence_node_list, suyin_list, suyin_node_list,exercise_code_list,doAttributeOrnot= get_evd(dataname, id, file_name, kn_list, suyin_timu)
    index = 0

    lastSuyinIndex = -1
    for i in range(len(doAttributeOrnot)):
        if doAttributeOrnot[i] == 1:
            lastSuyinIndex = i

    masterFileName = "./studentMasterOnEachKnowledge.txt"
    studentMaster = np.loadtxt(masterFileName, dtype=np.float)
    student_id_each_knowle_master_prob = studentMaster[id]

    model = constructTheBN(matrix, kn_list)
    item_number = 500    # 修改总图采样条数
    item_number_sub = 50    # 学习子图参数采样条数
    countOfexercise_code_list = 0
    countOfEvidence = 0     #evidence的计数
    counfOfAttribute = 0    #suyin_evidence的计数

    #doAttributeOrnot是一个维数为（题目数，1）的list，里面1表示需要做，0表示不用做
    answerItemNumber = len(doAttributeOrnot)      #可以用这种方式得到题目数量
    suyin_evidence_number = len(suyin_list)

    #用于训练的题目数量
    trainanswerItemNumber = round(answerItemNumber*train_scale)

    if dataname == 'user_id':
        for i in range(trainanswerItemNumber):
            if doAttributeOrnot[i] == 0:
                evidence = evidence_list[countOfEvidence]
                evidenceNode = evidence_node_list[countOfEvidence]
                model = all_causal(item_number, evidence, model, evidenceNode, False, file_path,rate)
                countOfEvidence += 1
            elif doAttributeOrnot[i] == 1:
                start_time = datetime.now()
                timu_code = exercise_code_list[countOfexercise_code_list]
                countOfexercise_code_list += 1
                suyin_evidence = suyin_list[counfOfAttribute]
                suyin_evidenceNode = suyin_node_list[counfOfAttribute]

                model = all_causal(item_number, suyin_evidence, model, suyin_evidenceNode, False, file_path,rate)
                if suyin_evidence[-1] != 1:
                    write_node = ""
                    for node in suyin_evidenceNode:
                        write_node += node
                        write_node += " "
                    f_evidence.write("exerciseCode: " + str(timu_code) + "\n")
                    f_evidence.write("evidence: " + str(suyin_evidence) + "\n")
                    f_evidence.write("evidenceNode: " + write_node + "\n")
                    f_log.write("exerciseCode: " + str(timu_code) + "\n")
                    f_log.write("evidence: " + str(suyin_evidence) + "\n")
                    f_log.write("evidenceNode: " + write_node + "\n")
                    # print(suyin_evidence)
                    # print(suyin_evidenceNode)
                    mat,nodelistOfMat = sub_causal(matrix, suyin_evidence, kn_list, model, suyin_evidenceNode, item_number_sub, True,
                                     file_path,rate)
                    #该题学生的掌握情况
                    currentProblemMaster = []


                    #找溯因图中过滤后的值
                    countOfSum = 0
                    sumOfMat = 0
                    maxWrongProbnode = []
                    for j in range(len(mat)):
                        for k in range(len(mat)):
                            if mat[j][k] != 0:
                                countOfSum += 1
                                sumOfMat += mat[j][k]
                    averageOfMat = sumOfMat/countOfSum
                    for j in range(len(mat)):
                        for k in range(len(mat)):
                            if mat[j][k] >= averageOfMat:
                                maxWrongProbnode.append(nodelistOfMat[j])


                    for eachNode in nodelistOfMat[:-1] :
                        currentProblemMaster.append(student_id_each_knowle_master_prob[kn_list.index(eachNode)])
                    min_master_prob_knowle = nodelistOfMat[currentProblemMaster.index(min(currentProblemMaster))]

                    with open(csvFilePath, 'a+',newline='') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow([id, i, maxWrongProbnode, min_master_prob_knowle, min_master_prob_knowle in maxWrongProbnode])
                        f.close()

                    mat_str = ""
                    mat_str += "[ "
                    for i in range(len(suyin_evidenceNode)):
                        for j in range(len(suyin_evidenceNode)):
                            mat_str = mat_str + str(mat[i][j]) + " "
                        if i != suyin_evidence_number - 1:
                            mat_str += "\n"
                    mat_str += " ]\n"
                    f_suyin.write(mat_str)
                    f_log.write(mat_str)
                end_time = datetime.now()
                print('Duration: {}'.format(end_time - start_time))
                counfOfAttribute += 1


        #测试部分，需要用到不在训练题目里面的数据
        nodeList = model.nodes
        data_construct = []
        for i in range(len(nodeList)):
            data_construct.append(0)

        df = pd.DataFrame([data_construct], columns=nodeList, index=[0])
        data_copy = df.copy()
        data_copy.drop(nodeList, axis=1, inplace=True)

        prob = model.predict_probability(data_copy)
        print(prob)

        #将对应的知识点掌握情况提取出来
        cur_student_knowledge_master = []
        for item in kn_list:
            cur_student_knowledge_master.append(prob[item+"_1"].values[0])

        #学生知识点掌握
        our_studentMaster.append(cur_student_knowledge_master)


        testanswerItemNumber = answerItemNumber - trainanswerItemNumber

        for testProblemNum in range(trainanswerItemNumber, answerItemNumber):
            #当前题目所需知识点list
            thisProblemKnowledge = Q_matrix[testProblemNum]
            # 学生对应当前题目要求知识点掌握情况
            student_master_this_problem_list = []
            for know_index in len(thisProblemKnowledge):
                if thisProblemKnowledge[know_index] == 1:
                    student_master_this_problem_list.append(our_studentMaster[know_index])

            #学生做对该题概率（取所有要求的知识点里面掌握度最小值）
            student_master_this_problem = min(student_master_this_problem_list)
            predictStudentAnswerRight = (1-slip[testProblemNum])*student_master_this_problem + student_master_this_problem*guess[testProblemNum]
            trueStudentAnswerRight = raw_data[id][testProblemNum]

            predict_student_score.append(predictStudentAnswerRight)
            true_student_score.append(trueStudentAnswerRight)


    elif dataname == "school_id":
        # 溯因
        suyin_evidenceNode = suyin_node_list[0]
        suyin_node_number = len(suyin_evidenceNode)
        suyin_evidence_number = len(suyin_list)
        for i in range(suyin_evidence_number):
            print(i)
            suyin_evidence = suyin_list[i]
            model = all_causal(item_number, suyin_evidence, model, suyin_evidenceNode, False, file_path,rate)
        suyin_evidence = [0,0,1,0,0,0]
        sub_causal(matrix, suyin_evidence, kn_list, model, suyin_evidenceNode, item_number_sub, True, file_path,rate)
    f_log.close()
    f_suyin.close()
    f_evidence.close()
    # run you programme / algorithm here


def customizeCausalAttribution(datasetName, rate):
    #存储结果的文件名
    csvFilePath = "../math2015/" +datasetName +  "/" + datasetName + "predictAndGroundtruth.csv"
    headers = ['user_id', 'problem_id', 'perdict_wrong_knowledge', 'true_wrong_knowledge', 'is_right']
    with open(csvFilePath, 'a+',newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f.close()

    #模拟学生做题情况
    filename = "../math2015/" +datasetName +  "/" + datasetName + "QA_Generate.csv"
    df = pd.read_csv(filename)

    #每道题一个slip和guess
    slip = np.loadtxt("slip.txt")
    guess = np.loadtxt("guess.txt")

    #不同数据集中客观题个数
    obj_prob_num = {"FrcSub":20, "Math1": 15, "Math2": 16}
    Q_matrix = np.loadtxt("../math2015/" +datasetName +  "/" + "q.txt",dtype=np.int)
    Q_matrix = Q_matrix[range(obj_prob_num[datasetName]),:]

    #训练集比例
    train_scale = 0.8


    #本数据集学生人数
    student_number = max(df["user_id"].values.tolist())+1

    #our_studentMaster: BEAN算法得到的学生各知识点掌握情况
    our_studentMaster = []
    #预测学生做对概率
    predict_student_score = []
    #学生实际做题情况
    true_student_score = []
    #初始学生做题情况
    raw_data = np.loadtxt("../math2015/"+datasetName+"/data.txt")


    for user_id in range(student_number):
        print(user_id)
        dataname = "user_id"
        isCorrectdata = df.loc[df["user_id"]==user_id]['is_right_answer']
        isCorrectdata = isCorrectdata.values
        student_suyin_timu = []
        for i in range(len(isCorrectdata)):
            if isCorrectdata[i] == 0:
                student_suyin_timu.append(i)

        # user_id = 40735241896591616
        # dataname = "user_id"
        # student_suyin_timu = ['SX04B04020201A007', 'SX04B04020202A001', 'SX04B04020202A015', ' SX04B04020203B003', " SX04B04020203B011",
        #                       'SX04B04020301A009', 'SX04B04020302B017', 'SX04B04020303C005']
        #
        run(dataname, user_id, student_suyin_timu,csvFilePath, slip, guess, Q_matrix, train_scale,datasetName,
            our_studentMaster, predict_student_score, true_student_score, raw_data,rate)

    predict_student_score.to_csv("../math2015/" + datasetName + "/predictScore.csv", encoding='utf-8',index=False)
    true_student_score.to_csv("../math2015/" + datasetName + "/trueScore.csv", encoding='utf-8', index=False)
    our_studentMaster.to_csv("../math2015/" + datasetName + "/studentMaster.csv")






    #
    # school_suyin_timu = ['SX04B04020303C005']
    # dataname = "school_id"
    # school_id = 18217349272506368
    # run(dataname, school_id, school_suyin_timu)


if __name__ == "__main__":
    dataSetList = ["FrcSub", "Math1", "Math2"]
    rate = 9
    # for dataName in dataSetList:
    #     customizeCausalAttribution(dataName)
    customizeCausalAttribution("FrcSub", rate)
