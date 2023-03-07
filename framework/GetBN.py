# from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd
import random as rd
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
import numpy as np
import pandas as pd
import copy
import random as rd
from pgmpy.factors.discrete import TabularCPD
from Plot_dag import Plot_dag_for_Error_Attribution_Project
import itertools



#找到子节点的父节点
def giveSonReturnFather(matrix, nodeName, son_index):
    father_list = []
    nodeNumber = len(nodeName)
    for i in range(nodeNumber):
        if matrix[i][son_index] == 1:
            father_list.append(nodeName[i])

    return father_list


# 加参数表
# 输入： 邻接矩阵，节点名称
# 输出： 带参数表的贝叶斯网络
def constructTheBN(matrix, nodeName):
    # 结构
    model = BayesianModel()
    for i in nodeName:
        model.add_node(i)

    for i in range(len(nodeName)):
        for j in range(len(nodeName)):
            if matrix[i][j] == 1:
                model.add_edge(nodeName[i], nodeName[j])
    nodeNumber = len(nodeName)
    # 参数表
    '''
    grades_cpd = TabularCPD('grades', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
                                      [0.1,0.1,0.1,0.1,0.1,0.1],
                                      [0.8,0.8,0.8,0.8,0.8,0.8]],
                        evidence=['diff', 'intel'], evidence_card=[2, 3])


    x = itertools.product([0,1],[0,1],[0,1],[0,1])
    for i in x:
        print(list(i))
    '''
    # 对每一个节点node_i，第一个参数是节点名，后面的根据笛卡尔积构建，系数是为0的数目乘以1/个数，evidence是父节点list，最后一个是父节点个数的.append(2)
    cpd_list = []
    a = [0, 1]
    for i in range(nodeNumber):
        father_list = giveSonReturnFather(matrix, nodeName, i)
        fatherNumber = len(father_list)
        DescartesArray = []
        if fatherNumber == 0:
            cpd_i = TabularCPD(nodeName[i], 2, [[0.5], [0.5]])
            cpd_list.append(cpd_i)
        else:
            if fatherNumber == 1:
                evidence_card = [2]
                cpd_i = TabularCPD(nodeName[i], 2, [[0.1, 0.9], [0.9, 0.1]], evidence=father_list,
                                   evidence_card=evidence_card)
                cpd_list.append(cpd_i)
            elif fatherNumber == 2:
                x = itertools.product(a, a)
            elif fatherNumber == 3:
                x = itertools.product(a, a, a)
            elif fatherNumber == 4:
                x = itertools.product(a, a, a, a)
            elif fatherNumber == 5:
                x = itertools.product(a, a, a, a, a)
            elif fatherNumber == 6:
                x = itertools.product(a, a, a, a, a, a)
            if fatherNumber > 1:
                for _ in x:
                    DescartesArray.append(list(_))
                probabilityForZero = []
                for j in range(pow(2, fatherNumber)):
                    sum = np.sum(DescartesArray[j])
                    probabilityForZero.append((fatherNumber - sum) * (1 / fatherNumber))
                probabilityForOne = []
                for _ in probabilityForZero:
                    probabilityForOne.append(1 - _)
                evidence_card = []
                for __ in range(fatherNumber):
                    evidence_card.append(2)
                cpd_i = TabularCPD(nodeName[i], 2, [probabilityForZero, probabilityForOne], evidence=father_list,
                                   evidence_card=evidence_card)
                cpd_list.append(cpd_i)

    for _ in cpd_list:
        model.add_cpds(_)

    return model


# 根据要求的知识点，从知识点结构网络中采样出N条满足该知识点题目的数据
# model是整个知识点的Bayesian Model
# 输入：总知识点的贝叶斯网络，含参。
def constructItemForSubQuestion(model, submodel, evidence, evidenceNode, item_number,rate):
    #从总图中提取m条
    m = item_number*rate
    nodeList = list(model.nodes())  #所有知识点
    subNodeList = list(submodel.nodes())   #部分知识点+错
    diff = []
    for _ in nodeList:
        if _ not in subNodeList:
            diff.append(_)
    samplesFromAll = BayesianModelSampling(model).forward_sample(size=m)
    samplesFromAll = samplesFromAll.drop(diff, axis=1)

    # 找叶子节点
    subMatr = giveModelReturnMatrix(submodel)
    leaves = []
    subNodeNumber = len(subNodeList)-1   #不看错的那一行一列
    for i in range(subNodeNumber):
        for j in range(subNodeNumber):
            if subMatr[i][j] == 1:
                break
        leaves.append(subNodeList[i])
    list_duicuo = []

    for i in range(samplesFromAll.shape[0]):
        flag = 1
        for _ in leaves:
            if samplesFromAll.loc[i,[_]].values[0] == 0:
                list_duicuo.append(0)
                flag = 0
                break
        if flag == 1:
            list_duicuo.append(1)
    samplesFromAll['wrong'] = list_duicuo

    #将evidence构造完全
    for i in range(item_number):
        evidenceDf = pd.DataFrame([evidence[:-1]], columns=evidenceNode[:-1])
        y_pred = model.predict(evidenceDf,n_jobs = 1)
        y_pred_values = y_pred.values[0]

        new_evidence = []
        countForPredict = 0
        countForEvidence = 0
        for _ in subNodeList[:-1]:
            if _ in evidenceNode:
                new_evidence.append(evidenceDf.loc[0, _])
                countForEvidence += 1
            else:
                new_evidence.append(y_pred_values[countForPredict])
                countForPredict += 1
        new_evidence.append(evidence[-1])
        new_evidenceDf = pd.DataFrame([new_evidence], columns=samplesFromAll.columns.values)

        samplesFromAll = samplesFromAll.append(new_evidenceDf)

    return samplesFromAll


class data_of_BayesianNetwork:
    def __init__(self, matr, nodelist):
        self.matr = matr
        self.nodelist = nodelist


# 提取子图
# 输入： 总图的邻接矩阵，考察知识点，节点名称
# 输出： 不带参数的贝叶斯网络，也就是包含“错”的子图
#
def extractBN(matrix, subNodeList, nodeName):
    nodeSet = set()
    for _ in subNodeList:
        nodeSet.add(_)
    for j in range(len(nodeName)):
        for i in range(len(nodeName)):
            if(matrix[i][j] == 1 and nodeName[j] in nodeSet):
                if (nodeName[i] not in nodeSet):
                    nodeSet.add(nodeName[i])

    new_nodeName = []
    for _ in nodeName:
        if _ in nodeSet:
            new_nodeName.append(_)
    new_nodeName.append('wrong')
    number_subBN = len(new_nodeName)
    subBN = np.zeros((number_subBN,number_subBN))
    for i in range(len(nodeName)):
        for j in range(len(nodeName)):
            if matrix[i][j] == 1:
                if (nodeName[i] in new_nodeName) and (nodeName[j] in new_nodeName):
                    subBN[new_nodeName.index(nodeName[i])][new_nodeName.index(nodeName[j])] = 1

    # 连接错
    for i in range(number_subBN-1):
        flag = 1
        for j in range(number_subBN-1):
            if subBN[i][j] == 1:
                flag = 0
                break
        if flag == 1:
            subBN[i][number_subBN-1] = 1

    extractModel = BayesianNetwork()
    for i in new_nodeName:
        extractModel.add_node(i)

    for i in range(number_subBN):
        for j in range(number_subBN):
            if subBN[i][j] == 1:
                extractModel.add_edge(new_nodeName[i], new_nodeName[j])
    return extractModel


# 参数拟合
def fitModel(model, data):
    model.fit(data)
    return model


# 得到模型中，每个节点为0以及为1的概率
def get_prob(model):
    nodelist = list(model.nodes())
    data = []
    for i in range(len(nodelist)):
        data.append(0)
    data = [data]
    new_data = pd.DataFrame(data, columns=nodelist)
    new_data.drop(nodelist, axis=1, inplace=True)
    prob = model.predict_probability(new_data)
    return prob


# 得到节点取值的概率
# score表示是否得分，返回得分以及不得分的概率
def get_prob_of_node(prob, node_name, score=False):
    if score:
        node_name = node_name + '_1'
    else:
        node_name = node_name + '_0'
    if node_name not in prob.columns:
        return 0.001
    return prob[node_name].values[0]


# 得到带权重的网络
# 功能：通过CPT 计算所有边(如A->B P(A=0)*P(B=0|A=0))权重
# 输入： 带CPT的子图网络
# 输出： 类(表示权重的邻接矩阵, 节点名称）
def getTheErorAttributionGraph(model):
    nodelist = list(model.nodes())                 #节点列表
    # print(nodelist)
    nodeNumber = len(nodelist)                 #节点个数
    matrixAll = np.zeros((nodeNumber, nodeNumber))                 #贝叶斯网络邻接矩阵，先置为全0
    for i in range(nodeNumber):                 #若存在边，则赋值为1
        for j in range(nodeNumber):
            if (nodelist[i], nodelist[j]) in model.edges:
                matrixAll[i][j] = 1
    # print(matrixAll)                 #check一遍
    inference = VariableElimination(model)  #用来推断
    matrixErrorAttribution = np.zeros((nodeNumber, nodeNumber))                 #各父节点对子节点的影响权重矩阵，先置为全0
    #竖着读矩阵
    weight_ = []
    # '''
    # 对于溯因，也就是带错节点的
    # '''
    if "wrong" in nodelist:
        for i in range(nodeNumber):                 #对节点i
            nameList = []                 #节点i的父节点集合，目前为空
            for j in range(nodeNumber):                 #对矩阵的第i列
                if (matrixAll[j][i] != 0):                 #的所有元素，如果不为0，则表示从节点j到节点i有一条边，则节点j为其父节点
                    nameList.append(nodelist[j])   #将其父节点j加入到nodeList中
            if len(nameList) == 0:                  #如果没有父节点，则进入下一个节点的循环
                continue
            # cpd = model.get_cpds(nodelist[i])   #当前节点的CPD，包括其父节点的所有情况，在该情况下该节点取0,1的概率，但此处，哪一个块块是其取0，取1，待确认
            # print(cpd.values)                 #所有值
            # print(cpd)                 #带节点的所有值
            # print(cpd.variables)                 #包含变量
            weightList = []                 #对于当前节点i，所有父节点的边的权重组成的list
            if nodelist[i] != "wrong":
                for node in nameList:                #对其所有父节点
                    # print(type(cpd))
                    # cpdCopy = cpd.copy()                #拷贝其cpd
                    # print(type(cpdCopy))
                    phi_query = inference.query(variables  = [nodelist[i], node], evidence={'wrong':0})
                    weight = phi_query.values[0][0]
                    weight_.append(weight)
                    if weight == 0:
                        weight = -1
                    weightList.append(weight)               #将当前父节点取0，节点i取0的概率，除以其取0，取1，节点i取0的概率之和加入到节点i的所有父节点的权重list中
            else:
                for node in nameList:                #对其所有父节点
                    # print(type(cpd))
                    # cpdCopy = cpd.copy()                #拷贝其cpd
                    # print(type(cpdCopy))
                    phi_query = inference.query(variables  = [nodelist[i], node])
                    weight = phi_query.values[0][0]
                    weight_.append(weight)
                    if weight == 0:
                        weight = -1
                    weightList.append(weight)               #将当前父节点取0，节点i取0的概率，除以其取0，取1，节点i取0的


            # 赋值到matrixErrorAttribution
            for j in range(nodeNumber):
                if (matrixAll[j][i] != 0):
                    # matrixErrorAttribution[j][i] = weightList[0]/total               #对每个父节点，将其权重除以所有权重之和，此处需要检查，准备填入的节点的顺序，和weightList中的节点顺序，是否对应
                    matrixErrorAttribution[j][i] = weightList[0]
                    weightList.pop(0)               #将该值移除


        sum_weight = np.sum(weight_)
        #归一化
        # if sum_weight != 0:
        #     for i in range(len(weight_)):
        #         weight_[i] = weight_[i] / sum_weight
        threshold = np.mean(weight_)

        # for i in range(nodeNumber):
        #     for j in range(nodeNumber):
        #         matrixErrorAttribution[i][j] = matrixErrorAttribution[i][j]/sum_weight
        # print(matrixErrorAttribution)

        data_of = data_of_BayesianNetwork(matrixErrorAttribution, nodelist)
    #
    # '''
    # 对于总图
    # '''
    else:
        for i in range(nodeNumber):  # 对节点i
            nameList = []  # 节点i的父节点集合，目前为空
            for j in range(nodeNumber):  # 对矩阵的第i列
                if (matrixAll[j][i] != 0):  # 的所有元素，如果不为0，则表示从节点j到节点i有一条边，则节点j为其父节点
                    nameList.append(nodelist[j])  # 将其父节点j加入到nodeList中
            if len(nameList) == 0:  # 如果没有父节点，则进入下一个节点的循环
                continue
            # cpd = model.get_cpds(nodelist[i])   #当前节点的CPD，包括其父节点的所有情况，在该情况下该节点取0,1的概率，但此处，哪一个块块是其取0，取1，待确认
            # print(cpd.values)                 #所有值
            # print(cpd)                 #带节点的所有值
            # print(cpd.variables)                 #包含变量
            weightList = []  # 对于当前节点i，所有父节点的边的权重组成的list
            for node in nameList:  # 对其所有父节点
                # print(type(cpd))
                # cpdCopy = cpd.copy()                #拷贝其cpd
                # print(type(cpdCopy))
                phi_query = inference.query(variables=[nodelist[i], node])
                weight = phi_query.values[0][0]
                weight_.append(weight)
                if weight == 0:
                    weight = -1
                weightList.append(weight)  # 将当前父节点取0，节点i取0的概率，除以其取0，取1，节点i取0的概率之和加入到节点i的所有父节点的权重list中
            # 赋值到matrixErrorAttribution
            for j in range(nodeNumber):
                if (matrixAll[j][i] != 0):
                    # matrixErrorAttribution[j][i] = weightList[0]/total               #对每个父节点，将其权重除以所有权重之和，此处需要检查，准备填入的节点的顺序，和weightList中的节点顺序，是否对应
                    matrixErrorAttribution[j][i] = weightList[0]
                    weightList.pop(0)  # 将该值移除

        # sum_weight = np.sum(weight_)
        # 归一化
        # if sum_weight != 0:
        #     for i in range(len(weight_)):
        #         weight_[i] = weight_[i] / sum_weight
        threshold = np.mean(weight_)

        # for i in range(nodeNumber):
        #     for j in range(nodeNumber):
        #         matrixErrorAttribution[i][j] = matrixErrorAttribution[i][j]/sum_weight
        # print(matrixErrorAttribution)

        data_of = data_of_BayesianNetwork(matrixErrorAttribution, nodelist)

    return data_of, threshold


# 根据部分知识点采样出一条完整的数据
# model指总的知识点网络
# 输入：总知识点网络，带答题对错信息和题目包含知识点信息的一条记录，节点（含对错）
# 输出：一条根据总知识点网络采样得到的完整不含对错的item,格式为List
def constructOneItemWithEvidence(model, evidence, evidenceNode):
    nodeList = list(model.nodes)
    data_construct = []
    count = 0
    evidence_delete_cuo = evidence[:-1]
    evidenceNode_delete_cuo = evidenceNode[:-1]
    # 用0将空白补齐，构造一条数据
    for _ in nodeList:
        if _ in evidenceNode_delete_cuo:
            data_construct.append(evidence_delete_cuo[count])
            count += 1
        else:

            data_construct.append(0)

    df = pd.DataFrame([data_construct], columns=nodeList, index=[0])

    nodeList_copy = nodeList.copy()
    data_copy = df.copy()

    knowledge_should_predict = list(set(nodeList) - set(evidenceNode_delete_cuo))
    data_copy.drop(knowledge_should_predict, axis=1, inplace=True)

    prob = model.predict_probability(data_copy)
    predict_length = len(knowledge_should_predict)
    each_knowledge_rate = []            #每个知识点取0的概率
    for i in range(predict_length):
        each_knowledge_rate.append(prob.iat[0, 2*i])

    countForAll = 0
    countForPredict = 0
    for _ in nodeList:
        if _ in evidenceNode_delete_cuo:
            countForAll += 1
            continue
        else:
            if rd.random() < each_knowledge_rate[countForPredict]:
                data_construct[countForAll] = 0
            else:
                data_construct[countForAll] = 1
            countForPredict += 1
            countForAll += 1
    return data_construct


#工具函数
#输入：贝叶斯网络model
#输出：model对应的邻接矩阵，格式为np.array
def giveModelReturnMatrix(model):
    nodes = list(model.nodes)
    nodes_number = len(nodes)
    matrix = np.zeros((nodes_number, nodes_number))
    for i in range(nodes_number):
        for j in range(nodes_number):
            if (nodes[i], nodes[j]) in model.edges:
                matrix[i][j] = 1

    return matrix




# data_of里面的是matr，nodelist
def read_from_csv(filename):
    read_csv2 = pd.read_csv(filename)
    list_name = read_csv2.columns.tolist()[1:]
    samples = read_csv2[list_name]
    return (samples, list_name)


# 对子问题进行溯因
# 1. 传入作答记录， 提取问题知识点
# 2. 提取子图
# 3. 计算权重
def sub_causal(matrix, evidence, nodeName, model, evidenceNode, item_number, plot, file_path,rate):
    # 单条作答记录
    submodel = extractBN(matrix, evidenceNode, nodeName)
    samples = constructItemForSubQuestion(model, submodel, evidence, evidenceNode, item_number,rate)
    submodel.fit(samples)
    data_of, threshold = getTheErorAttributionGraph(submodel)
    if plot:
        Plot_dag_for_Error_Attribution_Project(data_of, "Causal_structure_sub", 0, plot, file_path)
        Plot_dag_for_Error_Attribution_Project(data_of, "Error_attri", threshold, plot, file_path)
    return data_of.matr, data_of.nodelist


# 更新总图CPT
def all_causal(item_number, evidence, model, evidenceNode, plot, file_path,rate):
    samples = []
    for i in range(item_number):
        samples.append(constructOneItemWithEvidence(model, evidence, evidenceNode))
    samples = pd.DataFrame(samples, columns=list(model.nodes))
    model.fit_update(samples, n_prev_samples=rate*item_number)
    # 得到全图的带权重的溯因网络
    data_of, threshold = getTheErorAttributionGraph(model)
    Plot_dag_for_Error_Attribution_Project(data_of, "Causal_structure_all", 0, plot, file_path)
    return model




