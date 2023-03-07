from graphviz import Digraph
import json
import numpy as np
import os


# 根据阈值得到溯因网络
def getCansOfErrorAttri(data_of, threshold):
    can = {}
    nodenumber = len(data_of.nodelist)
    for i in range(nodenumber):
        can[data_of.nodelist[i]] = []
    # print(data_of.matr)
    for i in range(nodenumber):
        for j in range(nodenumber):
            if data_of.matr[i][j] != 0 and (data_of.matr[i][j]) > threshold - 0.001:
                can[data_of.nodelist[i]].append(data_of.nodelist[j])
            if int(data_of.matr[i][j]) < 0:
                data_of.matr[i][j] = 0
                can[data_of.nodelist[i]].append(data_of.nodelist[j])
    # print(can)
    return can


def Trans_dot(list_, nodeList):
    theListForNodes = []  # 存放节点权重的编码，如【0.4, 0.1, 0.9, 0.6】
    theList = []    # 存放个体的编码
    # 将两段编码填入该去的位置
    for i in range(len(nodeList)):
        theListForNodes.append(list_[i])
    for i in range(len(nodeList), len(list_)):
        theList.append(list_[i])
    nodeNumber = len(nodeList) # 节点个数

    # 将个体编码list转换成matrix
    arr = np.array(theList)
    networkMatrix = arr.reshape(nodeNumber, nodeNumber)

    can = {}
    for i in range(nodeNumber):
        can[nodeList[i]] = []

    # 只考虑上三角，不取对角线
    for i in range(nodeNumber):
        for j in range(nodeNumber):
            if networkMatrix[i][j] == 1:# 大于0.5表示有边，小于表示没边
                # 添加边，i为0为例，此时看69行注释，0对应第三个，表示C, j同样的找法，这一层循环，完成C到其连接的所有节点的连接
                #can[nodeList[sortIndex[i]]].append(nodeList[sortIndex[j]])
                can[nodeList[theListForNodes[i]]].append(nodeList[theListForNodes[j]])
    return can


def Plot_dag_for_Error_Attribution_Project(data_of_BN, datanameOfPDF, threshold, plot, file_path):
    can = getCansOfErrorAttri(data_of_BN, threshold)
    dot = Digraph()
    for k, v in can.items():
        # if k not in dot.body:
        #     dot.node(k,fontname="Microsoft YaHei")
        # if v:
        #     for v_ele in v:
        #         if v_ele not in dot.body:
        #             dot.node(v_ele,fontname="Microsoft YaHei")
        #         dot.edge(k, v_ele, label=str(round(data_of_BN.matr[data_of_BN.nodelist.index(k)][data_of_BN.nodelist.index(v_ele)],3)),fontname="Microsoft YaHei")

        # if k not in dot.body:
        #     dot.node(k,fontname="Microsoft YaHei")
        if v:
            for v_ele in v:
                if v_ele not in dot.body:
                    dot.node(v_ele,fontname="Microsoft YaHei")
                if k not in dot.body:
                    dot.node(k, fontname="Microsoft YaHei")
                dot.edge(k, v_ele, label=str(round(data_of_BN.matr[data_of_BN.nodelist.index(k)][data_of_BN.nodelist.index(v_ele)],3)),fontname="Microsoft YaHei")

    index = 0
    file_path += datanameOfPDF + '/'
    file = file_path + datanameOfPDF + str(index) + '/'  + str(index)
    filename = file + '.gv'
    while os.path.exists(filename):
        index += 1
        file = file_path + datanameOfPDF + str(index) + '/' + str(index)
        filename = file + '.gv'
    if plot:
        dot.render(filename, view=False)
    else:
        dot.render(filename, view=False)

    # save the result
    with open(file + 'result.json', 'w') as fp:
        json.dump(can, fp)
