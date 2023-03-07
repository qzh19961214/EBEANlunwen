import numpy as np
import pandas as pd
import copy
from sklearn import metrics
from sklearn.metrics import auc


def readCsv(fileName):
    df = pd.read_csv(fileName, header=None)
    return df

def combineCsv(datasetName, length):
    preList = ["predictScore", "studentMaster", "trueScore"]

    for pre in preList:
        fileName = "C:\\Users\\ybz\\Desktop\\ResultOfBean1\\" + datasetName + "\\" + pre + ".csv"
        res = readCsv(fileName)
        for i in range(1,length):
            fileName1 = "C:\\Users\\ybz\\Desktop\\ResultOfBean1\\" + datasetName + "\\" + pre + " (" + str(i) +")"+ ".csv"
            res1  = readCsv(fileName1)
            res = res.append(res1)

        storeFileName = "C:\\Users\\ybz\\Desktop\\ResultOfBean1\\" + datasetName + "\\" + pre + "Whole.csv"
        res.to_csv(storeFileName, index=False, header=False)





if __name__ == "__main__":
    lengthList = [2, 9, 8]
    datasetList = ["FrcSub", "Math1", "Math2"]
    for index in range(0,3):
        combineCsv(datasetList[index], lengthList[index])
