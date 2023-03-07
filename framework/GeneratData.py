from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling


def write_data_to_csv(fileName):
    reader = BIFReader(fileName)
    model = reader.get_model()
    sample_all = BayesianModelSampling(model).forward_sample(size=1000)
    print(type(sample_all))
    sample_all.to_csv('..\\rankData\\' + fileName + 'all' + '.csv')


    sample_typeA = BayesianModelSampling(model).forward_sample(size=1000)   #子问题A1,A2,A5,A7
    print(sample_typeA.columns)
    sample_typeA = sample_typeA.drop(["A3","A4","A6"], axis= 1)
    print(sample_typeA)
    cuo = []
    for i in sample_typeA.loc[:,"A7"]:
        if int(i) == 0:
            cuo.append(0)
        else:
            cuo.append(1)

    sample_typeA["dui/cuo"] = cuo
    print(sample_typeA)
    sample_typeA.to_csv('..\\rankData\\' + fileName + 'typeA' + '.csv')


    sample_typeB = BayesianModelSampling(model).forward_sample(size=1000)   #子问题A1,A3,A5,A7
    print(sample_typeB.columns)
    sample_typeB = sample_typeB.drop(["A2","A4","A6"], axis= 1)
    print(sample_typeB)
    cuo = []
    for i in sample_typeB.loc[:,"A7"]:
        if int(i) == 0:
            cuo.append(0)
        else:
            cuo.append(1)

    sample_typeB["dui/cuo"] = cuo
    print(sample_typeB)
    sample_typeB.to_csv('..\\rankData\\' + fileName + 'typeB' + '.csv')



    sample_typeC = BayesianModelSampling(model).forward_sample(size=1000)   #子问题A1,A4,A6
    print(sample_typeC.columns)
    sample_typeC = sample_typeC.drop(["A2","A3","A5","A7"], axis= 1)
    print(sample_typeC)
    cuo = []
    for i in sample_typeC.loc[:,"A6"]:
        if int(i) == 0:
            cuo.append(0)
        else:
            cuo.append(1)

    sample_typeC["dui/cuo"] = cuo
    print(sample_typeC)
    sample_typeC.to_csv('..\\rankData\\' + fileName + 'typeC' + '.csv')


name_list = []
for i in range(6):
    name_list.append("factor"+str(i)+".bif")

for _ in name_list:
    write_data_to_csv(_)

