from pgmpy.readwrite import BIFReader, BIFWriter
from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
import pandas as pd
from pgmpy.readwrite import BIFReader

the_data = [[1,1,0,0,0],
            [1,1,1,0,0],
            [1,0,0,0,0]]

knowledgePoints = ['A1','A2','A3','A4','A5','A6','A7']

#啥都不会
def factor0():
    factor_b = BayesianNetwork([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
                             (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    # factor_b.add_edges_from([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
    #                          (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    cpd_A1 = TabularCPD(knowledgePoints[0], 2, [[0.999], [0.001]])
    cpd_A2 = TabularCPD(knowledgePoints[1],variable_card=2, values=[[0.999, 0.95], [0.001, 0.05]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A3 = TabularCPD(knowledgePoints[2],variable_card=2, values=[[0.999, 0.95], [0.001, 0.05]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A4 = TabularCPD(knowledgePoints[3], variable_card=2, values=[[0.999, 0.95], [0.001, 0.05]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A5 = TabularCPD(knowledgePoints[4], variable_card=2, values=[[0.999, 0.99,0.99, 0.98], [0.001, 0.01, 0.01, 0.02]],evidence=[knowledgePoints[1],knowledgePoints[2]], evidence_card=[2,2])
    cpd_A6 = TabularCPD(knowledgePoints[5], variable_card=2,values=[[0.999, 0.95], [0.001, 0.05]],evidence=[knowledgePoints[3]], evidence_card=[2])
    cpd_A7 = TabularCPD(knowledgePoints[6], variable_card=2,values=[[0.999, 0.95], [0.001, 0.05]],evidence=[knowledgePoints[4]], evidence_card=[2])
    factor_b.add_cpds(cpd_A1, cpd_A2, cpd_A3, cpd_A4,cpd_A5,cpd_A6,cpd_A7)
    writer = BIFWriter(factor_b)
    writer.write_bif(filename='factors/factor0.bif')

    return factor_b

#会20%
def factor1():
    factor_b = BayesianNetwork()
    factor_b.add_edges_from([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
                             (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    cpd_A1 = TabularCPD(knowledgePoints[0], 2, [[0.8], [0.2]])
    cpd_A2 = TabularCPD(knowledgePoints[1],variable_card=2, values=[[0.8, 0.76], [0.2, 0.24]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A3 = TabularCPD(knowledgePoints[2],variable_card=2, values=[[0.8, 0.79], [0.2, 0.21]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A4 = TabularCPD(knowledgePoints[3], variable_card=2, values=[[0.8, 0.76], [0.2, 0.24]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A5 = TabularCPD(knowledgePoints[4], variable_card=2, values=[[0.75, 0.7,0.7, 0.6], [0.25, 0.3, 0.3, 0.4]],evidence=[knowledgePoints[1],knowledgePoints[2]], evidence_card=[2,2])
    cpd_A6 = TabularCPD(knowledgePoints[5], variable_card=2,values=[[0.8, 0.79], [0.2, 0.21]],evidence=[knowledgePoints[3]], evidence_card=[2])
    cpd_A7 = TabularCPD(knowledgePoints[6], variable_card=2,values=[[0.8, 0.79], [0.2, 0.21]],evidence=[knowledgePoints[4]], evidence_card=[2])
    factor_b.add_cpds(cpd_A1, cpd_A2, cpd_A3, cpd_A4,cpd_A5,cpd_A6,cpd_A7)
    writer = BIFWriter(factor_b)
    writer.write_bif(filename='factors/factor1.bif')
    return factor_b

#会40%
def factor2():
    factor_b = BayesianNetwork()
    factor_b.add_edges_from([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
                             (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    cpd_A1 = TabularCPD(knowledgePoints[0], 2, [[0.6], [0.4]])
    cpd_A2 = TabularCPD(knowledgePoints[1],variable_card=2, values=[[0.6, 0.55], [0.4, 0.45]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A3 = TabularCPD(knowledgePoints[2],variable_card=2, values=[[0.6, 0.55], [0.4, 0.45]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A4 = TabularCPD(knowledgePoints[3], variable_card=2, values=[[0.6, 0.55], [0.4, 0.45]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A5 = TabularCPD(knowledgePoints[4], variable_card=2, values=[[0.6, 0.5,0.5, 0.4], [0.4, 0.5, 0.5, 0.6]],evidence=[knowledgePoints[1],knowledgePoints[2]], evidence_card=[2,2])
    cpd_A6 = TabularCPD(knowledgePoints[5], variable_card=2,values=[[0.6, 0.55], [0.4, 0.45]],evidence=[knowledgePoints[3]], evidence_card=[2])
    cpd_A7 = TabularCPD(knowledgePoints[6], variable_card=2,values=[[0.6, 0.55], [0.4, 0.45]],evidence=[knowledgePoints[4]], evidence_card=[2])
    factor_b.add_cpds(cpd_A1, cpd_A2, cpd_A3, cpd_A4,cpd_A5,cpd_A6,cpd_A7)
    writer = BIFWriter(factor_b)
    writer.write_bif(filename='factors/factor2.bif')
    return factor_b

#会60%
def factor3():
    factor_b = BayesianNetwork()
    factor_b.add_edges_from([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
                             (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    cpd_A1 = TabularCPD(knowledgePoints[0], 2, [[0.4], [0.6]])
    cpd_A2 = TabularCPD(knowledgePoints[1],variable_card=2, values=[[0.4, 0.35], [0.6, 0.65]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A3 = TabularCPD(knowledgePoints[2],variable_card=2, values=[[0.4, 0.35], [0.6, 0.65]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A4 = TabularCPD(knowledgePoints[3], variable_card=2, values=[[0.4, 0.35], [0.6, 0.65]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A5 = TabularCPD(knowledgePoints[4], variable_card=2, values=[[0.4, 0.35,0.35, 0.3], [0.6, 0.65, 0.65, 0.7]],evidence=[knowledgePoints[1],knowledgePoints[2]], evidence_card=[2,2])
    cpd_A6 = TabularCPD(knowledgePoints[5], variable_card=2,values=[[0.4, 0.35], [0.6, 0.65]],evidence=[knowledgePoints[3]], evidence_card=[2])
    cpd_A7 = TabularCPD(knowledgePoints[6], variable_card=2,values=[[0.4, 0.35], [0.6, 0.65]],evidence=[knowledgePoints[4]], evidence_card=[2])
    factor_b.add_cpds(cpd_A1, cpd_A2, cpd_A3, cpd_A4,cpd_A5,cpd_A6,cpd_A7)
    writer = BIFWriter(factor_b)
    writer.write_bif(filename='factors/factor3.bif')
    return factor_b

#会80%
def factor4():
    factor_b = BayesianNetwork()
    factor_b.add_edges_from([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
                             (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    cpd_A1 = TabularCPD(knowledgePoints[0], 2, [[0.2], [0.8]])
    cpd_A2 = TabularCPD(knowledgePoints[1],variable_card=2, values=[[0.2, 0.15], [0.8, 0.85]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A3 = TabularCPD(knowledgePoints[2],variable_card=2, values=[[0.2, 0.15], [0.8, 0.85]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A4 = TabularCPD(knowledgePoints[3], variable_card=2, values=[[0.2, 0.15], [0.8, 0.85]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A5 = TabularCPD(knowledgePoints[4], variable_card=2, values=[[0.2, 0.15,0.15, 0.1], [0.8, 0.85, 0.85, 0.9]],evidence=[knowledgePoints[1],knowledgePoints[2]], evidence_card=[2,2])
    cpd_A6 = TabularCPD(knowledgePoints[5], variable_card=2,values=[[0.2, 0.15], [0.8, 0.85]],evidence=[knowledgePoints[3]], evidence_card=[2])
    cpd_A7 = TabularCPD(knowledgePoints[6], variable_card=2,values=[[0.2, 0.15], [0.8, 0.85]],evidence=[knowledgePoints[4]], evidence_card=[2])
    factor_b.add_cpds(cpd_A1, cpd_A2, cpd_A3, cpd_A4,cpd_A5,cpd_A6,cpd_A7)
    writer = BIFWriter(factor_b)
    writer.write_bif(filename='factors/factor4.bif')
    return factor_b

#都会
def factor5():
    factor_b = BayesianNetwork()
    factor_b.add_edges_from([(knowledgePoints[0], knowledgePoints[1]), (knowledgePoints[0], knowledgePoints[2]),(knowledgePoints[0],knowledgePoints[3]), (knowledgePoints[1], knowledgePoints[4]), (knowledgePoints[2], knowledgePoints[4]),
                             (knowledgePoints[3], knowledgePoints[5]),(knowledgePoints[4], knowledgePoints[6])])
    cpd_A1 = TabularCPD(knowledgePoints[0], 2, [[0.001], [0.999]])
    cpd_A2 = TabularCPD(knowledgePoints[1],variable_card=2, values=[[0.002, 0.001], [0.998, 0.999]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A3 = TabularCPD(knowledgePoints[2],variable_card=2, values=[[0.002, 0.001], [0.998, 0.999]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A4 = TabularCPD(knowledgePoints[3], variable_card=2, values=[[0.002, 0.001], [0.998, 0.999]],evidence=[knowledgePoints[0]], evidence_card=[2])
    cpd_A5 = TabularCPD(knowledgePoints[4], variable_card=2, values=[[0.003, 0.002,0.002, 0.001], [0.997, 0.998, 0.998, 0.999]],evidence=[knowledgePoints[1],knowledgePoints[2]], evidence_card=[2,2])
    cpd_A6 = TabularCPD(knowledgePoints[5], variable_card=2,values=[[0.002, 0.001], [0.998, 0.999]],evidence=[knowledgePoints[3]], evidence_card=[2])
    cpd_A7 = TabularCPD(knowledgePoints[6], variable_card=2,values=[[0.002, 0.001], [0.998, 0.999]],evidence=[knowledgePoints[4]], evidence_card=[2])
    factor_b.add_cpds(cpd_A1, cpd_A2, cpd_A3, cpd_A4,cpd_A5,cpd_A6,cpd_A7)
    writer = BIFWriter(factor_b)
    writer.write_bif(filename='factors/factor5.bif')
    return factor_b

def fitModel(model):
    name_list = model.nodes
    data = pd.DataFrame(data = [the_data[0],the_data[1]], columns=name_list)
    print(data)
    model.fit(data)
    for i in model.get_cpds():
        print(i)


factor0()
factor1()
factor2()
factor3()
factor4()
factor5()
