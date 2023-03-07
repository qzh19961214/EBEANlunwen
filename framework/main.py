from causal.framework.GetBN import constructTheBN, sub_causal, all_causal, read_from_csv


if __name__ == '__main__':
    matrix = [[0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]
    kn = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    file_name = "../rankData/factor2.bifall.csv"
    samples, list_name = read_from_csv(file_name)
    model = constructTheBN(matrix, kn)
    threshold = 0.6
    evidence = [1, 0, 0, 0]
    evidenceNode = ['A2', 'A3', 'A5', 'dui/cuo']
    item_number = 500
    # sub_causal(matrix, evidence, list_name, model, evidenceNode, item_number, threshold, True)
    all_causal(item_number, evidence, model, evidenceNode, True)
