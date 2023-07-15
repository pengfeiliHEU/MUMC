import os
import argparse
from utils import Logger
from vqaTools.vqa import *
from vqaTools.vqaEval import *

def compute_vqa_acc(answer_list_path, epoch=40, res_file_path=30):
    quesFile = answer_list_path
    all_result_list = []
    vqa = VQA(quesFile, quesFile)
    for i in range(epoch):
        resFile = res_file_path.replace('<epoch>', str(i))
        print(resFile)

        # create vqa object and vqaRes object
        vqaRes = vqa.loadRes(resFile, quesFile)

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
        # evaluate results
        vqaEval.evaluate()

        # print accuracies
        acc_dict = {}
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        acc_dict['Epoch'] = i + 1
        acc_dict['Overall'] = vqaEval.accuracy['overall']
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            acc_dict[ansType] = vqaEval.accuracy['perAnswerType'][ansType]
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

        # save evaluation results to ./results folder
        accuracyFile = resFile.replace('.json', '_acc.json')
        json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
        compareFile = resFile.replace('.json', '_compare.json')
        json.dump(vqaEval.ansComp, open(compareFile, 'w'))
        all_result_list.append(acc_dict)
    index = res_file_path.rfind('/')
    compareFile = res_file_path[0:index]
    compareFile = os.path.join(compareFile, 'all_acc.json')
    all_result_list.sort(key=lambda x: x['Overall'])
    for result in all_result_list:
        print(result)
    json.dump(all_result_list, open(compareFile, 'w'))
    print('All accurary file saved to: ', compareFile)