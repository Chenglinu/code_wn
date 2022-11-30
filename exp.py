import pickle
from hmmlearn import hmm
import random
import pandas as pd
import numpy as np
from copy import deepcopy
import warnings
import sys
import time
warnings.filterwarnings("ignore")

# 导入训练好的模型
with open("chunwo.pkl", "rb") as file:
    model = pickle.load(file)

# 数据处理
Logic_ID = []
# leftsteps=[5,4,3,2,1,0]
timelist = ['approved_at_po', 'created_at_po', 'approved_at_qr', 'created_at_qr', 'approved_at_pr', 'created_at_pr']

op_names = ['sum_pr', 'nextstep_pr_qr_watingtime', 'sum_qr', 'nextstep_qr_po_watingtime', 'sum_po']
data = pd.read_csv('0968_crane_pipe.csv', index_col=0)

data = data.dropna(axis=0, how='any')
data['true_time'] = (pd.to_datetime(data.loc[:, 'approved_at_po']) - pd.to_datetime(
    data.loc[:, 'created_at_pr'])) / np.timedelta64(1, 'D')
data['left_steps'] = 5 / 5
data['father_index'] = 0
data['true_avaiabletime'] = data.loc[:, 'avaiabletime']

father_index = None
for i in data.index:
    if data.loc[i, 'item'] == 2:
        father_index = i
    else:
        data.loc[i, 'father_index'] = father_index
        father_finishTime = pd.to_datetime(data.loc[father_index, 'approved_at_po'])
        for j in range(len(timelist)):
            if father_finishTime > pd.to_datetime(data.loc[i, timelist[j]]):
                data.loc[i, 'left_steps'] = j / 5
                break
        true_time = (pd.to_datetime(data.loc[i, 'approved_at_po']) - father_finishTime) / np.timedelta64(1, 'D')
        if true_time < 0:
            true_time = 0.01
        true_avaliable_time = (pd.to_datetime(
            data.loc[i, 'required_delivery_date']) - father_finishTime) / np.timedelta64(1, 'D')
        if true_avaliable_time < 0:
            true_avaliable_time = 0.01
        # request_delivery_date
        data.loc[i, 'true_time'] = true_time
        data.loc[i, 'true_avaiabletime'] = true_avaliable_time

# for i in data.index:
#     print(data.loc[i,'item'],data.loc[i,'left_steps'],data.loc[i,'true_time'],data.loc[i,'true_avaiabletime'])
traindata = data[:100].copy()
testdata = data[103:].copy()
traintuple = [[a, b, c, d] for a, b, c, d in
              zip(list(traindata['item']), list(traindata['left_steps']), list(traindata['true_time']),
                  list(traindata['true_avaiabletime']))]
testss = [[a, b, c, d, e] for a, b, c, d, e in
          zip(list(testdata['item']), list(testdata['left_steps']), list(testdata['true_time']),
              list(testdata['true_avaiabletime']), list(testdata.index))]

# [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156]
indexlist = list(testdata.index)


biaoqian = [[False, 0,0,True] if (testss[i][2]-testss[i][3])>-1.5 or testss[i][3]==0.01 else [False, 0,0,False] for i in range(len(indexlist))]

testsque = []
listtt = [pd.to_datetime(testdata.loc[indexlist[i], 'approved_at_po']) for i in range(len(indexlist))]
# while starttime<= max(listtt):
#     for i in range(len(indexlist)):
#         if starttime>pd.to_datetime(testdata.loc[indexlist[i],'created_at_pr']):
#             if [i,deepcopy(testss[i])] not in testsque:
#                 testsque.append([i,deepcopy(testss[i])])
#     testsque.sort(key=lambda x: x[0])
#
#     for i in range(len(testsque)):
#         if pd.to_datetime(testdata.loc[indexlist[testsque[i][0]],'approved_at_po'])>=starttime:
#             continue
#         #取前十个历史数据作为hmm的输入
#         if i>10:
#             templist=deepcopy(testsque[i-9:i+1])
#         else:
#             templist=deepcopy(testsque[:i+1])
#         for j in range(len(timelist)):
#             if templist[-1][1][0]==1:
#                 if starttime>pd.to_datetime(testdata.loc[indexlist[templist[-1][0]],timelist[j]]):
#                     templist[-1][1][1] = j/5
#                     break
#         if templist[-1][1][0]==1:
#             if starttime<=

# starttime += np.timedelta64(1, 'D')
# print(testsque)


# item left_steps true_time true_avaiabletime
testseq = deepcopy(testss)
for i in range(len(testseq)):

    print("\r", end="")
    print("预测进度: {}%: ".format(round((i/(len(testseq)-1))*100,2)), "▋" * (i // 2), end="")
    sys.stdout.flush()


    starttime = pd.to_datetime(testdata.loc[indexlist[i], 'created_at_pr'])
    previous = []
    while starttime <= (pd.to_datetime(testdata.loc[indexlist[i], 'approved_at_po']) + np.timedelta64(2, 'D')):
        previous = []
        for j in range(len(indexlist)):
            if starttime >= pd.to_datetime(testdata.loc[indexlist[j], 'approved_at_po']):
                previous.append(deepcopy(testseq[j]))

        previous = [x for x in previous if x[4] <= indexlist[i]]
        if testseq[i][0] == 1 and deepcopy(testseq[testdata.loc[indexlist[i], 'father_index'] - 105]) not in previous:
            true_time_now = (starttime - pd.to_datetime(
                testdata.loc[testdata.loc[indexlist[i], 'father_index'], 'created_at_pr'])) / np.timedelta64(1, 'D')

            previous.append(deepcopy(testseq[testdata.loc[indexlist[i], 'father_index'] - 105]))
            previous[-1][2] = true_time_now

        previous.sort(key=lambda x: x[4])

        testprevious = [x[:4] for x in previous]
        testprevious.append(deepcopy(testseq[i][:4]))
        if testseq[i][0] == 1:
            nowavaliable_time = (pd.to_datetime(
                testdata.loc[indexlist[i], 'required_delivery_date']) - starttime) / np.timedelta64(1, 'D')
            if nowavaliable_time >= testseq[i][3]:
                testprevious[-1][3] = nowavaliable_time
            for j in range(len(timelist)):

                if starttime >= pd.to_datetime(testdata.loc[indexlist[i], timelist[j]]):
                    testprevious[-1][1] = j / 5
                    break
        if len(testprevious)>10:
            testprevious=testprevious[-10:]
        if testseq[i][0] == 2:

            tt = []
            for k in range(0, int(testprevious[-1][3]) + 2):
                if k == 0:
                    testprevious[-1][2] = 0.01
                else:
                    testprevious[-1][2] = k
                try:
                    tt.append([k, testprevious[-1][3], model.score(testprevious)])
                except:
                    continue

            tt.sort(key=lambda x: x[2], reverse=True)
            if tt[0][1] == 0.01:
                biaoqian[i] = [True, (pd.to_datetime(
                    testdata.loc[indexlist[i], 'required_delivery_date']) - starttime) / np.timedelta64(1, 'D'), (
                                       starttime - pd.to_datetime(
                                   testdata.loc[indexlist[i], 'created_at_pr'])) / np.timedelta64(1, 'D'),biaoqian[i][3]]
            elif tt[0][1] < tt[0][0]:
                biaoqian[i] = [True, (pd.to_datetime(
                    testdata.loc[indexlist[i], 'required_delivery_date']) - starttime) / np.timedelta64(1, 'D'), (
                                       starttime - pd.to_datetime(
                                   testdata.loc[indexlist[i], 'created_at_pr'])) / np.timedelta64(1, 'D'),biaoqian[i][3]]
            # print('here',tt)
            # print(testseq[i])
            break
        else:
            tt = []
            for k in range(0, int(testprevious[-1][3]) + 2):
                if k == 0:
                    testprevious[-1][2] = 0.01
                else:
                    testprevious[-1][2] = k
                try:
                    tt.append([k, testprevious[-1][3], model.score(testprevious)])
                except:
                    continue

            tt.sort(key=lambda x: x[2], reverse=True)
            # print(tt)
            # print(testseq[i])

            if tt[0][1] == 0.01:
                biaoqian[i] = [True, (pd.to_datetime(
                    testdata.loc[indexlist[i], 'required_delivery_date']) - starttime) / np.timedelta64(1, 'D'), (
                                           starttime - pd.to_datetime(
                                       testdata.loc[indexlist[i], 'created_at_pr'])) / np.timedelta64(1, 'D'),biaoqian[i][3]]
                break
            elif tt[0][1] < tt[0][0]:
                biaoqian[i] = [True, (pd.to_datetime(
                    testdata.loc[indexlist[i], 'required_delivery_date']) - starttime) / np.timedelta64(1, 'D'), (
                                       starttime - pd.to_datetime(
                                   testdata.loc[indexlist[i], 'created_at_pr'])) / np.timedelta64(1, 'D'),biaoqian[i][3]]
                break

            # print('here', tt)
            # print(testseq[i])

        starttime += np.timedelta64(1, 'D')

print("写入结果到文件")
content='id,预测是否延误,提前天数,第几天发出警报,实际是否延误'
for i in range(len(biaoqian)):
    content+='\n'+str(testdata.loc[indexlist[i],'id_po'])+','+str(biaoqian[i][0])+','+str(round(biaoqian[i][1],1))+','+str(biaoqian[i][2])+','+str(biaoqian[i][3])

f=open('resultnew.csv','w')
f.write(content)
f.close()
print('文件保存结束')
