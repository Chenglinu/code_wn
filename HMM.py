from hmmlearn import hmm
import random
import pandas as pd
import numpy as np
# model=hmm.GMMHMM(n_components=5)
# model.n_features=2
# model.fit([[random.random(),(i%5)+random.random()] for i in range(200)],lengths=[10 for i in range(20)])
#
#
#
# print(sum([model.score([[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()]]) for i in range(10)]))
# print(sum([model.score([[random.random(),random.random()],[random.random(),1+random.random()],[random.random(),2+random.random()],[random.random(),3+random.random()],[random.random(),4+random.random()]]) for i in range(10)]))
# print(sum([model.score([[random.random(),random.random()],[random.random(),1+random.random()],[random.random(),2+random.random()],[random.random(),3+random.random()],[random.random(),1+random.random()]]) for i in range(10)]))

Logic_ID=[]
#leftsteps=[5,4,3,2,1,0]
timelist=['approved_at_po', 'created_at_po', 'approved_at_qr', 'created_at_qr', 'approved_at_pr','created_at_pr']

op_names=['sum_pr','nextstep_pr_qr_watingtime','sum_qr','nextstep_qr_po_watingtime','sum_po']
data = pd.read_csv('0968_crane_pipe.csv', index_col=0)

data=data.dropna(axis=0,how='any')
data['true_time']=(pd.to_datetime(data.loc[:,'approved_at_po'])-pd.to_datetime(data.loc[:,'created_at_pr']))/np.timedelta64(1, 'D')
data['left_steps']=5/5
data['father_index']=0
data['true_avaiabletime']=data.loc[:,'avaiabletime']




father_index=None
for i in data.index:
    if data.loc[i,'item']==2:
        father_index=i
    else:
        data.loc[i,'father_index']=father_index
        father_finishTime=pd.to_datetime(data.loc[father_index,'approved_at_po'])
        for j in range(len(timelist)):
            if father_finishTime>pd.to_datetime(data.loc[i,timelist[j]]):
                data.loc[i, 'left_steps'] = j/5
                break
        true_time=(pd.to_datetime(data.loc[i,'approved_at_po'])-father_finishTime)/np.timedelta64(1, 'D')
        if true_time<0:
            true_time=0.01
        true_avaliable_time=(pd.to_datetime(data.loc[i,'required_delivery_date'])-father_finishTime)/np.timedelta64(1, 'D')
        if true_avaliable_time<0:
            true_avaliable_time=0.01
        #request_delivery_date
        data.loc[i,'true_time']=true_time
        data.loc[i, 'true_avaiabletime']=true_avaliable_time

# for i in data.index:
#     print(data.loc[i,'item'],data.loc[i,'left_steps'],data.loc[i,'true_time'],data.loc[i,'true_avaiabletime'])


traindata = data[:100].copy()
testdata = data[103:].copy()



traintuple=[[a, b,c,d] for a, b,c,d in zip(list(traindata['item']), list(traindata['left_steps']), list(traindata['true_time']), list(traindata['true_avaiabletime']))]
testss=[[a, b,c,d] for a, b,c,d in zip(list(testdata['item']), list(testdata['left_steps']), list(testdata['true_time']), list(testdata['true_avaiabletime']))]
print(traintuple)
model=hmm.GMMHMM(n_components=5,n_mix=3,covariance_type="tied")
model.fit(traintuple,lengths=[10 for i in range(10)])
print('-----------------')
print(model.transmat_)
print(model.score(testss))
print(testss)
testss[1][2]=1
print(model.score(testss))
