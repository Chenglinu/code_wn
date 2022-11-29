import pandas as pd
import numpy as np
import math
from sklearn.mixture import GaussianMixture




#计算 分布（X+Y）=s的概率 其中X的分布为modela Y为modelb
def probSum(modela:GaussianMixture,modelb:GaussianMixture,s,avt):
    result=0
    f_x_np=np.array([[avt,x/10] for x in range(800)])
    f_s_x_np = np.array([[avt, s-(x / 10)] for x in range(800)])

    f_x=e**modela.score_samples(f_x_np)
    f_s_x=e**modelb.score_samples(f_s_x_np)
    result+=f_x*f_s_x*0.1

    return np.sum(result)







e=math.e
# n=15

dates = pd.read_csv('dataset.csv', index_col=0)



dates=dates.dropna(axis=0,how='any')


'''
参数在这
参数在这
参数在这
testdates=dates[4000:]分割训练和测试的边界
yz是阈值，决定着有多容易确定其为延误，越大越容易
n是gmm参数，越大gmm越复杂
'''

#前4000训练，后2000+测试
testdates=dates[4000:]
yz=0.12
n=15


d1=dates.loc[:,'sum_pr']+dates.loc[:,'nextstep_pr_qr_watingtime']+dates.loc[:,'sum_qr']+dates.loc[:,'nextstep_qr_po_watingtime']+dates.loc[:,'sum_po']
d2=dates.loc[:,'nextstep_pr_qr_watingtime']+dates.loc[:,'sum_qr']+dates.loc[:,'nextstep_qr_po_watingtime']+dates.loc[:,'sum_po']
d3=dates.loc[:,'sum_qr']+dates.loc[:,'nextstep_qr_po_watingtime']+dates.loc[:,'sum_po']
d4=dates.loc[:,'nextstep_qr_po_watingtime']+dates.loc[:,'sum_po']
d5=dates.loc[:,'sum_po']






ds=[list(dates.loc[:,'sum_pr']),list(dates.loc[:,'nextstep_pr_qr_watingtime']),list(dates.loc[:,'sum_qr']),list(dates.loc[:,'nextstep_qr_po_watingtime']),list(dates.loc[:,'sum_po'])]

dms = [np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(d1))]),
       np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(d2))]),
       np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(d3))]),
       np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(d4))]),
       np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(d5))])]
ms = [GaussianMixture(n_components=n).fit(dm) for dm in dms]




op_names=['sum_pr','nextstep_pr_qr_watingtime','sum_qr','nextstep_qr_po_watingtime','sum_po']

def ceshi(steps_finished,ava_t,t,already_t_now,dates):

    max_t = ava_t - t - 4
    if max_t <= 0:
        return 0

    if steps_finished < 4:
        dx = dates.loc[:, op_names[steps_finished]] - already_t_now
        tdx = np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(dx)) if b > 0])
        if len(tdx)<200:
            tdx = np.array([[a, b] for a, b in zip(list(dates['avaiabletime']), list(dx))])
        model1 = GaussianMixture(n_components=n).fit(tdx)
        sumP = 0
        sumF = 0
        tt = 1
        for x in range(0, 800):
            tt += 1
            truex = x / 10
            sumP += probSum(model1, ms[steps_finished + 1], truex, ava_t) * 0.1
            if int(x) == int(max_t * 10):
                sumF = sumP
    else:
        sumP = 0
        sumF = 0
        tt = 1

        sumP_np=np.array([[ava_t,x/10] for x in range(800)])
        sumF_np=np.array([[ava_t,x/10] for x in range(max_t*10)])

        # for x in range(0, 500):
        #     tt += 1
        #     truex = x / 5
        #     sumP += (e ** ms[4].score([[ava_t, truex]])) * 0.2
        #     if int(x) == int(max_t * 5):
        #         sumF = sumP
        sumP=np.sum(e**ms[4].score_samples(sumP_np))
        sumF=np.sum(e**ms[4].score_samples(sumF_np))
    return sumF/sumP









#单纯GMM
sumt=0

suml=0
testl_t=0
testl_f=0
sumnl=0
tiqian={i:0 for i in range(200)}
tiqiant={i:0 for i in range(200)}
for x in testdates['id_pr']:

    time_d = []
    s = 0

    #time_d 按时间顺序的时间结点排序
    for op in op_names:
        s += list(testdates[testdates['id_pr'] == x][op])[0]
        time_d.append(s)



    ava_t = int(list(testdates[testdates['id_pr'] == x]['avaiabletime'])[0])
    l_t = int(list(testdates[testdates['id_pr'] == x]['left_time_for_dev'])[0])

    if ava_t<=4:
        continue
    else:
        sumt+=1
        print('检测订单标号：', x)
        print('是否延误：', l_t < 4)
        if l_t < 4:
            suml+=1
        else:
            sumnl+=1
        start=ava_t-20
        if start<0:
            start=0
        end=int(time_d[-1])
        if end>=ava_t:
            end=ava_t
        for t in range(start,end+1):
            if (ava_t-4-t)>0:
                print('提前天数:',ava_t-4-t)
            else:
                break
            steps_finished = 0
            while t > time_d[steps_finished] and steps_finished < len(time_d):
                steps_finished += 1

            if steps_finished>0:
                already_t_now = t - time_d[steps_finished - 1]
            else:
                already_t_now=t

            if ceshi(steps_finished,ava_t,t,already_t_now,dates)<=yz:
                tiqian[ava_t - t] += 1
                if l_t < 4:
                    tiqiant[ava_t-t]+=1
                    testl_t+=1
                else:
                    testl_f+=1
                break

    print(sumt,suml)
    print('总体延误率:',suml/sumt)
    print('报答率:',testl_t/suml)
    print('正答率',testl_t/(testl_f+testl_t))


content=','
for i in range(20):
    content+=str(i)+','
content+='\n正确数目'
for i in range(20):
    content+=str(tiqiant[i])+','
content+='\n预测数目'
for i in range(20):
    content += str(tiqian[i]) + ','

content+='\n总预测数目,'+str(sumt)
content+='\n总延误数目,'+str(suml)
content+='\n延误正确总数,'+str(testl_t)
content+='\n判断为延误总数,'+str(testl_f+testl_t)
content+='\n判断为非延误总数,'+str(sumt-testl_f-testl_t)

f=open('result.csv','w')
f.write(content)
f.close()
















