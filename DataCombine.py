import copy
import random

import pylab
from pandas import read_csv
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
from fitter import Fitter
from pylab import linspace, plot
import scipy.stats



purchase_requests=read_csv('purchase_requests.csv')
purchase_requests_index=list(purchase_requests)
quotation_request_managements=read_csv('quotation_request_managements.csv')
quotation_request_managements_index=list(quotation_request_managements)
purchase_orders=read_csv('purchase_orders.csv')
purchase_orders_index=list(purchase_orders)
delivery_orders=read_csv('delivery_orders.csv')
delivery_orders_index=list(delivery_orders)
pr_po_map=read_csv('pr_po_mappings.csv')
pr_po_map_index=list(pr_po_map)



m1=pd.merge(purchase_requests,quotation_request_managements,left_on='id',right_on='purchase_request_id',suffixes=('_pr','_qr'))

m2=pd.merge(purchase_orders,delivery_orders,on='link_id',suffixes=('_po','_do'))

m3=pd.merge(m2,pr_po_map,left_on='id_po',right_on='purchase_order_id')

m4=pd.merge(m1,m3,left_on='id_pr',right_on='purchase_request_id')

m4=m4.drop_duplicates(subset=['id_do'],keep='first')


f=open('attributes.txt','r')
attr=f.read().split(',')
m4=m4[attr]


#预期时间减去创建订单时间 能利用的所有时间长度+pd.Timedelta(days=1)
avaiabletime=pd.to_datetime(m4.loc[:,'request_delivery_date'])-pd.to_datetime(m4.loc[:,'created_at_pr'])
#留给运送的时间
left_time=pd.to_datetime(m4.loc[:,'request_delivery_date'])-pd.to_datetime(m4.loc[:,'approved_at_po'])
#建立购买请求市场，1:从创建到请求同意时间，2：从请求同意到同意时间，sum：总时间
pr_intern1_create=pd.to_datetime(m4.loc[:,'submitter_for_approval_date_pr'])-pd.to_datetime(m4.loc[:,'created_at_pr'])
pr_intern2_aprove=pd.to_datetime(m4.loc[:,'approved_at_pr'])-pd.to_datetime(m4.loc[:,'submitter_for_approval_date_pr'])
sum_pr=pr_intern1_create+pr_intern2_aprove

#pr结束后等到qr创建所需时间
nextstep_pr_qr_watingtime=pd.to_datetime(m4.loc[:,'created_at_qr'])-pd.to_datetime(m4.loc[:,'approved_at_pr'])

#同
qr_intern1_create=pd.to_datetime(m4.loc[:,'submitter_for_approval_date_qr'])-pd.to_datetime(m4.loc[:,'created_at_qr'])
qr_intern2_aprove=pd.to_datetime(m4.loc[:,'approved_at_qr'])-pd.to_datetime(m4.loc[:,'submitter_for_approval_date_qr'])
sum_qr=qr_intern1_create+qr_intern2_aprove

nextstep_qr_po_watingtime=pd.to_datetime(m4.loc[:,'created_at_po'])-pd.to_datetime(m4.loc[:,'approved_at_qr'])

po_intern1_create=pd.to_datetime(m4.loc[:,'submitter_for_approval_date_po'])-pd.to_datetime(m4.loc[:,'created_at_po'])
po_intern2_aprove=pd.to_datetime(m4.loc[:,'approved_at_po'])-pd.to_datetime(m4.loc[:,'submitter_for_approval_date_po'])
sum_po=po_intern1_create+po_intern2_aprove


m4['pr_intern1_create']=pr_intern1_create/np.timedelta64(1, 'D')
m4['pr_intern2_aprove']=pr_intern2_aprove/np.timedelta64(1, 'D')
m4['qr_intern1_create']=qr_intern1_create/np.timedelta64(1, 'D')
m4['qr_intern2_aprove']=qr_intern2_aprove/np.timedelta64(1, 'D')
m4['po_intern1_create']=po_intern1_create/np.timedelta64(1, 'D')
m4['po_intern2_aprove']=po_intern2_aprove/np.timedelta64(1, 'D')
m4['left_time_for_dev']=left_time/np.timedelta64(1, 'D')
m4['avaiabletime']=avaiabletime/np.timedelta64(1, 'D')

m4['nextstep_pr_qr_watingtime']=nextstep_pr_qr_watingtime/np.timedelta64(1, 'D')
m4['nextstep_qr_po_watingtime']=nextstep_qr_po_watingtime/np.timedelta64(1, 'D')


m4['sum_pr']=sum_pr/np.timedelta64(1, 'D')
m4['sum_po']=sum_po/np.timedelta64(1, 'D')
m4['sum_qr']=sum_qr/np.timedelta64(1, 'D')



attributes=['project_id_pr','internal_contact_id_pr']



pr_qr_watingtime_attribute=['project_id_pr','subsidiary_id_pr','delivery_address_id_pr','internal_contact_id_pr']
qr_po_watingtime_attribute=['project_id_pr','subsidiary_id_pr','delivery_address_id_pr','internal_contact_id_pr']


pr_time=['pr_intern1_create','pr_intern2_aprove','avaiabletime']
qr_time=['qr_intern1_create','qr_intern2_aprove','avaiabletime']
po_time=['po_intern1_create','po_intern2_aprove','avaiabletime']

pr_states_dic={}
for x in m4.index:

    pr_state=list(m4.loc[x,attributes])
    pr_state=[int(x) for x in pr_state]

    if str(pr_state) in pr_states_dic:
        pr_states_dic[str(pr_state)].append(list(m4.loc[x,pr_time]))
    else:
        pr_states_dic[str(pr_state)]=[list(m4.loc[x,pr_time])]

print(len(m4))

m4=m4.drop_duplicates(subset=['sum_pr', 'sum_po', 'sum_qr'])
dates=m4
unnormal_data_index=list(dates[dates['sum_pr'] < 0].index) + list(dates[dates['sum_po'] < 0].index) + list(
    dates[dates['sum_qr'] < 0].index) + list(dates[dates['nextstep_pr_qr_watingtime'] < 0].index) + list(
    dates[dates['nextstep_qr_po_watingtime'] < 0].index)
unnormal_data_index=set(unnormal_data_index)
dates=dates.drop(index=unnormal_data_index)

print(len(dates))
dates.to_csv('dataset.csv')








