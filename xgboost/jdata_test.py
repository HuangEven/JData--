import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn.model_selection import train_test_split
import random

import xgboost as xgb
# 模型调参工具
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
# 模型保存工具
from sklearn.externals import joblib
#Error metrics
from sklearn.metrics import accuracy_score,recall_score,mean_squared_error, r2_score

import get_feats as gf

# 将所有特征合并，并将预测目标分离
def make_train_set(end_date, days=30):
    user = gf.get_basic_uesr_feat()
    product = gf.get_basic_product_feat()
    comment = gf.get_basic_comment_feat(end_date,days)
    others_action = gf.get_others_action_feat(end_date,days)
    user_order = gf.get_user_order_ratio(end_date,days)
    product_conv = gf.get_product_conversion(end_date,days)
    shop = gf.get_basic_shop_feat()

    # 时间窗口设置
    days_window = (1,3,7,14,30)
    actions = gf.get_action_feat(end_date, days_window)
    actions = pd.merge(actions, others_action, how='left', on=['user_id', 'sku_id'])
    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_order, how='left', on='user_id')
    actions = pd.merge(actions, product, how='left', on='sku_id')

    # 增加（用户，商店）对出现次数的特征
    shop_time = actions.groupby(['user_id','shop_id']).size().reset_index(name='user_shop_time')
    actions = pd.merge(actions, shop_time, how='left',on=['user_id','shop_id'])

    actions = pd.merge(actions, product_conv, how='left', on='sku_id')
    actions = pd.merge(actions, shop,how='left',on='shop_id') 
    print('最后一个表合并之前:',end='\t')
    print(actions.shape)   
    actions = pd.merge(actions, comment, how='left', on='sku_id')
    print('最后一个表合并之后:',end='\t')
    print(actions.shape)

    actions = actions.fillna(0)

    target = actions[['user_id','sku_id','shop_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['shop_id']

    return (target, actions)

# 返回end_date前days天内(用户-商品对)的label(即是否下单）
def get_labels(end_date, days=7):
    actions = gf.get_actions(end_date, days)
    product = gf.get_basic_product_feat()
    
    actions = actions[actions['type'] == 2]
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    actions = pd.merge(actions,product,how='left',on='sku_id')
    actions['label'] = 1
    actions = actions[['user_id', 'sku_id',  'shop_id', 'label']]    

    return  (actions)

# 将make_train_set得到的(用户-商品-商店)标注其在测试日期内是否下过单
def get_test_labels(target, end_date, days=7):
    labels = get_labels(end_date, days)
    labels = pd.merge(target, labels, how='left', on=['user_id','sku_id','shop_id'])
    labels = labels.fillna(0)
    labels = labels.astype(int)

    return (labels)

# 将一个dataframe对象分割成n个dataframe对象
# 返回一个由n个dataframe对象组成的list对象
def split_dataframe(dataframe,n):
    step_length = int(len(dataframe)/n)
    datalist=[]
    i=0
    while(i<n):
        sub_dataframe=dataframe[i*step_length:(i+1)*step_length-1]
        datalist.append(sub_dataframe)
        i+=1
    datalist.append(dataframe[n*step_length:])

    return datalist
   
def add_pos_sample(label,train_data):
    # 获得正样本索引
    label_pos_index=label.index[label['label']==1].tolist()
    label_pos_select=label.loc[label_pos_index]
    train_data_selct=train_data.loc[label_pos_index]
    repeat_time=5
    slices=20
    label_list=split_dataframe(label_pos_select,slices)
    train_data_list=split_dataframe(train_data_selct,slices)
    for j in np.arange(repeat_time): 
        print('第%d次插入正样本集。'%(j))
        for (sub_label,sub_train_data,i) in zip(label_list,train_data_list,np.arange(slices+1)):          
            print('随机插入了第%d个子正样本集'%(i+1))
            # 随机选择一个要插入的位置
            index=random.randint(1,label.shape[0])
            # 分成上下两部分
            label_above=label[:index]
            label_below=label[index+1:]
            train_data_above=train_data[:index]
            train_data_below=train_data[index+1:]
            # 合并成新数据帧
            label=pd.concat([label_above,sub_label,label_below],ignore_index=True)
            train_data=pd.concat([train_data_above,sub_train_data,train_data_below],ignore_index=True)
    
    return (label,train_data)

def sub_neg_sample(label,train_data):
    # 负样本索引
    label_neg_index=label[label==0].index.tolist()
    drop_num=400000
    neg_index_drop=random.sample(label_neg_index,drop_num)

    label.drop(neg_index_drop,inplace=True)
    train_data.drop(neg_index_drop,inplace=True)

    return (label,train_data)

def xgboost_pred():
    train_end_date = '2018-04-09'
    test_end_date = '2018-04-16'
    
    sub_end_date = '2018-04-16'
    
    target_index, train_data = make_train_set(train_end_date)
    label = get_test_labels(target_index, test_end_date)
    
    print('平衡正负样本前：')
    print(u'训练集和验证集中正样本数：%d, 负样本数：%d, 特征总数：%d' % (label[label['label'] == 1].shape[0], label[label['label'] == 0].shape[0],train_data.shape[1]))    # 考虑到正负样本分布极不平衡，增加正样本数减少负样本数平衡正负样本
    label,train_data=sub_neg_sample(label,train_data)
    label,train_data=add_pos_sample(label,train_data)
    print('平衡正负样本后：')
    print(u'训练集和验证集中正样本数：%d, 负样本数：%d, 特征总数：%d' % (label[label['label'] == 1].shape[0], label[label['label'] == 0].shape[0],train_data.shape[1]))    
    
    label = label['label'].astype(int)
    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                        label,
                                                        test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtest = xgb.DMatrix(X_test, label = y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,  # 此处用函数使用过max_depth的取值，为3时得分最高
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 400
    param['nthread'] = 30
    param['eval_metric'] = 'auc'
    param_list = list(param.items())
    param_list += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param_list, dtrain, num_round, evallist, verbose_eval=49)
    
    '''
    #测试
    y_test_pred=bst.predict(X_test)
    accuracy=accuracy_score(y_test,y_test_pred)
    recall=recall_score(y_test,y_test_pred)
    print(u'测试集准确率：%.2f%%, 测试集召回率：%.2f%%'%(accuracy*100,recall*100))
    '''

    # 预测
    sub_user_index, sub_training_data = make_train_set(sub_end_date)
    sub_training_data = xgb.DMatrix(sub_training_data)
    y = bst.predict(sub_training_data)
    
    pred = sub_user_index.copy()
    pred['label'] = y

    pred.to_csv('result.csv',index=False)

    acceptance=0.1
    get_pred_result(pred,acceptance)

def get_pred_result(pred,acceptance):
    pred_accept=pred[pred['label']>=acceptance]
    pred_accept.to_csv('result_%s.csv'%(acceptance),index=False)
    pred_accept=pred_accept[['user_id','sku_id','shop_id']]
    product=pd.read_csv('jdata_product.csv')
    pred_result=pd.merge(pred_accept,product,on=['sku_id','shop_id'])
    pred_result=pred_result[['user_id','cate','shop_id']]

    pred_result=pred_result.astype(int)
    pred_result=pred_result.drop_duplicates()

    pred_result.to_csv('pred_result.csv',index=False)

if __name__ == '__main__':
    xgboost_pred()
