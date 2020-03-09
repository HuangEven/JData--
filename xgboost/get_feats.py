import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import time

user_path='jdata_user.csv'
action_path='jdata_action.csv'
product_path='jdata_product.csv'
shop_path='jdata_shop.csv'
comment_path='jdata_comment.csv'

# 所有用户特征
# 注册时间，认为对结果影响较小？毕竟购买行为是需求驱动的，有需求不管注册多久还是会购买
def get_basic_uesr_feat():
    user = pd.read_csv(user_path)
    user['age'] = user['age'].fillna(-1)
    user['sex'] = user['sex'].fillna(2)

    age_dummies = pd.get_dummies(user['age'], prefix='age')
    sex_dummies = pd.get_dummies(user['sex'], prefix='sex')
    user_lv_dummies = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')
    user = pd.concat([user[['user_id','city_level']], age_dummies, sex_dummies, user_lv_dummies], axis=1)

    return (user)

# 所有商品特征
def get_basic_product_feat():
    product = pd.read_csv(product_path)
    product = transfer_time(product,'market_time')
    
    return (product)

def transfer_time(data,atrribute):
    timeSeries=data[atrribute]
    timeList=list(timeSeries)
    len(timeList)

    times=[]
    for i in timeList:
        sturct_time=time.strptime(i,'%Y-%m-%d %X.0')
        timeStamp = int(time.mktime(sturct_time))
        times.append(timeStamp)
    times_array=np.array(times)
    print(times_array)
    min_time=min(times_array)
    max_time=max(times_array)
    print(min_time)
    times_array=(times_array-min_time)/(max_time-min_time)
    print(times_array)
    data[atrribute]=pd.Series(times_array)
    return data

# 获取店铺的特征
def get_basic_shop_feat():
    shop = pd.read_csv(shop_path)

    shop['main_cate']=shop['cate']
    del shop['vender_id']
    del shop['shop_reg_tm']

    return (shop)

# 返回end_date前days天数内的总评论数指数、是否有差评、及平均差评比率
def get_basic_comment_feat(end_date, days):
    comments = pd.read_csv(comment_path)
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - dt.timedelta(days= days)
    start_date = start_date.strftime('%Y-%m-%d')
    comments = comments[(comments.dt >= start_date)&(comments.dt < end_date)]
    comments = comments.groupby(['sku_id'], as_index=False).sum()
    # comments['bad_comment_ratio'] = comments.bad_comments / comments.comments
    comments['good_comment_ratio'] = comments.good_comments / comments.comments
    comments['comments']=comments['comments'].astype(int)
    del comments['bad_comments']
    del comments['good_comments']

    return (comments)

# 获取end_date之前days天的行为数据
def get_actions(end_date,days):
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - dt.timedelta(days=days)
    start_date = start_date.strftime('%Y-%m-%d')
    actions = pd.read_csv(action_path)
    actions=actions[~(actions['type']==5)]
    actions = actions[(actions.action_time >= start_date) & (actions.action_time < end_date)]
    actions['user_id'] = actions['user_id'].astype(int)
    attributes=['user_id', 'sku_id', 'action_time', 'type']
    actions = actions[attributes]

    return (actions)

# 获取end_date之前按时间窗口days_window划分的时间区间内的行为特征
# 例end_date = '2016-04-06', days_window = (1, 3, 7, 14, 30),
# 本函数返回 "用户-商品"对 从2016-04-06往前1天内、1~3天内、3~7天内、7~14天内、14~30天内的浏览和、加购物车和、下单和等 
def get_action_feat(end_date, days_window):

    actions_raw = get_actions(end_date, days_window[-1])
    attributes=['user_id', 'sku_id', 'type', 'action_time']
    actions_raw = actions_raw[attributes]
    action_feat = None
    end_day = 0
    action_end_date = end_date
    for start_day in days_window:
        action_start_date=datetime.strptime(end_date, '%Y-%m-%d') - dt.timedelta(days =start_day)
        action_start_date=action_start_date.strftime('%Y-%m-%d')
        actions = actions_raw[(actions_raw.action_time >= action_start_date) & (actions_raw.action_time < action_end_date)]
        actions = actions[['user_id','sku_id','type']]

        # “用户-商品”对
        dummies = pd.get_dummies(actions['type'], prefix='type_%d_%d'%(end_day,start_day))
        tutp_df = pd.concat([actions[['user_id','sku_id']],dummies],axis=1)
        tutp_df = tutp_df.groupby(['user_id','sku_id'], as_index = False).sum()

        end_day = start_day
        action_end_date = action_start_date

        if action_feat is None:
            action_feat = tutp_df
        else:
            action_feat = pd.merge(action_feat, tutp_df, how='outer',on=['user_id', 'sku_id'])

    return (action_feat)

# 获取用户热度和商品热度
# 即对于每个（用户-商品对），返回该用户对除所研究商品外其它商品的浏览和、加购物车和、下单和等，
# 及该商品除所研究用户外其他用户的浏览和、加购物车和、下单和等
# 不是直接使用该用户对所有商品的行为特征和所有用户对该商品的行为特征和是为了避免特征冗余
def get_others_action_feat(end_date, days = 30):
    
    actions = get_actions(end_date, days)
    actions = actions[['user_id', 'sku_id', 'type']]
        
    feats = ['type_1', 'type_2', 'type_3','type_4']
        
    # this user this product
    tutp_feats = ['tutp_%s' % x for x in feats]
    dummies = pd.get_dummies(actions['type'], prefix='tutp_type')
    tutp_df = pd.concat([actions[['user_id', 'sku_id']], dummies], axis=1)
    tutp_df = tutp_df.groupby(['user_id', 'sku_id'], as_index=False).sum()

    # this user all product
    tuap_feats = ['tuap_%s' % x for x in feats]
    dummies = pd.get_dummies(actions['type'], prefix='tuap_type')
    tuap_df = pd.concat([actions[['user_id']], dummies], axis=1)
    tuap_df = tuap_df.groupby(['user_id'], as_index=False).sum()
            
    # all user this product
    autp_feats = ['autp_%s' % x for x in feats]
    dummies = pd.get_dummies(actions['type'], prefix='autp_type')
    autp_df = pd.concat([actions[['sku_id']], dummies], axis=1)
    autp_df = autp_df.groupby(['sku_id'], as_index=False).sum()
            
    df = pd.merge(tutp_df, tuap_df, how = 'left', on = ['user_id'])
    df = pd.merge(df, autp_df, how = 'left', on = ['sku_id'])
            
    # this user other product
    tuop_feats = ['tuop_%s' % x for x in feats]
    
    # other user this product
    outp_feats = ['outp_%s' % x for x in feats]
    for i in range(len(feats)):
        df[tuop_feats[i]] = df[tuap_feats[i]] - df[tutp_feats[i]]
        df[outp_feats[i]] = df[autp_feats[i]] - df[tutp_feats[i]]
    others_action = df[['user_id', 'sku_id'] + tuop_feats + outp_feats]

    return (others_action)

# 用户每个行为转化为下单行为的可能比率
def get_user_order_ratio(end_date, days):
    feature = ['user_id', 'user_type_1_ratio', 'user_type_3_ratio', 'user_type_4_ratio']
    actions = get_actions(end_date, days)
    
    dummies = pd.get_dummies(actions['type'], prefix='type')
    actions = pd.concat([actions[['user_id']], dummies], axis=1)
    actions = actions.groupby(['user_id'], as_index=False).sum()
    
    actions['user_type_1_ratio'] = actions['type_2'] / actions['type_1']
    actions['user_type_3_ratio'] = actions['type_2'] / actions['type_3']
    actions['user_type_4_ratio'] = actions['type_2'] / actions['type_4']
    # actions['user_type_5_ratio'] = actions['type_2'] / actions['type_5']
    
    actions = actions[feature]
    actions = actions.fillna(0).replace(np.inf, 0)
    return (actions)

# 对每个商品的行为转化为下单行为的比率
def get_product_conversion(end_date, days = 30):
    feature = ['sku_id', 'product_type_1_ratio', 'product_type_3_ratio', 'product_type_4_ratio']
    actions = get_actions(end_date, days)
    
    dummies = pd.get_dummies(actions['type'], prefix='type')
    actions = pd.concat([actions[['sku_id']], dummies], axis=1)
    actions = actions.groupby(['sku_id'], as_index=False).sum()
    
    actions['product_type_1_ratio'] = actions['type_2'] / actions['type_1']
    actions['product_type_3_ratio'] = actions['type_2'] / actions['type_3']
    actions['product_type_4_ratio'] = actions['type_2'] / actions['type_4']
    # actions['product_type_5_ratio'] = actions['type_2'] / actions['type_5']
    
    actions = actions[feature]
    actions = actions.fillna(0).replace(np.inf, 0)
       
    return(actions)