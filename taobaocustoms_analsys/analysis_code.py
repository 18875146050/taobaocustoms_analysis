from datetime import datetime
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
import re
data_user = pd.read_csv(r'data/tianchi_mobile_recommend_train_user.csv')
data_user.head()
######################################
#缺失值处理
missingTotal = data_user.isnull().sum()
#计算数据内每列的缺失值
missingExist = missingTotal[missingTotal>0]
#提取出存在缺失值的列
missingExist = missingExist.sort_values(ascending=False)
#对存在缺失值的列索引进行降序排序
#print(missingTotal)
####################################################
#一致化处理
data_user['date'] = data_user['time'].map(lambda s:re.compile(' ').split(s)[0])
data_user['hour'] = data_user['time'].map(lambda s:re.compile(' ').split(s)[1])
#print(data_user.head())
#map:根据输入对应关系映射Series的值
#用于将系列中的每个值替换为另一个值，可以从func，dict、series
#compil用于编译正则表达式生成一个pattern对象
#例如2014-11-23 23，以空格作为pattern分为两部分，第一部分为data,第二部分为hour
#查看数据类型
data_user.dtypes
#数据类型转换
data_user['date'] = pd.to_datetime(data_user['date'])
data_user['time'] = pd.to_datetime(data_user['time'])
data_user['hour'] = data_user['hour'].astype('int64')
data_user.dtypes
################################################################
#异常值处理
data_user = data_user.sort_values(by='time',ascending=True)#按时间列升序排序
data_user = data_user.reset_index(drop=True)
'''reset_index
生成一个新的带有重置索引的DataFrame或Series。
当索引需要被视为列时，
或者索引无意义并且需要在其他操作之前重置为默认值时，此功能很有用。
drop=False重置索引，将原始索引作为列插入新的DataFrame中。
drop=True重置索引，不将原始索引作为列插入新的DataFrame中
'''
data_user.describe()
'''生成描述性统计分析,描述性统计数据包括总结数据集分布的集中趋势，
离散度和形状的统计数据，但不包括“ NaN”值。
count mean std min max 25% 50% 75%
'''
##############################################################
#######################用户行为分析###########################
#pv和uv分析
#pv_daily记录用户每天操作次数，uv_daily记录每天不同得上线用户数量
#现在的data_user是按照时间排序重置索引之后的数据
pv_daily=data_user.groupby('date')['user_id'].count().reset_index().rename(columns={'user_id':'pv'})
#每天的用户数量计算
#reset_index()为数据框添加索引
#rename重命名
#pv_daily为按照日期排序的每天点击量，有索引
uv_daily = data_user.groupby('date')['user_id'].apply(lambda x:x.drop_duplicates().count()).reset_index().rename(columns={'user_id':'uv'})
# subset : column label or sequence of labels, optional
# 用来指定特定的列，默认所有列
# keep : {‘first’, ‘last’, False}, default ‘first’
# 删除重复项并保留第一次出现的项
# inplace : boolean, default False
# 是直接在原来数据上修改还是保留一个副本
fig,axes = plt.subplots(2,1,sharex=True)
# 函数返回一个figure图像和子图ax的array列表。
#这里的2和1参数对应的行列，在这里是将图按行方向拆分成两个子图
# 如果想要设置子图的宽度和高度可以在函数内加入figsize值
# fig, ax = plt.subplots(1,3,figsize=(15,7))，这样就会有1行3个15x7大小的子图。
pv_daily.plot(x='date',y='pv',ax = axes[0])
uv_daily.plot(x='date',y='uv',ax = axes[1])
print(axes[0].set_title('pv_daily'))
print(axes[1].set_title('uv_daily'))
#plt.show()
#小时访问量分析
#pv_hour记录每个小时用户操作数量，uv_hour记录每小时不同的上线用户数量
pv_hour = data_user.groupby('hour')['user_id'].count().reset_index().rename(columns={'user_id':'pv'})
#print(pv_hour.head(20))
#pv_hour记录的是从0-23小时每个小时的user_id的数量
uv_hour = data_user.groupby('hour')['user_id'].apply(lambda x:x.drop_duplicates().count()).reset_index().rename(columns={'user_id':'uv'})
#print(uv_hour.head(20))
##uv_hour记录的是从0-23小时每个小时的去重user_id的数量
fig,axes = plt.subplots(2,1,sharex=True)
pv_hour.plot(x='hour',y='pv',ax=axes[0])
uv_hour.plot(x='hour',y='uv',ax=axes[1])
axes[0].set_title('pv_hour')
axes[1].set_title('uv_hour')
#plt.show()
#不同行为类型用户pv分析
pv_detail = data_user.groupby(['behavior_type','hour'])['user_id'].count().reset_index().rename(columns={'user_id':'total_pv'})
#不同的behavior_type下0-23小时内user_id 的count
fig,axes=plt.subplots(2,1,sharex=True)
sns.pointplot(x='hour',y='total_pv',hue='behavior_type',data=pv_detail,ax=axes[0])
sns.pointplot(x='hour',y='total_pv',hue='behavior_type',data=pv_detail[pv_detail.behavior_type!=1],ax=axes[1])
axes[0].set_title('pv_different_behavior_type')
axes[1].set_title('pv_different_behavior_type_except1')
#plt.show()
#用户购买次数情况分析
data_user_buy=data_user[data_user.behavior_type==4].groupby('user_id')['behavior_type'].count()
#记录每个人买了几次
plt.figure(figsize=(8,4))#绘制画布
sns.distplot(data_user_buy,kde=False)
#柱状图，kde=F,不绘制高斯核密度估计
plt.title('daily_user_buy')
#plt.show()
##日ARPPU()平均付费额度，用平均消费次数来代表
#人均消费次数=总消费次数/总消费人数
data_user_buy1 = data_user[data_user.behavior_type==4].groupby(['date','user_id'])['behavior_type'].count().reset_index().rename(columns={'behavior_type':'total'})
#每一天每一个用户买过东西的次数
plt.figure(figsize=(8,4))#绘制画布
data_user_buy1.groupby('date').apply(lambda x:x.total.sum()/x.total.count()).plot()
#每天，所有人一共买东西的次数/买东西的人数
plt.title('daily_ARPPU')
#plt.show()
#日ARPU，平均每个用户消费金额
#活跃用户平均消费次数=消费总次数/活跃用户数
data_user['operation']=1
data_user_buy2=data_user.groupby(['date','user_id','behavior_type'])['operation'].count().reset_index().rename(columns={'operation':'total'})
#每天每一个用户每一种操作的操作次数
plt.figure(figsize=(8,4))#绘制画布
data_user_buy2.groupby('date').apply(lambda x:x[x.behavior_type==4].total.sum()/len(x.user_id.unique())).plot()
#按天数画表，每天总共买了多少次/每天买的人数
plt.title('daily_ARPU')
# plt.show()
#付费率
plt.figure(figsize=(8,4))#绘制画布
data_user_buy2.groupby('date').apply(lambda x:x[x.behavior_type==4].total.count()/len(x.user_id.unique())).plot()
#付费率=付费人数/活跃用户总人数
plt.title('daily_afford_rate')
# plt.show()
#同一时间段用户消费次数分布
data_user_buy3=data_user[data_user.behavior_type==4].groupby(['user_id','date','hour'])['operation'].sum().rename('buy_count')
#整理出每个用户在哪一天哪一个小时有过购买行为然后计算该用户在这期间总共购买次数
plt.figure(figsize=(8,4))#绘制画布
sns.distplot(data_user_buy3)
# plt.show()
print('大多数用户消费：{}次'.format(data_user_buy3.mode()[0]))
######################################################################
############复购情况分析
date_rebuy = data_user[data_user.behavior_type==4].groupby('user_id')['date'].apply(lambda x:len(x.unique())).rename('rebuy_count')
# 每个id买了多少次
print('复购率：',round(date_rebuy[date_rebuy>=2].count()/date_rebuy.count(),4))
#购买次数大于2的用户数量/所有用户次数，保留四位小数
plt.figure(figsize=(8,4))#绘制画布
sns.distplot(date_rebuy-1)
plt.title('rebuy_user')
# plt.show()
print('多数用户复购次数：{}'.format((date_rebuy-1).mode()[0]))
#所有复购时间间隔消费次数分布
data_day_buy = data_user[data_user.behavior_type==4].groupby(['user_id','date']).operation.count().reset_index()
# 每个用户在哪些天购买的次数
data_user_buy4=data_day_buy.groupby('user_id').date.apply(lambda x:x.sort_values().diff(1).dropna())
#用户购买行为的时间差
data_user_buy4=data_user_buy4.map(lambda x:x.days)
#变成数字
plt.figure(figsize=(8,4))#绘制画布
data_user_buy4.value_counts().plot(kind='bar')
#返回一个包含唯一值计数的系列。
plt.title('time_gap')
plt.xlabel('gap_day')
plt.ylabel('gap_count')
# plt.show()
#不同用户平均复购时间分析
plt.figure(figsize=(8,4))#绘制画布
sns.distplot(data_user_buy4.reset_index().groupby('user_id').date.mean())
#不同用户的平均复购时间间隔
# plt.show()
#################################################################
######################漏斗流失分析
data_user_count=data_user.groupby(['behavior_type']).count()
print(data_user_count.head())
#计算出各种操作的数量，operation来计数
pv_all=data_user['user_id'].count()
#没一次操作都算在浏览量里，算出总点击量
print(pv_all)
###########################################################################
################用户行为与商品种类关系分析
data_category = data_user[data_user.behavior_type!=2].groupby(['item_category','behavior_type']).operation.count().unstack(1).rename(columns={1:'点击量',3:'加入购物车',4:'购买量'}).fillna(0)
#每一种商品下用户产生了什么行为，产生行为的次数
print(data_category.head())
#转化率计算
data_category['转换率']=data_category['购买量']/data_category['点击量']
print(data_category.head())
#异常值处理
data_category=data_category.fillna(0)
data_category=data_category[data_category['转换率']<=1]
#转化率绘图
plt.figure(figsize=(8,4))#绘制画布
# sns.distplot(data_category['转换率'])
# plt.show()
sns.distplot(data_category[data_category['转换率']>0]['转换率'],kde=False)
plt.title('conversion rate')
# plt.show()
#感兴趣率
data_category['感兴趣比率']=data_category['加入购物车']/data_category['点击量']
print(data_category.head())
#异常值处理
data_category=data_category[data_category['感兴趣比率']<=1]
plt.figure(figsize=(8,4))#绘制画布
# sns.distplot(data_category['感兴趣比率'])
sns.distplot(data_category[data_category['感兴趣比率']>0]['感兴趣比率'],kde=False)
plt.title('Interest rate')
plt.show()
##将转化率分三类查看各类占比例
data_convert_rate=pd.cut(data_category['转换率'],[-1,0,0.1,1]).value_counts()
#将转换率分区间计数，pd.cut(要分类得, 指定区间)
data_convert_rate=data_convert_rate/data_convert_rate.sum()
print(data_convert_rate)
##将感兴趣比率分三类查看各类占比例
data_interest_rate=pd.cut(data_category['感兴趣比率'],[-1,0,0.1,1]).value_counts()
data_interest_rate=data_interest_rate/data_interest_rate.sum()
print(data_interest_rate)
########################################################################
#二八理论和长尾理论
print(data_category.head())
data_category=data_category[data_category['购买量']>0]
value_8=data_category['购买量'].sum()*0.8
value_10=data_category['购买量'].sum()
data_category=data_category.sort_values(by='购买量',ascending=False)
# print(data_category.head())
data_category['累计购买量']=data_category['购买量'].cumsum()
#返回序列得累加值
# print(data_category.head())
data_category['分类']=data_category['累计购买量'].map(lambda x:'前80%' if x<=value_8 else '后20%')
print(data_category.head())
result = data_category.groupby('分类')['分类'].count()/data_category['分类'].count()
#对商品种类进行计数
print(result)
#################################################################################################
#################################################### RFM模型分析
datenow=datetime(2014,12,20)
#每位用户最近购买时间
recent_buy_time=data_user[data_user.behavior_type==4].groupby('user_id').date.apply(lambda x:datetime(2014,12,20)-x.sort_values().iloc[-1]).reset_index().rename(columns={'date':'recent'})
# iloc函数：通过行号来取行数据,对每一个用户得每一次购买行为发生得时间与datenow计算差值，降序排序，
# iloc[-1]取最后一个时间差，也就是最新时间差
recent_buy_time.recent=recent_buy_time.recent.map(lambda x:x.days)
#返回x得days属性得值
#每个用户消费频率
buy_freq=data_user[data_user.behavior_type==4].groupby('user_id').date.count().reset_index().rename(columns={'date':'freq'})
#统计每个用户购买次数
rfm=pd.merge(recent_buy_time,buy_freq,left_on='user_id',right_on='user_id',how='outer')
#rfm=索引  user_id  recent  freq
#how=outer取并集
#将各维度分成两个程度,分数越高越好
rfm['recent_value']=pd.qcut(rfm.recent,2,labels=['2','1'])
rfm['freq_value']=pd.qcut(rfm.freq,2,labels=['1','2'])
# pandas的qcut可以把一组数字按大小区间进行分区
#把data分成两份，小得标签2大的标签1
#qcut()方法第一个参数是数据,第二个参数定义区间的分割方法
#第三个参数是要替换的值,就是对应区间的值应该替换成什么值,
# 顺序和区间保持一致就好了,
# 注意有几个区间,就要给几个值,不能多也不能少.
rfm['rfm']=rfm['recent_value'].str.cat(rfm['freq_value'])
#拼接
print(rfm.head())


