#-*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
legend_font = FontProperties(fname='K:\PycharmProjects\\regression\Deng.ttf', size=8)

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
# ax = plt.gca()


df_r = pd.DataFrame()
FileList_ = os.listdir('K:\PycharmProjects\\regression\\fig_in_article\多元数据')
FileList = []

for i in range(len(FileList_)):
    file_name = FileList_[i].replace('.xlsx', '')
    FileList.append(file_name)
print(FileList)

X_list1 = []
X_list2 = []
X_list3 = []
X_list4 = []
X_list5 = []
df_list = []
y_list = []
# print('X_list' + str(0 + 1))
# df = pd.read_excel('H:\PycharmProjects\\regression\\fig_in_article\多元数据\\1LZ_D_z.xlsx')
# print(df.iloc[:, 0:2])
# print(type(df.iloc[:, 1:2].values))
# exit()
for i in range(len(FileList)):
    file_path = 'K:\PycharmProjects\\regression\\fig_in_article\多元数据\\' + FileList_[i]
    file_path = file_path.replace('\\', '\\\\')
    df = pd.read_excel(file_path)
    # df = df.apply(lambda x: (x - np.mean(x)) / np.std(x))
    df_list.append(df)
    y = df['bxs2'].values
    y_list.append(y)
    # for j in range(5):
    #     x = df.iloc[:, 0:j+1]
    #     if j % 5 == 0:
    #         X_list.append(x)
    x1 = df.iloc[:, 1:2].values
    x2 = df.iloc[:, 0:2].values
    x3 = df.iloc[:, 0:3].values
    x4 = df.iloc[:, 0:4].values
    x5 = df.iloc[:, 0:5].values
    X_list1.append(x1)
    X_list2.append(x2)
    X_list3.append(x3)
    X_list4.append(x4)
    X_list5.append(x5)

data_number = []
for i in range(len(FileList)):
    m, n = df_list[i].shape
    data_number.append(m)
print('*' * 100)
# 一元线性回归bxs2与yxs2
R1 = []
for i in range(len(FileList)):
    # print('第%s次循环开始'%(i+1))
    model = linear_model.Ridge()
    model.fit(X_list1[i], y_list[i])
    df_list[i]['1_predict'] = model.predict(X_list1[i]).tolist()
    r1 = df_list[i]['bxs2'].corr(df_list[i]['1_predict'])
    R1.append(r1)
df_r['Model1'] = R1
# 二元线性回归bxs2与yxs2，yxs1
R2 = []
for i in range(len(FileList)):
    # print('第%s次循环开始'%(i+1))
    model = linear_model.Ridge()
    model.fit(X_list2[i], y_list[i])
    df_list[i]['2_predict'] = model.predict(X_list2[i]).tolist()
    r2 = df_list[i]['bxs2'].corr(df_list[i]['2_predict'])
    R2.append(r2)
df_r['Model2'] = R2

R3 = []
for i in range(len(FileList)):
    # print('第%s次循环开始'%(i+1))
    model = linear_model.Ridge()
    model.fit(X_list3[i], y_list[i])
    df_list[i]['3_predict'] = model.predict(X_list3[i]).tolist()
    r3 = df_list[i]['bxs2'].corr(df_list[i]['3_predict'])
    R3.append(r3)
df_r['Model3'] = R3

R4 = []
for i in range(len(FileList)):
    # print('第%s次循环开始'%(i+1))
    model = linear_model.Ridge()
    model.fit(X_list4[i], y_list[i])
    df_list[i]['4_predict'] = model.predict(X_list4[i]).tolist()
    r4 = df_list[i]['bxs2'].corr(df_list[i]['4_predict'])
    R4.append(r4)
df_r['Model4'] = R4

R5 = []
for i in range(len(FileList)):
    # print('第%s次循环开始'%(i+1))
    model = linear_model.Ridge()
    model.fit(X_list5[i], y_list[i])
    df_list[i]['5_predict'] = model.predict(X_list5[i]).tolist()
    r5 = df_list[i]['bxs2'].corr(df_list[i]['5_predict'])
    R5.append(r5)
df_r['Model5'] = R5
# df_r = df_r.apply(lambda x : (x ** 2))
# 画10条R^2折线图
plt.figure(figsize=(8, 6))
for i in range(len(FileList)):
    X_R = df_r.iloc[i]
    plt.plot(X_R, label=FileList[i])

df_r['data_number'] = data_number
print(df_r)

plt.legend(fontsize='xx-small', prop=legend_font)
plt.ylabel('r', fontsize=18)
# plt.title('汉字', fontproperties=font)
# ax.invert_yaxis()






plt.show()