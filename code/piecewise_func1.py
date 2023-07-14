import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

def sum_shape(num, shape):
    sum_num = 0
    LEFT_NUM = 0
    for i in range(num):
        sum_num += shape[i]
        LEFT_NUM = sum(shape) - sum_num
    return LEFT_NUM

# 标准方程法函数
def weights(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat # 矩阵乘法
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    # xTx.I为xTx的逆矩阵
    ws = xTx.I*xMat.T*yMat
    return ws

# 岭回归函数
def Ridge_Regression(xArr, yArr):
    alphas_to_test = np.linspace(0.000001, 1)
    model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
    model.fit(xArr, yArr)
    y_pred = model.predict(xArr)
    # y_pred = model.predict(yArr)
    # alphas_val_ = model.alpha_
    COEF_ = model.coef_
    intercept_ = model.intercept_
    ws = [intercept_, COEF_]
    return ws, y_pred

def select_last_number(list):
    L = len(list)
    last_number = 0
    if L == 0:
        print('list为空')
    elif L == 1:
        last_number = list[0]
    else:
        last_number = list[len(list) - 1]
    return last_number

def sum_num(list):
    NUM = 0
    i = 0
    while NUM < 10:
        NUM += list[i]
        i += 1
        # print(NUM)
    NUM_list = [NUM, i]
    # print('循环了%s次' % i)
    return NUM_list

def list_sum(list):
    new_list = []
    M = 0
    for i in range(len(list)):
        M += list[i]
        new_list.append(M)
    return new_list

def compute_error(a, b, yxs2, bxs2):
    totalError = 0
    for i in range(0, len(yxs2)):
        totalError += (bxs2[i] - yxs2[i] / (a * yxs2[i] + b)) ** 2
    return totalError

def eval_list(list, a , b):
    y_eval_list= []
    for i in range(len(list)):
        y_evaluate = (b + list[i] * a).tolist()[0][0]
        y_eval_list.append(y_evaluate)
    return y_eval_list

def combine_list(list):
    temp_list =[]
    for i in range(len(list)):
        temp_list.extend(list[i])
    return temp_list

# 读取文件列表
file_folder_path = 'k:\PycharmProjects\\regression\\fig_in_article\非线性数据修改'
df_r = pd.DataFrame()
FileList_ = os.listdir(file_folder_path)
FileList = []
for i in range(len(FileList_)):
    file_name = FileList_[i].replace('.xlsx', '')
    FileList.append(file_name)
print(FileList)

# 读取数据
X_list = []
df_list = []
y_list = []

for i in range(len(FileList)):
    file_path = file_folder_path + '\\' + FileList_[i]
    file_path = file_path.replace('\\', '\\\\')
    df = pd.read_excel(file_path)
    # df = df.apply(lambda x: (x - np.mean(x)) / np.std(x))
    # df = df.apply(lambda x: (x - np.min(x)) / np.max(x) - np.min(x))
    df_list.append(df)
    y = df['bxs2'].values
    y_list.append(y)
    x2 = df['yxs2'].values
    X_list.append(x2)


data_number = []
for i in range(len(FileList)):
    m, n = df_list[i].shape
    data_number.append(m)

print('*' * 100)
# print(df_list[0])

df = df_list[0]
m , n = df.shape
C = np.ones(m)
df['C'] = C     # 添加“1”列
total_data = m
print('total of data :', m)

# 对yxs2数据进行分析
yxs2 = df['yxs2'].values
print('the max value of yxs2 is :', max(yxs2))
print('round up the max value: :', np.ceil(max(yxs2)))

# 对x轴进行分段
S = 10
average_interval = np.ceil(max(yxs2)) / S
interval = np.ceil(np.ceil(max(yxs2)) / S)
print('we will divide the func into %s segments' %S)
print('the average value of interval is', average_interval)
print('interval is', interval)

# 确定被平分后每个区间内数据的个数
shape_list = []
for i in range(1, S + 1):
    df_ = df[(df['yxs2'] >= (i - 1) * interval) & (df['yxs2'] < i* interval)]
    shape_list.append(df_.shape[0])
print('the original shape_list is :', shape_list)
print(50 * '+')

if total_data <= 20 and total_data > 0:
    print('数据量太少，无法使用分段拟合，直接使用多元线性拟合')
elif total_data > 20:
    print()
else:
    print('数据量异常，无法完成拟合！')

# 筛选数据个数，及确定各个区间
I = 1
NUM_list = []
cycle_num = []
while sum_shape(1, shape_list) > 5:
    print(50 * '+')
    print('主函数循环次数：', I)
    print('剩余个数：', sum_shape(1, shape_list))
    print(shape_list)
    # if shape_list[0] < 10:
        # print('第一个数为：', shape_list[0])

    print('提取列表', shape_list[: sum_num(shape_list)[1]])
    print('循环次数：', sum_num(shape_list)[1])
    NUM_list.append(shape_list[: sum_num(shape_list)[1]])
    cycle_num.append(sum_num(shape_list)[1])
    shape_list = shape_list[sum_num(shape_list)[1] :]
    print(shape_list)
    I =+ 1

print('处理后的shape_list:', shape_list)
print('提取的NUM_list:', NUM_list)
NUM_list.append(shape_list)
print('总的NUM_list:', NUM_list)
print('提取的cycle_number:', cycle_num)
cycle_num.append(len(shape_list))
print('总的cycle_number:', cycle_num)

# 确定每段函数区间
condition = []
for i in range(len(list_sum(cycle_num))):
    con_down = (list_sum(cycle_num)[i] - len(NUM_list[i])) * interval
    con_up = list_sum(cycle_num)[i] * interval
    con_list = [con_down, con_up]
    condition.append(con_list)
print('各个函数区间为：', condition)




# 对每段函数进行岭回归拟合
bx_list = []
COEF_list = []
intercept = []
y_pred_list = []
for i in range(len(condition)):
    # print('开始第%s段函数拟合'%(i+1))
    condition_down = float(condition[i][0])
    condition_up = float(condition[i][1])
    # print('第%s段函数下限为%s'%(i + 1, condition_down))
    # print('第%s段函数上限为%s' % (i + 1, condition_up))
    df1 = df[(df['yxs2'] >= condition_down) & (df['yxs2'] < condition_up)]
    yxs2_1 = df1['yxs2'].values
    bxs2_1 = df1['bxs2'].values
    X_data = df1[['yxs1', 'yxs2', 'yxpg', 'opi']].values
    y_data = df1['bxs2'].values
    # X_data = df1[['C', 'yxs2']].values
    # y_data = bxs2_1
    # ws = weights(X_data, y_data)
    ws, y_pred = Ridge_Regression(X_data, y_data)
    COEF_list.append(ws[1])
    intercept.append(ws[0])
    y_pred_list.append(y_pred)



    plt.plot(yxs2_1, bxs2_1, 'b.')
    plt.plot(yxs2_1, y_pred, 'r^')

    y_temp = float(max(bxs2_1))
    x_temp = float(max(yxs2_1))
    plt.plot([x_temp, x_temp], [0, y_temp], 'g--')
    plt.plot(x_temp, 0, 'ro')
    plt.text(x_temp - 0, - 0, r'$%s$'%round(x_temp,2), fontdict={'size': '8', 'color': 'm'})
    # plt.show()
    # exit()





y_pred_list = combine_list(y_pred_list)
df['bxs2_eval'] = y_pred_list
MSE_ = np.sqrt(mean_squared_error(df['bxs2'], df['bxs2_eval']))
R_SCORE_ = r2_score(df['bxs2'], df['bxs2_eval'])
print(MSE_)
print(R_SCORE_)

# df.to_csv('new_data.csv')
# print(df)
# print(df.corr())
plt.xlabel('rock debris (mg/g)')
plt.ylabel('sidewall coring (mg/g)')
plt.legend()
plt.ylim(0)
print('拟合完毕！将函数分成%s段'%len(condition))
# plt.savefig('fig of piecewise.eps', dpi=300, format='eps')
plt.show()





