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
def standard_func(xArr, yArr):
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
    # model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
    model = linear_model.Ridge()
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

def list_mean_merge(list1, list2, S):
    """
    :param list1: 第一个列表
    :param list2: 第二个列表
    :param S: 分段个数
    :return: 第一个列表后一个值与第二个列表前一个值的平均值，列表最后一个值与第二个列表最后一个值相同
    """
    interval_value = []
    for i in range(S - 1):
        # print(list1[i + 1], list2[i])
        new_value = round(1 / 2 * (list1[i + 1] + list2[i]), 2)
        interval_value.append(new_value)
    interval_value.append(round(list2[S - 1], 2))
    return interval_value

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
data_num_input = 10
S = m // data_num_input
print('we will divide the func into %s segments' %S)
# print(df)
df_sort = df.sort_values(by='yxs2', axis=0, ascending=True)
# print(df_sort)
df_seg_list = []
interval_value_first = []
interval_value_last = []
if m % data_num_input == 0:
    print('分段拟合数据点可以被总数居整除')
    for i in range(S):
        df_seg = df_sort[i * data_num_input: (i + 1) * data_num_input]
        interval_value_first.append(df_seg['yxs2'].values[0])
        interval_value_last.append(df_seg['yxs2'].values[-1])
        df_seg_list.append(df_seg)
else:
    print('分段拟合数据点不可以被总数居整除，剩余%s个数据点'%(m % data_num_input))
    for i in range(S - 1):
        df_seg = df_sort[i * data_num_input: (i + 1) * data_num_input]
        df_seg_list.append(df_seg)
    df_seg_list.append(df_sort.iloc[-((m % data_num_input) + data_num_input) : ])
    for i in range(len(df_seg_list)):
        interval_value_first.append(df_seg_list[i]['yxs2'].values[0])
        interval_value_last.append(df_seg_list[i]['yxs2'].values[-1])
print(interval_value_first)
print(interval_value_last)
interval_value = list_mean_merge(interval_value_first, interval_value_last, S)
print(interval_value)

bx_list = []
COEF_list = []
intercept = []
y_pred_list = []
for i in range(len(df_seg_list)):
    # print(df_seg_list[i])
    yxs2_1 = df_seg_list[i]['yxs2'].values
    bxs2_1 = df_seg_list[i]['bxs2'].values
    X_data = df_seg_list[i][['C', 'yxs2']].values
    y_data = df_seg_list[i]['bxs2'].values
    # X_data = df1[['C', 'yxs2']].values
    # y_data = bxs2_1
    # ws = weights(X_data, y_data)
    ws = standard_func(X_data, y_data)
    bx_list.append(eval_list(yxs2_1, ws[1], ws[0]))

    x_test = np.array([[min(yxs2_1)], [max(yxs2_1)]])
    y_test = ws[0] + x_test * ws[1]
    print(ws[1], ws[0])
    # 打印函数
    if ws[0] > 0 and ws[1] > 0:
        print('第%s段函数为：bx_%s = %.2f * yx + %.2f' % (i + 1, i + 1, ws[1], ws[0]))
    elif ws[0] < 0 and ws[1] > 0:
        print('第%s段函数为：bx_%s = %.2f * yx - %.2f' % (i + 1, i + 1, ws[1], abs(ws[0])))
    elif ws[0] < 0 and ws[1] < 0:
        print('第%s段函数为：bx_%s = - %.2f * yx - %.2f' % (i + 1, i + 1, abs(ws[1]), abs(ws[0])))
    else:
        print('第%s段函数为：bx_%s = - %.2f * yx + %.2f' % (i + 1, i + 1, abs(ws[1]), ws[0]))
    plt.plot(yxs2_1, bxs2_1, 'b.')
    plt.plot(x_test, y_test, 'r')
    y_temp = float(max(bxs2_1))
    x_temp = float(max(yxs2_1))
    plt.plot([x_temp, x_temp], [0, y_temp], 'g--')
    plt.plot(x_temp, 0, 'ro')
    plt.text(x_temp - 1, -2, r'$%s$' % round(x_temp, 2), fontdict={'size': '8', 'color': 'm'})

print(bx_list)
bx_list = combine_list(bx_list)
print(bx_list)
df_sort['bxs2_eval'] = bx_list
# df.to_csv('new_data.csv')
print(df_sort)
print(df_sort.corr())
plt.xlabel('rock debris (mg/g)')
plt.ylabel('sidewall coring (mg/g)')
plt.ylim(0)

# plt.savefig('fig of piecewise.eps', dpi=300, format='eps')
plt.show()





