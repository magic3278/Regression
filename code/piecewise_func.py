import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x

def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def sum_shape(num, shape):
    sum_num = 0
    for i in range(num):
        sum_num += shape[i]
        LEFT_NUM = sum(shape) - sum_num
    return LEFT_NUM

def sum_shape1(num, i):
    sum_num = shape_list[i]
    # print(sum_num)
    for j in range(i, num - 1):
        # print(j)
        # print(shape_list[j])
        sum_num += shape_list[j+1]
        if sum_num >= 10 and m - sum_shape(i +1) > 20:
            next_i1 = j + 1
            # print('加到shape_list中的第%s个数，使得此区间个数大于等于10个,个数为%s'%((j+1), sum_num))
            break
        elif sum_num >= 10 and m - sum_shape(i +1) <= 20:
            next_i2 = j + 1
            # print('加到shape_list中的第%s个数，使得此区间个数大于等于10个,个数为%s，相加后，剩下数据个数小于等于20个'%((j+1), sum_num))
            break
    return next_i1

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

def select_last_number(list):
    L = len(list)
    if L == 0:
        print('list为空')
    elif L == 1:
        last_number = list[0]
    else:
        last_number = list[len(list) - 1]
    return last_number

df = pd.read_csv('H:\PycharmProjects\\regression\\7mx_Z.csv',
                 sep = ',',
                 # nrows=15
                 )

m , n = df.shape
C = np.ones(m)
df['C'] = C
print(df)

yxs2 = df['yxs2'].values
print('the max value of yxs2 is :', max(yxs2))
print('round up the max value: :', np.ceil(max(yxs2)))

S = 20
print('we will divide the func into %s segments' %S)

print(np.ceil(max(yxs2)) / S)

interval = np.ceil(np.ceil(max(yxs2)) / S)
print('interval is', interval)

shape_list = []
for i in range(1, S + 1):
    df_ = df[(df['yxs2'] >= (i - 1) * interval) & (df['yxs2'] < i* interval)][['yxs2', 'bxs2', 'C']]
    shape_list.append(df_.shape[0])

print(shape_list)
total_data = m
print('total data:', m)
print(50 * '+')

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
# print(sum_num(shape_list[6:])[0])
# exit()

if total_data <= 20 and total_data > 0:
    print('数据量太少，无法使用分段拟合，直接使用多元线性拟合')
elif total_data > 20:
    print()
else:
    print('数据量异常，无法完成拟合！')
print(50 * '+')
#
I = 1
NUM_list = []
cycle_num = []
# list1 = [2, 4, 3, 1, 0, 1, 0, 3, 0, 3, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
# print(sum_shape(1, list1))
# print(sum(list1))
# exit()
# left_num = m - sum_shape(I + 1, shape_list)
while sum_shape(1, shape_list) > 15:
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

        # while shape_list[0]  < 10:
        #     new_sum = shape_list[0]
        #     if new_sum <10:
        #         print('小于10的new_sum为', new_sum)
        #         while new_sum < 10:
        #             i = 0
        #             new_sum = 0
        #             print(shape_list[i])
        #             new_sum += shape_list[i]
        #             print('new_sum is', new_sum)
        #             i += 1
        #             # print(new_sum)
        #             break
        #         # print(i)



            # elif new_sum >= 10:
            #     # print('第一次合并的数据个数为：', new_sum)
            #     # print('加到第%s个数据后，数据个数大于10' % (i + 1))
            #     print('提取列表', shape_list[: i + 2])
            #     # print('还剩%s个数据点' % (m - sum_shape(i + 2, shape_list)))
            #     # left_num = m - sum_shape(i + 2, shape_list)
            #     shape_list = shape_list[(i + 2):]
            #
            #     break

    # print(shape_list)


        # print(shape_list)
        # print(sum_shape(1, shape_list))
        # I += 1
        # continue
    # if shape_list[0] >= 10:
    #     # print('第一个数为：', shape_list[0])
    #     print('提取列表***：', shape_list[0])
    #     shape_list = shape_list[1 : ]
    # print(shape_list)
    # I += 1


print('处理后的shape_list:', shape_list)
print('提取的NUM_list:', NUM_list)
NUM_list.append(shape_list)
print('总的NUM_list:', NUM_list)
print('提取的cycle_number:', cycle_num)
cycle_num.append(len(shape_list))
print('总的cycle_number:', cycle_num)

def list_sum(list):
    new_list = []
    M = 0
    for i in range(len(list)):
        M += list[i]
        new_list.append(M)
    return new_list

condition = []
for i in range(len(list_sum(cycle_num))):
    con_down = (list_sum(cycle_num)[i] - len(NUM_list[i])) * interval
    con_up = list_sum(cycle_num)[i] * interval
    con_list = [con_down, con_up]
    condition.append(con_list)
print(condition)
#
# if shape_list[I] < 10:
#     while shape_list[i] < 10:
#         print(shape_list[i])
#         new_num = shape_list[i] + shape_list[i + 1]
#         if new_num >= 10:
#             print('第一次合并的数据个数为：', new_num)
#             print('加到第%s个数据后，数据个数大于10' % (i + 1))
#             print('第一段函数区间为：[%s , %s)' % (0, i + 2))
#             print('还剩%s个数据点'%(m - sum_shape(i + 2, shape_list)))
#             left_num = m - sum_shape(i + 2, shape_list)
#             new_shape_list = shape_list[(i + 2):]
#             break
#         i += 1
#     print(new_shape_list)
#
# exit()

    # print(50 * '+')
    # print(i + 2)
    # j = 0
    # while new_shape_list[j] >= 10:
    #     # print(j)
    #     # print(new_shape_list[j])
    #     print('第%s段函数区间为：[%s , %s)' % (j + 2, j + i + 2, j + i + 3))
    #     j += 1
    # new_shape_list1 = new_shape_list[j:]
    # print(new_shape_list1)
    # print(50 * '+')




# exit()
# while shape_list[i] < 10:
#     print(shape_list[i])
#     new_num = shape_list[i] + shape_list[i + 1]
#     if new_num >= 10:
#         print('第一次合并的数据个数为：' , new_num)
#         print('加到第%s个数据后，数据个数大于10'%(i + 1))
#         print('第一段函数区间为：[%s , %s)'%(0 , i + 2))
#         new_shape_list = shape_list[(i + 2) : ]
#         print(new_shape_list)
#         break
#     i += 1
#
# print(50 * '+')
# print(i + 2)
# print(new_shape_list)
# j = 0
# while new_shape_list[j] >= 10:
#     print(j)
#     print(new_shape_list[j])
#     print('第%s段函数区间为：[%s , %s)'%(j + 2, j + i + 2 , j + i + 3))
#     j += 1
#
# print(50 * '+')
# new_shape_list1 = new_shape_list[j : ]
# print(new_shape_list1)
# k = 0
# while new_shape_list1[k] < 10:
#     print(new_shape_list1[k])
#     new_num = new_shape_list1[k] + new_shape_list1[k + 1]
#     if new_num >= 10:
#         print('第一次合并的数据个数为：', new_num)
#         print('加到第%s个数据后，数据个数大于10' % (k + 1))
#         print('第一段函数区间为：[%s , %s)' % (0, k + 2))
#         new_shape_list2 = new_shape_list1[(k + 2):]
#         print(new_shape_list2)
#         break
#     k += 1
#
#
# exit()
#
#
# condition_list1 = []
# condition_list2 = []
# condition_list3 = []
# condition_list4 = []
# I1 = []
# I2 = []
# I3 = []
# I4 = []
# print(range(len(shape_list)))
# print(shape_list[1])
# print(m - sum_shape(1 + 1))
# print(50 * '+')
#
# # print(sum_shape1(S, 2))
# # exit()
#
# for i in range(len(shape_list)):
#     if shape_list[i] >= 10 and m - sum_shape(i + 1) > 20:
#         condition1 = [i * section, (i + 1) * section]
#         condition_list1.append(condition1)
#         I1.append(i)
#     elif shape_list[i] > 10 and m - sum_shape(i + 1) <= 20:
#         condition2 = [i * section, (i + 1) * section]
#         condition_list2.append(condition2)
#         I2.append(i)
#     elif shape_list[i] < 10 and m - sum_shape(i + 1) > 20:
#         # next_I1 = sum_shape1(S, i)
#         # print('nextI1 is:', next_I1)
#         # print('nextI2 is:', next_I2)
#         I3.append(i)
#     elif shape_list[i] < 10 and m - sum_shape(i + 1) <= 20:
#         condition2 = [i * section, (S) * section]
#         condition_list4.append(condition2)
#         I4.append(i)
#
# print(I1)
# print(I2)
# print(I3)
# print(I4)
# print(condition_list1)
# print(condition_list2)
# print(condition_list3)
# print(condition_list4[0])
# exit()
# ws_list = []
# print(50 * '+')
# for i in range(1, S + 1):
#     df_ = df[(df['yxs2'] >= (i - 1) * section) & (df['yxs2'] < i* section)][['yxs2', 'bxs2', 'C']]

    # yxs2_ = df_['yxs2'].values
    # bxs2_ = df_['bxs2'].values
    # X_data = df_[['C', 'yxs2']].values
    # y_data = bxs2_
    # ws = weights(X_data, y_data)
    # x_test = np.array([[min(yxs2_)], [max(yxs2_)]])
    # y_test = ws[0] + x_test * ws[1]
    # plt.plot(yxs2_, bxs2_, 'b.')
    # plt.plot(x_test, y_test, 'r')
    # ws_list.append(ws)
# print(len(ws_list))

# for i in range(len(ws_list)):

# 函数拟合
# print(type(float(condition[0][0])))
# exit()

for i in range(len(condition)):
    print('开始第%s段函数拟合'%(i+1))
    condition_down = float(condition[i][0])
    condition_up = float(condition[i][1])
    print('第%s段函数下限为%s'%(i + 1, condition_down))
    print('第%s段函数上限为%s' % (i + 1, condition_up))
    df1 = df[(df['yxs2'] >= condition_down) & (df['yxs2'] < condition_up)][['yxs2', 'bxs2', 'C']]
    yxs2_1 = df1['yxs2'].values
    bxs2_1 = df1['bxs2'].values
    X_data = df1[['C', 'yxs2']].values
    y_data = bxs2_1
    ws = weights(X_data, y_data)
    x_test = np.array([[min(yxs2_1)], [max(yxs2_1)]])
    y_test = ws[0] + x_test * ws[1]
    plt.plot(yxs2_1, bxs2_1, 'b.')
    plt.plot(x_test, y_test, 'r')
    y_temp = float(max(bxs2_1))
    x_temp = float(max(yxs2_1))
    plt.plot([x_temp, x_temp], [0, y_temp], 'g--')
    plt.plot(x_temp, 0, 'ro')
    plt.text(x_temp - 1, -2, r'$%s$'%round(x_temp,2), fontdict={'size': '8', 'color': 'm'})





plt.xlabel('rock debris (mg/g)')
plt.ylabel('sidewall coring (mg/g)')
plt.ylim(0)
plt.savefig('fig of piecewise.eps', dpi=300, format='eps')
plt.show()
exit()



# 第一段函数拟合
print(df[(df['yxs2'] >= 0) & (df['yxs2'] < section)][['yxs2', 'bxs2', 'C']])
df1 = df[(df['yxs2'] >= 0) & (df['yxs2'] < section)][['yxs2', 'bxs2', 'C']]
yxs2_1 = df1['yxs2'].values
bxs2_1 = df1['bxs2'].values
X_data = df1[['C', 'yxs2']].values
y_data = bxs2_1
ws = weights(X_data,y_data)
x_test = np.array([[0],[max(yxs2_1)]])
y_test = ws[0] + x_test*ws[1]
plt.plot(yxs2_1, bxs2_1, 'b.')
plt.plot(x_test, y_test, 'r')

# 第二段函数
print(df[(df['yxs2'] >= section) & (df['yxs2'] < section + section)][['yxs2', 'bxs2', 'C']])
df2 = df[(df['yxs2'] >= section) & (df['yxs2'] < section + section)][['yxs2', 'bxs2', 'C']]

yxs2_2 = df2['yxs2'].values
bxs2_2 = df2['bxs2'].values
X_data_2 = df2[['C', 'yxs2']].values
y_data_2 = bxs2_2
ws = weights(X_data_2,y_data_2)
x_test_2 = np.array([[max(yxs2_1)],[max(yxs2_2)]])
y_test_2 = ws[0] + x_test_2*ws[1]
plt.plot(yxs2_2, bxs2_2, 'b.')
plt.plot(x_test_2, y_test_2, 'r')




plt.show()