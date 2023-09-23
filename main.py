import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


def do_cal():
    # 定义数据集的起始年份和年份数量
    start_year = 2
    num_year = 10
    # 定义区域人口总量用于制图（预测），并转秩为列向量
    P_o = np.array([7869.34, 8022.99, 8119.81, 8192.44, 8281.09, 8315.11, 8381.47, 8423.50, 8446.19, 8469.09, 8477.26])
    P_o = P_o.reshape(-1, 1)
    # 定义区域GDP用于制图（预测），并转秩为列向量
    G_o = np.array([41383.87, 45952.65, 50660.20, 55580.11, 60359.43, 65552.00, 70665.71, 75752.20, 80827.71, 85556.13, 88683.21])
    G_o = P_o.reshape(-1, 1)
    # 定义区域人口总量用于训练，因需要匹配增长率的数据个数，该数据比_o少第一位数据，并转秩为列向量
    P = np.array(P_o[1:])
    # 定义区域GDP用于训练，因需要匹配增长率的数据个数，该数据比_o少第一位数据，并转秩为列向量
    G = np.array(G_o[1:])

    # 构造模型训练的特征矩阵，拼接人口总量与GDP总量的向量，形成一个n行2列的矩阵
    X = np.concatenate((P, G), axis=1)

    len_p = len(P)
    len_g = len(G)

    # 构造dp/dt，即人口增长率，人口总量对时间的一阶导，并转秩为列向量
    Pt = np.zeros((1, len_p))
    i = 0
    while i in range(0, len_p):
        Pt[0][i] = (P_o[i+1][0] - P_o[i][0]) / P_o[i][0]
        i = i + 1
    Pt = Pt.reshape(-1, 1)

    # 构造dg/dt，即GDP增长率，GDP对时间的一阶导，并转秩为列向量
    Gt = np.zeros((1, len_g))
    i = 0
    while i in range(0, len_g):
        Gt[0][i] = (G_o[i + 1][0] - G_o[i][0]) / G_o[i][0]
        i = i + 1
    Gt = Gt.reshape(-1, 1)

    # 构造二次多项式模型参数，以下方法中degree控制阶数，include_bias控制是否增加偏置常量
    model_Pt_poly_features = PolynomialFeatures(degree=2, include_bias=False)
    model_Gt_poly_features = PolynomialFeatures(degree=2, include_bias=False)
    # 构造人口增长率预测模型，使用logistic回归
    model_Pt = LinearRegression()
    pipline_Pt = Pipeline([('poly_feature', model_Pt_poly_features), ('model', model_Pt)])
    # 构造GDP增长率预测模型，使用logistic回归
    model_Gt = LinearRegression()
    pipline_Gt = Pipeline([('poly_feature', model_Gt_poly_features), ('model', model_Gt)])

    # 训练
    pipline_Pt.fit(X, Pt)
    pipline_Gt.fit(X, Gt)

    # 预测，此处X可以重新构造为预测数据集，此处使用的仍是2010～2020的训练数据集
    Pt_predict = pipline_Pt.predict(X)
    Gt_predict = pipline_Gt.predict(X)
    # 用实际增长率与预测增长率计算均方差以评估拟合情况
    mse_Pt = mean_squared_error(Pt, Pt_predict)
    print(mse_Pt)
    mse_Gt = mean_squared_error(Gt, Gt_predict)
    print(mse_Gt)

    # 打印模型权重和偏置，权重值顺序同标准多项式公式中
    print(model_Pt.coef_)
    print(model_Pt.intercept_)
    print(model_Pt.coef_)
    print(model_Pt.intercept_)

    # 构造时间坐标轴
    t = np.linspace(0, 1, num_year)
    i = 0
    while i in range(0, num_year):
        t[i] = start_year + i
        i = i + 1

    # 打印制图，横轴是时间，纵轴是人口总量或GDP
    plt.figure()
    plt.title("population")
    plt.scatter(t, (1+Pt)*P, marker='o', c='r')
    plt.plot(t, (1+Pt_predict)*P, 'g-')
    plt.figure()
    plt.title("GDP")
    plt.scatter(t, (1 + Gt) * G, marker='o', c='r')
    plt.plot(t, (1 + Gt_predict) * G, 'g-')
    plt.show()




if __name__ == '__main__':
    do_cal()
