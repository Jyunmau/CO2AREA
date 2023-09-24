import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


def do_cal(p_o, g_o, energy_o, co2_o, start_year, num_year, pre_num_year):
    # 定义数据集的起始年份和年份数量
    start_year = start_year
    num_year = num_year
    # 定义区域人口总量用于制图（预测），并转秩为列向量
    P_o = p_o
    P_o = P_o.reshape(-1, 1)
    # 定义区域GDP用于制图（预测），并转秩为列向量
    G_o = g_o
    G_o = G_o.reshape(-1, 1)
    # 定义区域能源总量用于制图（预测），并转秩为列向量
    energy_o = energy_o
    energy_o = energy_o.reshape(-1, 1)
    # 定义区域碳排放量用于制图（预测），并转秩为列向量
    co2_o = co2_o
    co2_o = co2_o.reshape(-1, 1)
    # 定义区域人口总量用于训练，因需要匹配增长率的数据个数，该数据比_o少第一位数据，并转秩为列向量，此处可重新定义为所需的训练数据集
    P = np.array(P_o[1:])
    # 定义区域GDP用于训练，因需要匹配增长率的数据个数，该数据比_o少第一位数据，并转秩为列向量，此处可重新定义为所需的训练数据集
    G = np.array(G_o[1:])
    # 定义区域能源消耗量用于训练，因需要匹配增长率的数据个数，该数据比_o少第一位数据，并转秩为列向量，此处可重新定义为所需的训练数据集
    E = np.array(energy_o[1:])
    # 定义区域碳排放量用于训练，因需要匹配增长率的数据个数，该数据比_o少第一位数据，并转秩为列向量，此处可重新定义为所需的训练数据集
    C = np.array(co2_o[1:])

    # 由于要构造人口总量与GDP的乘积以及制图标点需要，年份、人口总量和GDP数据的个数必须相等
    len_p = len(P)
    len_g = len(G)
    len_e = len(E)
    len_c = len(C)
    if len_p != len_g:
        print('error: length of P and G should equal!')
        return
    if len_p != len_e:
        print('error: length of P and E should equal!')
        return
    if len_p != len_c:
        print('error: length of P and C should equal!')
        return
    elif len_p != num_year:
        print('error: length of P and num_year should equal!')
        return

    # 构造需提前运算的特征向量
    # 定义人口总量与GDP的乘积向量, 并做归一化以避免二阶量数值过大对结果产生不力影响
    PG = np.zeros((1, len_p))
    i = 0
    while i in range(0, len_p):
        PG[0][i] = P[i][0] * G[i][0]
        i = i + 1
    minmaxscaler = MinMaxScaler()
    PG = minmaxscaler.fit_transform(PG)
    PG = PG.reshape(-1, 1)
    # 定义GDP的平方向量, 并做归一化以避免二阶量数值过大对结果产生不力影响
    GG = np.zeros((1, len_p))
    i = 0
    while i in range(0, len_p):
        GG[0][i] = G[i][0] * G[i][0]
        i = i + 1
    # GG = minmaxscaler.fit_transform(GG)
    GG = GG.reshape(-1, 1)
    # 定义能源消耗量对数的向量
    ln_e = np.log(E)
    # 定义GDP对数的向量
    ln_g = np.log(G)
    # 定义人口总量对数的向量
    ln_p = np.log(P)
    # 定义GDP对数的向量
    ln_gg = np.log(GG)
    # 定义碳排放量对数的向量
    ln_c = np.log(C)
    # 定义人口总量与能源消耗量的乘积向量
    PE = np.zeros((1, len_p))
    i = 0
    while i in range(0, len_p):
        PE[0][i] = P[i][0] * E[i][0]
        i = i + 1
    PE = minmaxscaler.fit_transform(PE)
    PE = PE.reshape(-1, 1)
    # 构造人口增长率模型训练的特征矩阵，拼接人口总量、GDP、人口总量与GDP乘积的向量，形成一个n行3列的矩阵
    X_pt = np.concatenate((P, G, PG), axis=1)
    # 构造GPD增长率模型训练的特征矩阵，拼接GDP、人口总量与GDP乘积的向量，形成一个n行2列的矩阵
    X_gt = np.concatenate((G, PG), axis=1)
    # 构造碳排放量模型训练的特征矩阵，拼接能源消耗量对数、GDP对数、人口总量对数的向量，形成一个n行3列的矩阵
    X_lnc = np.concatenate((ln_e, ln_g, ln_p), axis=1)
    # 构造能源消耗量模型训练的特征矩阵，拼接能源消耗量、人口总量与能源消耗量乘积的向量，形成一个n行2列的矩阵
    X_et = np.concatenate((E, PE), axis=1)
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
    # 构造de/dt，即能源消耗量增长率，能源消耗量对时间的一阶导，并转秩为列向量
    Et = np.zeros((1, len_g))
    i = 0
    while i in range(0, len_g):
        Et[0][i] = (energy_o[i + 1][0] - energy_o[i][0]) / energy_o[i][0]
        i = i + 1
    Et = Et.reshape(-1, 1)

    # 构造二次多项式模型参数，以下方法中degree控制阶数，include_bias控制是否增加偏置常量
    model_Pt_poly_features = PolynomialFeatures(degree=1, include_bias=False)
    model_Gt_poly_features = PolynomialFeatures(degree=1, include_bias=False)
    model_lnc_poly_features = PolynomialFeatures(degree=1, include_bias=False)
    model_Et_poly_features = PolynomialFeatures(degree=1, include_bias=False)
    # 构造人口增长率预测模型，使用logistic回归
    model_Pt = LinearRegression()
    pipline_Pt = Pipeline([('poly_feature', model_Pt_poly_features), ('model', model_Pt)])
    # 构造GDP增长率预测模型，使用logistic回归
    model_Gt = LinearRegression()
    pipline_Gt = Pipeline([('poly_feature', model_Gt_poly_features), ('model', model_Gt)])
    # 构造碳排放量预测模型，使用logistic回归
    model_lnc = LinearRegression()
    pipline_lnc = Pipeline([('poly_feature', model_lnc_poly_features), ('model', model_lnc)])
    # 构造能源消耗量预测模型，使用logistic回归
    model_Et = LinearRegression()
    pipline_Et = Pipeline([('poly_feature', model_Et_poly_features), ('model', model_Et)])

    # 训练
    pipline_Pt.fit(X_pt, Pt)
    pipline_Gt.fit(X_gt, Gt)
    pipline_lnc.fit(X_lnc, ln_c)
    pipline_Et.fit(X_et, Et)

    # 预测，此处X可以重新构造为预测数据集，此处使用的仍是2010～2020的训练数据集
    Pt_predict = pipline_Pt.predict(X_pt)
    Gt_predict = pipline_Gt.predict(X_gt)
    lnc_predict = pipline_lnc.predict(X_lnc)
    Et_predict = pipline_Et.predict(X_et)

    # 用实际增长率与预测增长率计算均方差以评估拟合情况
    mse_Pt = mean_squared_error(Pt, Pt_predict)
    print('人口增长率预测方差')
    print(mse_Pt)
    mse_Gt = mean_squared_error(Gt, Gt_predict)
    print('GDP增长率预测方差')
    print(mse_Gt)
    mse_lnc = mean_squared_error(ln_c, lnc_predict)
    print('碳排放量预测方差')
    print(mse_lnc)
    mse_Et = mean_squared_error(Et, Et_predict)
    print('能源消耗量预测方差')
    print(mse_Et)

    # 打印模型权重和偏置，权重值顺序同标准多项式公式
    print('人口增长率模型参数，参考多项式w1*P+W2*G+W3*P*G+b，输出权重向量为[w1 w2 w3]，偏置向量为[b]')
    print(model_Pt.coef_)
    print(model_Pt.intercept_)
    print('GDP增长率模型参数，参考多项式W1*G+W2*P*G+b，输出权重向量为[w1 w2]，偏置向量为[b]')
    print(model_Gt.coef_)
    print(model_Gt.intercept_)
    print('碳排放量模型参数，参考多项式W1*LEU+W2*LGDP+W3*LP+b，输出权重向量为[w1 w2 w3]，偏置向量为[b]')
    print(model_lnc.coef_)
    print(model_lnc.intercept_)
    print('能源消耗量模型参数，参考多项式W1*E+W2*P*E+b，输出权重向量为[w1 w2]，偏置向量为[b]')
    print(model_Et.coef_)
    print(model_Et.intercept_)

    # 构造制图用的时间坐标轴
    t = np.linspace(0, 1, num_year)
    i = 0
    while i in range(0, num_year):
        t[i] = start_year + i
        i = i + 1
    t_predict = np.linspace(0, 1, num_year+pre_num_year)
    i = 0
    while i in range(0, num_year+pre_num_year):
        t_predict[i] = start_year + i
        i = i + 1
    # 构造后n年的预测数值
    P_predict_ex = np.zeros((1, pre_num_year+1))
    P_predict_ex[0][0] = P[len_p-1]
    G_predict_ex = np.zeros((1, pre_num_year+1))
    G_predict_ex[0][0] = G[len_p-1]
    lnC_predict_ex = np.zeros((1, pre_num_year + 1))
    lnC_predict_ex[0][0] = ln_c[len_p - 1]
    E_predict_ex = np.zeros((1, pre_num_year + 1))
    E_predict_ex[0][0] = E[len_p - 1]
    Pt_predict_ex = np.zeros((1, pre_num_year))
    Gt_predict_ex = np.zeros((1, pre_num_year))
    Et_predict_ex = np.zeros((1, pre_num_year))
    i = 0
    while i in range(0, pre_num_year):
        Pt_predict_ex[0][i] = model_Pt.coef_[0][0] * P_predict_ex[0][i] + model_Pt.coef_[0][1] * G_predict_ex[0][i] + model_Pt.coef_[0][2] * P_predict_ex[0][i] * G_predict_ex[0][i] + model_Pt.intercept_[0]
        Gt_predict_ex[0][i] = model_Gt.coef_[0][0] * G_predict_ex[0][i] + model_Gt.coef_[0][1] * P_predict_ex[0][i] * G_predict_ex[0][i] + model_Gt.intercept_[0]
        Et_predict_ex[0][i] = model_Et.coef_[0][0] * E_predict_ex[0][i] + model_Et.coef_[0][1] * P_predict_ex[0][i] * E_predict_ex[0][i] + model_Et.intercept_[0]
        P_predict_ex[0][i+1] = (1+Pt_predict_ex[0][i])*P_predict_ex[0][i]
        G_predict_ex[0][i+1] = (1+Gt_predict_ex[0][i])*G_predict_ex[0][i]
        E_predict_ex[0][i+1] = (1+Et_predict_ex[0][i])*E_predict_ex[0][i]
        lnC_predict_ex[0][i+1] = model_lnc.coef_[0][0] * np.log(E_predict_ex[0][i]) + model_lnc.coef_[0][1] * np.log(G_predict_ex[0][i]) + model_lnc.coef_[0][2] * np.log(P_predict_ex[0][i])
        i = i + 1
    P_predict_ex = P_predict_ex.reshape(-1,1)
    P_predict = np.concatenate((P[:-1], P_predict_ex))
    G_predict_ex = G_predict_ex.reshape(-1,1)
    G_predict = np.concatenate((G[:-1], G_predict_ex))
    E_predict_ex = E_predict_ex.reshape(-1, 1)
    E_predict = np.concatenate((E[:-1], E_predict_ex))
    lnC_predict_ex = lnC_predict_ex.reshape(-1, 1)
    lnC_predict = np.concatenate((ln_c[:-1], lnC_predict_ex))


    # 打印制图，横轴是时间，纵轴是人口总量或GDP
    # 人口
    plt.figure()
    plt.title("population")
    plt.scatter(t, (1 + Pt) * P_o[:-1], marker='o', c='r')
    plt.plot(t_predict, P_predict[:], 'g-')
    plt.draw()
    # GDP
    plt.figure()
    plt.title("GDP")
    plt.scatter(t, (1 + Gt) * G_o[:-1], marker='o', c='r')
    plt.plot(t_predict, G_predict[:], 'g-')
    plt.draw()
    # 碳排放
    plt.figure()
    plt.title("CO2")
    plt.scatter(t, np.exp(ln_c), marker='o', c='r')
    plt.plot(t_predict, np.exp(lnC_predict), 'g-')
    plt.draw()
    # 能源
    plt.figure()
    plt.title("energy")
    plt.scatter(t, (1 + Et) * energy_o[:-1], marker='o', c='r')
    plt.plot(t_predict, E_predict[:], 'g-')
    plt.draw()

    plt.show()
