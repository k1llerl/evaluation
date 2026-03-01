import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bottleneck as bn
from tqdm import tqdm
from factor.operations import *
from datetime import datetime

"""
# 提供了较多函数，比较灵活，平时可根据自己需要去写一些额外的回测结果, 公司统一按目前的get_factor_stats获取因子统计值
# 目前所有年化均是基于输入因子和收益率为日频数据
lym 240514 添加因子覆盖度
"""
#################################################
# 公司统一提交使用如下参数
#     num_layers = 10
#     rankIC_method = "spearman"
#     return_method = "minus_sum"
#################################################

# IC相关
# method: str, "spearman":rankIC   "pearson":IC

# get_ic_stats: 获取因子的截面IC序列统计量                    :return: pd.Series, "因子方向" , "IC平均值" "IC标准差" "ICIR" "IC胜率"
# get_ic_series: 获取因子的截面IC序列                        :return: pd.Series,截面IC序列
# get_ic_stats_by_series： 按截面IC序列获取因子IC的统计值      :return: pd.Series, "因子方向" , "IC平均值" "IC标准差" "ICIR" "IC胜率"

# 分层相关
# method: "minus_sum": （每日收益率-基准收益率）再累加  "minus_prod" (每日收益率-基准收益率)再加1累乘再减1
#         "prod_div": 加1累乘收益率 再除 加1累乘基准收益率再减1

# get_layer_return:  返回num_layers因子分组的每日收益率           :return: pd.DataFrame 因子分组的每日收益率
# get_layer_ex_cum_return:  返回num_layers因子分组的超额累计收益率 :return: pd.DataFrame 因子分组的超额累计收益率
# get_layer_ex_return:  返回num_layers因子分组的每日超额收益率     :return: pd.DataFrame 因子分组的超额收益率
# get_layer_turnover:  返回num_layers因子分组的每日换手率          :return: pd.DataFrame 因子分组的每日换手率
# get_group_return_stats: 返回num_layers因子分组收益的统计量       :return: pd.Series 因子分组收益的统计量

# 其他
# get_factor_stats:  返回因子的所有回测的结果  :return: pd.Series 因子分组收益的统计量
# return_plot:  对分组每日收益率绘图并保存

################################################################################################
# get_factor_stats返回回测结果解释
# factor_name：因子名 direction：因子IC正负 meanIC：截面IC日度均值 stdIC：IC日度标准差 ICIR：年化ICIR winRateIC：IC胜率
# t_Return：多头端因子值最大组年化收益 t_maxDraw：回测时间段内最大回撤 t_Sharpe：年化夏普比 t_Turnover：日换手率均值
# b_Return：多头端分组收益最大组年化收益 b_maxDraw：回测时间段内最大回撤 b_Sharpe：年化夏普比 b_Turnover：日换手率均值
# bestGroup:多头端分组收益最大组组别 isMono：因子端日均收益是否单调
################################################################################################


dayInYear = 250


# fee = 5e-4

def get_coverage_ratio(fac, stock_returns):
    """
    计算因子的平均覆盖度
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)
    close = stock_returns
    fac = fac.replace([np.inf, -np.inf], np.nan)
    x = fac.isnull().sum(axis=1)
    y = close.isnull().sum(axis=1)
    coverage = (x - y) / (~close.isnull()).sum(axis=1)
    coverage = coverage.replace([np.inf, -np.inf], np.nan)
    coverage = coverage.dropna(how = "all")
    coverage = coverage.mean()
    return 1 - coverage


def get_ic_stats(fac, stock_returns, method="spearman"):
    """
    获取因子的截面IC序列统计量
    :param fac: pd.DataFrame, 因子值
    :param stock_returns: pd.DataFrame, 对应未来收益率
    :param method: str, "spearman":rankIC   "pearson":IC
    :return: pd.Series, "因子方向" , "IC平均值" "IC标准差" "ICIR" "IC胜率"
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)

    icSeries = get_ic_series(fac, stock_returns, method)
    icStats = get_ic_stats_by_series(icSeries)
    return icStats


def get_ic_series(fac, stock_returns, method="spearman"):
    """
    获取因子的截面IC序列
    :param fac: pd.DataFrame, 因子值
    :param stock_returns: pd.DataFrame, 对应未来收益率
    :param method: str, "spearman":rankIC   "pearson":IC
    :return: pd.Series,截面IC序列
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)

    icSeries = fac.corrwith(stock_returns, method=method, axis=1)
    return icSeries


def get_ic_stats_by_series(icSeries):
    """
    按截面IC序列获取因子IC的统计值
    :param icSeries: pd.Series,截面IC序列
    :return: pd.Series, "因子方向" , "IC平均值" "IC标准差" "ICIR" "IC胜率"
    """
    mean_ = np.nanmean(icSeries)
    res = {"direction": np.sign(mean_), "meanIC": mean_, "stdIC": np.nanstd(icSeries)}
    res["ICIR"] = res["meanIC"] / res["stdIC"] * np.sqrt(dayInYear)
    if res["meanIC"] > 0:
        res["winRateIC"] = len(np.array(icSeries)[np.where(np.array(icSeries) > 0)]) / len(icSeries)
    elif res["meanIC"] <= 0:
        res["winRateIC"] = len(np.array(icSeries)[np.where(np.array(icSeries) <= 0)]) / len(icSeries)

    return pd.Series(res)


def get_layer_return(fac, stock_returns, num_layers=10):
    """
    返回num_layers因子分组的每日收益率
    :param fac: df，测试的因子
    :param stock_returns: df, 对应未来收益率
    :param num_layers: 因子分层数
    :return: pd.DataFrame 因子分组的每日收益率
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)

    fac_rank = ((fac.rank(axis=1).div(((fac.count(axis=1) + 1) / num_layers), axis=0)).fillna(-1).astype(int).
                replace(-1, np.nan).astype(float))
    lst = []
    for i in range(num_layers):
        df1 = (fac_rank == i)
        tem = stock_returns[df1].mean(axis=1)
        # tem = (df1 * stock_returns).mean(axis=1)
        lst.append(tem)
    group_returns = pd.concat(lst, axis=1)
    group_returns.columns = [int(x) + 1 for x in group_returns.columns]
    if group_returns.dropna(how="all").isnull().sum().sum() / len(group_returns) >= 0.05:
        raise ValueError("This factor can not be grouped well. Please check if this factor has too many same values "
                         "at same time.")
    return group_returns


def get_layer_ex_cum_return(fac, stock_returns, num_layers=10, method="minus_sum"):
    """
    返回num_layers因子分组的超额累计收益率
    :param fac: df，测试的因子
    :param stock_returns: df, 对应未来收益率
    :param num_layers: 因子分层数
    :param method: "minus_sum": （每日收益率-基准收益率）再累加  "minus_prod" (每日收益率-基准收益率)再加1累乘再减1
                   "prod_div": 加1累乘收益率 再除 加1累乘基准收益率再减1
    :return: 因子分组的超额累计收益率
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)

    errMsg = "请检查method为以下选项之一\n 'minus_sum': (每日收益率 - 基准收益率)再累加 'minus_prod': (每日收益率 - 基准收益率)再加1累乘再减1'prod_div': 加1累乘收益率再除加1累乘基准收益率再减1"
    if method == "minus_sum":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        group_returns = group_returns.sub(bn.nanmean(group_returns, axis=1), axis=0)
        return group_returns.cumsum() + 1

    elif method == "minus_prod":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        group_returns = group_returns.sub(bn.nanmean(group_returns, axis=1), axis=0)
        return (group_returns + 1).cumprod()

    elif method == "prod_div":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        bench_returns = group_returns.mean(axis=1)
        return ((group_returns + 1).cumprod()).div((bench_returns + 1).cumprod(), axis=0)

    else:
        raise ValueError(errMsg)


def get_layer_ex_return(fac, stock_returns, num_layers=10, method="minus_sum"):
    """
    返回num_layers因子分组的每日超额收益率
    :param fac: df，测试的因子
    :param stock_returns: df, 对应未来收益率
    :param num_layers: 因子分层数
    :param method: "minus_sum": （每日收益率-基准收益率）再累加  "minus_prod" (每日收益率-基准收益率)再加1累乘再减1
                   "prod_div": 加1累乘收益率 再除 加1累乘基准收益率再减1
    :return: pd.DataFrame 因子分组的超额收益率
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)

    errMsg = "请检查method为以下选项之一\n 'minus_sum': (每日收益率 - 基准收益率)再累加 'minus_prod': (每日收益率 - 基准收益率)再加1累乘再减1  'prod_div': 加1累乘收益率再除加1累乘基准收益率再减1"

    if method == "minus_sum":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        group_returns = group_returns.sub(bn.nanmean(group_returns, axis=1), axis=0)
        return group_returns

    elif method == "minus_prod":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        group_returns = group_returns.sub(bn.nanmean(group_returns, axis=1), axis=0)
        return group_returns

    elif method == "prod_div":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        bench_returns = group_returns.mean(axis=1)
        return ((group_returns + 1).cumprod()).div((bench_returns + 1).cumprod(), axis=0).pct_change()
    else:
        raise ValueError(errMsg)

def get_layer_num(fac, num_layers=10):
    fac_rank = (fac.rank(axis=1).div(((fac.count(axis=1) + 1) / num_layers), axis=0)
                ).fillna(-1).astype(int).replace(-1, np.nan).astype(float)
    return fac_rank
def get_layer_turnover(fac, num_layers=10):
    """
    返回num_layers因子分组的每日换手率
    :param fac: df，测试的因子
    :param num_layers: 因子分层数
    :return: pd.DataFrame 因子分组的每日换手率
    """

    fac_rank = (fac.rank(axis=1).div(((fac.count(axis=1) + 1) / num_layers), axis=0)
                ).fillna(-1).astype(int).replace(-1, np.nan).astype(float)
    lst = []
    for i in range(num_layers):
        df1 = (fac_rank == i)
        df2 = (fac_rank.shift(1) == i)
        fac_rank_turnover = 1 - ((df2 & df1).sum(axis=1)) / df2.sum(axis=1)
        lst.append(fac_rank_turnover)
    turnover = pd.concat(lst, axis=1)
    turnover.columns = [int(x) + 1 for x in turnover.columns]
    return turnover


def get_group_return_stats(fac, stock_returns, direction, num_layers=10, method="minus_sum"):
    """
    返回num_layers因子分组收益的统计量
    :param fac: df，测试的因子
    :param stock_returns: df, 对应未来收益率
    :param direction: -1 or 1
    :param num_layers: 因子分层数
    :param method: "minus_sum": （每日收益率-基准收益率）再累加  "minus_prod" (每日收益率-基准收益率)再加1累乘再减1
                   "prod_div": 加1累乘收益率 再除 加1累乘基准收益率再减1
    :return: pd.Series 因子分组收益的统计量
    """

    def monotonic(mean_bp, num_layers):
        des = mean_bp.diff().dropna()
        des = np.where(des >= 0, 1, -1)
        sig = des.sum()
        if (sig == num_layers - 1 or sig == -num_layers + 1):
            sig = 1
        else:
            sig = 0
        return sig

    # 定义函数计算最大回撤
    def max_drawdown(cum_returns):
        cum_roll_max = cum_returns.cummax()
        drawdown = cum_roll_max - cum_returns
        max_drawdown = drawdown.max()
        return max_drawdown

    # 定义函数计算年化夏普比
    def annualized_sharpe_ratio(returns, annualziedReturn):
        std = returns.std() * np.sqrt(dayInYear)
        return annualziedReturn / std

    errMsg = "请检查method为以下选项之一\n 'minus_sum': (每日收益率 - 基准收益率)再累加 'minus_prod': (每日收益率 - 基准收益率)再加1累乘再减1  'prod_div': 加1累乘收益率再除加1累乘基准收益率再减1"

    if method == "minus_sum":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        group_returns_daily = group_returns.sub(bn.nanmean(group_returns, axis=1), axis=0)
        group_returns = group_returns_daily.cumsum() + 1
    elif method == "minus_prod":
        group_returns = get_layer_return(fac, stock_returns, num_layers=num_layers)
        group_returns_daily = group_returns.sub(bn.nanmean(group_returns, axis=1), axis=0)
        group_returns = (group_returns_daily + 1).cumprod()

    elif method == "prod_div":
        group_returns_daily = get_layer_return(fac, stock_returns, num_layers=num_layers)
        bench_returns = group_returns_daily.mean(axis=1)
        group_returns = ((group_returns_daily + 1).cumprod()).div((bench_returns + 1).cumprod(), axis=0)
        group_returns_daily = group_returns.pct_change()
    else:
        raise ValueError(errMsg)

    turn_over = get_layer_turnover(fac, num_layers=num_layers)
    turn_over_mean = turn_over.iloc[:-1].mean()

    mean_bp = group_returns_daily.sum() / len(group_returns)
    res = {}

    group1_returns = group_returns[group_returns.columns[0]].dropna().values[-1]
    group2_returns = group_returns[group_returns.columns[-1]].dropna().values[-1]

    if group1_returns >= group2_returns:
        topReturns = group_returns[group_returns.columns[0]]
        topReturns_daily = group_returns_daily[group_returns.columns[0]]

    else:
        topReturns = group_returns[group_returns.columns[-1]]
        topReturns_daily = group_returns_daily[group_returns.columns[-1]]

    # res["t_totalReturn"] = topReturns.dropna().values[-1] - 1
    if method == "minus_sum":
        res["t_Return"] = (topReturns.dropna().values[-1] - 1) / len(topReturns) * dayInYear
    elif method == "minus_prod":
        res["t_Return"] = (topReturns.dropna().values[-1]) ** (dayInYear / len(topReturns)) - 1
    elif method == "prod_div":
        res["t_Return"] = (topReturns.dropna().values[-1]) ** (dayInYear / len(topReturns)) - 1
    res["t_maxDraw"] = max_drawdown(topReturns)
    res["t_Sharpe"] = annualized_sharpe_ratio(topReturns_daily, res["t_Return"])
    if direction == -1:
        res["t_Turnover"] = turn_over_mean.iloc[0]
    elif direction == 1:
        res["t_Turnover"] = turn_over_mean.iloc[-1]
    bestGroup = mean_bp.idxmax()
    bestReturns = group_returns[bestGroup]
    bestReturns_daily = group_returns_daily[bestGroup]

    # res["b_totalReturn"] = (bestReturns.dropna().values[-1] - 1)
    if method == "minus_sum":
        res["b_Return"] = (bestReturns.dropna().values[-1] - 1) / len(bestReturns) * dayInYear
    elif method == "minus_prod":
        res["b_Return"] = (bestReturns.dropna().values[-1]) ** (dayInYear / len(bestReturns)) - 1
    elif method == "prod_div":
        res["b_Return"] = (bestReturns.dropna().values[-1]) ** (dayInYear / len(bestReturns)) - 1

    res["b_maxDraw"] = max_drawdown(bestReturns)
    res["b_Sharpe"] = annualized_sharpe_ratio(bestReturns_daily, res["b_Return"])
    res["b_Turnover"] = turn_over_mean[bestGroup]
    res["bestGroup"] = bestGroup

    res["isMono"] = monotonic(mean_bp, num_layers)

    res["coverage"] = get_coverage_ratio(fac, stock_returns)

    return pd.Series(res)


def get_factor_stats(fac, stock_returns, num_layers=10, rankIC_method="spearman", return_method="minus_sum"):
    """
    因子的所有回测的结果
    :param fac: df，测试的因子
    :param stock_returns: df, 对应未来收益率
    :param num_layers: 因子分层数
    :param return_method: "spearman" or "pearson"
    :param rankIC_method: "minus_sum": （每日收益率-基准收益率）再累加  "minus_prod" (每日收益率-基准收益率)再加1累乘再减1
                   "prod_div": 加1累乘收益率 再除 加1累乘基准收益率再减1
    :return: 该因子的所有回测的结果
    """
    fac = fac.dropna(how = "all")
    stock_returns = stock_returns.reindex(columns = fac.columns, index = fac.index)


    icStats = get_ic_stats(fac, stock_returns, method=rankIC_method)
    direction = icStats["direction"]
    layer_result = get_group_return_stats(fac, stock_returns, direction, num_layers=num_layers, method=return_method)

    return pd.concat([icStats, layer_result])


def return_plot(daily_return, save_path=None):
    """
    对分组每日收益率绘图
    :param daily_return: df, 分组每日收益率绘图
    :param save_path: str or None, 图片保存路径
    """
    log_word = "(sum)"
    net_log =  ((daily_return ).cumsum())*100
    cum_return = (daily_return + 1).cumprod()
    fig, axs = plt.subplots(4, 1, figsize=(8, 16))
    axs[0].bar(daily_return.columns, 10000 * daily_return.mean(), width=2.5 / len(daily_return.columns))
    axs[0].set_title("Mean Return By Factor Quantile")
    axs[0].set_ylabel("bps")
    # 添加刻度线
    axs[0].grid(linestyle='--', color='gray')
    for i in range(len(cum_return.columns)):
        axs[1].plot(cum_return[cum_return.columns[i]], label=cum_return.columns[i])
    axs[1].legend(loc="best")
    axs[1].grid(linestyle='--', color='gray')
    axs[1].set_title(f"Cumulative (prod) Return By Quantile")
    axs[1].set_ylabel("PnL")

    for i in range(len(net_log.columns)):
        axs[2].plot(net_log[net_log.columns[i]], label=net_log.columns[i])
    axs[2].legend(loc="best")
    axs[2].grid(linestyle='--', color='gray')
    axs[2].set_title(f"Cumulative{log_word} Return By Quantile")
    axs[2].set_ylabel("Percent")

    axs[3].plot((daily_return[5] - daily_return[1]).cumsum()*100)
    axs[3].grid(linestyle='--', color='gray')
    axs[3].set_title(f"Long/Short portfolio (sum) Cumulative Return")
    # 调整子图之间的间距
    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    #######################################################################################
    # 公司统一提交使用如下参数
    num_layers = 5
    rankIC_method = "spearman"
    return_method = "minus_sum"

    #######################################################################################
    # 以测单个因子为例

    #######################################################################################
    # 读数据，确保得到fac因子值和对应的未来收益率 一一 对应
    price_volumns_path = "E:\ds\data_d"
    d = {
          "close": pd.read_parquet(os.path.join(price_volumns_path, "close.pq")),

          "up_down_limit": pd.read_parquet(os.path.join(price_volumns_path, "up_down_limit.pq")),

          "size": np.log(pd.read_parquet(os.path.join(price_volumns_path, "market_cap_2.pq")))

          }
    close = d["close"]
    up_down_limit = d["up_down_limit"]
    # up_st = pd.read_parquet("\\\\BF-202309011640\\ds\\data_d\\up_down_limit.pq")
    factor = pd.read_parquet(r"\\BF-202309011640\ds\data_model\xgb.pq")
    factor = factor.replace([np.inf, -np.inf], np.nan)

    stock_returns = close.pct_change().shift(-2)
    stock_returns = stock_returns.reindex(index=factor.index, columns=factor.columns)

    # factor = factor.reindex(index=size.index, columns=size.columns)
    # factor = ols(factor, size)
    factor = factor * up_down_limit
    #######################################################################################

    fac_stats = get_factor_stats(factor, stock_returns, num_layers=num_layers, rankIC_method = rankIC_method, return_method = return_method)

    print(fac_stats)  # 公司所需的单因子的回测结果
    save_path = "E:\\factors\\单因子回测数据"
    fac_stats.to_excel(os.path.join(save_path, "单因子回测.xlsx"))
    ########################################################################################
    ex_return_df = get_layer_ex_return(factor, stock_returns, num_layers=num_layers, method=return_method)
    return_plot(ex_return_df, os.path.join(save_path, "超额收益图.png"))
    return_df = get_layer_return(factor, stock_returns, num_layers=num_layers)
    return_plot(return_df, os.path.join(save_path, "绝对收益图.png"))

    #######################################################################################
    # 以批量测试为例

    #######################################################################################
    # def list_files(directory):
    #     file_list = []
    #     fac_file_list = []
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             file_list.append(os.path.join(root, file))
    #             fac_file_list.append(file[:-3])
    #     return file_list, fac_file_list

    # close = pd.read_parquet("\\\\BF-202309011640\\ds\\data_d\\close.pq")
    # up_st = pd.read_parquet("\\\\BF-202309011640\\ds\\data_d\\up_down_limit.pq")
    # stock_returns = close.pct_change().shift(-2)

    # price_volumns_path = "D:\stocks\ds\data_d"
    # d = {
    #     "close": pd.read_parquet(os.path.join(price_volumns_path, "close.pq")),
    #
    #     "up_down_limit": pd.read_parquet(os.path.join(price_volumns_path, "up_down_limit.pq")),
    #
    #     "size": np.log(pd.read_parquet(os.path.join(price_volumns_path, "market_cap_2.pq")))
    #
    # }
    # close = d["close"]
    # up_down_limit = d["up_down_limit"]
    # path = r"D:\stocks\ds\data_dataset\zz500.xlsx"
    # stock_returns = close.pct_change().shift(-2)
    # size = d["size"]
    # factor_list = pd.read_excel(path)["zz_500"].tolist()
    # dic = {}
    # for fac in tqdm(factor_list):
    #     factor = pd.read_parquet(os.path.join(price_volumns_path, f"{fac}.pq"))
    #     factor = factor.replace([np.inf, -np.inf], np.nan)
    #     stock_returns_reindex = stock_returns.reindex(index=factor.index, columns=factor.columns)
    #
    #     # 计算换手率
    #     # 1. 原始因子
    #     factor_raw = factor * up_down_limit
    #     fac_stats = get_factor_stats(factor_raw, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_raw"] = fac_stats
    #
    #     factor_mean = ts_mean(factor, 20)
    #     factor_mean = factor_mean * up_down_limit
    #     fac_stats = get_factor_stats(factor_mean, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_raw_mean"] = fac_stats
    #
    #     factor_ewm = ts_ewm(factor, 20)
    #     factor_ewm = factor_ewm * up_down_limit
    #     fac_stats = get_factor_stats(factor_ewm, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_raw_ewm"] = fac_stats
    #
    #     factor_wma = ts_wma(factor, 20)
    #     factor_wma = factor_wma * up_down_limit
    #     fac_stats = get_factor_stats(factor_wma, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_raw_wma"] = fac_stats
    #
    #     # 2. 市值中性化后的因子
    #
    #     factor = factor.reindex(index=size.index, columns=size.columns)
    #     factor = factor.replace([np.inf, -np.inf], np.nan)
    #
    #     factor = ols(factor, size)
    #     factor_raw = factor * up_down_limit
    #     fac_stats = get_factor_stats(factor_raw, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_cap"] = fac_stats
    #
    #     factor_mean = ts_mean(factor, 20)
    #     factor_mean = factor_mean * up_down_limit
    #     fac_stats = get_factor_stats(factor_mean, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_cap_mean"] = fac_stats
    #
    #     factor_ewm = ts_ewm(factor, 20)
    #     factor_ewm = factor_ewm * up_down_limit
    #     fac_stats = get_factor_stats(factor_ewm, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_cap_ewm"] = fac_stats
    #
    #     factor_wma = ts_wma(factor, 20)
    #     factor_wma = factor_wma * up_down_limit
    #     fac_stats = get_factor_stats(factor_wma, stock_returns, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                  return_method=return_method)
    #     dic[f"{fac}_cap_wma"] = fac_stats
    #
    # out_put = pd.DataFrame(dic).T
    # today = datetime.now().strftime("%Y-%m-%d")
    # out_put.to_excel(rf"D:\stocks\ds\data_output\{today}_zz500.xlsx")

    ######################################################################################
    # 以批量测试为例

    ######################################################################################
    # def list_files(directory):
    #     file_list = []
    #     fac_file_list = []
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             file_list.append(os.path.join(root, file))
    #             fac_file_list.append(file[:-3])
    #     return file_list, fac_file_list
    #
    # close = pd.read_parquet("\\\\BF-202309011640\\ds\\data_d\\close.pq")
    # up_st = pd.read_parquet("\\\\BF-202309011640\\ds\\data_d\\up_down_limit.pq")
    # stock_returns = close.pct_change().shift(-2)
    #
    # factor_path ="\\\\BF-202309011640\\ds\\factor_wzx_eff"
    # save_path = "E:\\factors\\单因子回测数据"
    #
    # file_list, fac_file_list = list_files(factor_path)
    # stats_list = []
    #
    # for IF, fname, in enumerate(tqdm(fac_file_list,desc="因子测试中")):
    #     try:
    #         fac = pd.read_parquet(file_list[IF])
    #         fac = fac.dropna(how="all")
    #         fac = fac.replace([-np.inf, np.inf], np.nan)
    #         stock_returns_reindex = stock_returns.reindex(index=fac.index, columns=fac.columns)
    #         up_st_reindex = up_st.reindex(index=fac.index, columns=fac.columns)
    #         fac = fac * up_st_reindex
    #         fac_stats = get_factor_stats(fac, stock_returns_reindex, num_layers=num_layers, rankIC_method=rankIC_method,
    #                                      return_method=return_method)
    #         fac_stats.name = fname
    #         stats_list.append(fac_stats)
    #     except Exception as e:
    #         print(f"Got error when calculating {fname} with error msg {e}")
    #         continue
    #
    # fac_stats_all = pd.concat(stats_list, axis=1).T
    # fac_stats_all.index.name = "factor_name"
    # fac_stats_all.reset_index().to_excel(os.path.join(save_path, "因子回测结果.xlsx"))

    ###################################################################################################################
