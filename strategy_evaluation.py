import os
import sys
# sys.path.append("../../")
import pandas as pd
from quant_utils.utility import load_json
import numpy as np
import matplotlib.pyplot as plt

"""
240425 添加了跟踪误差
"""

class StrategyEvaluator:
    def __init__(self, rets, bench_rets, leverage=0.8, excess_method="minus_sum",name=None):
        """

        :param rets: pd.Series: daily returns
        :param bench_rets: pd.Series: Benchmark rets
        :param leverage: leverage of returns
        :param excess_method: "minus_sum" "minus_prod" "prod_div"
        """
        self.name=name
        if not (isinstance(rets, pd.Series) and isinstance(rets.index, pd.core.indexes.datetimes.DatetimeIndex)):
            raise TypeError("Please make sure your input 'rets' is a pd.Series with pd.DatetimeIndex.")
        if not (isinstance(bench_rets, pd.Series) and isinstance(bench_rets.index,
                                                                 pd.core.indexes.datetimes.DatetimeIndex)):
            raise TypeError("Please make sure your input 'bench_rets' is a pd.Series with pd.DatetimeIndex.")
        if isinstance(leverage, int):
            leverage = float(leverage)
        if not (isinstance(leverage, float) and (leverage > 0)):
            raise TypeError("Please make sure your input 'leverage' is a float with value >0.")
        if not (isinstance(excess_method, str) and (excess_method in ['minus_sum', 'minus_prod', 'prod_div'])):
            raise TypeError("Please make sure your input 'excess_method' is a string within " +
                            "['minus_sum','minus_prod','prod_div'].")
        if len(rets)>len(bench_rets):
            rets = rets.reindex(index=bench_rets.index)

        else:
            bench_rets = bench_rets.reindex(index = rets.index)
        bench_rets = bench_rets.reindex(index=rets.index) # Todo lym
        self.dayInYear = 250
        self.result = None
        self.year_result = None
        # 加载数据
        self.rets = rets
        self.bench_rets = bench_rets
        self.excess_method = excess_method
        self.leverage = leverage
        (self.excess_rets, self.cum_rets, self.cum_bench_rets,
         self.cum_excess_rets) = self.__get_cum_returns(rets, bench_rets, leverage=leverage,
                                                        excess_method=excess_method)

    def get_return(self, excess=True):
        """
        获取年化收益
        :param excess: 超额或绝对收益
        :return:
        """
        if excess:
            rets = (self.cum_excess_rets.dropna().values[-1] - 1) / len(self.cum_excess_rets) * self.dayInYear
        else:
            rets = (self.cum_rets.dropna().values[-1] - 1) / len(self.cum_rets) * self.dayInYear
        return rets

    def get_sharpe(self, excess=True):
        """
        获取年化夏普
        :param excess: 超额或绝对收益
        :return:
        """
        if excess:
            rets = (self.cum_excess_rets.dropna().values[-1] - 1) / len(self.cum_excess_rets) * self.dayInYear
            std = self.excess_rets.std() * np.sqrt(self.dayInYear)
            sharpe = rets / std
        else:
            rets = (self.cum_rets.dropna().values[-1] - 1) / len(self.cum_rets) * self.dayInYear
            std = self.rets.std() * np.sqrt(self.dayInYear)
            sharpe = rets / std
        return sharpe

    def get_std(self,excess = True):
        if excess:
            std = self.excess_rets.std() * np.sqrt(self.dayInYear)
        else:
            std = self.rets.std() * np.sqrt(self.dayInYear)
        return std
    def get_maxDrawDown(self, excess=True):
        """
        获取最大回撤
        :param excess: 超额或绝对收益
        :return:
        """
        if excess:
            cum_returns = self.cum_excess_rets
        else:
            cum_returns = self.cum_rets
        cum_roll_max = cum_returns.cummax()
        drawdown = cum_roll_max - cum_returns
        max_drawdown = drawdown.max()
        return max_drawdown

    def get_maxDrawDown_time(self, excess=True):
        """
        获取最大回撤时间
        :param excess: 超额或绝对收益
        :return:
        """
        if excess:
            cum_returns = self.cum_excess_rets
        else:
            cum_returns = self.cum_rets

        cum_roll_max = cum_returns.cummax()
        drawdown = cum_roll_max - cum_returns
        pos = drawdown.argmax()
        end_time = cum_returns.index[pos].strftime("%Y-%m-%d")
        start_time = cum_returns.index[cum_returns[np.arange(len(cum_returns)) < pos].argmax()].strftime("%Y-%m-%d")
        return start_time, end_time

    def get_return_by_year(self, excess=True):
        """
        获取每年的年化收益
        :param excess:
        :return:
        """
        return_dic = {}
        if excess:
            returns = self.cum_excess_rets
        else:
            returns = self.cum_rets
        grouped_by_year = returns.groupby(returns.index.year)
        for year, group in grouped_by_year:
            if self.excess_method == "minus_sum":
                return_dic[str(year) + "_return"] = (group.dropna().values[-1] - group.dropna().values[0]) / len(
                    group) * self.dayInYear
            else:
                return_dic[str(year) + "_return"] = (group.dropna().values[-1] / group.dropna().values[0]) ** (
                        self.dayInYear / len(group)) - 1
        return pd.Series(return_dic)

    def get_result(self):
        """
        获取策略表现评价
        :return:
        """
        if self.result is None:
            self.result = {"return": self.get_return(False), "std":self.get_std(False),"sharpe": self.get_sharpe(False),
                           "maxDrawDown": self.get_maxDrawDown(False),
                           "startDrawDown": (self.get_maxDrawDown_time(False))[0],
                           "endDrawDown": (self.get_maxDrawDown_time(False))[1]}
            return_dic = self.get_return_by_year(False)

            ex_result = {"ex_return": self.get_return(True), "ex_std":self.get_std(True),
                         "ex_sharpe": self.get_sharpe(True),
                         "ex_maxDrawDown": self.get_maxDrawDown(True),
                         "ex_startDrawDown": (self.get_maxDrawDown_time(True))[0],
                         "ex_endDrawDown": (self.get_maxDrawDown_time(True))[1]}

            self.result.update(ex_result)
            self.result.update(return_dic.to_dict())
            self.result = pd.Series(self.result)
        return self.result

    def get_result_by_year(self):
        """
        分年获取策略表现评价
        :param excess:
        :return:
        """

        def _get_maxDrawDown_time(cum_rets):
            cum_roll_max = cum_rets.cummax()
            drawdown = cum_roll_max - cum_rets
            pos = drawdown.argmax()
            max_drawdown = drawdown.iloc[pos]
            end_time = cum_rets.index[pos].strftime("%Y%m%d")
            start_time = cum_returns.index[cum_returns[np.arange(len(cum_returns)) < pos].argmax()].strftime("%Y%m%d")
            return max_drawdown, start_time, end_time

        # 获取returns数据
        cum_excess_returns = self.cum_excess_rets
        excess_returns = self.excess_rets
        cum_returns = self.cum_rets
        returns = self.rets

        groups = returns.groupby(returns.index.year).groups

        # 存储结果
        results = pd.DataFrame([], index=["return", "std", "sharpe", "maxDrawDown", "startDrawDown", "endDrawDown",
                                          "ex_return", "ex_std", "ex_sharpe", "ex_maxDrawDown",
                                          "ex_startDrawDown", "ex_endDrawDown"])
        # 计算总体结果
        all_result = {}

        if self.excess_method == "minus_sum":
            all_result["return"] = (cum_returns.dropna().values[-1] - cum_returns.dropna().values[0]) / len(
                cum_returns) * self.dayInYear
        else:
            all_result["return"] = (cum_returns.dropna().values[-1] / cum_returns.dropna().values[0]) ** (
                    self.dayInYear / len(cum_returns)) - 1
        std = returns.std() * np.sqrt(self.dayInYear)
        all_result["std"] = std
        all_result["sharpe"] = all_result["return"] / std

        all_result["maxDrawDown"], all_result["startDrawDown"], all_result["endDrawDown"] = _get_maxDrawDown_time(
            cum_returns)

        if self.excess_method == "minus_sum":
            all_result["ex_return"] = (
                    (cum_excess_returns.dropna().values[-1] - cum_excess_returns.dropna().values[0]) /
                    len(cum_excess_returns) * self.dayInYear)
        else:
            all_result["ex_return"] = (cum_excess_returns.dropna().values[-1] / cum_excess_returns.dropna().values[0]
                                       ) ** (self.dayInYear / len(cum_excess_returns)) - 1
        std = excess_returns.std() * np.sqrt(self.dayInYear)
        all_result["ex_std"] = std
        all_result["ex_sharpe"] = all_result["ex_return"] / std

        all_result["ex_maxDrawDown"], all_result["ex_startDrawDown"], all_result["ex_endDrawDown"] = (
            _get_maxDrawDown_time(cum_excess_returns))
        results["all"] = pd.Series(all_result)

        # 计算分年结果
        for year, index in groups.items():
            year_result = {}
            year_returns = returns[index]
            year_cum_returns = cum_returns[index]
            year_excess_returns = excess_returns[index]
            year_cum_excess_returns = cum_excess_returns[index]
            if self.excess_method == "minus_sum":
                year_result["return"] = (year_cum_returns.dropna().values[-1] - year_cum_returns.dropna().values[
                    0]) / len(year_cum_returns) * self.dayInYear
            else:
                year_result["return"] = (year_cum_returns.dropna().values[-1] / year_cum_returns.dropna().values[
                    0]) ** (self.dayInYear / len(year_cum_returns)) - 1
            std = year_returns.std() * np.sqrt(self.dayInYear)
            year_result["sharpe"] = year_result["return"] / std

            year_result["maxDrawDown"], year_result["startDrawDown"], year_result[
                "endDrawDown"] = _get_maxDrawDown_time(year_cum_returns)

            if self.excess_method == "minus_sum":
                year_result["ex_return"] = ((year_cum_excess_returns.dropna().values[-1] -
                                             year_cum_excess_returns.dropna().values[0]) / len(
                    year_cum_excess_returns) *
                                            self.dayInYear)
            else:
                year_result["ex_return"] = ((year_cum_excess_returns.dropna().values[-1] /
                                             year_cum_excess_returns.dropna().values[0]) **
                                            (self.dayInYear / len(year_cum_excess_returns)) - 1)
            std = year_excess_returns.std() * np.sqrt(self.dayInYear)
            year_result["ex_sharpe"] = year_result["ex_return"] / std

            year_result["ex_maxDrawDown"], year_result["ex_startDrawDown"], year_result["ex_endDrawDown"] = (
                _get_maxDrawDown_time(year_cum_excess_returns))

            results[year] = pd.Series(year_result)

        self.year_result = results

        return results

    def save_result(self, path):
        """保存回测结果"""
        if self.result is None:
            self.result = self.get_result()
        if self.year_result is None:
            self.year_result = self.get_result_by_year()

        self.result.to_excel(os.path.join(path, f"result_{self.name}.xlsx"))
        self.year_result.to_excel(os.path.join(path, f"result_by_year_{self.name}.xlsx"))

    def plot(self, path=None):
        """画图"""
        from matplotlib.gridspec import GridSpec
        def draw_table(cell_data, ax, cell_colors=None):
            table = ax.table(cellText=cell_data, fontsize=8, edges='open',
                             cellLoc='center', loc='upper center')
            table.auto_set_font_size(False)
            table.set_fontsize(14)
            table.scale(1.2, 1.2)  # 控制表格的大小

        if self.result is None:
            self.result = self.get_result()
        data = self.result
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 1, height_ratios=[0.1, 0.1, 0.8, 0.8])

        # 第一个子图（表格）
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')  # 去掉轴
        ax1.set_title(f'Strategy Performance')

        data = [
            ['', 'anReturn','anStd', 'anSharpe', 'maxDD', 'maxDDStart', 'maxDDEnd'],
            ['absolute', f'{data["return"] * 100:.2f}%', f'{data["std"] * 100:.2f}%', f'{data["sharpe"]:.2f}',
             f'{data["maxDrawDown"] * 100:.2f}%',
             data["startDrawDown"], data["endDrawDown"]],
            ['excess', f'{data["ex_return"] * 100:.2f}%', f'{data["ex_std"] * 100:.2f}%', f'{data["ex_sharpe"]:.2f}',
             f'{data["ex_maxDrawDown"] * 100:.2f}%',
             data["ex_startDrawDown"], data["ex_endDrawDown"]],
        ]

        # 可选的单元格颜色示例（将为每个单元格提供颜色）
        # colors = [
        #     ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray'],
        #     ['lightgray', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen'],
        #     ['lightgray', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen'],
        #     ['salmon', 'salmon', 'salmon', 'salmon', 'salmon']
        # ]
        # 在第一个子图上画表格
        draw_table(data, ax1)
        # 在第二个子图上画表格
        year_return_list1 = ['absolute']
        year_return_list2 = ['excess']
        year_list = ['']
        return_abs = self.get_return_by_year(excess=False)
        return_exc = self.get_return_by_year(excess=True)

        for i in range(len(self.result)):
            if i < len(return_abs):
                year_list += [return_abs.index[-1 - i][:4]]
                year_return_list1 += [f"{return_abs.iloc[-1 - i] * 100:.2f}%"]
                year_return_list2 += [f"{return_exc.iloc[-1 - i] * 100:.2f}%"]
            else:
                year_list += ['']
                year_return_list1 += ['']
                year_return_list2 += ['']
            if i >= 5:
                break

        data2 = [
            year_list,
            year_return_list1,
            year_return_list2
        ]
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')  # 去掉轴
        ax2.set_title(f'Return Each Year')

        draw_table(data2, ax2)

        return_cums = pd.concat([self.cum_rets, self.cum_bench_rets, self.cum_excess_rets], axis=1)
        return_cums.columns = ['port', 'bench', 'excess']

        return_df = pd.concat([self.rets, self.bench_rets, self.excess_rets], axis=1)
        return_df.columns = ['port', 'bench', 'excess']

        # 第三个子图（收益率图像）
        ax3 = fig.add_subplot(gs[2])
        ax3.set_xlabel('time')
        ax3.set_ylabel('PnL')
        ax3.plot(return_cums[return_cums.columns[0]], label=return_cums.columns[0], color="#D32F2F")#红色 strategy
        ax3.plot(return_cums[return_cums.columns[1]], label=return_cums.columns[1], color="#1976D2")#蓝色 benchmark
        ax3.set_title('Absolute Returns')
        ax3.grid(color="none", zorder=0)
        plt.legend()

        # 第四个子图（收益率图像）
        ax4 = fig.add_subplot(gs[3])
        ax4.set_xlabel('time')
        ax4.set_ylabel('PnL')
        ax4.set_title(f'Excess Returns with Leverage={self.leverage}')
        ax4.plot(return_cums[return_cums.columns[2]], alpha=0.9, label=return_cums.columns[2], color="#333333")
        ax4.grid(color="none", zorder=0)
        plt.tight_layout()
        plt.legend()
        if path is not None:
            plt.savefig(os.path.join(path, f"strategy_evaluation_{self.name}.jpg"))
        plt.show()


    @staticmethod
    def __get_cum_returns(port_ret, bench_ret, leverage=0.8, excess_method="minus_sum"):
        errMsg = "请检查method为以下选项之一\n 'minus_sum': (每日收益率 - 基准收益率)再累加 'minus_prod': (每日收益率 - 基准收益率)再加1累乘再减1  'prod_div': 加1累乘收益率再除加1累乘基准收益率再减1"

        if excess_method == "minus_sum":
            cum_rets = port_ret.cumsum() + 1
            cum_bench_rets = bench_ret.cumsum() + 1
            excess_rets = leverage * (port_ret - bench_ret)
            cum_excess_rets = excess_rets.cumsum() + 1
        elif excess_method == "minus_prod":
            cum_rets = port_ret.cumprod() + 1
            cum_bench_rets = bench_ret.cumprod() + 1
            excess_rets = leverage * (port_ret - bench_ret)
            cum_excess_rets = (excess_rets + 1).cumprod()
        elif excess_method == "prod_div":
            cum_rets = port_ret.cumprod() + 1
            cum_bench_rets = bench_ret.cumprod() + 1
            cum_excess_rets = leverage * (cum_rets / cum_bench_rets)
            excess_rets = cum_excess_rets.pct_change()

        else:
            raise ValueError(errMsg)
        return excess_rets, cum_rets, cum_bench_rets, cum_excess_rets


if __name__ == "__main__":

    close = pd.read_parquet("\\\\BF-202309011640\\ds\\data_d\\close.pq")
    strategy_ret = close[close.columns[0]].pct_change()
    benchmark_ret = close[close.columns[1]].pct_change()

    SE = StrategyEvaluator(strategy_ret, benchmark_ret, leverage=0.8, excess_method="minus_sum")  # %time: 1.02ms
    returns = SE.get_return()  # %time: 66.1 µs
    sharpe = SE.get_sharpe()  # %time: 89.2 µs
    std = SE.get_std()
    maxDD = SE.get_maxDrawDown()  # %time: 97.9 µs
    maxDDT = SE.get_maxDrawDown_time()  # %time: 222 µs
    return_by_years = SE.get_return_by_year()  # %time: 1.84 ms
    overall_results = SE.get_result()  # %time: 65.5 ns
    year_results = SE.get_return_by_year()  # %time: 1.83 ms
    SE.save_result(path="\\\\BF-202309011640\\ds\\test")
    SE.plot(path="\\\\BF-202309011640\\ds\\test")
