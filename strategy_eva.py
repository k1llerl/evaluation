import matplotlib.pyplot as plt
import empyrical as em
import pandas as pd



class EvaluationPortfolio():
    """
    计算评价组合常见的指标
    输入：
                    bench	strategy	excess
    date			
    2017-02-28	0.000000	0.000000	0.000000
    2017-03-31	-0.000400	0.010922	0.011322
    2017-04-30	-0.039468	-0.027725	0.011743
    2017-05-31	-0.071886	-0.063709	0.008177
    
    不同策略的 收益率值
    
    当是策略输出时,策略的超额指定列为excess
    
    date: 时间格式  
    """
    
    def __init__(self, port_ret, period="monthly"):
        self.em = em
        self.port_ret = port_ret
        self.period = period
        self.risk_free = 0.0

    def _sum_ret(self):
        sumret = self.em.cum_returns_final(self.port_ret, starting_value=0)
        return pd.DataFrame(sumret, columns=["cum_ret"])
        
    def _ann_ret(self ):
        annret = self.em.stats.annual_return(self.port_ret, self.period )
        annret = pd.DataFrame(annret, columns= ["ann_ret"])         
        return annret
    
    def _sharpe(self):
        shape = self.em.sharpe_ratio(self.port_ret, risk_free=self.risk_free, period=self.period)
        columns = self.port_ret.columns
        return pd.DataFrame(shape, index=columns, columns=["sharpe"])
    
    def _max_drawdown(self):
        maxdrwa= self.em.max_drawdown(self.port_ret)
        return pd.DataFrame(maxdrwa.values, index=self.port_ret.columns, columns=["max_drwadown"])
    
    def _ann_std(self):
        std = self.em.annual_volatility(self.port_ret, self.period)         
        return pd.DataFrame(std, index=self.port_ret.columns, columns=["ann_std"])
    
    def plot(self):
        """
        画出每个策略的净值曲线图,若有excess列,则将该列的净值曲线放在右侧坐标轴
        """
        strategy_value = (self.port_ret+1).cumprod()
        if "超额收益" in self.port_ret.columns:           

            for col in self.port_ret.columns.drop('超额收益'):
                plt.plot(strategy_value.index, strategy_value[col] , label= col)               
            plt.legend()
            plt.twinx()            
            plt.plot(strategy_value.index, strategy_value["超额收益"] , c="r", label= '超额收益')                
            plt.legend(loc = 4)
            plt.show()
        else:
            for col in self.port_ret.columns:
                plt.plot(strategy_value.index, strategy_value[col], label= col)               
            plt.legend()
            plt.show()
        
    def group_year(self):
        # group by year
        df = self.port_ret.groupby(self.port_ret.index.year).apply(lambda dt: self.em.stats.cum_returns_final(dt))
        # df.columns =  ["指数等权", "策略", "超额收益"]
        return df

    def summary(self):
        cumret = self._sum_ret()
        annret = self._ann_ret()
        sharpe = self._sharpe()
        max_dd = self._max_drawdown()
        ann_std = self._ann_std()
        # ann_std = Pd.Dataframe()
        df = pd.concat([cumret, annret, sharpe, max_dd, ann_std], axis=1)    
        df.columns = ["累计收益率", "年化收益率", "夏普比率", "最大回撤", "年化波动率"]     
        # df.index = ["指数等权", "策略", "超额收益"]
        return df
    
    def turnover(self, portfolio):
        """
        计算策略换手率：每期权重相减，取绝对值，求和,月度换手
        datetime stockid 1D
        2020-01-02 000001 0.001
        """

        portfolio["1D"] = 1
        ddd = portfolio.unstack(level =1)
        ddd = ddd.fillna(0)
        s = ddd.values/ddd.values.sum(axis=1).reshape(-1, 1)
        weight = pd.DataFrame(s, index=ddd.index, columns=ddd.columns)
        del_weight = weight - weight.shift(1)
        turnover = del_weight.abs().sum(axis=1).mean()*12
        del_weight.abs().sum(axis=1).plot()
        return turnover
    
