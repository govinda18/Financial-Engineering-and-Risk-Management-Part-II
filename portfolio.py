"""
The module contains various classes for the construction of portfolio.
"""

import numpy as np
import pandas as pd
import risk_kit as erk
from scipy.optimize import minimize


class MeanVarPortfolio(object):
    """
    Portfolio model based on the modern portfolio theory. 
    The model helps to run a backtest on the mean variance
    portfolio strategy and plots the efficient frontier.
    
    Three major mean-variance strategies are implemented:
    - Maximum Sharpe Ratio Portfolio
    - Equally Weighted Portfolio
    - Global Minimum Variance Portfolio

    ...

    Parameters
    ----------
    
    er: pd.Series
        A series of expected returns for different assets

    cov: Matrx
        The covariance matrix for different assets 
    """

    def __init__(self, er, covmat):

        self.er = er

        self.cov = covmat

    @property
    def er(self):
        """
        Get the expected returns
        """

        return self._er

    @er.setter
    def er(self, er):
        self._er = er

    @property
    def cov(self):
        """
        Get the covariance matrix.
        """

        return self._cov

    @cov.setter
    def cov(self, covmat):
        self._cov = covmat

    def minimize_vol(
        self, target_return,
    ):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix

        Parameters
        ----------
        target_return (float): The targetted return of the portfolio

        Returns
        -------
        pd.Series: The optimal weight assignment that minimizes the 
        volatility for the given set of expected returns.
        """

        # Number of assets
        n = self.er.shape[0]

        # Random guess to start with.
        init_guess = np.repeat(1 / n, n)

        # Bounds of the weights
        bounds = ((0.0, 1.0),) * n

        # Constriant for sum of weights to be equal to 1
        weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1,
        }

        # Constraint that total return should be equal to target return
        return_is_target = {
            "type": "eq",
            "args": (self.er,),
            "fun": lambda weights, er: target_return
            - erk.portfolio_return(weights, self.er,),
        }

        weights = minimize(
            erk.portfolio_vol,
            init_guess,
            args=(self.cov,),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1, return_is_target,),
            bounds=bounds,
        )

        return weights.x

    def get_cml(self, riskfree_rate=0.0):
        """
        Returns the parameters of the capital market line.

        Parameters:
        ----------
        riskfree_rate (float): The riskfree_rate of the market

        Returns:
        -------

        tuple - Returns a tuple of (slope, y_intercept) of the CML.
                The y_intercept would be equal to the riskfree_rate.
        """

        wt_msr = self.msr(riskfree_rate=riskfree_rate)
        ret, vol = self.get_point(wt_msr)

        slope = (ret - riskfree_rate) / vol
        y_intercept = riskfree_rate

        return (slope, y_intercept)

    def max_return_cml(self, vol, riskfree_rate=0.0):
        """
        The maximum return earned by the captial market line model
        for the given volatility.

        Parameters:
        ----------

        vol (float): The volatility corresponding to which the maximum return 
                        has to be evaluated.

        riskfree_rate (float): The risk free rate of the market

        Returns:
        -------

        float: The return of capital market line corresponding to the given volatility.
        """

        slope, intercept = self.get_cml(riskfree_rate)
        sigma = 0.05

        exp_ret = slope * sigma + intercept

        return exp_ret


    def get_ef_weights(self, n_points):
        """
        Returns a set of optimal weights for n_points equally spaced 
        target returns from minimum expected return to 
        maximum expected return. 
        The weights are a set of weights on the efficient frontier.

        Parameters
        ----------
        n_points (int): The number of equally spaced weights to be 
                        considered

        Returns
        -------
        [pd.Series]: The list of n_points equally spaced weights. 
        """

        target_rs = np.linspace(self.er.min(), self.er.max(), n_points,)
        weights = [self.minimize_vol(target_return) for target_return in target_rs]

        return weights

    def msr(self, riskfree_rate=0.0, use_er=True):
        """
        Returns the weights corresponding to the maximum sharpe ratio 
        portfolio.

        Parameters
        ----------
        riskfree_rate (float): The risk free rate of return. Defaults to 0.0.

        use_er (bool): Uses the expected return attribute if set to true. 
                        Assumes equal expected returns and gives weights 
                        for global minimum variance if set False.
                        Defaults to True. 

        Returns
        -------
        pd.Series: The optimal weight assignment that minimizes the 
                    volatility for the given set of expected returns.
        """

        # Number of stocks
        n = self.er.shape[0]

        # Uses equally weighted expected returns if use_er is set to False
        rets = self.er if use_er else np.repeat(1, n)

        # Inital guess of weights
        init_guess = np.repeat(1 / n, n)

        # Bounds of the weights
        bounds = ((0.0, 1.0),) * n

        # Constraint for the sum of weights to be equal to 1
        weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1,
        }

        def neg_sharpe(
            weights, riskfree_rate, er, cov,
        ):
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            r = erk.portfolio_return(weights, er)
            vol = erk.portfolio_vol(weights, cov)

            return -(r - riskfree_rate) / vol

        weights = minimize(
            neg_sharpe,
            init_guess,
            args=(riskfree_rate, rets, self.cov,),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1,),
            bounds=bounds,
        )
        return weights.x

    def ew(self):
        """
        Returns the weights corresponding to an equally weighted portfolio.

        Returns
        -------
        pd.Series: The weights for an equally weighted portfolio. 
        """

        n = self.er.shape[0]

        wt = np.repeat(1 / n, n)

        return wt

    def gmv(self):
        """
        Returns the weights corresponding to the global minimum 
        variance portfolio. The weights only depends on covariance matrix.

        Returns
        -------
        pd.Series: The optimal weight assignment corresponding to the 
                    global minimum variance portfolio. 
        """

        # Number of assets
        n = self.er.shape[0]

        # Risk free rate would not be used, so we can take it 0
        return self.msr(0.0, use_er=False)

    def get_point(self, weights):
        """
        Returns the return and risk in terms of volatility 
        for the given assignemt of weights

        Parameters
        ----------
        weights (pd.Series): The weights assigned to different asset in the portfolio.

        Returns
        -------
        (float, float): A tuple representing the return and colatility respectively.
        """

        ret = erk.portfolio_return(weights, self.er)
        vol = erk.portfolio_vol(weights, self.cov)

        return (ret, vol)

    def plot_ef(
        self,
        n_points,
        style=".-",
        show_cml=False,
        riskfree_rate=0.0,
        show_ew=False,
        show_gmv=False,
    ):
        """
        Plots the efficient frontier for the given expected returns 
        and covariance matrix.

        Parameters
        ----------
        n_points (int): The number of equally spaced weights to be 
                        considered

        style (matplotlib.style): The style to be used for plotting 
                                    the efficient frontier. Defaults to '.-'.

        show_cml (bool): Plots the capital market line if set true. 
                            Defauts to False.

        risk_free_rate (float): Risk free rate to be used for plotting 
                                the capital market line. Defualts to 0.0

        show_ew (bool): Plots the equally weghted portfolio point
                            Defauts to False.

        show_gmv (bool): Plots the global minimum variance portfolio
                            Defauts to False.

        Returns
        -------
        matplotlib.plot: The plot of the efficient frontier. 
        """
        weights = self.get_ef_weights(n_points)

        rets = [erk.portfolio_return(w, self.er,) for w in weights]

        vols = [erk.portfolio_vol(w, self.cov,) for w in weights]

        ef = pd.DataFrame({"Returns": rets, "Volatility": vols,})

        ax = ef.plot.line(x="Volatility", y="Returns", style=style)
        ax.set_xlim(left=0)

        if show_cml:

            wt_msr = self.msr(riskfree_rate)
            ret_msr, vol_msr = self.get_point(wt_msr)

            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, ret_msr]

            ax.plot(
                cml_x,
                cml_y,
                color="green",
                marker="o",
                linestyle="dashed",
                linewidth=2,
                markersize=10,
            )

        if show_ew:
            wt_ew = self.ew()
            ret_ew, vol_ew = self.get_point(wt_ew)

            ax.plot(
                vol_ew, ret_ew, color="goldenrod", marker="o", markersize=10,
            )

        if show_gmv:
            wt_gmv = self.gmv()
            ret_gmv, vol_gmv = self.get_point(wt_gmv)

            ax.plot(
                vol_gmv, ret_gmv, color="midnightblue", marker="o", markersize=10,
            )

        return ax
