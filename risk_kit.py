import pandas as pd
import numpy as np
from scipy.stats import norm
from collections import Iterable

def portfolio_return(weights, returns):
    """
    Computes the net return for the given weights assigned to corresponding returns.

    Parameters - 
    :weights (np.array/N x 1 Matrix) - Describes the weights assigned to different assets

    :returns (np.array/1 x N Matrix) - Expected returns of different assets

    Return - 
    float - The net return of the given assets for the given weights
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the net volatility of the returns for the given weights.

    For weights w and covariance matrix S, the volatility is given by the square root of wTSw, 
    where wT represents w transpose.

    Parameters - 
    :weights (np.array/N x 1 Matrix) - Describes the weights assigned to different assets

    :returns (N x N Matrix) - Covariance matrix for the given assets

    Return - 
    float - The net volatility of the given assets for the given weights
    """

    return (weights.T @ covmat @ weights) ** 0.5


def discount(time, discount_rate):
    """
    Returns the discounted price of a dollar at the given discount rate
    for the given time periods.

    For a time period t, the discounted price of a dollar is given by
    1/(1 + t) ^ discount_rate.

    Parameters:
    time (Iterable) - The time periods for which discounted price is to
                        be calculated.

    discount_rate (scalar/pd.Series) - Discount rate(s) per period.     

    Return:
    (pd.DataFrame) - Returns a |t| x |r| dataframe of discounted prices.
    """

    if not isinstance(time, Iterable):
        discounts = (1 + discount_rate) ** (-time)
    else:
        discounts = pd.DataFrame(
            [(1 + discount_rate) ** (-t) for t in time], index=time
        )

    return discounts


def present_value(flows, discount_rate, periods=None):
    """
    Returns the persent discounted value of future cashflows.

    Parameters:
    flows (pd.Series) - A series of future cash flows

    discount_rate (scalar/pd.Series) - Discount rate(s) per period.

    periods (pd.Series) - The time period of flows. 
                            Considers index if as to None.

    Return:
    (float) - The present value of the set of future cash flows.
    """
    if periods is None:
        periods = flows.index
        indexed_flows = flows
    else:
        indexed_flows = pd.Series(list(flows), index=periods)

    discounts = discount(periods, discount_rate)
    pv = discounts.multiply(indexed_flows, axis="index").sum()

    return pv
