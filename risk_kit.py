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