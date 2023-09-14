
# ------------------------------------------------------------------------------------------
# Author: GIAN MARCO ODDO
# Module: SupportFunctions
#     - Part of the pyriskmgmt package (support module)
# Contact Information:
#   - Email: gian.marco.oddo@usi.ch
#   - LinkedIn: https://www.linkedin.com/in/gian-marco-oddo-8a6b4b207/
#   - GitHub: https://github.com/GianMarcoOddo
# Feel free to reach out for any questions or further clarification on this code.
# ------------------------------------------------------------------------------------------

import numpy as np ; import threading ; import time ; import sys

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# DotPrinter <<<<<<<<<<<<<<<

class DotPrinter(threading.Thread):
    """
    A custom thread class that prints a growing and shrinking string of dots in the console,
    to give the visual effect of a task in progress.

    Attributes
    ----------
    running : bool : A flag that indicates whether the dot printing should continue.
    direction : int : A value of 1 or -1 that determines whether the number of dots should increase or decrease.
    dot_count : int : The current number of dots to be printed.
    process_name : str : The name of the process to be displayed.

    Methods
    -------
    start(): Starts the dot printing thread.
    run(): Prints the dots in a cycle while the `running` flag is set to True.
    stop(): Stops the dot printing thread by setting the `running` flag to False.
    """

    def __init__(self, process_name="Minimization Process"):
        super(DotPrinter, self).__init__()

        self.running = False 
        self.direction = 1 
        self.dot_count = 0
        self.process_name = process_name

    def start(self):
        self.running = True 
        super(DotPrinter, self).start()

    def run(self):
        while self.running:
            print(f"\r{self.process_name} " + "." * self.dot_count, end=" " * 90)
            sys.stdout.flush()
            if self.dot_count == 10:
                self.direction = -1
            elif self.dot_count == 0:
                self.direction = 1
            self.dot_count += self.direction
            time.sleep(0.2)

    def stop(self):
        self.running = False

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# EVT GPD log likelihood  [SUPPORT FUNCTION] <<<<<<<<<<<<<<<

def gpd_log_likelihood(params, right_hand_losses, threshold):
    """
    This function calculates the log-likelihood of the Generalized Pareto Distribution (GPD) parameters given data.
    
    The GPD is often used in extreme value theory and is suitable for modeling the tail of a loss distribution.

    Parameters
    -----------
    - params (list or tuple): A list or tuple containing the shape parameter (ξ) and scale parameter (β) of the GPD.
                            The shape parameter controls the heaviness of the tail of the distribution.
                            The scale parameter is a scaling factor that expands or contracts the distribution.
    - right_hand_losses (numpy.ndarray): A numpy array of the loss values.
    - threshold (float): A selected threshold, above which we consider losses as extreme. Only losses greater than 
                       this threshold are fitted to the GPD.

    Returns
    --------
    - float: The calculated log-likelihood of the GPD parameters given the data. A higher log-likelihood means 
           that the parameters are more likely to have generated the data, given the model.

    Note
    ---- 
    The negative sign in front of the sum is because in the context of optimization, optimizers minimize the function.
    But in the case of maximum likelihood estimation, we want to maximize the likelihood. The negative sign converts
    this maximization problem to a minimization one.
    """
    
    import numpy as np

    ξ, β = params

    return -np.sum(np.log((1 / β) * ((1 + (ξ * (right_hand_losses - threshold) / β)) ** (-1 / ξ - 1))))

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

#  DCC GARCH MODEL [SUPPORT FUNCTION] <<<<<<<<<<<< USING the R package 'rmgarch' in Python

def dccCondCovGarch(returns, interval, p, q):
    """
    This function estimates the conditional covariance matrix of returns using the Dynamic Conditional Correlation
    GARCH (DCC-GARCH) model, which is a multivariate variant of the GARCH model that allows for time-varying
    correlations in the data. The model is fitted and forecasted using the 'rmgarch' package from R.

    Parameters
    ----------
    - returns (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame of return series. 
    - interval (int): The time interval for aggregating returns. If interval is 1, no aggregation is done. 
    - p (int): The lag order of the GARCH model's conditional variance.
    - q (int): The lag order of the GARCH model's squared residuals.

    Returns
    --------
    - cov_forecast_np (numpy.ndarray): The forecasted conditional covariance matrix from the DCC-GARCH model.
    - residuals_np (numpy.ndarray): The residuals from the fitted DCC-GARCH model.

    Raises
    ------
    - Exception: If the length of the aggregated return data or the original return data is less than 100. 

    Note
    ---- 
    - This function uses the R 'rmgarch' package for DCC-GARCH model estimation and forecasting. Hence, this 
    package needs to be installed in your R distribution to use this function.
    """

    import rpy2.robjects as robjects ; from rpy2.robjects.packages import importr ; from rpy2.robjects import pandas2ri
    import numpy as np ; import pandas as pd

    # enabling conversion between pandas dataframes and R dataframes
    pandas2ri.activate()

    # importing R's "base" and "rmgarch" package
    base = importr('base') ; rmgarch = importr('rmgarch')

    rets_df = pd.DataFrame(returns)

    # --------------------------------------------------------------------------------------

    def reshape_and_aggregate_returns(returns, interval):
        returns = returns.values 
        num_periods = len(returns) // interval * interval 
        trimmed_returns = returns[len(returns) - num_periods:]
        reshaped_returns = trimmed_returns.reshape((-1, interval)) 
        aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1

        if len(aggregated_returns) < 100:
            raise Exception(f"""
            This Garch(p,q) estimation requires a minimum of 100 data points to provide reliable results. 
            Currently, the aggregated return data only contains {len(aggregated_returns)} points. 
            Please increase the data length or decrease the interval of the risk measures calculations to obtain solid results.""")

        return aggregated_returns
    
    # ----------------------------------------------------------------------------------------------------------------------------------------

    if interval == 1:

        if rets_df.shape[0] < 100:
            raise Exception(f"""
            This Garch(p,q) estimation requires a minimum of 100 data points to provide reliable results. 
            The present dataset ('returns') comprises of only {rets_df.shape[0]} data points. 
            To rectify this, consider collecting additional data or switch to a simpler risk estimation method.""")

        rets_df = rets_df

    elif interval > 1:

        rets_df = rets_df.apply(lambda x: reshape_and_aggregate_returns(x, interval)) 

    # ------------------------------------------------------------------------------------------------------------------

    # just some aesthetics
    from pyriskmgmt.SupportFunctions import DotPrinter

    # starting the animation
    my_dot_printer = DotPrinter(f"R language: DCC-GARCH ({p},{q}) Model - Full VCV Estimation")
    my_dot_printer.start()

    # R code 
    rets_r = pandas2ri.py2rpy(rets_df)
    num_cols = rets_df.shape[1]

    r_code = """
    library(rmgarch)
    uspec <- ugarchspec(mean.model=list(armaOrder=c({},{})), variance.model=list(garchOrder=c({},{})))
    spec1 = dccspec(uspec=multispec(replicate({}, uspec)), dccOrder=c({},{}), distribution="mvnorm")
    fit1 <- dccfit(spec1, data={})
    forecast <- dccforecast(fit1, n.ahead = 1)
    cov_forecast <- forecast@mforecast$H[[1]]
    residuals <- residuals(fit1)
    """.format( p,q, p, q, num_cols, p, q, 'rets_r' )

    robjects.r.assign('rets_r', rets_r)
    robjects.r(r_code)

    cov_forecast = robjects.r['cov_forecast'] 
    residuals = robjects.r['residuals']

   # stopping the animation when minimization is done
    my_dot_printer.stop()
    print(f"\rVCV Estimation --> Done", end=" " * 60)

    cov_forecast_np = np.array(cov_forecast).squeeze()
    residuals_np = np.array(residuals).squeeze()

    return cov_forecast_np, residuals_np

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

#  DCC GARCH MODEL FACTOR [SUPPORT FUNCTION] <<<<<<<<<<<< USING the R package 'rmgarch' in Python

def dccCondCovGarch_Factors(returns, interval, p, q): # NO DOTTED PROGRESS
    """
    This function estimates the conditional covariance matrix of returns using the Dynamic Conditional Correlation
    GARCH (DCC-GARCH) model, which is a multivariate variant of the GARCH model that allows for time-varying
    correlations in the data. The model is fitted and forecasted using the 'rmgarch' package from R.

    Parameters
    ----------
    - returns (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame of return series. 
    - interval (int): The time interval for aggregating returns. If interval is 1, no aggregation is done. 
    - p (int): The lag order of the GARCH model's conditional variance.
    - q (int): The lag order of the GARCH model's squared residuals.

    Returns
    --------
    - cov_forecast_np (numpy.ndarray): The forecasted conditional covariance matrix from the DCC-GARCH model.
    - residuals_np (numpy.ndarray): The residuals from the fitted DCC-GARCH model.

    Raises
    ------
    - Exception: If the length of the aggregated return data or the original return data is less than 100. 

    Note
    ---- 
    - This function uses the R 'rmgarch' package for DCC-GARCH model estimation and forecasting. Hence, this 
    package needs to be installed in your R distribution to use this function.
    """

    import rpy2.robjects as robjects ; from rpy2.robjects.packages import importr ; from rpy2.robjects import pandas2ri
    import numpy as np ; import pandas as pd

    # enabling conversion between pandas dataframes and R dataframes
    pandas2ri.activate()

    # importing R's "base" and "rmgarch" package
    base = importr('base') ; rmgarch = importr('rmgarch')

    rets_df = pd.DataFrame(returns)

    # ------------------------------------------------------------------------------------------------------------------

    def reshape_and_aggregate_returns(returns, interval):
        returns = returns.values 
        num_periods = len(returns) // interval * interval 
        trimmed_returns = returns[len(returns) - num_periods:]
        reshaped_returns = trimmed_returns.reshape((-1, interval)) 
        aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1

        if len(aggregated_returns) < 100:
            raise Exception(f"""
            This Garch(p,q) estimation requires a minimum of 100 data points to provide reliable results. 
            Currently, the aggregated return data only contains {len(aggregated_returns)} points. 
            Please increase the data length or decrease the interval of the risk measures calculations to obtain solid results.""")

        return aggregated_returns
    
    # ------------------------------------------------------------------------------------------------------------------

    if interval == 1:

        if rets_df.shape[0] < 100:
            raise Exception(f"""
            This Garch(p,q) estimation requires a minimum of 100 data points to provide reliable results. 
            The present dataset ('returns') comprises of only {rets_df.shape[0]} data points. 
            To rectify this, consider collecting additional data or switch to a simpler risk estimation method.""")

        rets_df = rets_df

    elif interval > 1:

        rets_df = rets_df.apply(lambda x: reshape_and_aggregate_returns(x, interval)) 

    # ------------------------------------------------------------------------------------------------------------------

    # R code 
    rets_r = pandas2ri.py2rpy(rets_df) 
    num_cols = rets_df.shape[1]

    r_code = """
    library(rmgarch)
    uspec <- ugarchspec(mean.model=list(armaOrder=c({},{})), variance.model=list(garchOrder=c({},{})))
    spec1 = dccspec(uspec=multispec(replicate({}, uspec)), dccOrder=c({},{}), distribution="mvnorm")
    fit1 <- dccfit(spec1, data={})
    forecast <- dccforecast(fit1, n.ahead = 1)
    cov_forecast <- forecast@mforecast$H[[1]]
    residuals <- residuals(fit1)
    """.format( p,q, p, q, num_cols, p, q, 'rets_r' )

    robjects.r.assign('rets_r', rets_r) 
    robjects.r(r_code)

    cov_forecast = robjects.r['cov_forecast'] 
    residuals = robjects.r['residuals']

    cov_forecast_np = np.array(cov_forecast).squeeze() 
    residuals_np = np.array(residuals).squeeze()

    return cov_forecast_np, residuals_np

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

#  EWMA COND COV [SUPPORT FUNCTION] <<<<<<<<<<<<

def ewmaCondCov(returns, interval, lambda_ewma):
    """
    This function calculates the Exponentially Weighted Moving Average (EWMA) conditional covariance matrix of returns.
    EWMA is a method used to compute variance and covariance by applying more weight to recent observations in a time series.
    
    Parameters
    ----------
    - returns (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame of return series.
    - interval (int): The time interval for aggregating returns. If interval is 1, no aggregation is done.
    - lambda_ewma (float): The decay factor in the EWMA model, which determines the weight of observations in the past. 
                         This value should be between 0 and 1, where a value closer to 1 places more emphasis on past observations.

    Returns
    -------
    - cov_forecast_np (numpy.ndarray): The EWMA conditional covariance matrix. The diagonal elements represent the variances 
                                     and the off-diagonal elements represent the covariances.

    Note
    ---- 
    - This function uses the EWMA method to compute the conditional covariance matrix, which does not explicitly assume a constant correlation structure. 
    - Instead, it gives more weight to recent observations, allowing for potentially changing correlations over time.
    """

    import numpy as np ; import pandas as pd

    # ------------------------------------------------------------------------------------------------------------------------

    def reshape_and_aggregate_returns(returns, interval):
        returns = returns.values 
        num_periods = len(returns) // interval * interval 
        trimmed_returns = returns[len(returns) - num_periods:]
        reshaped_returns = trimmed_returns.reshape((-1, interval)) 
        aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
        return aggregated_returns
    
    # -------------------------------------------------------------------------------------------------------------------------------------------

    if interval == 1:
        returns = returns

    if interval > 1:
        returns = pd.DataFrame(returns) 
        returns = returns.apply(lambda x: reshape_and_aggregate_returns(x, interval)) 
        returns = returns.values

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    def ewma_cond_variance(rets, lambda_ewma):
        
        # creating the fitted time series of the ewma(lambda_ewma) variances 

        variance_time_series = np.repeat(rets[0]**2, len(rets))
        for t in range(1,len(rets)):
            variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * rets[t-1]**2
        variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * rets[-1]**2  

        return variance_t_plus1
    
    # ------------------------------------------------------------------------------------------------------------------------

    def ewma_cond_covariance(rets_1,rets_2, lambda_ewma):

        # creatting the fitted time series of the ewma(lambda_ewma) variances 

        covariance_time_series = np.repeat(rets_1[0] * rets_2[0], len(rets_1))
        for t in range(1,len(rets_1)):
            covariance_time_series[t] = lambda_ewma * covariance_time_series[t-1] + (1-lambda_ewma) * rets_1[t-1] * rets_2[t-1]
        covariance_t_plus1 = lambda_ewma * covariance_time_series[-1] + (1-lambda_ewma) * rets_1[-1] * rets_2[-1]
        return covariance_t_plus1
        
    # --------------------------------------------------------------------------------------------------------------------------------

    # variance_cov_matrix estimation

    # just some aesthetics
    from pyriskmgmt.SupportFunctions import DotPrinter

    # starting the animation
    my_dot_printer = DotPrinter(f"\rEWMA MODEL ({lambda_ewma}) - Full VCV Estimation") ; my_dot_printer.start()

    variance_cov_matrix = np.zeros((returns.shape[1],returns.shape[1]))

    for i in range(returns.shape[1]):
        for j in range(returns.shape[1]):
            if i == j:
                variance_cov_matrix[i,j] = ewma_cond_variance(returns[:,i],lambda_ewma = lambda_ewma)
            else:
                variance_cov_matrix[i,j] = ewma_cond_covariance(returns[:,i],returns[:,j],lambda_ewma = lambda_ewma)

    # stopping the animation when minimization is done
    my_dot_printer.stop()
    print(f"\rVCV Estimation --> Done", end=" " * 60)

    cov_forecast_np = variance_cov_matrix

    return cov_forecast_np

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

#  EWMA COND COV [SUPPORT FUNCTION] <<<<<<<<<<<<

def ewmaCondCov__Factors(returns, interval, lambda_ewma): # NO DOTTED PROGRESS
    """
    This function calculates the Exponentially Weighted Moving Average (EWMA) conditional covariance matrix of returns.
    EWMA is a method used to compute variance and covariance by applying more weight to recent observations in a time series.
    
    Parameters
    ----------
    - returns (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame of return series.
    - interval (int): The time interval for aggregating returns. If interval is 1, no aggregation is done.
    - lambda_ewma (float): The decay factor in the EWMA model, which determines the weight of observations in the past. 
                         This value should be between 0 and 1, where a value closer to 1 places more emphasis on past observations.

    Returns
    -------
    - cov_forecast_np (numpy.ndarray): The EWMA conditional covariance matrix. The diagonal elements represent the variances 
                                     and the off-diagonal elements represent the covariances.

    Note
    ---- 
    - This function uses the EWMA method to compute the conditional covariance matrix, which does not explicitly assume a constant correlation structure. 
    - Instead, it gives more weight to recent observations, allowing for potentially changing correlations over time.
    """

    import numpy as np ; import pandas as pd

    # ------------------------------------------------------------------------------------------------------------------------

    def reshape_and_aggregate_returns(returns, interval):
        returns = returns.values 
        num_periods = len(returns) // interval * interval 
        trimmed_returns = returns[len(returns) - num_periods:]
        reshaped_returns = trimmed_returns.reshape((-1, interval)) 
        aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
        return aggregated_returns
    
    # -------------------------------------------------------------------------------------------------------------------------------------------

    if interval == 1:
        returns = returns

    if interval > 1:
        returns = pd.DataFrame(returns) 
        returns = returns.apply(lambda x: reshape_and_aggregate_returns(x, interval)) 
        returns = returns.values

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    def ewma_cond_variance(rets, lambda_ewma):
        
            # creating the fitted time series of the ewma(lambda_ewma) variances 

        variance_time_series = np.repeat(rets[0]**2, len(rets))
        for t in range(1,len(rets)):
            variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * rets[t-1]**2
        variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * rets[-1]**2  

        return variance_t_plus1
    
    # ------------------------------------------------------------------------------------------------------------------------

    def ewma_cond_covariance(rets_1,rets_2, lambda_ewma):

        # creatting the fitted time series of the ewma(lambda_ewma) variances 

        covariance_time_series = np.repeat(rets_1[0] * rets_2[0], len(rets_1))
        for t in range(1,len(rets_1)):
            covariance_time_series[t] = lambda_ewma * covariance_time_series[t-1] + (1-lambda_ewma) * rets_1[t-1] * rets_2[t-1]
        covariance_t_plus1 = lambda_ewma * covariance_time_series[-1] + (1-lambda_ewma) * rets_1[-1] * rets_2[-1]
        return covariance_t_plus1
        
    # --------------------------------------------------------------------------------------------------------------------------------

    # variance_cov_matrix estimation

    variance_cov_matrix = np.zeros((returns.shape[1],returns.shape[1]))

    for i in range(returns.shape[1]):
        for j in range(returns.shape[1]):
            if i == j:
                variance_cov_matrix[i,j] = ewma_cond_variance(returns[:,i],lambda_ewma = lambda_ewma)
            else:
                variance_cov_matrix[i,j] = ewma_cond_covariance(returns[:,i],returns[:,j],lambda_ewma = lambda_ewma)

    cov_forecast_np = variance_cov_matrix

    return cov_forecast_np

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# [FAMA FRENCH 3] MKT, HML and SMB <<<<<<<<<<<<<<<<<< from 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html'

def ff3Downloader( start_date: str, end_date: str, freq: str = "daily"):
    """
    Main
    ----
    This method retrieves Fama French 3 factors (market, size, value) for a particular date range and frequency. 
    The Fama French 3 factors, retrieved from Ken French's data library, are widely used in finance for risk factor model development. 
    The three factors are market factor, size factor (SMB - Small Minus Big), and value factor (HML - High Minus Low).
    if freq == "daily":
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    elif freq == "weekly":
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip'
    elif freq == "monthly":
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'

    Parameters
    ----------
    - start_date (str): The beginning date of the range for which factors need to be fetched.
    - end_date (str): The end date of the range for which factors need to be fetched.
    - freq (str, optional): The frequency of the data. Options include "daily", "weekly", and "monthly". Defaults to "daily".
        
    Returns
    -------
    - df (pd.DataFrame): A pandas DataFrame containing the Fama French 3 factors for the specified date range and frequency.
    
    Notes
    ------
    - This method connects to the internet to fetch data. Ensure that your machine has a stable internet connection.
    - The method requires the requests, io, zipfile, and pandas libraries. Make sure these libraries are installed and imported in your environment.
    - The three factors, namely market, SMB, and HML, are returned as percentage values. For example, a return of 0.01 corresponds to a 1% return.
    - The date range specified should fall within the availability of the data in the Fama French data library. For daily data, this typically goes back to 1962. 
    - For monthly data, this goes back to 1926.
    - This function is a part of the EquityPrmPort class, a toolkit for portfolio risk management. 
    - It serves to fetch and process the Fama French 3 factors which can be used for further analysis or model development.
    """

    import pandas as pd ; import requests ; import io ; from zipfile import ZipFile

    from pyriskmgmt.inputsControlFunctions import (check_and_convert_dates,check_freq)

    # check procedure
    start_date, end_date = check_and_convert_dates(start_date, end_date) ; freq = check_freq(freq, allowed_values=["daily", "weekly", "monthly"])

    url = ''
    if freq == "daily":
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    elif freq == "weekly":
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip'
    elif freq == "monthly":
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
    
    response = requests.get(url) # request 

    with ZipFile(io.BytesIO(response.content)) as thezip:

        for zipinfo in thezip.infolist():

            with thezip.open(zipinfo) as thefile:

                # freq = "daily" -------------------------------------------------------------------------------------------------------------------

                if freq == "daily":

                    df = pd.read_csv(thefile, skiprows=4) ; df = df.iloc[:-1]  # removing the last row
                    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%Y%m%d')

                    # setting the date as the index and selecting the "MKT","HML" and "SMB" columns
                    df.set_index('Unnamed: 0', inplace=True) ; df.index.name = 'Date' ; df["MKT"] = df["Mkt-RF"] + df["RF"] ; df = df[["MKT","HML","SMB"]]

                    # convert them to numeric
                    df['MKT'] = pd.to_numeric(df['MKT'], errors='coerce')                       
                    df['SMB'] = pd.to_numeric(df['SMB'], errors='coerce') ;  df['HML'] = pd.to_numeric(df['HML'], errors='coerce')

                    # dividing these columns by 100
                    df['MKT'] = df['MKT'] / 100 ; df['SMB'] = df['SMB'] / 100 ; df['HML'] = df['HML'] / 100

                    df = df[start_date:end_date] # slicing
                    
                # freq = "weekly" ---------------------------------------------------------------------------------------------------------------------

                if freq == "weekly":

                    df = pd.read_csv(thefile, skiprows=4) ; df = df.iloc[:-1]  # removing the last row
                    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%Y%m%d')

                    # setting the date as the index and selecting the "MKT","HML" and "SMB" columns
                    df.set_index('Unnamed: 0', inplace=True) ; df.index.name = 'Date' ; df["MKT"] = df["Mkt-RF"] + df["RF"] ; df = df[["MKT","HML","SMB"]]

                    # convert them to numeric
                    df['MKT'] = pd.to_numeric(df['MKT'], errors='coerce')                       
                    df['SMB'] = pd.to_numeric(df['SMB'], errors='coerce') ;  df['HML'] = pd.to_numeric(df['HML'], errors='coerce')

                    # dividing these columns by 100
                    df['MKT'] = df['MKT'] / 100 ; df['SMB'] = df['SMB'] / 100 ; df['HML'] = df['HML'] / 100

                    df = df[start_date:end_date]  # slicing

                # freq = "monthly" ------------------------------------------------------------------------------------------------------------------------

                elif freq == "monthly":

                    df = pd.read_csv(thefile, skiprows=3) ; df = df.iloc[:-1]  # removing the last row
                    
                    # converting the DataFrame to string to detect lines with specific string
                    df = df.astype(str)
                    
                    # finding a specific row 
                    target_rows = df[df.iloc[:, 0].str.contains("Annual Factors: January-December")]

                    if not target_rows.empty:
                        target_index = target_rows.index[0]
                        # selecting rows from the start until the target row
                        df = df.loc[:target_index-1]
                        # the name of the first column is 'Unnamed: 0'
                        df['Unnamed: 0'] = '01' + df['Unnamed: 0'] ; df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%d%Y%m')

                        # setting the date as the index and selecting the "MKT","HML" and "SMB" columns
                        df.set_index('Unnamed: 0', inplace=True) ; df.index.name = 'Date' ; df['Mkt-RF'] = pd.to_numeric(df['Mkt-RF'], errors='coerce') 
                        
                        df['RF'] = pd.to_numeric(df['RF'], errors='coerce')  ; df["MKT"] = df["Mkt-RF"] + df["RF"] ; df = df[["MKT","HML","SMB"]]
                    
                        # convert them to numeric
                        df['SMB'] = pd.to_numeric(df['SMB'], errors='coerce') ;  df['HML'] = pd.to_numeric(df['HML'], errors='coerce')

                        # dividing these columns by 100
                        df['MKT'] = df['MKT'] / 100 ; df['SMB'] = df['SMB'] / 100 ; df['HML'] = df['HML'] / 100

                        df = df[start_date:end_date] # slicing
                        
    return df
    
# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# SHARPE DIAGONAL VAR - [SINGLE] FACTOR SUPPORT FUNCTION <<<<<<<<<<<<<<<<<<

def sharpeDiagSingle(returns, positions, interval, alpha, factor_rets: np.ndarray = None, vol: str = "simple", p: int = 1, q: int = 1,
                        lambda_ewma: float = 0.94, warning: bool = True):
    """
    This function implements the Sharpe Single Index Model to estimate Value at Risk (VaR) and Expected Shortfall (ES). 
    It uses three methods to estimate volatility: simple historical volatility, GARCH (p, q), and EWMA (lambda).

    Parameters
    ----------
    - returns (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame of return series.
    - positions (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame representing the positions in different assets.
    - interval (int): The time interval for aggregating returns. If interval is 1, no aggregation is done.
    - alpha (float): The significance level for the VaR and ES estimates. This should be a number between 0 and 1.
    - factor_rets (numpy.ndarray): The return series of the single risk factor in the Sharpe Single Index Model.
    - vol (str): The method to use for estimating volatility. Options are "simple", "garch", or "ewma".
    - p (int): The lag order of the GARCH model's conditional variance. This is used if vol="garch".
    - q (int): The lag order of the GARCH model's squared residuals. This is used if vol="garch".
    - lambda_ewma (float): The decay factor in the EWMA model, which determines the weight of observations in the past. This is used if vol="ewma".
    - warning (bool): If True, the function will print warnings for certain conditions. 

    Returns
    -------
    - dict: A dictionary containing the following:
        "var": The Value at Risk estimate.
        "es": The Expected Shortfall estimate.
        "T": The time interval used for aggregating returns and make predictions.
        "p": The lag order of the GARCH model's conditional variance. Returned if vol="garch".
        "q": The lag order of the GARCH model's squared residuals. Returned if vol="garch".
        "lambda_ewma": The decay factor in the EWMA model, if vol="ewma".
        "PosFactor": The position in the factor calculated as the dot product of positions and betas.
        "sigmaPi": The standard deviation of the portfolio calculated as the product of the position in the factor and the factor's standard deviation.

    Raises
    ------
    - Exception: If factor_rets is None or the lengths of factor_rets and returns are not equal.
    - ValueError: If all columns in returns are equal to factor_rets.
    """

    import numpy as np

    from pyriskmgmt.inputsControlFunctions import (validate_returns_single,check_vol,check_warning)
    
    # check factor_rets
    if  factor_rets is None:
        raise Exception("factor_rets = None. Please, insert the returns for chosen risk factor")
    
    # check factor_rets --- validity
    factor_rets = validate_returns_single(factor_rets)

    # checking if factor_rets and self.returns are of the same dimension
    if len(factor_rets) != len(returns):
        raise ValueError("Input 'factor_rets' and 'self.returns' should have the same number of rows.")
    
    # Check if all columns in self.returns are equal to factor_rets
    if all(np.array_equal(returns[:,i], factor_rets) for i in range(returns.shape[1])):
        raise ValueError("All columns in self.returns are equal to factor_rets. Further factor analysis is not required.")

    # check procedure
    vol = check_vol(vol) ; check_warning(warning)
    
    # warning if len of self.positions is less than 30
    if len(positions) < 30 and warning:
        print("WARNING:")
        print("The portfolio has less than 30 assests. Idiosyncratic components do not play a marginal role. This function could generate misleading results.")
        print("To mute this warning, add 'warning = False' as an argument of the sharpeDiag() function.")

    # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    if vol == "simple":

        if interval == 1: ###########
                
                factor_variance_betas = np.var(factor_rets, ddof=1)
                
                # calculating the beta for each stock ----------------------------------------

                betas = []

                for i in range(returns.shape[1]): # loop over each column
                    stock_rets = returns[:, i] 
                    covariance = np.cov(stock_rets, factor_rets)[0, 1]
                    beta = covariance / (factor_variance_betas)
                    betas.append(beta)

        elif interval > 1: ###########

            # -------------------------------------------------------------------------------------

            def reshape_and_aggregate_returns(returns, interval):
                import numpy as np
                num_periods = len(returns) // interval * interval 
                trimmed_returns = returns[len(returns) - num_periods:] 
                reshaped_returns = trimmed_returns.reshape((-1, interval)) 
                aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                return aggregated_returns
            
            # -------------------------------------------------------------------------------------------------------------------------------------
            
            aggregated_factor_returns = reshape_and_aggregate_returns(factor_rets, interval) 
            factor_variance_betas = np.var(aggregated_factor_returns, ddof=1) 

            # calculating the beta for each stock -------------------------------------------------------------------------------------------------

            betas = []

            for i in range(returns.shape[1]): # loop over each column
                stock_rets = reshape_and_aggregate_returns(returns[:, i], interval) 
                covariance = np.cov(stock_rets, aggregated_factor_returns)[0, 1]
                beta = covariance / (factor_variance_betas)
                betas.append(beta)

        # mapping procedure ------------------------------------------------------------------------------------------------------------------------------------

        factor_position = np.dot(positions, betas)

        # sigma of the factor -----------------------------------------------------------------------------------------------------------------------------------

        factor_sigma = np.sqrt(factor_variance_betas) # this follows already the specified interval

        from scipy.stats import norm

        # Value at Risk ----------------------------------

        q = norm.ppf(1-alpha, loc=0, scale=1) ; VaR = q * abs(factor_position) * factor_sigma 

        VaR =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ( ( np.exp( - (q**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-alpha)))) 
        ES = q_es * abs(factor_position) * factor_sigma 

        ES = max(0,ES)

        SHRP_DIAG = {"var" : round(VaR,4),
                     "es" : round(ES,4),
                     "T" : interval,
                     "PosFactor" : round(factor_position,4),
                     "sigmaPi" : round(abs(factor_position) * factor_sigma,5)}
        
        
    # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    if vol == "garch":

        if warning:
            from pyriskmgmt.SupportFunctions import garch_warnings
            garch_warnings("sharpeDiag()")

        from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

        # check procedure
        p = check_p(p) ; q = check_q(q)

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing the support function dccCondCovGarch_Factors(), from the SupportFunctions module

        from pyriskmgmt.SupportFunctions import dccCondCovGarch_Factors

        DDC_GARCH = dccCondCovGarch_Factors

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # computing conditional betas and variances

        cond_betas = np.zeros(returns.shape[1]) 
        cond_variance_factor = np.zeros(returns.shape[1]) 

        total_loops = returns.shape[1]
        for i in range(total_loops):
            # Display the progress
            print(f'\rR language: DCC-GARCH ({p},{q}) Model - Mapping Procedure -> 1 Factor: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end='                ')

            if np.array_equal(returns[:, i], factor_rets):
                cond_betas[i] = 1.0 # beta with itself is 1
            else:
                rets_stock_factor = np.column_stack((returns[:, i], factor_rets))
                cond_cov_rets_stock_factor = DDC_GARCH(rets_stock_factor, interval = interval, p=p, q=q)[0]
                cond_covariance = cond_cov_rets_stock_factor[0,1] 
                cond_var_factor = cond_cov_rets_stock_factor[1,1]
                cond_variance_factor[i] = cond_var_factor
                cond_betas[i] = cond_covariance / cond_var_factor

        # mapping procedure ----------------------------------------------------------------------------

        factor_position = np.dot(positions, cond_betas) 
        factor_sigma = np.sqrt(cond_variance_factor[-1]) # this follows already the specified interval

        from scipy.stats import norm

        # Value at Risk ----------------------------------

        quantile = norm.ppf(1-alpha, loc=0, scale=1) 
        VaR = quantile * abs(factor_position) * factor_sigma 

        VaR =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ((np.exp(-(quantile**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha)))) 
        ES = q_es * abs(factor_position) * factor_sigma 

        ES = max(0,ES)

        SHRP_DIAG = {"var" : round(VaR,4),
                     "es" : round(ES,4),
                     "T" : interval,
                     "p" : p,
                     "q" : q,
                     "PosFactor" : round(factor_position,4),
                     "sigmaPi" : round(abs(factor_position) * factor_sigma,5)}


    # EWMA (lamdba = lambda_ewma) Volatility ------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    if vol == "ewma":

        from pyriskmgmt.inputsControlFunctions import (check_lambda_ewma)

        # check procedure
        lambda_ewma = check_lambda_ewma(lambda_ewma)

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing the support function ewmaCondCov(), from the SupportFunctions module

        from pyriskmgmt.SupportFunctions import ewmaCondCov__Factors

        EWMA_COV = ewmaCondCov__Factors

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Compute conditional betas and variances

        cond_betas = np.zeros(returns.shape[1])
        cond_variance_factor = np.zeros(returns.shape[1]) 

        total_loops = returns.shape[1]

        for i in range(total_loops):
            # Display the progress
            print(f'\rEWMA MODEL ({lambda_ewma}) - Mapping Procedure -> 1 Factor: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end='           ')
            if np.array_equal(returns[:, i], factor_rets):
                cond_betas[i] = 1.0 # beta with itself is 1
            else:
                rets_stock_factor = np.column_stack((returns[:, i], factor_rets))
                cond_cov_rets_stock_factor = EWMA_COV(rets_stock_factor, interval = interval, lambda_ewma = lambda_ewma)
                cond_covariance = cond_cov_rets_stock_factor[0,1] 
                cond_var_factor = cond_cov_rets_stock_factor[1,1]
                cond_variance_factor[i] = cond_var_factor
                cond_betas[i] = cond_covariance / cond_var_factor

        # mapping procedure ----------------------------------------------------------------------------

        factor_position = np.dot(positions, cond_betas) 
        factor_sigma = np.sqrt(cond_variance_factor[-1]) # this follows already the specified interval

        from scipy.stats import norm

        # Value at Risk ----------------------------------

        q = norm.ppf(1-alpha, loc=0, scale=1) 
        VaR = q * abs(factor_position) * factor_sigma 

        VaR =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ( ( np.exp( - (q**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-alpha)))) 
        ES = q_es * abs(factor_position) * factor_sigma 

        ES = max(0,ES)

        SHRP_DIAG = {"var" : round(VaR,4),
                     "es" : round(ES,4),
                     "T" : interval,
                     "lambda_ewma" : lambda_ewma,
                     "PosFactor" : round(factor_position,4),
                     "sigmaPi" : round(abs(factor_position) * factor_sigma,5)}
        
    return SHRP_DIAG

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# SHARPE DIAGONAL VAR - [MULTI] FACTORS SUPPORT FUNCTION <<<<<<<<<<<<<<<<<<

def sharpeDiagMulti(returns, positions, interval, alpha, factors_rets : np.ndarray = None, vol: str = "simple", p: int = 1, q: int = 1,
                        lambda_ewma: float = 0.94, warning: bool = True):
    
    """
    This function implements a multi-factor Sharpe Model to estimate Value at Risk (VaR) and Expected Shortfall (ES). 
    It uses three methods to estimate volatility: simple historical volatility, GARCH (p, q), and EWMA (lambda).

    Parameters
    ----------
    - returns (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame of return series.
    - positions (numpy.ndarray or pandas.DataFrame): A two-dimensional array or DataFrame representing the positions in different assets.
    - interval (int): The time interval for aggregating returns. If interval is 1, no aggregation is done.
    - alpha (float): The significance level for the VaR and ES estimates. This should be a number between 0 and 1.
    - factors_rets (numpy.ndarray): The return series of the multiple risk factors in the Sharpe Model.
    - vol (str): The method to use for estimating volatility. Options are "simple", "garch", or "ewma".
    - p (int): The lag order of the GARCH model's conditional variance. This is used if vol="garch".
    - q (int): The lag order of the GARCH model's squared residuals. This is used if vol="garch".
    - lambda_ewma (float): The decay factor in the EWMA model, which determines the weight of observations in the past. This is used if vol="ewma".
    - warning (bool): If True, the function will print warnings for certain conditions. 

    Returns
    -------
    - dict: A dictionary containing the following:
        "var": The Value at Risk estimate.
        "es": The Expected Shortfall estimate.
        "T": The time interval used for aggregating returns.
        "p": The lag order of the GARCH model's conditional variance. Returned if vol="garch".
        "q": The lag order of the GARCH model's squared residuals. Returned if vol="garch".
        "lambda_ewma": The decay factor in the EWMA model. Returned if vol="ewma".
        "sigmaPi": The standard deviation of the portfolio calculated as the product of the position in the factor and the factor's standard deviation.
        "PosFactor{i}": The position in the i-th factor, calculated as the dot product of positions and betas.

    Raises
    ------
    - Exception: If factors_rets is None or the lengths of factors_rets and returns are not equal.
    - ValueError: If all columns in returns are equal to any of the columns in factors_rets.
    """

    import numpy as np
    from pyriskmgmt.inputsControlFunctions import (validate_returns_port,check_vol,check_warning)
    
    # check factors_rets
    if  factors_rets is None:
        raise Exception("factors_rets = None. Please, insert the returns for chosen risk factors")
    
    # check factors_rets --- validity
    factors_rets = validate_returns_port(factors_rets)

    # checking if factors_rets and self.returns have the same number of rows
    if factors_rets.shape[0] != returns.shape[0]:
        raise ValueError("Input 'factors_rets' and 'self.returns' should have the same number of rows.")
    
    for j in range(factors_rets.shape[1]):
        if all(np.array_equal(returns[:,i], factors_rets[:j]) for i in range(returns.shape[1])):
            raise ValueError(f"All columns in self.returns are equal to the colum {j} of the factors_rets array. Further factors analysis is not required.")
        
    # check procedure
    vol = check_vol(vol) ; check_warning(warning)
    
    # warning if len of self.positions is less than 30
    if len(positions) < 30 and warning:
        print("WARNING:")
        print("The portfolio has less than 30 assests. Idiosyncratic components do not play a marginal role. This function could generate misleading results.")
        print("To mute this warning, add 'warning = False' as an argument of the sharpeDiag() function.")


    # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    if vol == "simple":

        if interval == 1: ###########
            factor_variances = np.var(factors_rets, axis=0, ddof=1) 
            factor_variances_betas = factor_variances.tolist()
            variance_cov_matrix_factors = np.cov(factors_rets, rowvar=False)

            # calculating the beta for each stock ##################################################################
            
            betas_factors = []
            # transposing the returns to match the dimensions with factors
            trans_returns = returns.T
            for j in range(factors_rets.shape[1]):
                # computing the covariance matrix between returns and each factor
                covariances = np.cov(trans_returns, factors_rets[:,j])[:-1,-1]
                # calculating betas
                betas = covariances / factor_variances_betas[j]
                betas_factors.append(betas.tolist())
            betas_factors_array = np.array(betas_factors) 
            factors_positions = betas_factors_array @ positions 

        elif interval > 1: #########

            # -----------------------------------------------------------------------------------

            def reshape_and_aggregate_returns(column, interval):
                    num_periods = len(column) // interval * interval
                    trimmed_returns = column[len(column) - num_periods:]
                    reshaped_returns = trimmed_returns.reshape((-1, interval))
                    aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                    
                    return aggregated_returns
            
            # -----------------------------------------------------------------------------------

            def aggregate_returns(returns, factors_rets, interval):

                aggregated_stock_rets = []
                for i in range(returns.shape[1]):
                    agg_rets = reshape_and_aggregate_returns(returns[:,i], interval)
                    aggregated_stock_rets.append(agg_rets)
                aggregated_stock_rets = np.array(aggregated_stock_rets).T

                aggregated_factor_rets = []
                for i in range(factors_rets.shape[1]):
                    agg_rets = reshape_and_aggregate_returns(factors_rets[:,i], interval)
                    aggregated_factor_rets.append(agg_rets)
                aggregated_factor_rets = np.array(aggregated_factor_rets).T

                return aggregated_stock_rets, aggregated_factor_rets

            # -----------------------------------------------------------------------------------

            aggregated_stock_rets, aggregated_factor_rets = aggregate_returns(returns, factors_rets, interval)

            factor_variances = np.var(aggregated_factor_rets, axis=0, ddof=1) 
            factor_variances_betas = factor_variances.tolist()

            variance_cov_matrix_factors = np.cov(aggregated_factor_rets, rowvar=False)

            # calculating the beta for each stock ##################################################################
            
            betas_factors = []
            # transposing the returns to match the dimensions with factors
            trans_returns = aggregated_stock_rets.T
            for j in range(aggregated_factor_rets.shape[1]):
                # computing the covariance matrix between returns and each factor
                covariances = np.cov(trans_returns, aggregated_factor_rets[:,j])[:-1,-1]
                # calculating betas
                betas = covariances / factor_variances_betas[j]
                betas_factors.append(betas.tolist())

        # -------------------------------------------------------------------------

            betas_factors_array = np.array(betas_factors) 
            factors_positions = betas_factors_array @ positions 

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        from scipy.stats import norm

        # Value at Risk ----------------------------------

        var_port = np.dot(np.dot(factors_positions,variance_cov_matrix_factors),factors_positions.T) 
        sigma = np.sqrt(var_port) # this follows already the interval

        q = norm.ppf(1-alpha, loc=0, scale=1) ; VaR = q * sigma 

        VaR =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ((np.exp(-(q**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha)))) 
        ES = q_es * sigma 

        ES = max(0,ES)

        SHRP_DIAG = {
            "var" : round(VaR, 4),
            "es" : round(ES, 4),
            "T" : interval,
            "sigmaPi" : round(sigma,5)
        }

        for i, factor in enumerate(factors_positions, start=1):
            SHRP_DIAG[f"PosFactor{i}"] = round(factor, 4)

    # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    if vol == "garch":

        if warning:
            from pyriskmgmt.SupportFunctions import garch_warnings
            garch_warnings("sharpeDiag()")

        from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

        # check procedure
        p = check_p(p) ; q = check_q(q)

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing the support function dccCondCovGarch_Factors(), from the SupportFunctions module

        from pyriskmgmt.SupportFunctions import dccCondCovGarch_Factors

        DDC_GARCH = dccCondCovGarch_Factors

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # computing conditional betas and variances

                # computing conditional betas and variances

        betas_factors = np.zeros((returns.shape[1], factors_rets.shape[1])) 

        total_loops = returns.shape[1]
        
        for i in range(total_loops):
            # displaying  the progress
            print(f'\rR language: DCC-GARCH ({p},{q}) Model - Mapping Procedure -> {factors_rets.shape[1]} Factors: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end='                                   ')
            rets_stock_factors = np.column_stack((returns[:, i], factors_rets))

            # flagging for columns to drop in case of duplicates 
            cols_to_drop = []
            # checking for duplicates
            for col in range(rets_stock_factors.shape[1] -1):
                if np.array_equal(rets_stock_factors[:, 0], rets_stock_factors[:, col + 1]):
                    cols_to_drop.append(col + 1) 
            rets_stock_factors_checked = np.delete(rets_stock_factors, cols_to_drop, axis=1)

            if rets_stock_factors_checked.shape[1] == rets_stock_factors.shape[1]: # no columns have been dropped ////////////////////////
                cond_var_cov_matrix_inner = DDC_GARCH(rets_stock_factors_checked, interval = interval, p=p, q=q)[0]

                inner_betas = np.zeros(factors_rets.shape[1])

                for j in range(1, cond_var_cov_matrix_inner.shape[1]):
                    beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j] 
                    inner_betas[j - 1] = beta
                betas_factors[i,:] = inner_betas

            elif rets_stock_factors_checked.shape[1] != rets_stock_factors.shape[1]: # the cols_to_drop column has been dropped /////////
                # fitting the GARCH_PROCESS on n-1 columns
                cond_var_cov_matrix_inner = DDC_GARCH(rets_stock_factors_checked, interval = interval, p=p, q=q)[0]
    
                inner_betas = np.zeros(factors_rets.shape[1]-1)

                for j in range(1, cond_var_cov_matrix_inner.shape[1]):
                    beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j]
                    inner_betas[j - 1] = beta

                if cols_to_drop[0] > rets_stock_factors_checked.shape[1]-1:
                    inner_betas = np.append(inner_betas, 1)
                else:
                    inner_betas = np.insert(inner_betas, cols_to_drop[0], 1)
                betas_factors[i,:] = inner_betas

    # -------------------------------------------------------------------------------------------------------------------------

        # ensuring positions is a 2D array with shape (n, 1)
        positions = positions.reshape(-1, 1) 

        # performing the matrix multiplication
        betas_factors_array = np.array(betas_factors) 
        factors_positions = betas_factors_array.T @ positions 
        factors_positions = factors_positions.reshape(-1)

        # now that I have the garch(p,q) positions in the factors, I need the garch(p,q) variance_cov_matrix_factors

        variance_cov_matrix_factors = DDC_GARCH(factors_rets, interval = interval, p=p, q=q)[0]

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        from scipy.stats import norm

        # Value at Risk ----------------------------------

        var_port = np.dot(np.dot(factors_positions,variance_cov_matrix_factors),factors_positions.T)  
        sigma = np.sqrt(var_port)

        quantile = norm.ppf(1-alpha, loc=0, scale=1) ; VaR = quantile * sigma 

        VaR =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ((np.exp(-(quantile**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha)))) 
        ES = q_es * sigma 

        ES = max(0,ES)

        SHRP_DIAG = {
            "var" : round(VaR, 4),
            "es" : round(ES, 4),
            "T" : interval,
            "p" : p,
            "q" : q,
            "sigmaPi" : round(sigma,5)
        }

        for i, factor in enumerate(factors_positions, start=1):
            SHRP_DIAG[f"PosFactor{i}"] = round(factor, 4)

    # EWMA (lamdba = lambda_ewma) Volatility --------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    if vol == "ewma":

        from pyriskmgmt.inputsControlFunctions import (check_lambda_ewma)

        # check procedure
        lambda_ewma = check_lambda_ewma(lambda_ewma)

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing the support function ewmaCondCov__Factors(), from the SupportFunctions module

        from pyriskmgmt.SupportFunctions import ewmaCondCov__Factors

        EWMA_COV = ewmaCondCov__Factors

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # computing conditional betas and variances
        betas_factors = np.zeros((returns.shape[1], factors_rets.shape[1]))

        total_loops = returns.shape[1]

        for i in range(total_loops):
            # displaying  the progress
            print(f'\rEWMA MODEL ({lambda_ewma}) - Mapping Procedure -> {factors_rets.shape[1]} Factors: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end='                                   ')
            rets_stock_factors = np.column_stack((returns[:, i], factors_rets))
            cond_var_cov_matrix_inner = EWMA_COV(rets_stock_factors, interval = interval, lambda_ewma = lambda_ewma) 

            inner_betas = np.zeros(factors_rets.shape[1])

            for j in range(1, cond_var_cov_matrix_inner.shape[1]):
                beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j] ; inner_betas[j - 1] = beta
            betas_factors[i,:] = inner_betas

    # -------------------------------------------------------------------------------------------------------------------------

        # ensuring positions is a 2D array with shape (n, 1)
        positions = positions.reshape(-1, 1) 

        # performing the matrix multiplication
        betas_factors_array = np.array(betas_factors) 
        factors_positions = betas_factors_array.T @ positions 
        factors_positions = factors_positions.reshape(-1)

        # now that I have the ewma(lambda_ewma) positions in the factors, I need the ewma(lambda_lambda_ewma) variance_cov_matrix_factors

        variance_cov_matrix_factors = EWMA_COV(factors_rets, interval = interval,lambda_ewma = lambda_ewma)

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        from scipy.stats import norm

        # Value at Risk ----------------------------------

        var_port = np.dot(np.dot(factors_positions.T,variance_cov_matrix_factors),factors_positions)  
        sigma = np.sqrt(var_port)

        q = norm.ppf(1-alpha, loc=0, scale=1) 
        VaR = q * sigma 

        VaR =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ((np.exp(-(q**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha)))) 
        ES = q_es * sigma 

        ES = max(0,ES)

        SHRP_DIAG = {
            "var" : round(VaR, 4),
            "es" : round(ES, 4),
            "T" : interval,
            "lambda_ewma" : lambda_ewma,
            "sigmaPi" : round(sigma,5)
        }

        for i, factor in enumerate(factors_positions, start=1):
            SHRP_DIAG[f"PosFactor{i}"] = round(factor, 4)

    return SHRP_DIAG

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# BINOMIAL TREE <<<<<<<<<<<<<<<<<<

from abc import ABC

class BinomialTree(ABC): 
    """
    Implements an abstract class for the mulit-period binomial pricing models.
    """
    def __init__(self, n, q=0.5):
        self.n = n ; self.q = q
        self.tree = np.zeros([self.n + 1, self.n + 1])

    def printtree(self):
        for i in range(self.n + 1):
            print("t: " + str(i)) 
            print(self.tree[i][: i + 1].round(4)) 

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# STOCK PRICING <<<<<<<<<<<<<<<<<<
            
class StockPricing(BinomialTree): 
    """
    Implements the binomial stock pricing model. 
    """
    def __init__(self, n, S0, u, d, c=0.0):
        super().__init__(n)
        self.S0 = S0 ; self.u = u
        self.d = d ; self.c = c
        self._constructTree()

    def _constructTree(self):
        for i in range(self.n + 1):
            for j in range(i + 1):
                price = self.S0 * (self.u ** j) * (self.d ** (i - j))
                self.tree[i, j] = price

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# BLACK AND SCHOLES PRICE <<<<<<<<<<<<<<<<<<

def BlackAndScholesPrice(St, T, K, sigma, r, option_type):
    """
    Calculate the price of European call and put options using the Black-Scholes model.

    Parameters
    ----------
    - St : float : The spot price of the underlying asset.
    - T : float : The time to expiration of the option, expressed in years.
    - K : float : The strike price of the option.
    - sigma : float : The volatility of the underlying asset's returns.
    - r : float : The risk-free rate of return.
    - option_type : str : The type of the option. Can be either 'call' for call options or 'put' for put options.

    Returns
    -------
    - price : float : The calculated price of the option.

    Raises
    ------
    - ValueError : If `option_type` is not either 'call' or 'put'.
    """

    from scipy.stats import norm ; import numpy as np

    d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = St * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - St * norm.cdf(-d1)

    return price

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# BLACK AND SCHOLES PRICE VECTORIZATION <<<<<<<<<<<<<<<<<<

def BlackAndScholesPriceVectorization(St_s, T_s, K_s, sigma_s, r_s, option_type_s):
    """
    Calculate the price of European call and put options using the Black-Scholes model.
    This version of the function expects arrays as inputs and returns an array of prices.
    """

    from scipy.stats import norm ; import numpy as np

    d1 = (np.log(St_s / K_s) + (r_s + 0.5 * sigma_s ** 2) * T_s) / (sigma_s * np.sqrt(T_s)) 
    d2 = d1 - sigma_s * np.sqrt(T_s)

    # Using np.where to vectorize the 'if' condition
    prices = np.where(option_type_s == 'call', St_s * norm.cdf(d1) - K_s * np.exp(-r_s * T_s) * norm.cdf(d2),
                                        K_s * np.exp(-r_s * T_s) * norm.cdf(-d2) - St_s * norm.cdf(-d1))
    
    # Round the prices to 4 digits
    prices = np.around(prices, 4)

    return prices

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# BLACK AND SCHOLES GREEKS <<<<<<<<<<<<<<<<<<

def CalculateGreeks(St, T, K, sigma, r, option_type):
    """
    Calculate the Greeks (Delta, Gamma, Vega, Theta, Rho) of European call and put options using the Black-Scholes model.

    Parameters
    ----------
    - St : float: The spot price of the underlying asset.
    - T : float: The time to expiration of the option, expressed in years.
    - K : float: The strike price of the option.
    - sigma : float: The volatility of the underlying asset's returns.
    - r : float: The risk-free rate of return.
    - option_type : str: The type of the option. Can be either 'call' for call options or 'put' for put options.

    Returns
    -------
    - delta : float: The rate of change of option price with respect to the price of the underlying asset.
    - gamma : float : The rate of change of the option's Delta with respect to the price of the underlying asset.
    - vega : float: The rate of change of option price with respect to the asset's volatility.
    - theta : float : The rate of change of option price with respect to time to expiry.
    - rho : float: The rate of change of option price with respect to the risk-free interest rate.
    
    Raises
    ------
    - ValueError: If `option_type` is not either 'call' or 'put'.
    """

    from scipy.stats import norm ; import numpy as np

    # Compute d1, a part of the Black-Scholes formula. It incorporates the stock price (St), strike price (K),
    # risk-free rate (r), volatility (sigma), and time to expiration (T).
    d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # Compute d2, which is based on d1 and some other parameters.
    d2 = d1 - sigma * np.sqrt(T)

    # Compute the Delta Greek for a call or put option. Delta measures the rate of change of option price with respect
    # to changes in the underlying asset's price.
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

    # Compute the Gamma Greek. Gamma measures the rate of change in Delta with respect to changes in the underlying price.
    gamma = norm.pdf(d1) / (St * sigma * np.sqrt(T))

    # Compute the Vega Greek. Vega measures sensitivity to volatility.
    vega = St * norm.pdf(d1) * np.sqrt(T)

    # Compute the Theta Greek for a call or put option. Theta measures the rate of change of option price with respect
    # to a change in time to expiration.
    theta = (- (St * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'call' else (- (St * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))

    # Compute the Rho Greek for a call or put option. Rho measures the sensitivity of option price to the interest rate.
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else - K * T * np.exp(-r * T) * norm.cdf(-d2)

    # Return all the calculated Greeks.
    return delta, gamma, vega, theta, rho
    
# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# BINOMIAL AMERICAN OPTION PRICE <<<<<<<<<<<<<<<<<<

def BiAmericanOptionPrice(K , T, S0, r, sigma, option_type,  steps):
    """
    Calculate the price of American call and put options using the binomial tree model.

    Parameters
    ----------
    - K : float: The strike price of the option.
    - T : float : The time to expiration of the option, expressed in years.
    - S0 : float: The spot price of the underlying asset.
    - r : float : The risk-free rate of return.
    - sigma : float : The volatility of the underlying asset's returns.
    - option_type : str : The type of the option. Can be either 'call' for call options or 'put' for put options.
    - steps : int: The number of time steps in the binomial tree model.

    Returns
    -------
    - bi_price : float : The calculated price of the American option.

    Raises
    ------
    - ValueError: If `option_type` is not either 'call' or 'put'.
    """

    import numpy as np
    
    # Calculate the time increment, dt, for each step of the binomial tree, and then calculate the 'up' and 'down' factors (u and d).
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Calculate the risk-neutral probability, p, for an up-move in the underlying and discount factor for each time step.
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Initialize the stock price tree with zeros. This will be populated later.
    price_tree = np.zeros([steps+1, steps+1])

    # Populate the stock price tree for each node using the up and down factors.
    for i in range(steps+1):
        for j in range(i+1):
            price_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Initialize the option price tree with zeros. 
    option = np.zeros([steps+1, steps+1])

    # Calculate the option's intrinsic value at maturity for either call or put options.
    if option_type == 'call':
        option[:, steps] = np.maximum(np.zeros(steps+1), price_tree[:, steps]-K)
    elif option_type == 'put':
        option[:, steps] = np.maximum(np.zeros(steps+1), K-price_tree[:, steps])

    # Backward induction: Starting from the penultimate time step, recursively calculate option values going backwards.
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            # Calculate the expected option value based on the risk-neutral probability.
            option[j, i] = disc * (p * option[j, i+1] + (1-p) * option[j+1, i+1])

            # If the option is American style, it can be exercised early. This part checks if early exercise is optimal.
            if option_type == 'call':
                option[j, i] = np.maximum(option[j, i], price_tree[j, i] - K)
            elif option_type == 'put':
                option[j, i] = np.maximum(option[j, i], K - price_tree[j, i])

    # The option price at the initial node is the binomial model's price for the option.
    bi_price = round(option[0, 0], 4)

    # Return the calculated binomial price.
    return bi_price

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

def BiAmericanOptionPriceVectorized(K_s, T_s, S0_s, r_s, sigma_s, option_type_s, steps_s, inner_function):
    """
    Vectorized function to compute the prices of multiple American options using the binomial tree model.

    Parameters
    ----------
    - K_s: list or array-like : List of strike prices for each option.
    - T_s: list or array-like : List of times to expiration for each option, expressed in years.
    - S0_s: list or array-like : List of initial spot prices of the underlying asset for each option.
    - r_s: list or array-like : List of risk-free rates for each option.
    - sigma_s: list or array-like : List of volatilities of the underlying asset's returns for each option.
    - option_type_s: list or array-like : List of option types. Each value can be either 'call' for call options or 'put' for put options.
    - steps_s : list or array-like : List of number of time steps to be used in the binomial tree model for each option.
    - inner_function: function : Reference to the inner function used for the binomial tree pricing (usually the non-vectorized version of the American option pricing).

    Returns
    -------
    bi_prices : numpy.ndarray : Array of computed prices for each American option.

    Notes
    -----
    This function assumes that the length of each input list or array is the same and corresponds to the number of options being priced.
    """

    num_options = len(K_s)
    bi_prices = np.zeros(num_options)

    BiAmericanOptionPrice = inner_function

    for idx in range(num_options):
        bi_prices[idx] = BiAmericanOptionPrice(K_s[idx], T_s[idx], S0_s[idx], r_s[idx], sigma_s[idx], option_type_s[idx],
                                                steps_s[idx])
    
    return bi_prices

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# QUANTLIB AMERICAN OPTION PRICE <<<<<<<<<<<<<<<<<< 

import QuantLib as ql

def QuantLibAmericanOptionPrice(K: float, T: float, S0: float, r: float, sigma: float, option_type: str, steps: int) -> float:
    """
    Calculate the American Option price using QuantLib's BinomialVanillaEngine for given option parameters.

    Parameters
    ----------
    - K (float): Strike price.
    - T (float): Option maturity in years.
    - S0 (float): Initial stock price.
    - r (float): Risk-free rate.
    - sigma (float): Volatility.
    - option_type (str): Indicates the type of the option ('call' or 'put').
    - steps (int): Number of steps for the binomial tree.

    Returns
    -------
    - price (float): Option price.

    Note
    ----
    - The function utilizes QuantLib's tools and the BinomialVanillaEngine to calculate option prices. The pricing method 
    used is the binomial method with equal probabilities ("eqp") for the tree.
    """

    # Setting up the option's parameters
    calculation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calculation_date
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()

    # Defining option type
    option_type_ql = ql.Option.Call if option_type == 'call' else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type_ql, K)

    # Defining exercise date and option
    exercise_date = calendar.advance(calculation_date, ql.Period(int(T * 365), ql.Days))
    exercise = ql.AmericanExercise(calculation_date, exercise_date)
    american_option = ql.VanillaOption(payoff, exercise)

    # Setting up the processes and engine
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, r, day_count))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), day_count))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    bsm_process = ql.BlackScholesProcess(spot_handle, rate_handle, vol_handle)
    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "eqp", steps)
    
    american_option.setPricingEngine(binomial_engine)

    return american_option.NPV()

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# QUANTLIB AMERICAN OPTION PRICE Vectorized <<<<<<<<<<<<<<<<<< 

import QuantLib as ql

def QuantLibAmericanOptionPriceVectorized(K_s, T_s, S0_s, r_s, sigma_s, option_type_s, steps_s):

    """
    Calculate the American Option prices using QuantLib's BinomialVanillaEngine for multiple option parameters.

    Parameters
    ----------
    - K_s (list or numpy array): Array of strike prices.
    - T_s (list or numpy array): Array of option maturities in years.
    - S0_s (list or numpy array): Array of initial stock prices.
    - r_s (list or numpy array): Array of risk-free rates.
    - sigma_s (list or numpy array): Array of volatilities.
    - option_type_s (list or numpy array): Array indicating the type of the option ('call' or 'put').
    - steps_s (list or numpy array): Array of number of steps for the binomial tree.

    Returns
    -------
    - prices (numpy array): Array of option prices.

    Note
    ----
    - This function isn't truly vectorized as it still uses a loop to iterate over the given arrays. However, it facilitates
    processing multiple options at once by passing arrays of parameters.
    - The function utilizes QuantLib's tools and the BinomialVanillaEngine to calculate option prices. The pricing method 
    used is the binomial method with equal probabilities ("eqp") for the tree.
    """

    # preallocating a NumPy array for storing prices
    num_options = len(K_s) ; prices = np.zeros(num_options)

    # setting up the option's parameters
    calculation_date = ql.Date.todaysDate() ; ql.Settings.instance().evaluationDate = calculation_date ; calendar = ql.NullCalendar() ; day_count = ql.Actual365Fixed()

    # iteratating over arrays (not truly vectorized but we handle arrays)
    for idx, (K, T, S0, r, sigma, option_type, steps) in enumerate(zip(K_s, T_s, S0_s, r_s, sigma_s, option_type_s, steps_s)):

        option_type_ql = ql.Option.Call if option_type == 'call' else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(option_type_ql, K)

        exercise_date = calendar.advance(calculation_date, ql.Period(int(T * 365), ql.Days)) ; exercise = ql.AmericanExercise(calculation_date, exercise_date)
        american_option = ql.VanillaOption(payoff, exercise)

        rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, r, day_count))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), day_count))
        
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0)) ; bsm_process = ql.BlackScholesProcess(spot_handle, rate_handle, vol_handle)

        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "eqp", int(steps)) ; american_option.setPricingEngine(binomial_engine)

        # assigning the price directly to the preallocated array
        prices[idx] = american_option.NPV()

    return prices

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# FOWARD PRICE <<<<<<<<<<<<<<<<<< 

import math

def ForwardPrice(spot_price, risk_free_rate, dividend_yield, time_to_maturity):
    """
    Calculate the forward price of an asset.

    Parameters:
    - spot_price (float): Current price of the asset.
    - risk_free_rate (float): Annual risk-free rate.
    - dividend_yield (float): Annual continuous dividend yield.
    - time_to_maturity (float): Time to maturity of the forward contract in years.

    Returns:
    - (float) Forward price of the asset.
    """
    price = spot_price * math.exp((risk_free_rate - dividend_yield) * time_to_maturity)

    return price

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# FUTURE PRICE <<<<<<<<<<<<<<<<<< 

import math

def FuturesPrice(spot_price, risk_free_rate, dividend_yield, convenience_yield, storage_cost, time_to_maturity):
    
    """
    Calculate the futures price of an asset.

    Parameters:
    - spot_price (float): Current price of the asset.
    - risk_free_rate (float): Annual risk-free rate.
    - dividend_yield (float): Annual continuous dividend yield.
    - convenience_yield (float): Annual continuous convenience yield.
    - storage_cost (float): Annual continuous storage cost.
    - time_to_maturity (float): Time to maturity of the futures contract in years.

    Returns:
    - (float) Futures price of the asset.
    """
    price = spot_price * math.exp((risk_free_rate - dividend_yield + convenience_yield - storage_cost) * time_to_maturity)

    return price

# ******************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************

# TESTR <<<<<<<<<<<<<<<<<< 

def TestR():
    """
    Check the installation status of R and the 'rmgarch' package on the current machine.

    This function checks:
    1. If the R language is installed.
    2. If the 'rmgarch' package is installed in R.

    Prints:
        - Information about the presence or absence of R.
        - If R is installed, information about the presence or absence of the 'rmgarch' package.

    Raises:
        subprocess.CalledProcessError: If there's an error when running R.
        FileNotFoundError: If the R executable is not found.
    """

    import subprocess

    def is_r_installed():
        try:
            subprocess.run(["R", "--slave", "--vanilla", "-e", "0"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def is_r_package_installed(package_name):
        try:
            cmd = [
                "R", "--slave", "--vanilla", "-e",
                f"if (!requireNamespace('{package_name}', quietly = TRUE)) quit(status=1, save = 'no')"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    if is_r_installed():
        print("R is installed on this machine.")
        if is_r_package_installed("rmgarch"):
            print("The 'rmgarch' package is installed within your R environment.")
        else:
            print("The 'rmgarch' package is not installed within your R environment.")
    else:
        print("R is not installed on this machine.")


# ******************************************************************************************************************************************************************
# ****************************************************************************************************************************************************************** 

# garch_warnings <<<<<<<<<

def garch_warnings(string):
            
        print("WARNING:")
        print("When the 'garch' volatility is involved this method integrates Python and R for efficiency (especially due to DCC GARCH models in R).")
        print("Please ensure the following prerequisites are met:")
        print("- An active R environment is required.")
        print("- The 'rmgarch' package must be installed in your R environment.")
        print("If you haven't set this up:")
        print("1. Download and install R from:")
        print("- For macOS: https://cran.r-project.org/bin/macosx/")
        print("- For Windows: https://cran.r-project.org/bin/windows/base/")
        print("2. Run the 'install.packages(\"rmgarch\")' command within your R environment.")
        print(f"Once these steps are completed, the {string} function should operate correctly.")
        print(f"To mute this warning, add 'warning = False' as an argument of the {string} function")


# ******************************************************************************************************************************************************************
# ****************************************************************************************************************************************************************** 
