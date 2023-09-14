
# ------------------------------------------------------------------------------------------
# Author: GIAN MARCO ODDO
# Module: inputsControlFunctions
#     - Part of the pyriskmgmt package (support module)
# Contact Information:
#   - Email: gian.marco.oddo@usi.ch
#   - LinkedIn: https://www.linkedin.com/in/gian-marco-oddo-8a6b4b207/
#   - GitHub: https://github.com/GianMarcoOddo
# Feel free to reach out for any questions or further clarification on this code.
# ------------------------------------------------------------------------------------------

from typing import Union, List ; import numpy as np ; import pandas as pd

## check_position_single ################################################################################################################################################
############################################################################################################################################################################

def check_position_single(position): 
    """
    Check if a position value is valid.

    Args
    position: A position value to check.
    types: A tuple of types that the position value can be. 
    name: A string representing the name of the position value. 
    single_value: A boolean indicating whether the function should only accept a single value. 
    values: A list of valid position values to check against. 
    Returns:
    The validated position value as a string.
    Raises:
    TypeError: If the position value is not one of the types specified in `types`.
    ValueError: If `single_value` is `True` and more than one position value is provided, 
    or if `values` is provided and the position value is not in the list of valid values.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(position, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The position should be a number!")

    # Convert to list
    if isinstance(position, (list, pd.Series, np.ndarray)):
        position = list(position)
        if len(position) == 0: 
            raise ValueError("No position provided")
        # Check for single value
        if len(position) > 1:
            raise ValueError(f"More than one position has been provided. This function works only with one position at the time")
    
    if isinstance(position, (list)):
        position = position[0]

    if not isinstance(position, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"The position should be a number!")

    return position

## check_and_convert_dates ################################################################################################################################################
############################################################################################################################################################################

def check_and_convert_dates(start_date, end_date):

    """
    Check if the input dates are valid and convert them to a datetime object.

    Args:
    start_date: A string representing the start date in one of the following formats: 
    "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", or "%Y_%m_%d", or the string "today" to represent today's date.
    end_date: A string representing the end date in one of the following formats: 
    "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", or "%Y_%m_%d", or the string "today" to represent today's date.
    Returns:
    A tuple containing the start and end dates as datetime objects.
    Raises:
    ValueError: If the start_date or end_date is not in one of the valid formats, or if the 
    end_date is before the start_date, or if the end_date is after today's date, or if the start_date is equal to the end_date.
    """

    from datetime import datetime; import dateutil.parser ; from datetime import date 

    if start_date == "today":
        raise ValueError(f"Please insert in the function a valid start_date format: [%Y-%m-%d, %Y.%m.%d, %Y/%m/%d ,%Y_%m_%d]")
    
    # Function to convert a single date
    def convert_to_date(input, name):
        if input == "today":
            return datetime.today()
        try:
            return dateutil.parser.parse(input)
        except ValueError:
            raise ValueError(f"Please insert in the function a valid {name} format: [%Y-%m-%d, %Y.%m.%d, %Y/%m/%d ,%Y_%m_%d]")

    start_date = convert_to_date(start_date, "start_date")
    end_date = convert_to_date(end_date, "end_date")

    # Get today's date
    today = datetime.today()

    if end_date < start_date:
        raise ValueError("The end_date cannot be before the start_date")
    if end_date > today:
        raise ValueError(f"The end_date cannot be after today's date {str(date.today())}")
    if start_date == end_date:
            raise ValueError("The start_date cannot be equal to the end_date")
    
    return start_date, end_date

## check_alpha #############################################################################################################################################################
############################################################################################################################################################################

def check_alpha(alpha):
    """
    Check if an alpha value or a list of alpha values is valid.
    Args:
    alpha: A float, list, numpy array, or pandas series representing one or more alpha values.
    Returns:
    alpha: A float representing a valid alpha value.
    Raises:
    TypeError: If the alpha value is not a float, list, numpy array, or pandas series.
    ValueError: If more than one alpha value is provided, or if the alpha value is not within the range of 0 to 1.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(alpha, (float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The alpha should be one of the following types: float, list, np.ndarray, pd.Series")

    # Convert to list if input is pd.Series or np.ndarray
    if isinstance(alpha, (pd.Series, np.ndarray)):
        if len(alpha) == 0: 
            raise ValueError("No alpha provided")
        if len(alpha) > 1:
            raise ValueError("More than one alpha provided")
        alpha =  list(alpha)

    # Check if input is list and convert to single float
    if isinstance(alpha, list):
        if len(alpha) == 0: 
            raise ValueError("No alpha provided")
        if len(alpha) > 1:
            raise ValueError("More than one alpha provided")
        alpha = alpha[0]

    # Check alpha range
    if not isinstance(alpha, (float, np.float32, np.float64)):
        raise TypeError("The alpha value should be a number (float)")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("The alpha value should be between 0 and 1")

    return alpha

## check_alpha_test ################################################################################################################################################
############################################################################################################################################################################

def check_alpha_test(alpha_test):
    """
    Check if an alpha_test value or a list of alpha_test values is valid.
    Args:
    alpha_test: A float, list, numpy array, or pandas series representing one or more alpha_test values.
    Returns:
    alpha_test: A float representing a valid alpha value.
    Raises:
    TypeError: If the alpha_test value is not a float, list, numpy array, or pandas series.
    ValueError: If more than one alpha_test value is provided, or if the alpha_test value is not within the range of 0 to 1.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(alpha_test, (float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The alpha_test should be one of the following types: float, list, np.ndarray, pd.Series")

    # Convert to list if input is pd.Series or np.ndarray
    if isinstance(alpha_test, (pd.Series, np.ndarray)):
        if len(alpha_test) == 0: 
            raise ValueError("No alpha_test provided")
        if len(alpha_test) > 1:
            raise ValueError("More than one alpha_test provided")
        alpha =  list(alpha_test)

    # Check if input is list and convert to single float
    if isinstance(alpha_test, list):
        if len(alpha_test) == 0: 
            raise ValueError("No alpha_test provided")
        if len(alpha_test) > 1:
            raise ValueError("More than one alpha_test provided")
        alpha_test = alpha_test[0]

    # Check alpha range
    if not isinstance(alpha_test, (float, np.float32, np.float64)):
        raise TypeError("The alpha_test value should be a number (float)")
    if alpha_test <= 0 or alpha_test >= 1:
        raise ValueError("The alpha_test value should be between 0 and 1")

    return alpha_test

## check_freq ##############################################################################################################################################################
############################################################################################################################################################################

def check_freq(freq, allowed_values=["daily", "weekly","monthly"]):
    """
    Checks whether the freq argument is valid.

    Parameters:
    freq: str or array-like
        The freq to be checked. Should be one of the allowed_values.
        Can be a single string or an array-like object containing a single string.

    allowed_values: list of str, optional
        A list of the values that are allowed for the freq argument.
        Defaults to ["daily"]

    Raises:
    TypeError: If the method argument is not of the correct type.
    ValueError: If more than one method is provided or if the method is not in the list of allowed values.

    Returns:
    method: str
    The validated method argument.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: {types}")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at the time")

    def convert_to_list(input):
        if isinstance(input, (list, pd.Series, np.ndarray)):
            input =  list(input)
        if isinstance(input, list):
            input =  input[0]
        return input

    def check_in_values(input, values, name):
        if not isinstance(input, str):
            raise TypeError("The freq parameter should be a string!")
        if input not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(freq, (str, list, np.ndarray, pd.Series), "freq")
    check_single_value(freq, "freq")
    freq = convert_to_list(freq)
    check_in_values(freq, allowed_values, "freq")

    return freq

## check_method ############################################################################################################################################################
############################################################################################################################################################################

def check_method(method, allowed_values=["quantile", "bootstrap"]):
    """
    Checks whether the method argument is valid.

    Parameters:
    method: str or array-like
        The method to be checked. Should be one of the allowed_values.
        Can be a single string or an array-like object containing a single string.

    allowed_values: list of str, optional
        A list of the values that are allowed for the method argument.
        Defaults to ["quantile", "weighted", "bootstrap"].

    Raises:
    TypeError: If the method argument is not of the correct type.
    ValueError: If more than one method is provided or if the method is not in the list of allowed values.

    Returns:
    method: str
    The validated method argument.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: {types}")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at the time")

    def convert_to_list(input):
        if isinstance(input, (list, pd.Series, np.ndarray)):
            input =  list(input)
        if isinstance(input, list):
            input =  input[0]
        return input

    def check_in_values(input, values, name):
        if not isinstance(input, str):
            raise TypeError("The method parameter should be a string!")
        if input not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(method, (str, list, np.ndarray, pd.Series), "method")
    check_single_value(method, "method")
    freq = convert_to_list(method)
    check_in_values(method, allowed_values, "method")

    return method

## check_n_bootstrap_samples ###############################################################################################################################################
############################################################################################################################################################################

def check_n_bootstrap_samples(n_bootstrap_samples):
    """
    Checks the validity of the input n_bootstrap_samples.

    Parameters:
    n_bootstrap_samples: int, float, or array-like
        The number of bootstrap samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If n_bootstrap_samples is not of the correct type.
    ValueError: If no value is provided for n_bootstrap_samples or if more than one value is provided.

    Returns:
    n_bootstrap_samples: int or float
        The validated number of bootstrap samples.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(n_bootstrap_samples, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The n_bootstrap_samples should be a number!")

    # Convert to list
    if isinstance(n_bootstrap_samples, (list, pd.Series, np.ndarray)):
        n_bootstrap_samples = list(n_bootstrap_samples)
        if len(n_bootstrap_samples) == 0: 
            raise ValueError("No n_bootstrap_samples provided")
        # Check for single value
        if len(n_bootstrap_samples) > 1:
            raise ValueError(f"More than one n_bootstrap_samples has been provided. This function works only with one n_bootstrap_samples at the time")
    
    if isinstance(n_bootstrap_samples, (list)):
        n_bootstrap_samples = n_bootstrap_samples[0]

    if not isinstance(n_bootstrap_samples, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"The n_bootstrap_samples should be a number!")
    
    if n_bootstrap_samples <= 0:
        raise TypeError(f"The n_bootstrap_samples should be a number! higher than 0")

    return n_bootstrap_samples

## check_positions_port ####################################################################################################################################################
############################################################################################################################################################################

def check_positions_port(positions):
    """
    This function validates the input 'positions', ensuring that it is a one-dimensional array-like object (numpy array, pandas series, or list)
    that contains only real numbers, and has at least one non-zero element.

    Parameters
    ----------
    positions : np.ndarray, pd.core.series.Series, list
        The positions of a portfolio. This should be a one-dimensional array-like object containing only real numbers.

    Raises
    ------
    TypeError
        If 'positions' is not a numpy array, pandas Series or list.
        If 'positions' contains elements that are not numbers.
    Exception
        If 'positions' is empty.
        If 'positions' is multi-dimensional.
        If all elements of 'positions' are zero.

    Returns
    -------
    positions : np.ndarray
        The validated positions array.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(positions, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("Positions should be provided in the following object: [np.ndarray, pd.core.series.Series, list]")
    
    if isinstance(positions, pd.Series):
        positions = positions.to_numpy()

        if len(positions) == 0: 
            raise Exception("The Series with the positions is empty")
        
    if isinstance(positions, np.ndarray):
        if len(positions) == 0: 
            raise Exception("The array with the positions is empty")
        
        if positions.ndim > 1:
            raise Exception("The positions array should be a one-dimensional array")
        
        if not np.all(np.isreal(positions)):
            raise TypeError("The array of positions should contain only numbers")
        
    if isinstance(positions, list):
        if len(positions) == 0: 
            raise Exception("The list with the positions is empty")
        
        if np.ndim(positions) > 1:
            raise Exception("The positions list should be a one-dimensional list")
        
        if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in positions):
            raise TypeError("The list of positions should contain only numbers")
        
        positions = np.array(positions)

    # If all the positions are 0, raise an exception
    if np.all(positions == 0):
        raise Exception("All positions are 0. At least one position should be different from 0.")

    return positions

## validate_returns_single #################################################################################################################################################
############################################################################################################################################################################

def validate_returns_single(returns):
    """
    Check if a list, pandas series, numpy array or dataframe of returns is valid and return a numpy array of returns.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of returns for a single asset.
    Returns:
    A numpy array of returns.
    Raises:
    TypeError: If the returns are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of returns is empty.
    Exception: If the returns list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the returns contain non-numeric types.
    Exception: If the returns contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(returns,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("Returns must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(returns,list):
        if len(returns) == 0:
            raise Exception("The list of returns is empty")
        dim = np.ndim(returns)
        if dim > 1:
            raise Exception("The returns list should be one-dimensional or should contain returns for only one asset")
        if not all(isinstance(item, (float, np.float16, np.float32, np.float64)) for item in returns):
            raise TypeError("The list of returns should contain only numbers - (Percentages)")
        returns = np.asarray(returns)

    if isinstance(returns,pd.core.series.Series):
        returns = returns.values
        if len(returns) == 0:
            raise Exception("The Series of returns is empty")
        if not all(isinstance(item, (float, np.float16, np.float32, np.float64)) for item in returns):
            raise TypeError("The Series of returns should contain only numbers - (Percentages)")
        returns = np.asarray(returns)

    if isinstance(returns,np.ndarray):
        if len(returns) == 0:
            raise Exception("The array of returns is empty")
        dim = np.ndim(returns)
        if dim > 1:
            raise Exception("The returns array should be one-dimensional or should contain returns for only one asset")
        if not all(isinstance(item, (float, np.float16, np.float32, np.float64)) for item in returns):
            raise TypeError("The array of returns should contain only numbers")

    if isinstance(returns,pd.DataFrame):
        if returns.empty:
            raise Exception("The DataFrame with the returns is empty")
        if returns.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided, this function works only with one asset at the time")
        for col in returns.columns:
            if not all(isinstance(item, (float, np.float16, np.float32, np.float64)) for item in returns[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers - Percentages")
        if returns.shape[1] == 1:
            returns = returns.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(returns).any():
        raise Exception("The returns contain NaN values")

    return returns

## validate_returns_port ###################################################################################################################################################
############################################################################################################################################################################

def validate_returns_port(returns):
    """
    Validates the format of the returns for a portfolio of assets.
    Args:
    returns : np.ndarray or pd.DataFrame with the returns of a portfolio of assets. 
    Must be provided in the form of an np.array or pd.DataFrame.
    Returns:
    np.ndarray with the returns of a portfolio of assets as a np.ndarray.
    Raises:
    Exception
        If the returns are not provided as an np.array or pd.DataFrame.
        If the np.array with the returns contains less than one column. 
        If the np.array with the returns contains NaN values.
        If the DataFrame with the returns is empty.
        If the DataFrame with the returns contains less than two columns.
        If the DataFrame with the returns contains NaN values.
        If any column of the DataFrame with the returns contains values that are not numbers.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(returns,(np.ndarray, pd.DataFrame)):
        raise Exception("Returns must be provided in the form of: [np.array, pd.DataFrame]. This function works with multiple assets")

    # Array
    if isinstance(returns,np.ndarray):
        if returns.shape[1] < 1:
            raise Exception("The array with the returns contains less than or one column. This function works with more than one asset")

        if np.isnan(returns).any():
            raise Exception("The array with the returns contains NaN values")

        for col in range(returns.shape[1]):
            if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in returns[:,col]):
                raise Exception("The Array of returns should contain only numbers")

    # DataFrame
    if isinstance(returns,pd.DataFrame):
        if returns.empty:
            raise Exception("The DataFrame with the returns is empty")

        if returns.shape[1] == 1:
            raise Exception("The DataFrame with the returns should have more than one column")

        if returns.isna().values.any():
            raise Exception("The DataFrame with the returns contains NaN values")

        for col in returns.columns:
            if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in returns[col]):
                raise Exception(f"The DataFrame column {col} should contain only numbers")

        returns = returns.values

    return returns

## check_scale_factor #####################################################################################################################################################
############################################################################################################################################################################

def check_scale_factor(scale_factor):
    """
    Check if an scale_factor value or a list of scale_factor values is valid.
    Args:
    scale_factor: A float, list, numpy array, or pandas series representing one or more scale_factor values.
    Returns:
    scale_factor
    Raises:
    TypeError: If the scale_factor value is not a float, list, numpy array, or pandas series.
    ValueError: If more than one scale_factor value is provided, or if the scale_factor value is not within the range of 0 to 1.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: {types}")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            if len(input) == 0: 
                raise ValueError("No scale_factor provided")
            if len(input) > 1:
                raise Exception("More than one scale_factor provided")
            input=  list(input)
        if isinstance(input, list):
            if len(input) == 0: 
                raise ValueError("No scale_factor provided")
            if len(input) > 1:
                raise Exception("More than one scale_factor provided")
            input =  input[0]
        return input

    def check_scale_factor_range(scale_factor):
        if not isinstance(scale_factor, (float, np.float16, np.float32, np.float64)):
            raise TypeError("The scale_factor value should be a number (float)")
        if scale_factor <= 0 or scale_factor >= 1:
            raise ValueError("The scale_factor value should be between 0 and 1")

    check_input_type(scale_factor, (float, np.float16, np.float32, np.float64, list, np.ndarray, pd.Series), "scale_factor")
    scale_factor = convert_to_list(scale_factor)
    check_single_value(scale_factor, "scale_factor")
    check_scale_factor_range(scale_factor)

    return scale_factor

## check_interval #########################################################################################################################################################
############################################################################################################################################################################

def check_interval(interval):
    """
    This function checks if the provided interval parameter is valid for statistical analysis.
    Args:
    interval: (np.ndarray, pd.core.series.Series, list, number)
        The interval parameter to be checked for validity.
    Returns:
    interval: (int)
        The validated interval to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the interval parameter is a pandas DataFrame object.
        - If the interval parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the interval parameter is empty.
        - If the interval array/list is more than one-dimensional.
        - If there is more than one interval parameter provided.
        - If the interval parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if interval is a DataFrame
    if isinstance(interval, pd.core.frame.DataFrame):
        raise TypeError("The interval parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if interval is a Series
    if isinstance(interval, pd.core.series.Series):
        if len(interval) == 0:
            raise Exception("The Series with the interval parameter is empty")
        interval = list(interval)

    # Handle if interval is a list or ndarray
    if isinstance(interval, (np.ndarray, list)):
        if len(interval) == 0:
            raise Exception("The array/list with the interval parameter is empty")
        dim = np.ndim(interval)
        if dim > 1:
            raise Exception("The interval array/list should be one-dimensional")
        if len(interval) > 1:
            raise Exception("More than one interval provided")
        interval = interval[0]
        
    # Handle if interval is a number
    if not isinstance(interval, (int, np.int32, np.int64)):
        raise TypeError("The interval parameter must be an integer ")

    # Ensure the value of interval higher than one
    if interval < 1:
        raise Exception("Please, insert a correct value for interval parameter! > 1)")

    return interval

## check_quantile_threshold ################################################################################################################################################
############################################################################################################################################################################

def check_quantile_threshold(quantile_threshold):
    """
    This function validates the input 'quantile_threshold', ensuring that it is a scalar value within the range [0.90, 1] (inclusive).
    The input can be provided as a single-value numpy array, pandas Series, list, or number. If the input is a list or array-like 
    with more than one element, only the first element will be used.

    Parameters
    ----------
    quantile_threshold : np.ndarray, pd.core.series.Series, list, number
        The quantile threshold for the risk measure calculation. This should be a scalar value between 0.90 and 1. 

    Raises
    ------
    TypeError
        If 'quantile_threshold' is a DataFrame or a type not listed above.
        If 'quantile_threshold' is not a numeric type when provided as a single value.
    Exception
        If 'quantile_threshold' is empty.
        If 'quantile_threshold' contains more than one value.
        If 'quantile_threshold' is not within the range [0.90, 1].

    Returns
    -------
    quantile_threshold : float
        The validated quantile threshold value.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if quantile_threshold is a DataFrame
    if isinstance(quantile_threshold, pd.core.frame.DataFrame):
        raise TypeError("The quantile_threshold parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if quantile_threshold is a Series
    if isinstance(quantile_threshold, pd.core.series.Series):
        if len(quantile_threshold) == 0:
            raise Exception("The Series with the quantile_threshold parameter is empty")
        quantile_threshold = list(quantile_threshold)

    # Handle if quantile_threshold is a list or ndarray

    if isinstance(quantile_threshold, (np.ndarray, list)):
        if len(quantile_threshold) == 0:
            raise Exception("The array/list with the quantile_threshold parameter is empty")
        if len(quantile_threshold) > 1:
            raise Exception("The array/list with the quantile_threshold parameter should contain only one value")
        dim = np.ndim(quantile_threshold)
        if dim > 1:
            raise Exception("The quantile_threshold array/list should be one-dimensional")
        quantile_threshold = quantile_threshold[0]

    # Handle if quantile_threshold is a number
    if not isinstance(quantile_threshold, (float, np.float16, np.float32, np.float64)):
        raise TypeError("The quantile_threshold parameter must be a float number between 0 and 1")

    # Ensure the value of quantile_threshold is between 0.90 and 1
    if quantile_threshold > 1 or quantile_threshold < 0.90:
        raise Exception("Please, insert a correct value for quantile_threshold parameter! Between 0.90 and 1 (i.e for 95% insert 0.95)")

    return quantile_threshold

## check_lambda_ewma ######################################################################################################################################################
############################################################################################################################################################################

def check_lambda_ewma(lambda_ewma):
    """
    Check if an lambda_ewma value or a list of lambda_ewma values is valid.
    Args:
    lambda_ewma: A float, list, numpy array, or pandas series representing one or more lambda_ewma values.
    Returns:
    lambda_ewma
    Raises:
    TypeError: If the lambda_ewma value is not a float, list, numpy array, or pandas series.
    ValueError: If more than one lambda_ewma value is provided, or if the lambda_ewma value is not within the range of 0 to 1.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: [float, list, np.ndarray, pd.Series]")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            if len(input) == 0: 
                raise ValueError("No lambda_ewma provided")
            if len(input) > 1:
                raise Exception("More than one lambda_ewma provided")
            input=  list(input)
        if isinstance(input, list):
            if len(input) == 0: 
                raise ValueError("No lambda_ewma provided")
            if len(input) > 1:
                raise Exception("More than one lambda_ewma provided")
            input =  input[0]
        return input

    def check_lambda_ewma(lambda_ewma):
        if not isinstance(lambda_ewma, (float, np.float16, np.float32, np.float64)):
            raise TypeError("The lambda_ewma value should be a number (float)")
        if lambda_ewma <= 0 or lambda_ewma >= 1:
            raise ValueError("The lambda_ewma value should be between 0 and 1")

    check_input_type(lambda_ewma, (float, np.float16, np.float32, np.float64, list, np.ndarray, pd.Series), "lambda_ewma")
    lambda_ewma = convert_to_list(lambda_ewma)
    check_single_value(lambda_ewma, "lambda_ewma")
    check_lambda_ewma(lambda_ewma)

    return lambda_ewma

## check_number_simulations ################################################################################################################################################
############################################################################################################################################################################

def check_number_simulations(number_simulations):
    """
    Checks the validity of the input number_simulations.

    Parameters:
    number_simulations: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If number_simulations is not of the correct type.
    ValueError: If no value is provided for number_simulations or if more than one value is provided.

    Returns:
    number_simulations: int or float
        The validated number of number_simulations.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(number_simulations, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The number_simulations should be a number!")

    # Convert to list
    if isinstance(number_simulations, (list, pd.Series, np.ndarray)):
        number_simulations = list(number_simulations)
        if len(number_simulations) == 0: 
            raise ValueError("No number_simulations provided")
        # Check for single value
        if len(number_simulations) > 1:
            raise ValueError(f"More than one number_simulations has been provided. This function works only with one number_simulations at the time")
    
    if isinstance(number_simulations, (list)):
        number_simulations = number_simulations[0]

    if not isinstance(number_simulations, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"The number_simulations should be a number!")
    
    if number_simulations <= 0:
        raise TypeError(f"The number_simulations should be a number! higher than 0")

    return number_simulations

## check_vol ###############################################################################################################################################################
############################################################################################################################################################################

def check_vol(vol, allowed_values=["garch", "ewma","simple"]):
    """
    Check if a vol value or a list of vol values is valid.
    Args:
    vol: A string, list, numpy array, or pandas series representing one or more vol values.
    allowed_values: A list of valid vol values.
    Returns:
    None
    Raises:
    TypeError: If the vol value is not a string, list, numpy array, or pandas series.
    ValueError: If more than one vol value is provided, or if the vol value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(vol, (str, list, np.ndarray, pd.Series)):
        raise TypeError("The vol should be one of the following types: (str, list, np.ndarray, pd.Series)")

    if isinstance(vol, (pd.Series, np.ndarray)):
        vol = list(vol)

    if isinstance(vol, list):
        if len(vol) > 1:
            raise ValueError("More than one vol has been provided. This function works only with one vol at a time")
        if len(vol) == 1:
            vol = vol[0]

    if vol not in allowed_values:
        raise ValueError(f"Please, be sure to use a correct vol! {allowed_values}")
    
    return vol

## check_var ###############################################################################################################################################################
############################################################################################################################################################################

def check_var(var):
    """
    Checks the validity of the input variable 'var'.

    Parameters:
    var: int, float, or array-like
        The input to be checked. Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If 'var' is not of the correct type.
    ValueError: If no value is provided for 'var' or if more than one value is provided.

    Returns:
    var: int or float
        The validated variable.
    """
    import numpy as np ; import pandas as pd

    # Check input type
    if not isinstance(var, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError("'var' should be a number!")

    # Convert to list
    if isinstance(var, (list, np.ndarray, pd.Series)):
        var = list(var)
        if len(var) == 0: 
            raise ValueError("No value provided for 'var'")
        # Check for single value
        if len(var) > 1:
            raise ValueError("More than one value has been provided. This function works only with one number at the time")
        var = var[0]

    # Check for correct number type
    if not isinstance(var, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError("'var' should be a number!")

    # Check if var is non-positive
    if var <= 0:
        raise ValueError("'var' should be a positive number!")

    return var

## check_p #################################################################################################################################################################
############################################################################################################################################################################

def check_p(p):
    """
    This function checks if the provided p parameter is valid for statistical analysis.
    Args:
    p: (np.ndarray, pd.core.series.Series, list, number)
        The p parameter to be checked for validity.
    Returns:
    p: (int)
        The validated p to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the p parameter is a pandas DataFrame object.
        - If the p parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the p parameter is empty.
        - If the p array/list is more than one-dimensional.
        - If there is more than one p parameter provided.
        - If the p parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if p is a DataFrame
    if isinstance(p, pd.core.frame.DataFrame):
        raise TypeError("The p parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if p is a Series
    if isinstance(p, pd.core.series.Series):
        if len(p) == 0:
            raise Exception("The Series with the p parameter is empty")
        p = list(p)

    # Handle if p is a list or ndarray
    if isinstance(p, (np.ndarray, list)):
        if len(p) == 0:
            raise Exception("The array/list with the p parameter is empty")
        dim = np.ndim(p)
        if dim > 1:
            raise Exception("The p array/list should be one-dimensional")
        if len(p) > 1:
            raise Exception("More than one p provided")
        p = p[0]
        
    # Handle if p is a number
    if not isinstance(p, (int, np.int32, np.int64)):
        raise TypeError("The p parameter must be an integer ")

    # Ensure the value of p higher than one
    if p < 1:
        raise Exception("Please, insert a correct value for p parameter! > 1)")

    return p

## check_q #################################################################################################################################################################
############################################################################################################################################################################

def check_q(q):
    """
    This function checks if the provided q parameter is valid for statistical analysis.
    Args:
    q: (np.ndarray, pd.core.series.Series, list, number)
        The q parameter to be checked for validity.
    Returns:
    q: (int)
        The validated q to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the q parameter is a pandas DataFrame object.
        - If the q parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the q parameter is empty.
        - If the q array/list is more than one-dimensional.
        - If there is more than one q parameter provided.
        - If the q parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if q is a DataFrame
    if isinstance(q, pd.core.frame.DataFrame):
        raise TypeError("The q parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if q is a Series
    if isinstance(q, pd.core.series.Series):
        if len(q) == 0:
            raise Exception("The Series with the q parameter is empty")
        q = list(q)

    # Handle if q is a list or ndarray
    if isinstance(q, (np.ndarray, list)):
        if len(q) == 0:
            raise Exception("The array/list with the q parameter is empty")
        dim = np.ndim(q)
        if dim > 1:
            raise Exception("The q array/list should be one-dimensional")
        if len(q) > 1:
            raise Exception("More than one q provided")
        q = q[0]
        
    # Handle if q is a number
    if not isinstance(q, (int, np.int32, np.int64 )):
        raise TypeError("The q parameter must be an integer ")

    # Ensure the value of q higher than one
    if q < 1:
        raise Exception("Please, insert a correct value for q parameter! > 1)")

    return q

## check_num_stocks #######################################################################################################################################################
############################################################################################################################################################################

def check_num_stocks(num_stocks):
    """
    This function checks if the provided num_stocks parameter is valid for statistical analysis.
    Args:
    num_stocks: (np.ndarray, pd.core.series.Series, list, number)
        The num_stocks parameter to be checked for validity.
    Returns:
    num_stocks: (int)
        The validated num_stocks to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the num_stocks parameter is a pandas DataFrame object.
        - If the num_stocks parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the num_stocks parameter is empty.
        - If the num_stocks array/list is more than one-dimensional.
        - If there is more than one num_stocks parameter provided.
        - If the num_stocks parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if num_stocks is a DataFrame
    if isinstance(num_stocks, pd.core.frame.DataFrame):
        raise TypeError("The num_stocks parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if num_stocks is a Series
    if isinstance(num_stocks, pd.core.series.Series):
        if len(num_stocks) == 0:
            raise Exception("The Series with the num_stocks parameter is empty") 
        num_stocks = list(num_stocks)

    # Handle if num_stocks is a list or ndarray
    if isinstance(num_stocks, (np.ndarray, list)):
        if len(num_stocks) == 0:
            raise Exception("The array/list with the num_stocks parameter is empty")
        dim = np.ndim(num_stocks)
        if dim > 1:
            raise Exception("The num_stocks array/list should be one-dimensional")
        if len(num_stocks) > 1:
            raise Exception("More than one num_stocks provided")
        num_stocks = num_stocks[0]
        
    # Handle if num_stocks is a number
    if not isinstance(num_stocks, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError("The num_stocks parameter must be an number")

    return num_stocks

## check_S0 ################################################################################################################################################################
############################################################################################################################################################################

def check_S0(S0):
    """
    Checks the validity of the input S0.

    Parameters:
    S0: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If S0 is not of the correct type.
    ValueError: If no value is provided for S0 or if more than one value is provided.

    Returns:
    S0: int or float
        The validated number of S0.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(S0, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The S0 should be a number!")

    # Convert to list
    if isinstance(S0, (list, pd.Series, np.ndarray)):
        S0 = list(S0)
        if len(S0) == 0: 
            raise ValueError("No S0 parameter provided")
        # Check for single value
        if len(S0) > 1:
            raise ValueError(f"More than one S0 has been provided. This function works only with one S0 parameter at the time")
    
    if isinstance(S0, (list)):
        S0 = S0[0]

    if not isinstance(S0, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"S0 should be a number!")
    
    if S0 <= 0:
        raise TypeError(f"S0 should be a number! higher than 0 --- Stocks' prices cannot be negative...")

    return S0

## check_S0_initial ########################################################################################################################################################
############################################################################################################################################################################

def check_S0_initial(S0_initial):
    """
    Checks the validity of the input S0_initial.

    Parameters:
    S0_initial: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If S0_initial is not of the correct type.
    ValueError: If no value is provided for S0_initial or if more than one value is provided.

    Returns:
    S0: int or float
        The validated number of S0_initial.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(S0_initial, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The S0_initial should be a number!")

    # Convert to list
    if isinstance(S0_initial, (list, pd.Series, np.ndarray)):
        S0_initial = list(S0_initial)
        if len(S0_initial) == 0: 
            raise ValueError("No S0_initial parameter provided")
        # Check for single value
        if len(S0_initial) > 1:
            raise ValueError(f"More than one S0_initial has been provided. This function works only with one S0_initial parameter at the time")
    
    if isinstance(S0_initial, (list)):
        S0_initial = S0_initial[0]

    if not isinstance(S0_initial, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"S0_initial should be a number!")
    
    if S0_initial <= 0:
        raise TypeError(f"S0_initial should be a number! higher than 0 --- Stocks' prices cannot be negative...")

    return S0_initial

## check_zero_drift ########################################################################################################################################################
############################################################################################################################################################################
 
def check_zero_drift(zero_drift, allowed_values=[True, False]): # Not in use <<<< For future developement
    """
    Check if a zero_drift value or a list of zero_drift values is valid.
    Args:
    zero_drift: A boolean, string, list, numpy array, or pandas series representing one or more zero_drift values.
    allowed_values: A list of valid zero_drift values.
    Returns:
    None
    Raises:
    TypeError: If the zero_drift value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one zero_drift value is provided, or if the zero_drift value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: {types}")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(zero_drift, (bool, str, list, np.ndarray, pd.Series), "zero_drift")
    zero_drift = convert_to_list(zero_drift)
    check_single_value(zero_drift, "zero_drift")
    check_in_values(zero_drift, allowed_values, "zero_drift")

## check_mu ################################################################################################################################################################
############################################################################################################################################################################

def check_mu(mu, allowed_values=["zero", "constant", "moving_average"]):
    """
    Check if a mu value or a list of mu values is valid.
    Args:
    mu: A string, list, numpy array, or pandas series representing one or more mu values.
    allowed_values: A list of valid mu values.
    Returns:
    None
    Raises:
    TypeError: If the mu value is not a string, list, numpy array, or pandas series.
    ValueError: If more than one mu value is provided, or if the mu value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(mu, (str, list, np.ndarray, pd.Series)):
        raise TypeError("The mu should be one of the following types: (str, list, np.ndarray, pd.Series)")

    if isinstance(mu, (pd.Series, np.ndarray)):
        mu = list(mu)

    if isinstance(mu, list):
        if len(mu) > 1:
            raise ValueError("More than one mu has been provided. This function works only with one mu at a time")
        if len(mu) == 1:
            mu = mu[0]

    if mu not in allowed_values:
        raise ValueError(f"Please, be sure to use a correct mu! {allowed_values}")
    
    return mu 

## check_ma_window #########################################################################################################################################################
############################################################################################################################################################################

def check_ma_window(ma_window):
    """
    This function checks if the provided ma_window parameter is valid for statistical analysis.
    Args:
    ma_window: (np.ndarray, pd.core.series.Series, list, number)
        The ma_window parameter to be checked for validity.
    Returns:
    ma_window: (int)
        The validated ma_window to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the ma_window parameter is a pandas DataFrame object.
        - If the ma_window parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the ma_window parameter is empty.
        - If the ma_window array/list is more than one-dimensional.
        - If there is more than one ma_window parameter provided.
        - If the ma_window parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if ma_window is a DataFrame
    if isinstance(ma_window, pd.core.frame.DataFrame):
        raise TypeError("The ma_window parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if ma_window is a Series
    if isinstance(ma_window, pd.core.series.Series):
        if len(ma_window) == 0:
            raise Exception("The Series with the ma_window parameter is empty")
        ma_window = list(ma_window)

    # Handle if ma_window is a list or ndarray
    if isinstance(ma_window, (np.ndarray, list)):
        if len(ma_window) == 0:
            raise Exception("The array/list with the ma_window parameter is empty")
        dim = np.ndim(ma_window)
        if dim > 1:
            raise Exception("The ma_window array/list should be one-dimensional")
        if len(ma_window) > 1:
            raise Exception("More than one ma_window provided")
        ma_window = ma_window[0]
        
    # Handle if ma_window is a number
    if not isinstance(ma_window, (int, np.int32, np.int64)):
        raise TypeError("The ma_window parameter must be an integer ")

    # Ensure the value of mu_window higher than one
    if ma_window < 1:
        raise Exception("Please, insert a correct value for ma_window parameter! > 1)")

    return ma_window

## check_sharpe_diagonal ###################################################################################################################################################
############################################################################################################################################################################

def check_sharpe_diagonal(sharpe_diagonal, allowed_values=[True, False]):
    """
    Check if a sharpe_diagonal value or a list of sharpe_diagonal values is valid.
    Args:
    sharpe_diagonal: A boolean, string, list, numpy array, or pandas series representing one or more sharpe_diagonal values.
    allowed_values: A list of valid sharpe_diagonal values.
    Returns:
    None
    Raises:
    TypeError: If the sharpe_diagonal value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one sharpe_diagonal value is provided, or if the sharpe_diagonal value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: bool, str, list, np.ndarray, pd.Series")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(sharpe_diagonal, (bool, str, list, np.ndarray, pd.Series), "sharpe_diagonal")
    sharpe_diagonal = convert_to_list(sharpe_diagonal)
    check_single_value(sharpe_diagonal, "sharpe_diagonal")
    check_in_values(sharpe_diagonal, allowed_values, "sharpe_diagonal")

## check_warning ###########################################################################################################################################################
############################################################################################################################################################################

def check_warning(warning, allowed_values=[True, False]):
    """
    Check if a warning value or a list of warning values is valid.
    Args:
    warning: A boolean, string, list, numpy array, or pandas series representing one or more warning values.
    allowed_values: A list of valid warning values.
    Returns:
    None
    Raises:
    TypeError: If the warning value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one warning value is provided, or if the warning value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(warning, (bool, str, list, np.ndarray, pd.Series), "warning")
    warning = convert_to_list(warning)
    check_single_value(warning, "warning")
    check_in_values(warning, allowed_values, "warning")

## check_zero_mean #########################################################################################################################################################
############################################################################################################################################################################

def check_zero_mean(zero_mean, allowed_values=[True, False]):  # Not in use <<<< For future developement
    """
    Check if a zero_mean value or a list of zero_mean values is valid.
    Args:
    zero_mean: A boolean, string, list, numpy array, or pandas series representing one or more zero_mean values.
    allowed_values: A list of valid zero_mean values.
    Returns:
    None
    Raises:
    TypeError: If the zero_mean value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one zero_mean value is provided, or if the zero_mean value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(zero_mean, (bool, str, list, np.ndarray, pd.Series), "zero_mean")
    zero_mean = convert_to_list(zero_mean)
    check_single_value(zero_mean, "zero_mean")
    check_in_values(zero_mean, allowed_values, "zero_mean")

## check_sharpeDiag ########################################################################################################################################################
############################################################################################################################################################################

def check_sharpeDiag(sharpeDiag, allowed_values=[True, False]):
    """
    Check if a sharpeDiag value or a list of sharpeDiag values is valid.
    Args:
    sharpeDiag: A boolean, string, list, numpy array, or pandas series representing one or more sharpeDiag values.
    allowed_values: A list of valid sharpeDiag values.
    Returns:
    None
    Raises:
    TypeError: If the sharpeDiag value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one sharpeDiag value is provided, or if the sharpeDiag value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(sharpeDiag, (bool, str, list, np.ndarray, pd.Series), "sharpeDiag")
    sharpeDiag = convert_to_list(sharpeDiag)
    check_single_value(sharpeDiag, "sharpeDiag")
    check_in_values(sharpeDiag, allowed_values, "sharpeDiag")


## check_VCV ###############################################################################################################################################################
############################################################################################################################################################################

def check_VCV(VCV, allowed_values=[True, False]):
    """
    Check if a VCV value or a list of VCV values is valid.
    Args:
    VCV: A boolean, string, list, numpy array, or pandas series representing one or more VCV values.
    allowed_values: A list of valid VCV values.
    Returns:
    None
    Raises:
    TypeError: If the VCV value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one VCV value is provided, or if the VCV value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(VCV, (bool, str, list, np.ndarray, pd.Series), "VCV")
    VCV = convert_to_list(VCV)
    check_single_value(VCV, "VCV")
    check_in_values(VCV, allowed_values, "VCV")

## check_allow_short #######################################################################################################################################################
############################################################################################################################################################################

def check_allow_short(allow_short, allowed_values=[True, False]):
    """
    Check if a allow_short value or a list of allow_short values is valid.
    Args:
    allow_short: A boolean, string, list, numpy array, or pandas series representing one or more allow_short values.
    allowed_values: A list of valid allow_short values.
    Returns:
    None
    Raises:
    TypeError: If the allow_short value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one allow_short value is provided, or if the allow_short value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(allow_short, (bool, str, list, np.ndarray, pd.Series), "allow_short")
    allow_short = convert_to_list(allow_short)
    check_single_value(allow_short, "allow_short")
    check_in_values(allow_short, allowed_values, "allow_short")

## check_num_stocks_s #####################################################################################################################################################
############################################################################################################################################################################

def check_num_stocks_s(num_stocks_s):
    """
    Check if a list, pandas series, numpy array or dataframe of num_stocks_s is valid and return a numpy array of num_stocks_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of num_stocks_s for a multiple options.
    Returns:
    A numpy array of num_stocks_s.
    Raises:
    TypeError: If the num_stocks_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the num_stocks_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the num_stocks_s contain non-numeric types.
    Exception: If the num_stocks_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(num_stocks_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("num_stocks_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(num_stocks_s,list):
        if len(num_stocks_s) == 0:
            raise Exception("The list of num_stocks_s is empty")
        dim = np.ndim(num_stocks_s)
        if dim > 1:
            raise Exception("The num_stocks_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_stocks_s):
            raise TypeError("The list of num_stocks_s should contain only numbers")
        num_stocks_s = np.asarray(num_stocks_s)

    if isinstance(num_stocks_s,pd.core.series.Series):
        num_stocks_s = num_stocks_s.values
        if len(num_stocks_s) == 0:
            raise Exception("The Series of num_stocks_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_stocks_s):
            raise TypeError("The Series of num_stocks_s should contain only numbers")
        num_stocks_s = np.asarray(num_stocks_s)

    if isinstance(num_stocks_s,np.ndarray):
        if len(num_stocks_s) == 0:
            raise Exception("The array of num_stocks_s is empty")
        dim = np.ndim(num_stocks_s)
        if dim > 1:
            raise Exception("The num_stocks_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_stocks_s):
            raise TypeError("The array of num_stocks_s should contain only numbers")

    if isinstance(num_stocks_s,pd.DataFrame):
        if num_stocks_s.empty:
            raise Exception("The DataFrame with the num_stocks_s is empty")
        if num_stocks_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in num_stocks_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_stocks_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if num_stocks_s.shape[1] == 1:
            num_stocks_s = num_stocks_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(num_stocks_s).any():
        raise Exception("The num_stocks_s contain NaN values")

    return num_stocks_s

## check_K #################################################################################################################################################################
############################################################################################################################################################################

def check_K(K):
    """
    Checks the validity of the input K.

    Parameters:
    K: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If K is not of the correct type.
    ValueError: If no value is provided for K or if more than one value is provided.

    Returns:
    K: int or float
        The validated number of K.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(K, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The K parameter should be a number!")

    # Convert to list
    if isinstance(K, (list, pd.Series, np.ndarray)):
        K = list(K)
        if len(K) == 0: 
            raise ValueError("No K provided")
        # Check for single value
        if len(K) > 1:
            raise ValueError(f"More than one K has been provided. This function works only with one K at the time")
    
    if isinstance(K, (list)):
        K = K[0]

    if not isinstance(K, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"K should be a number!")
    
    if K <= 0:
        raise TypeError(f"K should be a number! higher than 0 --- Strike prices cannot be negative...")

    return K

## check_T #################################################################################################################################################################
############################################################################################################################################################################

def check_T(T):
    """
    Checks the validity of the input T.

    Parameters:
    T: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If T is not of the correct type.
    ValueError: If no value is provided for T or if more than one value is provided.

    Returns:
    T: int or float
        The validated number of T.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(T, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The T parameter should be a number!")

    # Convert to list
    if isinstance(T, (list, pd.Series, np.ndarray)):
        T = list(T)
        if len(T) == 0: 
            raise ValueError("No T provided")
        # Check for single value
        if len(T) > 1:
            raise ValueError(f"More than one T has been provided. This function works only with one T at the time")
    
    if isinstance(T, (list)):
        T = T[0]

    if not isinstance(T, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"T should be a number!")
    
    if T <= 0:
        raise TypeError(f"T should be a number! higher than 0 --- Time to maturity cannot be negative...")

    return T

## check_r #################################################################################################################################################################
############################################################################################################################################################################

def check_r(r):
    """
    Checks the validity of the input r.

    Parameters:
    r: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If r is not of the correct type.
    ValueError: If no value is provided for r or if more than one value is provided.

    Returns:
    r: int or float
        The validated number of r.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(r, (float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The r parameter should be a number!")

    # Convert to list
    if isinstance(r, (list, pd.Series, np.ndarray)):
        r = list(r)
        if len(r) == 0: 
            raise ValueError("No r parameter provided")
        # Check for single value
        if len(r) > 1:
            raise ValueError(f"More than one r has been provided. This function works only with one r at the time")
    
    if isinstance(r, (list)):
        r = r[0]

    if not isinstance(r, (float, np.float32, np.float64)):
        raise TypeError(f"r should be a number!")
    
    if r < 0:
        raise TypeError(f"r should be a number! higher than 0 --- this model works only with a positive r")

    return r

## check_sigma_deri ########################################################################################################################################################
############################################################################################################################################################################

def check_sigma_deri(sigma):
    """
    Checks the validity of the input sigma.

    Parameters:
    sigma: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If sigma is not of the correct type.
    ValueError: If no value is provided for sigma or if more than one value is provided.

    Returns:
    sigma: int or float
        The validated number of sigma.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(sigma, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The sigma parameter should be a number! float")

    # Convert to list
    if isinstance(sigma, (list, pd.Series, np.ndarray)):
        sigma = list(sigma)
        if len(sigma) == 0: 
            raise ValueError("No sigma parameter provided")
        # Check for single value
        if len(sigma) > 1:
            raise ValueError(f"More than one sigma has been provided. This function works only with one sigma at the time")
    
    if isinstance(sigma, (list)):
        sigma = sigma[0]

    if not isinstance(sigma, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"sigma should be a number! float")
    
    if sigma <= 0:
        raise TypeError(f"sigma should be a number! higher than 0 --- Annualized sigma cannot be negative...")

    return sigma

## check_option_type #######################################################################################################################################################
############################################################################################################################################################################

def check_option_type(option_type, allowed_values=["call", "put"]):
    """
    Check if a option_type value or a list of option_type values is valid.
    Args:
    option_type: A string, list, numpy array, or pandas series representing one or more option_type values.
    allowed_values: A list of valid option_type values.
    Returns:
    None
    Raises:
    TypeError: If the option_type value is not a string, list, numpy array, or pandas series.
    ValueError: If more than one option_type value is provided, or if the option_type value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(option_type, (str, list, np.ndarray, pd.Series)):
        raise TypeError("The option_type should be one of the following types: (str, list, np.ndarray, pd.Series)")

    if isinstance(option_type, (pd.Series, np.ndarray)):
        option_type = list(option_type)

    if isinstance(option_type, list):
        if len(option_type) > 1:
            raise ValueError("More than one option_type has been provided. This function works only with one option_type at a time")
        if len(option_type) == 1:
            option_type = option_type[0]

    if option_type not in allowed_values:
        raise ValueError(f"Please, be sure to use a correct option_type! {allowed_values}")
    
    return option_type

## check_num_options #######################################################################################################################################################
############################################################################################################################################################################

def check_num_options(num_options):
    """
    This function checks if the provided num_options parameter is valid for statistical analysis.
    Args:
    num_options: (np.ndarray, pd.core.series.Series, list, number)
        The num_options parameter to be checked for validity.
    Returns:
    num_options: (int)
        The validated num_options to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the num_options parameter is a pandas DataFrame object.
        - If the num_options parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the num_options parameter is empty.
        - If the num_options array/list is more than one-dimensional.
        - If there is more than one num_options parameter provided.
        - If the num_options parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if num_options is a DataFrame
    if isinstance(num_options, pd.core.frame.DataFrame):
        raise TypeError("The num_options parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if num_options is a Series
    if isinstance(num_options, pd.core.series.Series):
        if len(num_options) == 0:
            raise Exception("The Series with the num_options parameter is empty")
        num_options = list(num_options)

    # Handle if num_options is a list or ndarray
    if isinstance(num_options, (np.ndarray, list)):
        if len(num_options) == 0:
            raise Exception("The array/list with the num_options parameter is empty")
        dim = np.ndim(num_options)
        if dim > 1:
            raise Exception("The num_options array/list should be one-dimensional")
        if len(num_options) > 1:
            raise Exception("More than one num_options provided")
        num_options = num_options[0]
        
    # Handle if num_options is a number
    if not isinstance(num_options, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError("The num_options parameter must be an number")

    return num_options

## check_contract_size #####################################################################################################################################################
############################################################################################################################################################################

def check_contract_size(contract_size):
    """
    This function checks if the provided contract_size parameter is valid for statistical analysis.
    Args:
    contract_size: (np.ndarray, pd.core.series.Series, list, number)
        The contract_size parameter to be checked for validity.
    Returns:
    contract_size: (int)
        The validated contract_size to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the contract_size parameter is a pandas DataFrame object.
        - If the contract_size parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the contract_size parameter is empty.
        - If the contract_size array/list is more than one-dimensional.
        - If there is more than one contract_size parameter provided.
        - If the contract_size parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if contract_size is a DataFrame
    if isinstance(contract_size, pd.core.frame.DataFrame):
        raise TypeError("The contract_size parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if contract_size is a Series
    if isinstance(contract_size, pd.core.series.Series):
        if len(contract_size) == 0:
            raise Exception("The Series with the contract_size parameter is empty")
        contract_size = list(contract_size)

    # Handle if contract_size is a list or ndarray
    if isinstance(contract_size, (np.ndarray, list)):
        if len(contract_size) == 0:
            raise Exception("The array/list with the contract_size parameter is empty")
        dim = np.ndim(contract_size)
        if dim > 1:
            raise Exception("The contract_size array/list should be one-dimensional")
        if len(contract_size) > 1:
            raise Exception("More than one contract_size provided")
        contract_size = contract_size[0]
        
    # Handle if contract_size is a number
    if not isinstance(contract_size, (int, np.int32, np.int64 )):
        raise TypeError("The contract_size parameter must be an integer ")

    # Ensure the value of contract_size higher than one
    if contract_size < 1:
        raise Exception("Please, insert a correct value for contract_size parameter! (it should be at least 1)")

    return contract_size

## check_print_tree ########################################################################################################################################################
############################################################################################################################################################################

def check_print_tree(print_tree, allowed_values=[True, False]): # Not in use <<<< For future developement
    """
    Check if a print_tree value or a list of print_tree values is valid.
    Args:
    print_tree: A boolean, string, list, numpy array, or pandas series representing one or more print_tree values.
    allowed_values: A list of valid print_tree values.
    Returns:
    None
    Raises:
    TypeError: If the print_tree value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one print_tree value is provided, or if the print_tree value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(print_tree, (bool, str, list, np.ndarray, pd.Series), "print_tree")
    print_tree = convert_to_list(print_tree)
    check_single_value(print_tree, "print_tree")
    check_in_values(print_tree, allowed_values, "print_tree")

## check_adjust_underlying_rets ################################################################################################################################################
################################################################################################################################################################################

def check_adjust_underlying_rets(adjust_underlying_rets, allowed_values=[True, False]): # Not in use <<<< For future developement
    """
    Check if a adjust_underlying_rets value or a list of adjust_underlying_rets values is valid.
    Args:
    adjust_underlying_rets: A boolean, string, list, numpy array, or pandas series representing one or more adjust_underlying_rets values.
    allowed_values: A list of valid adjust_underlying_rets values.
    Returns:
    None
    Raises:
    TypeError: If the adjust_underlying_rets value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one adjust_underlying_rets value is provided, or if the adjust_underlying_rets value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(adjust_underlying_rets, (bool, str, list, np.ndarray, pd.Series), "adjust_underlying_rets")
    adjust_underlying_rets = convert_to_list(adjust_underlying_rets)
    check_single_value(adjust_underlying_rets, "adjust_underlying_rets")
    check_in_values(adjust_underlying_rets, allowed_values, "adjust_underlying_rets")

## check_return_mc_returns #################################################################################################################################################
############################################################################################################################################################################

def check_return_mc_returns(return_mc_returns, allowed_values=[True, False]):
    """
    Check if a return_mc_returns value or a list of return_mc_returns values is valid.
    Args:
    return_mc_returns: A boolean, string, list, numpy array, or pandas series representing one or more return_mc_returns values.
    allowed_values: A list of valid return_mc_returns values.
    Returns:
    None
    Raises:
    TypeError: If the return_mc_returns value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one return_mc_returns value is provided, or if the return_mc_returns value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(return_mc_returns, (bool, str, list, np.ndarray, pd.Series), "return_mc_returns")
    return_mc_returns = convert_to_list(return_mc_returns)
    check_single_value(return_mc_returns, "return_mc_returns")
    check_in_values(return_mc_returns, allowed_values, "return_mc_returns")

## check_return_gbm_path ###################################################################################################################################################
############################################################################################################################################################################

def check_return_gbm_path(return_gbm_path, allowed_values=[True, False]):
    """
    Check if a return_gbm_path value or a list of return_gbm_path values is valid.
    Args:
    return_gbm_path: A boolean, string, list, numpy array, or pandas series representing one or more return_gbm_path values.
    allowed_values: A list of valid return_gbm_path values.
    Returns:
    None
    Raises:
    TypeError: If the return_gbm_path value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one return_gbm_path value is provided, or if the return_gbm_path value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(return_gbm_path, (bool, str, list, np.ndarray, pd.Series), "return_gbm_path")
    return_gbm_path = convert_to_list(return_gbm_path)
    check_single_value(return_gbm_path, "return_gbm_path")
    check_in_values(return_gbm_path, allowed_values, "return_gbm_path")

## check_K_s ###############################################################################################################################################################
############################################################################################################################################################################

def check_K_s(K_s):
    """
    Check if a list, pandas series, numpy array or dataframe of K_s is valid and return a numpy array of K_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of K_s for a multiple options.
    Returns:
    A numpy array of K_s.
    Raises:
    TypeError: If the K_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of K_s is empty.
    Exception: If the K_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the K_s contain non-numeric types.
    Exception: If the K_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(K_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("K_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(K_s,list):
        if len(K_s) == 0:
            raise Exception("The list of K_s is empty")
        dim = np.ndim(K_s)
        if dim > 1:
            raise Exception("The K_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in K_s):
            raise TypeError("The list of K_s should contain only numbers")
        K_s = np.asarray(K_s)

    if isinstance(K_s,pd.core.series.Series):
        K_s = K_s.values
        if len(K_s) == 0:
            raise Exception("The Series of K_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in K_s):
            raise TypeError("The Series of K_s should contain only numbers - (Percentages)")
        K_s = np.asarray(K_s)

    if isinstance(K_s,np.ndarray):
        if len(K_s) == 0:
            raise Exception("The array of K_s is empty")
        dim = np.ndim(K_s)
        if dim > 1:
            raise Exception("The K_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in K_s):
            raise TypeError("The array of K_s should contain only numbers")

    if isinstance(K_s,pd.DataFrame):
        if K_s.empty:
            raise Exception("The DataFrame with the K_s is empty")
        if K_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in K_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in K_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if K_s.shape[1] == 1:
            K_s = K_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(K_s).any():
        raise Exception("The K_s contain NaN values")

    # Check if any value is less than zero
    if (K_s < 0).any():
        raise Exception("All values in K_s must be greater or equal to 0")
        
    return K_s

## check_T_s ###############################################################################################################################################################
############################################################################################################################################################################

def check_T_s(T_s):
    """
    Check if a list, pandas series, numpy array or dataframe of T_s is valid and return a numpy array of T_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of T_s for a multiple options.
    Returns:
    A numpy array of T_s.
    Raises:
    TypeError: If the T_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of T_s is empty.
    Exception: If the T_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the T_s contain non-numeric types.
    Exception: If the T_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(T_s,(list, pd.core.series.Series, np.ndarray, pd.DataFrame)):
        raise TypeError("T_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(T_s,list):
        if len(T_s) == 0:
            raise Exception("The list of T_s is empty")
        dim = np.ndim(T_s)
        if dim > 1:
            raise Exception("The T_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in T_s):
            raise TypeError("The list of T_s should contain only numbers")
        T_s = np.asarray(T_s)

    if isinstance(T_s,pd.core.series.Series):
        T_s = T_s.values
        if len(T_s) == 0:
            raise Exception("The Series of T_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in T_s):
            raise TypeError("The Series of T_s should contain only numbers - (Percentages)")
        T_s = np.asarray(T_s)

    if isinstance(T_s,np.ndarray):
        if len(T_s) == 0:
            raise Exception("The array of T_s is empty")
        dim = np.ndim(T_s)
        if dim > 1:
            raise Exception("The T_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in T_s):
            raise TypeError("The array of T_s should contain only numbers")

    if isinstance(T_s,pd.DataFrame):
        if T_s.empty:
            raise Exception("The DataFrame with the T_s is empty")
        if T_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in T_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in T_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if T_s.shape[1] == 1:
            T_s = T_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(T_s).any():
        raise Exception("The T_s contain NaN values")

    # Check if any value is less than zero
    if (T_s < 0).any():
        raise Exception("All values in T_s must be greater or equal to 0")
        
    return T_s

## check_S0_s ##############################################################################################################################################################
############################################################################################################################################################################

def check_S0_s(S0_s):
    """
    Check if a list, pandas series, numpy array or dataframe of S0_s is valid and return a numpy array of S0_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of S0_s for a multiple options.
    Returns:
    A numpy array of S0_s.
    Raises:
    TypeError: If the S0_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the S0_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the S0_s contain non-numeric types.
    Exception: If the S0_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(S0_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("S0_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(S0_s,list):
        if len(S0_s) == 0:
            raise Exception("The list of S0_s is empty")
        dim = np.ndim(S0_s)
        if dim > 1:
            raise Exception("The S0_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_s):
            raise TypeError("The list of S0_s should contain only numbers")
        S0_s = np.asarray(S0_s)

    if isinstance(S0_s,pd.core.series.Series):
        S0_s = S0_s.values
        if len(S0_s) == 0:
            raise Exception("The Series of S0_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_s):
            raise TypeError("The Series of S0_s should contain only numbers")
        S0_s = np.asarray(S0_s)

    if isinstance(S0_s,np.ndarray):
        if len(S0_s) == 0:
            raise Exception("The array of S0_s is empty")
        dim = np.ndim(S0_s)
        if dim > 1:
            raise Exception("The S0_s array should be one-dimensional. if you are using .tail(1).values make sure to do what follows: .tail(1).values.ravel()")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_s):
            raise TypeError("The array of S0_s should contain only numbers")

    if isinstance(S0_s,pd.DataFrame):
        if S0_s.empty:
            raise Exception("The DataFrame with the S0_s is empty")
        if S0_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in S0_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if S0_s.shape[1] == 1:
            S0_s = S0_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(S0_s).any():
        raise Exception("The S0_s contain NaN values")

    # Check if any value is less than zero
    if (S0_s < 0).any():
        raise Exception("All values in S0_s must be greater or equal to 0")
        
    return S0_s

## check_r_s ###############################################################################################################################################################
############################################################################################################################################################################

def check_r_s(r_s):
    """
    Check if a list, pandas series, numpy array or dataframe of r_s is valid and return a numpy array of r_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of r_s for a multiple options.
    Returns:
    A numpy array of r_s.
    Raises:
    TypeError: If the r_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of r_s is empty.
    Exception: If the r_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the r_s contain non-numeric types.
    Exception: If the r_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(r_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("r_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(r_s,list):
        if len(r_s) == 0:
            raise Exception("The list of r_s is empty")
        dim = np.ndim(r_s)
        if dim > 1:
            raise Exception("The r_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in r_s):
            raise TypeError("The list of r_s should contain only numbers")
        r_s = np.asarray(r_s)

    if isinstance(r_s,pd.core.series.Series):
        r_s = r_s.values
        if len(r_s) == 0:
            raise Exception("The Series of r_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in r_s):
            raise TypeError("The Series of r_s should contain only numbers")
        r_s = np.asarray(r_s)

    if isinstance(r_s,np.ndarray):
        if len(r_s) == 0:
            raise Exception("The array of r_s is empty")
        dim = np.ndim(r_s)
        if dim > 1:
            raise Exception("The r_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in r_s):
            raise TypeError("The array of r_s should contain only numbers - (%)")

    if isinstance(r_s,pd.DataFrame):
        if r_s.empty:
            raise Exception("The DataFrame with the r_s is empty")
        if r_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in r_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in r_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if r_s.shape[1] == 1:
            r_s = r_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(r_s).any():
        raise Exception("The r_s contain NaN values")

    # Check if any value is less than zero
    if (r_s < 0).any():
        raise Exception("All values in r_s must be greater or equal to 0")
        
    return r_s

## chek_sigma_deri_s #######################################################################################################################################################
############################################################################################################################################################################

def chek_sigma_deri_s(sigma_s):
    """
    Check if a list, pandas series, numpy array or dataframe of sigma_s is valid and return a numpy array of sigma_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of sigma_s for a multiple options.
    Returns:
    A numpy array of sigma_s.
    Raises:
    TypeError: If the sigma_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of sigma_s is empty.
    Exception: If the sigma_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the sigma_s contain non-numeric types.
    Exception: If the sigma_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(sigma_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("sigma_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(sigma_s,list):
        if len(sigma_s) == 0:
            raise Exception("The list of sigma_s is empty")
        dim = np.ndim(sigma_s)
        if dim > 1:
            raise Exception("The sigma_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in sigma_s):
            raise TypeError("The list of sigma_s should contain only numbers")
        sigma_s = np.asarray(sigma_s)

    if isinstance(sigma_s,pd.core.series.Series):
        sigma_s = sigma_s.values
        if len(sigma_s) == 0:
            raise Exception("The Series of sigma_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in sigma_s):
            raise TypeError("The Series of sigma_s should contain only numbers")
        sigma_s = np.asarray(sigma_s)

    if isinstance(sigma_s,np.ndarray):
        if len(sigma_s) == 0:
            raise Exception("The array of sigma_s is empty")
        dim = np.ndim(sigma_s)
        if dim > 1:
            raise Exception("The sigma_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in sigma_s):
            raise TypeError("The array of sigma_s should contain only numbers - (%)")

    if isinstance(sigma_s,pd.DataFrame):
        if sigma_s.empty:
            raise Exception("The DataFrame with the sigma_s is empty")
        if sigma_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in sigma_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in sigma_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if sigma_s.shape[1] == 1:
            sigma_s = sigma_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(sigma_s).any():
        raise Exception("The sigma_s contain NaN values")

    # Check if any value is less than zero
    if (sigma_s < 0).any():
        raise Exception("All values in sigma_s must be greater or equal to 0")
        
    return sigma_s

## check_option_type_s #####################################################################################################################################################
############################################################################################################################################################################

def check_option_type_s(option_type_s):
    """
    Check if a list, pandas series, numpy array or dataframe of option_type_s is valid and return a numpy array of option_type_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of option_type_s for a multiple options.
    Returns:
    A numpy array of option_type_s.
    Raises:
    TypeError: If the option_type_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of option_type_s is empty.
    Exception: If the option_type_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the option_type_s contain non-numeric types.
    Exception: If the option_type_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(option_type_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("option_type_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(option_type_s,list):
        if len(option_type_s) == 0:
            raise Exception("The list of option_type_s is empty")
        dim = np.ndim(option_type_s)
        if dim > 1:
            raise Exception("The option_type_s list should be one-dimensional")
        if not all(isinstance(item, (str)) for item in option_type_s):
            raise TypeError("The list of option_type_s should contain only strings ['call','put']")
        option_type_s = np.asarray(option_type_s)

    if isinstance(option_type_s,pd.core.series.Series):
        option_type_s = option_type_s.values
        if len(option_type_s) == 0:
            raise Exception("The Series of option_type_s is empty")
        if not all(isinstance(item, (str)) for item in option_type_s):
            raise TypeError("The Series of option_type_s should contain only strings ['call','put']")
        option_type_s = np.asarray(option_type_s)

    if isinstance(option_type_s,np.ndarray):
        if len(option_type_s) == 0:
            raise Exception("The array of option_type_s is empty")
        dim = np.ndim(option_type_s)
        if dim > 1:
            raise Exception("The option_type_s array should be one-dimensional")
        if not all(isinstance(item, (str)) for item in option_type_s):
            raise TypeError("The array of option_type_s should contain strings ['call','put']")

    if isinstance(option_type_s,pd.DataFrame):
        if option_type_s.empty:
            raise Exception("The DataFrame with the option_type_s is empty")
        if option_type_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in option_type_s.columns:
            if not all(isinstance(item, (str)) for item in option_type_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only strings ['call','put']")
        if option_type_s.shape[1] == 1:
            option_type_s = option_type_s.values
    
    from pyriskmgmt.inputsControlFunctions import check_option_type

    for index, value in enumerate(option_type_s):
        option_type_s[index] = check_option_type(value)

    return option_type_s

## check_num_options_s #####################################################################################################################################################
############################################################################################################################################################################

def check_num_options_s(num_options_s):
    """
    Check if a list, pandas series, numpy array or dataframe of num_options_s is valid and return a numpy array of num_options_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of num_options_s for a multiple options.
    Returns:
    A numpy array of num_options_s.
    Raises:
    TypeError: If the num_options_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the num_options_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the num_options_s contain non-numeric types.
    Exception: If the num_options_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(num_options_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("num_options_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(num_options_s,list):
        if len(num_options_s) == 0:
            raise Exception("The list of num_options_s is empty")
        dim = np.ndim(num_options_s)
        if dim > 1:
            raise Exception("The num_options_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_options_s):
            raise TypeError("The list of num_options_s should contain only numbers")
        num_options_s = np.asarray(num_options_s)

    if isinstance(num_options_s,pd.core.series.Series):
        num_options_s = num_options_s.values
        if len(num_options_s) == 0:
            raise Exception("The Series of num_options_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_options_s):
            raise TypeError("The Series of num_options_s should contain only numbers")
        num_options_s = np.asarray(num_options_s)

    if isinstance(num_options_s,np.ndarray):
        if len(num_options_s) == 0:
            raise Exception("The array of num_options_s is empty")
        dim = np.ndim(num_options_s)
        if dim > 1:
            raise Exception("The num_options_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_options_s):
            raise TypeError("The array of num_options_s should contain only numbers")

    if isinstance(num_options_s,pd.DataFrame):
        if num_options_s.empty:
            raise Exception("The DataFrame with the num_options_s is empty")
        if num_options_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in num_options_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_options_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if num_options_s.shape[1] == 1:
            num_options_s = num_options_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(num_options_s).any():
        raise Exception("The num_options_s contain NaN values")

    return num_options_s

## check_contract_size_s ###################################################################################################################################################
############################################################################################################################################################################

def check_contract_size_s(contract_size_s):
    """
    Check if a list, pandas series, numpy array or dataframe of contract_size_s is valid and return a numpy array of contract_size_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of contract_size_s for a multiple options.
    Returns:
    A numpy array of contract_size_s.
    Raises:
    TypeError: If the contract_size_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the contract_size_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the contract_size_s contain non-numeric types.
    Exception: If the contract_size_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(contract_size_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("contract_size_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(contract_size_s,list):
        if len(contract_size_s) == 0:
            raise Exception("The list of contract_size_s is empty")
        dim = np.ndim(contract_size_s)
        if dim > 1:
            raise Exception("The contract_size_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in contract_size_s):
            raise TypeError("The list of contract_size_s should contain only numbers")
        contract_size_s = np.asarray(contract_size_s)

    if isinstance(contract_size_s,pd.core.series.Series):
        contract_size_s = contract_size_s.values
        if len(contract_size_s) == 0:
            raise Exception("The Series of contract_size_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in contract_size_s):
            raise TypeError("The Series of contract_size_s should contain only numbers")
        contract_size_s = np.asarray(contract_size_s)

    if isinstance(contract_size_s,np.ndarray):
        if len(contract_size_s) == 0:
            raise Exception("The array of contract_size_s is empty")
        dim = np.ndim(contract_size_s)
        if dim > 1:
            raise Exception("The contract_size_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in contract_size_s):
            raise TypeError("The array of contract_size_s should contain only numbers")

    if isinstance(contract_size_s,pd.DataFrame):
        if contract_size_s.empty:
            raise Exception("The DataFrame with the contract_size_s is empty")
        if contract_size_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in contract_size_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in contract_size_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if contract_size_s.shape[1] == 1:
            contract_size_s = contract_size_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(contract_size_s).any():
        raise Exception("The contract_size_s contain NaN values")

    # Check if any value is less than zero
    if (contract_size_s < 1).any():
        raise Exception("All values in contract_size_s must be at least 1.")
        
    return contract_size_s

## check_return_simulated_rets ################################################################################################################################################
###############################################################################################################################################################################

def check_return_simulated_rets(return_simulated_rets, allowed_values=[True, False]):
    """
    Check if a return_simulated_rets value or a list of return_simulated_rets values is valid.
    Args:
    return_simulated_rets: A boolean, string, list, numpy array, or pandas series representing one or more return_simulated_rets values.
    allowed_values: A list of valid return_simulated_rets values.
    Returns:
    None
    Raises:
    TypeError: If the return_simulated_rets value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one return_simulated_rets value is provided, or if the return_simulated_rets value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(return_simulated_rets, (bool, str, list, np.ndarray, pd.Series), "return_simulated_rets")
    return_simulated_rets = convert_to_list(return_simulated_rets)
    check_single_value(return_simulated_rets, "return_simulated_rets")
    check_in_values(return_simulated_rets, allowed_values, "return_simulated_rets")

## check_S0_initial_s ######################################################################################################################################################
############################################################################################################################################################################

def check_S0_initial_s(S0_initial_s):
    """
    Check if a list, pandas series, numpy array or dataframe of S0_initial_s is valid and return a numpy array of S0_initial_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of S0_initial_s for a multiple options.
    Returns:
    A numpy array of S0_initial_s.
    Raises:
    TypeError: If the S0_initial_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_initial_s is empty.
    Exception: If the S0_initial_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the S0_initial_s contain non-numeric types.
    Exception: If the S0_initial_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(S0_initial_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("S0_initial_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(S0_initial_s,list):
        if len(S0_initial_s) == 0:
            raise Exception("The list of S0_initial_s is empty")
        dim = np.ndim(S0_initial_s)
        if dim > 1:
            raise Exception("The S0_initial_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_initial_s):
            raise TypeError("The list of S0_initial_s should contain only numbers")
        S0_initial_s = np.asarray(S0_initial_s)

    if isinstance(S0_initial_s,pd.core.series.Series):
        S0_initial_s = S0_initial_s.values
        if len(S0_initial_s) == 0:
            raise Exception("The Series of S0_initial_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_initial_s):
            raise TypeError("The Series of S0_initial_s should contain only numbers")
        S0_initial_s = np.asarray(S0_initial_s)

    if isinstance(S0_initial_s,np.ndarray):
        if len(S0_initial_s) == 0:
            raise Exception("The array of S0_initial_s is empty")
        dim = np.ndim(S0_initial_s)
        if dim > 1:
            raise Exception("The S0_initial_s array should be one-dimensional. if you are using .tail(1).values make sure to do what follows: .tail(1).values.ravel()")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_initial_s):
            raise TypeError("The array of S0_initial_s should contain only numbers")

    if isinstance(S0_initial_s,pd.DataFrame):
        if S0_initial_s.empty:
            raise Exception("The DataFrame with the S0_initial_s is empty")
        if S0_initial_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in S0_initial_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in S0_initial_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if S0_initial_s.shape[1] == 1:
            S0_initial_s = S0_initial_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(S0_initial_s).any():
        raise Exception("The S0_initial_s contain NaN values")

    # Check if any value is less than zero
    if (S0_initial_s < 0).any():
        raise Exception("All values in S0_initial_s must be greater or equal to 0")
        
    return S0_initial_s

## check_num_forward #############################################################################################################################################################
############################################################################################################################################################################

def check_num_forward(num_forward):
    """
    This function checks if the provided num_forward parameter is valid for statistical analysis.
    Args:
    num_forward: (np.ndarray, pd.core.series.Series, list, number)
        The num_forward parameter to be checked for validity.
    Returns:
    num_forward: (int)
        The validated num_forward to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the num_forward parameter is a pandas DataFrame object.
        - If the num_forward parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the num_forward parameter is empty.
        - If the num_forward array/list is more than one-dimensional.
        - If there is more than one num_forward parameter provided.
        - If the num_forward parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if num_forward is a DataFrame
    if isinstance(num_forward, pd.core.frame.DataFrame):
        raise TypeError("The num_forward parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if num_forward is a Series
    if isinstance(num_forward, pd.core.series.Series):
        if len(num_forward) == 0:
            raise Exception("The Series with the num_forward parameter is empty")
        num_forward = list(num_forward)

    # Handle if num_forward is a list or ndarray
    if isinstance(num_forward, (np.ndarray, list)):
        if len(num_forward) == 0:
            raise Exception("The array/list with the num_forward parameter is empty")
        dim = np.ndim(num_forward)
        if dim > 1:
            raise Exception("The num_forward array/list should be one-dimensional")
        if len(num_forward) > 1:
            raise Exception("More than one num_forward provided")
        num_forward = num_forward[0]
        
    # Handle if num_forward is a number
    if not isinstance(num_forward, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError("The num_forward parameter must be an number")

    return num_forward

## check_num_forward_s #####################################################################################################################################################
############################################################################################################################################################################

def check_num_forward_s(num_forward_s):
    """
    Check if a list, pandas series, numpy array or dataframe of num_forward_s is valid and return a numpy array of num_forward_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of num_forward_s for a multiple options.
    Returns:
    A numpy array of num_forward_s.
    Raises:
    TypeError: If the num_forward_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the num_forward_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the num_forward_s contain non-numeric types.
    Exception: If the num_forward_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(num_forward_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("num_forward_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(num_forward_s,list):
        if len(num_forward_s) == 0:
            raise Exception("The list of num_forward_s is empty")
        dim = np.ndim(num_forward_s)
        if dim > 1:
            raise Exception("The num_forward_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_forward_s):
            raise TypeError("The list of num_forward_s should contain only numbers")
        num_forward_s = np.asarray(num_forward_s)

    if isinstance(num_forward_s,pd.core.series.Series):
        num_forward_s = num_forward_s.values
        if len(num_forward_s) == 0:
            raise Exception("The Series of num_forward_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_forward_s):
            raise TypeError("The Series of num_forward_s should contain only numbers")
        num_forward_s = np.asarray(num_forward_s)

    if isinstance(num_forward_s,np.ndarray):
        if len(num_forward_s) == 0:
            raise Exception("The array of num_forward_s is empty")
        dim = np.ndim(num_forward_s)
        if dim > 1:
            raise Exception("The num_forward_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_forward_s):
            raise TypeError("The array of num_forward_s should contain only numbers")

    if isinstance(num_forward_s,pd.DataFrame):
        if num_forward_s.empty:
            raise Exception("The DataFrame with the num_forward_s is empty")
        if num_forward_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in num_forward_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_forward_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if num_forward_s.shape[1] == 1:
            num_forward_s = num_forward_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(num_forward_s).any():
        raise Exception("The num_forward_s contain NaN values")

    return num_forward_s

## check_dividend_yield ####################################################################################################################################################
############################################################################################################################################################################

def check_dividend_yield(dividend_yield):
    """
    Checks the validity of the input dividend_yield.

    Parameters:
    dividend_yield: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If dividend_yield is not of the correct type.
    ValueError: If no value is provided for dividend_yield or if more than one value is provided.

    Returns:
    dividend_yield: int or float
        The validated number of dividend_yield.
    """

    import pandas as pd ; import numpy as np 

    # Check input type
    if not isinstance(dividend_yield, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64,
                                        list, np.ndarray, pd.Series)):
        raise TypeError(f"The dividend_yield parameter should be a number!")

    # Convert to list
    if isinstance(dividend_yield, (list, pd.Series, np.ndarray)):
        dividend_yield = list(dividend_yield)
        if len(dividend_yield) == 0: 
            raise ValueError("No dividend_yield parameter provided")
        # Check for single value
        if len(dividend_yield) > 1:
            raise ValueError(f"More than one dividend_yield has been provided. This function works only with one dividend_yield at the time")
    
    if isinstance(dividend_yield, (list)):
        dividend_yield = dividend_yield[0]

    if not isinstance(dividend_yield, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError(f"dividend_yield should be a number!")
    
    if dividend_yield < 0:
        raise TypeError(f"dividend_yield should be a number! higher or equal than 0 --- this model works only with a positive dividend_yield")

    return dividend_yield

## check_dividend_yield_s ##################################################################################################################################################
############################################################################################################################################################################

def check_dividend_yield_s(dividend_yield_s):
    """
    Check if a list, pandas series, numpy array or dataframe of dividend_yield_s is valid and return a numpy array of dividend_yield_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of dividend_yield_s for a multiple options.
    Returns:
    A numpy array of dividend_yield_s.
    Raises:
    TypeError: If the dividend_yield_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of dividend_yield_s is empty.
    Exception: If the dividend_yield_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the dividend_yield_s contain non-numeric types.
    Exception: If the dividend_yield_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(dividend_yield_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("dividend_yield_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(dividend_yield_s,list):
        if len(dividend_yield_s) == 0:
            raise Exception("The list of dividend_yield_s is empty")
        dim = np.ndim(dividend_yield_s)
        if dim > 1:
            raise Exception("The dividend_yield_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in dividend_yield_s):
            raise TypeError("The list of dividend_yield_s should contain only numbers")
        dividend_yield_s = np.asarray(dividend_yield_s)

    if isinstance(dividend_yield_s,pd.core.series.Series):
        dividend_yield_s = dividend_yield_s.values
        if len(dividend_yield_s) == 0:
            raise Exception("The Series of dividend_yield_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in dividend_yield_s):
            raise TypeError("The Series of dividend_yield_s should contain only numbers")
        dividend_yield_s = np.asarray(dividend_yield_s)

    if isinstance(dividend_yield_s,np.ndarray):
        if len(dividend_yield_s) == 0:
            raise Exception("The array of dividend_yield_s is empty")
        dim = np.ndim(dividend_yield_s)
        if dim > 1:
            raise Exception("The dividend_yield_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in dividend_yield_s):
            raise TypeError("The array of dividend_yield_s should contain only numbers - (%)")

    if isinstance(dividend_yield_s,pd.DataFrame):
        if dividend_yield_s.empty:
            raise Exception("The DataFrame with the dividend_yield_s is empty")
        if dividend_yield_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in dividend_yield_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in dividend_yield_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if dividend_yield_s.shape[1] == 1:
            dividend_yield_s = dividend_yield_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(dividend_yield_s).any():
        raise Exception("The dividend_yield_s contain NaN values")

    # Check if any value is less than zero
    if (dividend_yield_s < 0).any():
        raise Exception("All values in dividend_yield_s must be greater or equal to 0")
        
    return dividend_yield_s

## check_convenience_yield #################################################################################################################################################
############################################################################################################################################################################

def check_convenience_yield(convenience_yield):
    """
    Checks the validity of the input convenience_yield.

    Parameters:
    convenience_yield: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If convenience_yield is not of the correct type.
    ValueError: If no value is provided for convenience_yield or if more than one value is provided.

    Returns:
    convenience_yield: int or float
        The validated number of convenience_yield.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(convenience_yield, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The convenience_yield parameter should be a number!")

    # Convert to list
    if isinstance(convenience_yield, (list, pd.Series, np.ndarray)):
        convenience_yield = list(convenience_yield)
        if len(convenience_yield) == 0: 
            raise ValueError("No convenience_yield parameter provided")
        # Check for single value
        if len(convenience_yield) > 1:
            raise ValueError(f"More than one convenience_yield has been provided. This function works only with one convenience_yield at the time")
    
    if isinstance(convenience_yield, (list)):
        convenience_yield = convenience_yield[0]

    if not isinstance(convenience_yield, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError(f"convenience_yield should be a number!")
    
    if convenience_yield < 0:
        raise TypeError(f"convenience_yield should be a number! higher or equal than 0 --- this model works only with a positive convenience_yield")

    return convenience_yield

## check_storage_cost ######################################################################################################################################################
############################################################################################################################################################################

def check_storage_cost(storage_cost):
    """
    Checks the validity of the input storage_cost.

    Parameters:
    storage_cost: int, float, or array-like
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If storage_cost is not of the correct type.
    ValueError: If no value is provided for storage_cost or if more than one value is provided.

    Returns:
    storage_cost: int or float
        The validated number of storage_cost.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(storage_cost, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The storage_cost parameter should be a number!")

    # Convert to list
    if isinstance(storage_cost, (list, pd.Series, np.ndarray)):
        storage_cost = list(storage_cost)
        if len(storage_cost) == 0: 
            raise ValueError("No storage_cost parameter provided")
        # Check for single value
        if len(storage_cost) > 1:
            raise ValueError(f"More than one storage_cost has been provided. This function works only with one storage_cost at the time")
    
    if isinstance(storage_cost, (list)):
        storage_cost = storage_cost[0]

    if not isinstance(storage_cost, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError(f"storage_cost should be a number!")
    
    if storage_cost < 0:
        raise TypeError(f"storage_cost should be a number! higher or equal than 0 --- this model works only with a positive storage_cost")

    return storage_cost

## check_num_future #######################################################################################################################################################
############################################################################################################################################################################

def check_num_future(num_future):
    """
    This function checks if the provided num_future parameter is valid for statistical analysis.
    Args:
    num_future: (np.ndarray, pd.core.series.Series, list, number)
        The num_future parameter to be checked for validity.
    Returns:
    num_future: (int)
        The validated num_future to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the num_future parameter is a pandas DataFrame object.
        - If the num_future parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the num_future parameter is empty.
        - If the num_future array/list is more than one-dimensional.
        - If there is more than one num_future parameter provided.
        - If the num_future parameter is not an integer greater than 1.
    """
    import pandas as pd ; import numpy as np

    # Raise TypeError if num_future is a DataFrame
    if isinstance(num_future, pd.core.frame.DataFrame):
        raise TypeError("The num_future parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if num_future is a Series
    if isinstance(num_future, pd.core.series.Series):
        if len(num_future) == 0:
            raise Exception("The Series with the num_future parameter is empty")
        num_future = list(num_future)

    # Handle if num_future is a list or ndarray
    if isinstance(num_future, (np.ndarray, list)):
        if len(num_future) == 0:
            raise Exception("The array/list with the num_future parameter is empty")
        dim = np.ndim(num_future)
        if dim > 1:
            raise Exception("The num_future array/list should be one-dimensional")
        if len(num_future) > 1:
            raise Exception("More than one num_future provided")
        num_future = num_future[0]
        
    # Handle if num_future is a number
    if not isinstance(num_future, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError("The num_future parameter must be an number")

    return num_future

## check_convenience_yield_s ################################################################################################################################################
#############################################################################################################################################################################

def check_convenience_yield_s(convenience_yield_s):
    """
    Check if a list, pandas series, numpy array or dataframe of convenience_yield_s is valid and return a numpy array of convenience_yield_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of convenience_yield_s for a multiple options.
    Returns:
    A numpy array of convenience_yield_s.
    Raises:
    TypeError: If the convenience_yield_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of convenience_yield_s is empty.
    Exception: If the convenience_yield_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the convenience_yield_s contain non-numeric types.
    Exception: If the convenience_yield_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(convenience_yield_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("convenience_yield_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(convenience_yield_s,list):
        if len(convenience_yield_s) == 0:
            raise Exception("The list of convenience_yield_s is empty")
        dim = np.ndim(convenience_yield_s)
        if dim > 1:
            raise Exception("The convenience_yield_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in convenience_yield_s):
            raise TypeError("The list of convenience_yield_s should contain only numbers")
        convenience_yield_s = np.asarray(convenience_yield_s)

    if isinstance(convenience_yield_s,pd.core.series.Series):
        convenience_yield_s = convenience_yield_s.values
        if len(convenience_yield_s) == 0:
            raise Exception("The Series of convenience_yield_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in convenience_yield_s):
            raise TypeError("The Series of convenience_yield_s should contain only numbers")
        convenience_yield_s = np.asarray(convenience_yield_s)

    if isinstance(convenience_yield_s,np.ndarray):
        if len(convenience_yield_s) == 0:
            raise Exception("The array of convenience_yield_s is empty")
        dim = np.ndim(convenience_yield_s)
        if dim > 1:
            raise Exception("The convenience_yield_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in convenience_yield_s):
            raise TypeError("The array of convenience_yield_s should contain only numbers - (%)")

    if isinstance(convenience_yield_s,pd.DataFrame):
        if convenience_yield_s.empty:
            raise Exception("The DataFrame with the convenience_yield_s is empty")
        if convenience_yield_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in convenience_yield_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in convenience_yield_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if convenience_yield_s.shape[1] == 1:
            convenience_yield_s = convenience_yield_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(convenience_yield_s).any():
        raise Exception("The convenience_yield_s contain NaN values")

    # Check if any value is less than zero
    if (convenience_yield_s < 0).any():
        raise Exception("All values in convenience_yield_s must be greater or equal to 0")
        
    return convenience_yield_s

## check_storage_cost_s ####################################################################################################################################################
############################################################################################################################################################################

def check_storage_cost_s(storage_cost_s):
    """
    Check if a list, pandas series, numpy array or dataframe of storage_cost_s is valid and return a numpy array of storage_cost_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of storage_cost_s for a multiple options.
    Returns:
    A numpy array of storage_cost_s.
    Raises:
    TypeError: If the storage_cost_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of storage_cost_s is empty.
    Exception: If the storage_cost_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the storage_cost_s contain non-numeric types.
    Exception: If the storage_cost_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(storage_cost_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("storage_cost_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(storage_cost_s,list):
        if len(storage_cost_s) == 0:
            raise Exception("The list of storage_cost_s is empty")
        dim = np.ndim(storage_cost_s)
        if dim > 1:
            raise Exception("The storage_cost_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in storage_cost_s):
            raise TypeError("The list of storage_cost_s should contain only numbers")
        storage_cost_s = np.asarray(storage_cost_s)

    if isinstance(storage_cost_s,pd.core.series.Series):
        storage_cost_s = storage_cost_s.values
        if len(storage_cost_s) == 0:
            raise Exception("The Series of storage_cost_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in storage_cost_s):
            raise TypeError("The Series of storage_cost_s should contain only numbers")
        storage_cost_s = np.asarray(storage_cost_s)

    if isinstance(storage_cost_s,np.ndarray):
        if len(storage_cost_s) == 0:
            raise Exception("The array of storage_cost_s is empty")
        dim = np.ndim(storage_cost_s)
        if dim > 1:
            raise Exception("The storage_cost_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in storage_cost_s):
            raise TypeError("The array of storage_cost_s should contain only numbers - (%)")

    if isinstance(storage_cost_s,pd.DataFrame):
        if storage_cost_s.empty:
            raise Exception("The DataFrame with the storage_cost_s is empty")
        if storage_cost_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in storage_cost_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in storage_cost_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if storage_cost_s.shape[1] == 1:
            storage_cost_s = storage_cost_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(storage_cost_s).any():
        raise Exception("The storage_cost_s contain NaN values")

    # Check if any value is less than zero
    if (storage_cost_s < 0).any():
        raise Exception("All values in storage_cost_s must be greater or equal to 0")
        
    return storage_cost_s

## check_num_future_s ######################################################################################################################################################
############################################################################################################################################################################

def check_num_future_s(num_future_s):
    """
    Check if a list, pandas series, numpy array or dataframe of num_future_s is valid and return a numpy array of num_future_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of num_future_s for a multiple options.
    Returns:
    A numpy array of num_future_s.
    Raises:
    TypeError: If the num_future_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the num_future_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the num_future_s contain non-numeric types.
    Exception: If the num_future_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(num_future_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("num_future_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(num_future_s,list):
        if len(num_future_s) == 0:
            raise Exception("The list of num_future_s is empty")
        dim = np.ndim(num_future_s)
        if dim > 1:
            raise Exception("The num_future_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_future_s):
            raise TypeError("The list of num_future_s should contain only numbers")
        num_future_s = np.asarray(num_future_s)

    if isinstance(num_future_s,pd.core.series.Series):
        num_future_s = num_future_s.values
        if len(num_future_s) == 0:
            raise Exception("The Series of num_future_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_future_s):
            raise TypeError("The Series of num_future_s should contain only numbers")
        num_future_s = np.asarray(num_future_s)

    if isinstance(num_future_s,np.ndarray):
        if len(num_future_s) == 0:
            raise Exception("The array of num_future_s is empty")
        dim = np.ndim(num_future_s)
        if dim > 1:
            raise Exception("The num_future_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_future_s):
            raise TypeError("The array of num_future_s should contain only numbers")

    if isinstance(num_future_s,pd.DataFrame):
        if num_future_s.empty:
            raise Exception("The DataFrame with the num_future_s is empty")
        if num_future_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in num_future_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_future_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if num_future_s.shape[1] == 1:
            num_future_s = num_future_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(num_future_s).any():
        raise Exception("The num_future_s contain NaN values")

    return num_future_s

## check_years_back #################################################################################################################################################################
############################################################################################################################################################################

def check_years_back(years_back):
    """
    This function checks if the provided years_back parameter is valid for statistical analysis.
    Args:
    years_back: (np.ndarray, pd.core.series.Series, list, number)
        The years_back parameter to be checked for validity.
    Returns:
    years_back: (int)
        The validated years_back to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the years_back parameter is a pandas DataFrame object.
        - If the years_back parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the years_back parameter is empty.
        - If the q array/list is more than one-dimensional.
        - If there is more than one years_back parameter provided.
        - If the q parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if years_back is a DataFrame
    if isinstance(years_back, pd.core.frame.DataFrame):
        raise TypeError("The years_back parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if years_back is a Series
    if isinstance(years_back, pd.core.series.Series):
        if len(years_back) == 0:
            raise Exception("The Series with the years_back parameter is empty")
        years_back = list(years_back)

    # Handle if years_back is a list or ndarray
    if isinstance(years_back, (np.ndarray, list)):
        if len(years_back) == 0:
            raise Exception("The array/list with the years_back parameter is empty")
        dim = np.ndim(years_back)
        if dim > 1:
            raise Exception("The years_back array/list should be one-dimensional")
        if len(years_back) > 1:
            raise Exception("More than one years_back provided")
        years_back = years_back[0]
        
    # Handle if years_back is a number
    if not isinstance(years_back, (int, np.int32, np.int64 )):
        raise TypeError("The years_back parameter must be an integer ")

    # Ensure the value of years_back higher than one
    if years_back < 1:
        raise Exception("Please, insert a correct value for years_back parameter! at least 1)")

    return years_back

## check_method_yield_curve ##############################################################################################################################################################
############################################################################################################################################################################

def check_method_yield_curve(method, allowed_values=["cubic_spline", "linear"]):
    """
    Checks whether the method argument is valid.

    Parameters:
    method: str or array-like
        The method to be checked. Should be one of the allowed_values.
        Can be a single string or an array-like object containing a single string.

    allowed_values: list of str, optional
        A list of the values that are allowed for the method argument.
        Defaults to ["daily"]

    Raises:
    TypeError: If the method argument is not of the correct type.
    ValueError: If more than one method is provided or if the method is not in the list of allowed values.

    Returns:
    method: str
    The validated method argument.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: {types}")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at the time")

    def convert_to_list(input):
        if isinstance(input, (list, pd.Series, np.ndarray)):
            input =  list(input)
        if isinstance(input, list):
            input =  input[0]
        return input

    def check_in_values(input, values, name):
        if not isinstance(input, str):
            raise TypeError("The method parameter should be a string!")
        if input not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(method, (str, list, np.ndarray, pd.Series), "method")
    check_single_value(method, "method")
    method = convert_to_list(method)
    check_in_values(method, allowed_values, "method")

    return method

## check_yield_curve_today_bool ###########################################################################################################################################################
############################################################################################################################################################################

def check_yield_curve_today_bool(yield_curve_today, allowed_values=[True, False]):
    """
    Check if a yield_curve_today value or a list of yield_curve_today values is valid.
    Args:
    yield_curve_today: A boolean, string, list, numpy array, or pandas series representing one or more yield_curve_today values.
    allowed_values: A list of valid yield_curve_today values.
    Returns:
    None
    Raises:
    TypeError: If the yield_curve_today value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one yield_curve_today value is provided, or if the yield_curve_today value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(yield_curve_today, (bool, str, list, np.ndarray, pd.Series), "yield_curve_today")
    yield_curve_today = convert_to_list(yield_curve_today)
    check_single_value(yield_curve_today, "yield_curve_today")
    check_in_values(yield_curve_today, allowed_values, "yield_curve_today")

## check_yield_curve_today #################################################################################################################################################
############################################################################################################################################################################

def check_yield_curve_today(yield_curve_today):
    """
    Checks the validity of the yield_curve_today DataFrame.
    
    Parameters
    ----------
    yield_curve_today : A DataFrame containing yield curve data for today.
        
    Returns
    -------
    yield_curve_today :  A checked DataFrame containing yield curve data for today.
    
    Raises
    ------
    ValueError: If any of the validation checks fail.
    """

    import pandas as pd ; import numpy as np
    
    # Check if yield_curve_today is a DataFrame
    if not isinstance(yield_curve_today, pd.DataFrame):
        raise ValueError("yield_curve_today must be a DataFrame.")
    
    # Check if yield_curve_today has only one row
    if yield_curve_today.shape[0] != 1:
        raise ValueError("yield_curve_today must have only one row: today's value of the yield curve")
    
    # Check if columns start from 1 and increment by 1
    expected_columns = list(range(1, yield_curve_today.shape[1] + 1))
    if list(yield_curve_today.columns) != expected_columns:
        raise ValueError("Columns must start from the 1 month value and increments by 1.")
    
    # Check if all column values are integers
    if not all(isinstance(col, (int, np.int32, np.int64, float, np.float32, np.float64)) for col in yield_curve_today.columns):
        raise ValueError("All column names must be either integers or floats.")
    
    # Check if all row values are either int, np.int32, np.int64, float, np.float32, or np.float64
    if not all(yield_curve_today.iloc[0].apply(lambda x: isinstance(x, (int, np.int32, np.int64, float, np.float32, np.float64)))):
        raise ValueError("All values in the row must be either int, np.int32, np.int64, float, np.float32, or np.float64.")
    
    return yield_curve_today

## check_maturity #############################################################################################################################################################
############################################################################################################################################################################

def check_maturity(maturity):
    """
    This function checks if the provided maturity parameter is valid for statistical analysis.
    Args:
    maturity: (np.ndarray, pd.core.series.Series, list, number)
        The maturity parameter to be checked for validity.
    Returns:
    maturity: (int)
        The validated maturity to be used for statistical analysis.
    Raises:
        TypeError:
            - If the maturity parameter is a pandas DataFrame object.
            - If the maturity parameter is not a number or an array with valid values.
        Exception:
            - If the Series or array/list with the maturity parameter is empty.
            - If the maturity array/list is more than one-dimensional.
            - If there is more than one maturity parameter provided.
            - If the maturity parameter is not positive and greater than 0.
            - If the maturity is not an integer or a value with a 0.5 decimal part.
    """
    
    import pandas as pd ; import numpy as np

    # Raise TypeError if maturity is a DataFrame
    if isinstance(maturity, pd.core.frame.DataFrame):
        raise TypeError("The maturity parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if maturity is a Series
    if isinstance(maturity, pd.core.series.Series):
        if len(maturity) == 0:
            raise Exception("The Series with the maturity parameter is empty")
        maturity = list(maturity)

    # Handle if maturity is a list or ndarray
    if isinstance(maturity, (np.ndarray, list)):
        if len(maturity) == 0:
            raise Exception("The array/list with the maturity parameter is empty")
        dim = np.ndim(maturity)
        if dim > 1:
            raise Exception("The maturity array/list should be one-dimensional")
        if len(maturity) > 1:
            raise Exception("More than one maturity provided")
        maturity = maturity[0]
        
    # Check if maturity is a number and if it's positive and greater than 0
    if not isinstance(maturity, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError("The maturity parameter must be a number")
    if maturity <= 0:
        raise Exception("The maturity parameter must be positive and greater than 0")

    # Check if maturity is either an integer or a number with a 0.5 decimal part
    if maturity != int(maturity) and (maturity - int(maturity)) != 0.5:
        raise Exception("The maturity (in years) must be either an integer or a number with a 0.5 decimal part")

    return maturity

## check_num_assets #############################################################################################################################################################
############################################################################################################################################################################

def check_num_assets(num_assets):
    """
    This function checks if the provided num_assets parameter is valid for statistical analysis.
    Args:
    num_assets: (np.ndarray, pd.core.series.Series, list, number)
        The num_assets parameter to be checked for validity.
    Returns:
    num_assets: (int)
        The validated num_assets to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the num_assets parameter is a pandas DataFrame object.
        - If the num_assets parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the num_assets parameter is empty.
        - If the num_assets array/list is more than one-dimensional.
        - If there is more than one num_assets parameter provided.
        - If the num_assets parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if num_assets is a DataFrame
    if isinstance(num_assets, pd.core.frame.DataFrame):
        raise TypeError("The num_assets parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if num_assets is a Series
    if isinstance(num_assets, pd.core.series.Series):
        if len(num_assets) == 0:
            raise Exception("The Series with the num_assets parameter is empty")
        num_assets = list(num_assets)

    # Handle if num_assets is a list or ndarray
    if isinstance(num_assets, (np.ndarray, list)):
        if len(num_assets) == 0:
            raise Exception("The array/list with the num_assets parameter is empty")
        dim = np.ndim(num_assets)
        if dim > 1:
            raise Exception("The num_assets array/list should be one-dimensional")
        if len(num_assets) > 1:
            raise Exception("More than one num_assets provided")
        num_assets = num_assets[0]
        
    # Handle if num_assets is a number
    if not isinstance(num_assets, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError("The num_assets parameter must be an number")

    return num_assets

## check_face_value ################################################################################################################################################################
############################################################################################################################################################################

def check_face_value(face_value):
    """
    Checks the validity of the input face_value.

    Parameters:
    face_value: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If face_value is not of the correct type.
    ValueError: If no value is provided for face_value or if more than one value is provided.

    Returns:
    face_value: int or float
        The validated number of face_value.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(face_value, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The face_value should be a number!")

    # Convert to list
    if isinstance(face_value, (list, pd.Series, np.ndarray)):
        face_value = list(face_value)
        if len(face_value) == 0: 
            raise ValueError("No face_value parameter provided")
        # Check for single value
        if len(face_value) > 1:
            raise ValueError(f"More than one face_value has been provided. This function works only with one face_value parameter at the time")
    
    if isinstance(face_value, (list)):
        face_value = face_value[0]

    if not isinstance(face_value, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"face_value should be a number!")
    
    if face_value <= 0:
        raise TypeError(f"face_value should be a number! higher than 0 --- face_values cannot be negative...")

    return face_value

## check_is_zcb ###########################################################################################################################################################
############################################################################################################################################################################

def check_is_zcb(is_zcb, allowed_values=[True, False]):
    """
    Check if a is_zcb value or a list of is_zcb values is valid.
    Args:
    is_zcb: A boolean, string, list, numpy array, or pandas series representing one or more is_zcb values.
    allowed_values: A list of valid is_zcb values.
    Returns:
    None
    Raises:
    TypeError: If the is_zcb value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one is_zcb value is provided, or if the is_zcb value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(is_zcb, (bool, str, list, np.ndarray, pd.Series), "is_zcb")
    is_zcb = convert_to_list(is_zcb)
    check_single_value(is_zcb, "is_zcb")
    check_in_values(is_zcb, allowed_values, "is_zcb")

## check_coupon_rate ################################################################################################################################################################
############################################################################################################################################################################

def check_coupon_rate(coupon_rate):
    """
    Checks the validity of the input coupon_rate.

    Parameters:
    coupon_rate: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If coupon_rate is not of the correct type.
    ValueError: If no value is provided for coupon_rate or if more than one value is provided.

    Returns:
    coupon_rate: int or float
        The validated number of coupon_rate.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(coupon_rate, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The coupon_rate should be a number!")

    # Convert to list
    if isinstance(coupon_rate, (list, pd.Series, np.ndarray)):
        coupon_rate = list(coupon_rate)
        if len(coupon_rate) == 0: 
            raise ValueError("No coupon_rate parameter provided")
        # Check for single value
        if len(coupon_rate) > 1:
            raise ValueError(f"More than one coupon_rate has been provided. This function works only with one coupon_rate parameter at the time")
    
    if isinstance(coupon_rate, (list)):
        coupon_rate = coupon_rate[0]

    if not isinstance(coupon_rate, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"coupon_rate should be a number!")
    
    if coupon_rate <= 0:
        raise TypeError(f"coupon_rate should be a number! higher than 0 --- coupon_rate cannot be negative...")

    return coupon_rate

## check_semi_annual_payment ###########################################################################################################################################################
############################################################################################################################################################################

def check_semi_annual_payment(semi_annual_payment, allowed_values=[True, False]):
    """
    Check if a semi_annual_payment value or a list of semi_annual_payment values is valid.
    Args:
    semi_annual_payment: A boolean, string, list, numpy array, or pandas series representing one or more semi_annual_payment values.
    allowed_values: A list of valid semi_annual_payment values.
    Returns:
    None
    Raises:
    TypeError: If the semi_annual_payment value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one semi_annual_payment value is provided, or if the semi_annual_payment value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(semi_annual_payment, (bool, str, list, np.ndarray, pd.Series), "semi_annual_payment")
    semi_annual_payment = convert_to_list(semi_annual_payment)
    check_single_value(semi_annual_payment, "semi_annual_payment")
    check_in_values(semi_annual_payment, allowed_values, "semi_annual_payment")

## check_yield_curve_df #################################################################################################################################################
############################################################################################################################################################################

def check_yield_curve_df(yield_curve):
    """
    Checks the validity of the yield_curve DataFrame.
    
    Parameters
    ----------
    yield_curve : A DataFrame containing yield curve data for today.
        
    Returns
    -------
    yield_curve :  A checked DataFrame containing yield curve data for today.
    
    Raises
    ------
    ValueError: If any of the validation checks fail.
    """

    import pandas as pd ; import numpy as np
    
    # Check if yield_curve is a DataFrame
    if not isinstance(yield_curve, pd.DataFrame):
        raise ValueError("yield_curve must be a DataFrame.")
    
    # Check if columns start from 1 and increment by 1
    expected_columns = list(range(1, yield_curve.shape[1] + 1))
    if list(yield_curve.columns) != expected_columns:
        raise ValueError("Columns must start from the 1 month value and increments by 1.")
    
    # Check if all column values are integers
    if not all(isinstance(col, (int, np.int32, np.int64, float, np.float32, np.float64)) for col in yield_curve.columns):
        raise ValueError("All column names must be either integers or floats.")
    
    # Check if all row and column values are of the required types
    if not yield_curve.applymap(lambda x: isinstance(x, (int, np.int32, np.int64, float, np.float32, np.float64))).all().all():
        raise ValueError("All values in the DataFrame must be either int, np.int32, np.int64, float, np.float32, or np.float64.")
    
    return yield_curve

## check_minimum_number_of_sim ################################################################################################################################################
###############################################################################################################################################################################

def check_minimum_number_of_sim(minimum_number_of_sim): # >>>>> not in use... for future developement <<<<<<
    """
    Checks the validity of the input minimum_number_of_sim.

    Parameters:
    minimum_number_of_sim: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If minimum_number_of_sim is not of the correct type.
    ValueError: If no value is provided for minimum_number_of_sim or if more than one value is provided.

    Returns:
    minimum_number_of_sim: int or float
        The validated number of minimum_number_of_sim.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(minimum_number_of_sim, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The minimum_number_of_sim should be a number!")

    # Convert to list
    if isinstance(minimum_number_of_sim, (list, pd.Series, np.ndarray)):
        minimum_number_of_sim = list(minimum_number_of_sim)
        if len(minimum_number_of_sim) == 0: 
            raise ValueError("No minimum_number_of_sim provided")
        # Check for single value
        if len(minimum_number_of_sim) > 1:
            raise ValueError(f"More than one minimum_number_of_sim has been provided. This function works only with one minimum_number_of_sim at the time")
    
    if isinstance(minimum_number_of_sim, (list)):
        minimum_number_of_sim = minimum_number_of_sim[0]

    if not isinstance(minimum_number_of_sim, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"The minimum_number_of_sim should be a number!")
    
    if minimum_number_of_sim <= 0:
        raise TypeError(f"The minimum_number_of_sim should be a number! higher than 0")

    return minimum_number_of_sim

## check_maturity_s ####################################################################################################################################################
############################################################################################################################################################################

def check_maturity_s(maturity_s):
    """
    This function validates the input 'maturity_s', ensuring that it is a one-dimensional array-like object (numpy array, pandas series, or list)
    that contains only real numbers, and has at least one non-zero element.

    Parameters
    ----------
    maturity_s : np.ndarray, pd.core.series.Series, list
        The maturities of a portfolio of bonds. This should be a one-dimensional array-like object containing only real numbers.

    Raises
    ------
    TypeError
        If 'maturity_s' is not a numpy array, pandas Series or list.
        If 'maturity_s' contains elements that are not numbers.
    Exception
        If 'maturity_s' is empty.
        If 'maturity_s' is multi-dimensional.
        If all elements of 'maturity_s' are zero.

    Returns
    -------
    maturity_s : np.ndarray
        The validated maturity_s array.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(maturity_s, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("maturity_s should be provided in the following object: [ np.ndarray, pd.core.series.Series, list ]")
    
    if isinstance(maturity_s, pd.Series):
        maturity_s = maturity_s.to_numpy()

        if len(maturity_s) == 0: 
            raise Exception("The Series with the maturity_s is empty")
        
    if isinstance(maturity_s, np.ndarray):
        if len(maturity_s) == 0: 
            raise Exception("The array with the maturity_s is empty")
        
        if maturity_s.ndim > 1:
            raise Exception("The maturity_s array should be a one-dimensional array")
        
        if not np.all(np.isreal(maturity_s)):
            raise TypeError("The array of maturity_s should contain only numbers")
        
    if isinstance(maturity_s, list):
        if len(maturity_s) == 0: 
            raise Exception("The list with the maturity_s is empty")
        
        if np.ndim(maturity_s) > 1:
            raise Exception("The maturity_s list should be a one-dimensional list")
        
        if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in maturity_s):
            raise TypeError("The list of maturity_s should contain only numbers")
        
        maturity_s = np.array(maturity_s)

    if np.all(maturity_s <= 0):
        raise Exception("All elements in maturity_s must be greater than 0.")
        
    # Check if the numbers are integers or float with .5 increment
    if not all((x > 0 and (x == int(x) or (x - int(x)) == 0.5)) for x in maturity_s):
        raise Exception("All elements in maturity_s should be positive integers or floats ending in 0.5.")

    return maturity_s

## check_num_assets_s #####################################################################################################################################################
############################################################################################################################################################################

def check_num_assets_s(num_assets_s):
    """
    Check if a list, pandas series, numpy array or dataframe of num_assets_s is valid and return a numpy array of num_assets_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of num_assets_s for a multiple options.
    Returns:
    A numpy array of num_assets_s.
    Raises:
    TypeError: If the num_assets_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the num_assets_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the num_assets_s contain non-numeric types.
    Exception: If the num_assets_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(num_assets_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("num_assets_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(num_assets_s,list):
        if len(num_assets_s) == 0:
            raise Exception("The list of num_assets_s is empty")
        dim = np.ndim(num_assets_s)
        if dim > 1:
            raise Exception("The num_assets_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s):
            raise TypeError("The list of num_assets_s should contain only numbers")
        num_assets_s = np.asarray(num_assets_s)

    if isinstance(num_assets_s,pd.core.series.Series):
        num_assets_s = num_assets_s.values
        if len(num_assets_s) == 0:
            raise Exception("The Series of num_assets_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s):
            raise TypeError("The Series of num_assets_s should contain only numbers")
        num_assets_s = np.asarray(num_assets_s)

    if isinstance(num_assets_s,np.ndarray):
        if len(num_assets_s) == 0:
            raise Exception("The array of num_assets_s is empty")
        dim = np.ndim(num_assets_s)
        if dim > 1:
            raise Exception("The num_assets_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s):
            raise TypeError("The array of num_assets_s should contain only numbers")

    if isinstance(num_assets_s,pd.DataFrame):
        if num_assets_s.empty:
            raise Exception("The DataFrame with the num_assets_s is empty")
        if num_assets_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in num_assets_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_forward_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if num_assets_s.shape[1] == 1:
            num_assets_s = num_assets_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(num_assets_s).any():
        raise Exception("The num_assets_s contain NaN values")

    return num_assets_s

## check_face_value_s ####################################################################################################################################################
############################################################################################################################################################################

def check_face_value_s(face_value_s):
    """
    This function validates the input 'face_value_s', ensuring that it is a one-dimensional array-like object (numpy array, pandas series, or list)
    that contains only real numbers, and has at least one non-zero element.

    Parameters
    ----------
    face_value_s : np.ndarray, pd.core.series.Series, list
        The maturities of a portfolio of bonds. This should be a one-dimensional array-like object containing only real numbers.

    Raises
    ------
    TypeError
        If 'face_value_s' is not a numpy array, pandas Series or list.
        If 'face_value_s' contains elements that are not numbers.
    Exception
        If 'face_value_s' is empty.
        If 'face_value_s' is multi-dimensional.
        If all elements of 'face_value_s' are zero.

    Returns
    -------
    face_value_s : np.ndarray
        The validated face_value_s array.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(face_value_s, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("face_value_s should be provided in the following object: [ np.ndarray, pd.core.series.Series, list ]")
    
    if isinstance(face_value_s, pd.Series):
        face_value_s = face_value_s.to_numpy()

        if len(face_value_s) == 0: 
            raise Exception("The Series with the face_value_s is empty")
        
    if isinstance(face_value_s, np.ndarray):
        if len(face_value_s) == 0: 
            raise Exception("The array with the face_value_s is empty")
        
        if face_value_s.ndim > 1:
            raise Exception("The face_value_s array should be a one-dimensional array")
        
        if not np.all(np.isreal(face_value_s)):
            raise TypeError("The array of face_value_s should contain only numbers")
        
    if isinstance(face_value_s, list):
        if len(face_value_s) == 0: 
            raise Exception("The list with the face_value_s is empty")
        
        if np.ndim(face_value_s) > 1:
            raise Exception("The face_value_s list should be a one-dimensional list")
        
        if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in face_value_s):
            raise TypeError("The list of face_value_s should contain only numbers")
        
        face_value_s = np.array(face_value_s)

    if np.all(face_value_s <= 0):
        raise Exception("All elements in face_value_s must be greater than 0.")
        
    if not all((x > 0) for x in face_value_s):
        raise Exception("All elements in face_value_s should be positive integers or floats - face value cannot be negative")

    return face_value_s

## check_is_zcb_s #####################################################################################################################################################
############################################################################################################################################################################

def check_is_zcb_s(is_zcb_s):
    """
    Check if a list, pandas series, numpy array, or dataframe of is_zcb_s is valid and return a numpy array of is_zcb_s.
    
    Parameters
    ----------
    is_zcb_s : list, pd.Series, np.ndarray, pd.DataFrame
        A list, pandas series, numpy array, or dataframe of is_zcb_s for multiple options.

    Raises
    ------
    TypeError
        If the is_zcb_s are not provided in a list, pandas series, numpy array, or dataframe.
        If the is_zcb_s contain non-boolean types.
    Exception
        If the list, series, array, or dataframe of is_zcb_s is empty.

    Returns
    -------
    np.ndarray
        A numpy array of is_zcb_s with dtype np.bool_.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(is_zcb_s, (list, pd.Series, np.ndarray, pd.DataFrame)):
        raise TypeError("is_zcb_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    # Convert different types to numpy array for uniform handling
    is_zcb_s = np.asarray(is_zcb_s)

    # Check for empty array
    if is_zcb_s.size == 0:
        raise Exception("The array of is_zcb_s is empty")
    
    # Check for bool dtype
    if is_zcb_s.dtype != np.bool_:
        raise TypeError("The array of is_zcb_s should contain only bools [True,False]")

    return is_zcb_s

## check_coupon_rate_s ###############################################################################################################################################################
############################################################################################################################################################################

def check_coupon_rate_s(coupon_rate_s):
    """
    Check if a list, pandas series, numpy array or dataframe of coupon_rate_s is valid and return a numpy array of coupon_rate_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of coupon_rate_s for a multiple options.
    Returns:
    A numpy array of coupon_rate_s.
    Raises:
    TypeError: If the coupon_rate_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of coupon_rate_s is empty.
    Exception: If the coupon_rate_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the coupon_rate_s contain non-numeric types.
    Exception: If the coupon_rate_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(coupon_rate_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("coupon_rate_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(coupon_rate_s,list):
        if len(coupon_rate_s) == 0:
            raise Exception("The list of coupon_rate_s is empty")
        dim = np.ndim(coupon_rate_s)
        if dim > 1:
            raise Exception("The coupon_rate_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s):
            raise TypeError("The list of coupon_rate_s should contain only numbers")
        coupon_rate_s = np.asarray(coupon_rate_s)

    if isinstance(coupon_rate_s,pd.core.series.Series):
        coupon_rate_s = coupon_rate_s.values
        if len(coupon_rate_s) == 0:
            raise Exception("The Series of coupon_rate_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s):
            raise TypeError("The Series of coupon_rate_s should contain only numbers")
        coupon_rate_s = np.asarray(coupon_rate_s)

    if isinstance(coupon_rate_s,np.ndarray):
        if len(coupon_rate_s) == 0:
            raise Exception("The array of coupon_rate_s is empty")
        dim = np.ndim(coupon_rate_s)
        if dim > 1:
            raise Exception("The coupon_rate_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s):
            raise TypeError("The array of coupon_rate_s should contain only numbers - (%)")

    if isinstance(coupon_rate_s,pd.DataFrame):
        if coupon_rate_s.empty:
            raise Exception("The DataFrame with the coupon_rate_s is empty")
        if coupon_rate_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in coupon_rate_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if coupon_rate_s.shape[1] == 1:
            coupon_rate_s = coupon_rate_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(coupon_rate_s).any():
        raise Exception("The coupon_rate_s contain NaN values")

    # Check if any value is less than zero
    if (coupon_rate_s < 0).any():
        raise Exception("All values in coupon_rate_s must be greater or equal to 0")
        
    return coupon_rate_s

## check_semi_annual_payment_s #####################################################################################################################################################
############################################################################################################################################################################

def check_semi_annual_payment_s(semi_annual_payment_s):
    """
    Check if a list, pandas series, numpy array, or dataframe of semi_annual_payment_s is valid and return a numpy array of semi_annual_payment_s.
    
    Parameters
    ----------
    semi_annual_payment_s : list, pd.Series, np.ndarray, pd.DataFrame
        A list, pandas series, numpy array, or dataframe of semi_annual_payment_s for multiple options.

    Raises
    ------
    TypeError
        If the semi_annual_payment_s are not provided in a list, pandas series, numpy array, or dataframe.
        If the semi_annual_payment_s contain non-boolean types.
    Exception
        If the list, series, array, or dataframe of semi_annual_payment_s is empty.

    Returns
    -------
    np.ndarray
        A numpy array of semi_annual_payment_s with dtype np.bool_.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(semi_annual_payment_s, (list, pd.Series, np.ndarray, pd.DataFrame)):
        raise TypeError("semi_annual_payment_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    # Convert different types to numpy array for uniform handling
    semi_annual_payment_s = np.asarray(semi_annual_payment_s)

    # Check for empty array
    if semi_annual_payment_s.size == 0:
        raise Exception("The array of semi_annual_payment_s is empty")
    
    # Check for bool dtype
    if semi_annual_payment_s.dtype != np.bool_:
        raise TypeError("The array of semi_annual_payment_s should contain only bools [True,False]")

    return semi_annual_payment_s

    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 


