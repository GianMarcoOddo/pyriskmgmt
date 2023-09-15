
# ------------------------------------------------------------------------------------------
# Author: GIAN MARCO ODDO
# Module: equity_models
#     - Part of the pyriskmgmt package.
# Contact Information:
#   - Email: gian.marco.oddo@usi.ch
#   - LinkedIn: https://www.linkedin.com/in/gian-marco-oddo-8a6b4b207/ 
#   - GitHub: https://github.com/GianMarcoOddo
# Feel free to reach out for any questions or further clarification on this code.
# ------------------------------------------------------------------------------------------

from typing import Union, List ; import numpy as np

"""
####################################################################### NON PARAMETRIC APPROACH ##############################################################################
"""

##########################################################################################################################################################################
##########################################################################################################################################################################
## 1. EquityNprmSingle ###################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EquityNprmSingle:

    """
    Main
    ----
    This class is designed to model Non-Parametric Risk Measures for a single asset. The class performs calculations
    for the Value at Risk (VaR) and Expected Shortfall (ES) using either the quantile or bootstrap method.
    Furthermore, this class also allows for fitting the generalized Pareto distribution (GPD) to the right-hand tail 
    of the loss distribution to perform an Extreme Value Theory (EVT) analysis.

    The interval of the VaR/Es calculation follows the frequency of the provided `returns`
    
    Initial Parameters
    ------------------
    - returns: Array with returns for the asset.
    - position: The position on the asset. A positive value indicates a long position, a negative value indicates a short position.  
      - The position you possess can be determined by multiplying the asset's present-day price by the quantity you hold.
    - alpha: The significance level for the risk measure calculations (VaR and ES). Alpha is the probability of the
    occurrence of a loss greater than or equal to VaR or ES. Default is 0.05.
    - method: The method to calculate the VaR and ES. The possible values are "quantile" or "bootstrap". Default is "quantile".
    - n_bootstrap_samples: The number of bootstrap samples to use when method is set to "bootstrap". Default is 1000.

    Methods
    --------
    - summary(): Returns a dictionary containing the summary of the calculated risk measures.
    - evt(): Performs an Extreme Value Theory (EVT) analysis by fitting the Generalized Pareto Distribution 
    to the right-hand tail of the loss distribution. 
    Returns the calculated VaR and ES under EVT along with the shape and scale parameters of the fitted GPD.

    Attributes
    ----------
    - var: float: The calculated Value at Risk.
    - es: float: The calculated Expected Shortfall.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************
    
    def __init__(self, returns: np.ndarray,
                    position: Union[int, float], 
                    alpha: Union[int, float] = 0.05, 
                    method: str = "quantile", 
                    n_bootstrap_samples: int = 10000):
    

        from pyriskmgmt.inputsControlFunctions import (validate_returns_single, check_position_single, check_alpha, 
                                                       check_method, check_n_bootstrap_samples)
        # check procedure
        self.returns = validate_returns_single(returns) ; self.position = check_position_single(position) ; self.alpha = check_alpha(alpha)
        self.method = check_method(method); self.n_bootstrap_samples = check_n_bootstrap_samples(n_bootstrap_samples) 

        self.var = None ; self.es = None 

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        # computing Value at Risk (VaR) using different methods

        # using the "quantile" method
        if self.method == "quantile":

            # for positive positions
            if self.position > 0:  
                # finding the alpha quantile of the returns
                quantile_alpha = np.quantile(self.returns, self.alpha)
                # determining VaR using the alpha quantile
                self.var = self.position * - quantile_alpha 

            # for negative positions
            elif self.position < 0: 
                # finding the (1-alpha) quantile of the returns
                quantile_alpha = np.quantile(self.returns, 1 - self.alpha)
                # determining VaR using the (1-alpha) quantile
                self.var = - self.position * quantile_alpha 

            # for a zero position
            elif self.position == 0: 
                self.var = 0

        # using the "bootstrap" method
        elif self.method == "bootstrap":

            # generating bootstrap samples from the returns
            bootstrap_samples = np.random.choice(self.returns, (self.n_bootstrap_samples, len(self.returns)), replace=True)

            # for positive positions
            if self.position > 0: 
                # determining the alpha percentile for each bootstrap sample
                bootstrap_VaRs = np.percentile(bootstrap_samples, 100*self.alpha, axis=1)
                # calculating the average of the bootstrap VaRs
                VaR_estimate = np.mean(bootstrap_VaRs)
                # computing VaR using the average bootstrap VaR
                self.var = - self.position * VaR_estimate

            # for negative positions
            elif self.position < 0:
                # determining the (1-alpha) percentile for each bootstrap sample
                bootstrap_VaRs = np.percentile(bootstrap_samples, 100*(1-self.alpha), axis=1)
                # calculating the average of the bootstrap VaRs
                VaR_estimate = np.mean(bootstrap_VaRs)
                # computing VaR using the average bootstrap VaR
                self.var = - self.position * VaR_estimate

            # for a zero position
            elif self.position == 0:
                self.var = 0

        # Computing Expected Shortfall (ES)

        # using the "quantile" method
        if self.method == "quantile":

            # for positive positions
            if self.position > 0:
                # finding the alpha quantile of the returns
                quantile_alpha = np.quantile(self.returns, self.alpha)
                # computing ES based on the mean of returns below the alpha quantile
                self.es = np.mean(self.returns[self.returns < quantile_alpha]) * - self.position

            # for negative positions
            if self.position < 0:
                # finding the (1-alpha) quantile of the returns
                quantile_alpha = np.quantile(self.returns, 1 - self.alpha)
                # computing ES based on the mean of returns above the (1-alpha) quantile
                self.es = np.mean(self.returns[self.returns > quantile_alpha]) * - self.position

            # for a zero position
            elif self.position == 0:
                self.es = 0

        # using the "bootstrap" method
        elif self.method == "bootstrap":

            # generating bootstrap samples from the returns
            bootstrap_samples = np.random.choice(self.returns, (self.n_bootstrap_samples, len(self.returns)), replace=True)

            # for positive positions
            if self.position > 0:
                # determining the alpha percentile for each bootstrap sample
                bootstrap_VaRs = np.percentile(bootstrap_samples, 100*self.alpha, axis=1)
                # calculating the average of returns less than or equal to the bootstrap VaR for each sample
                bootstrap_ESs = [np.mean(sample[sample <= VaR]) for sample, VaR in zip(bootstrap_samples, bootstrap_VaRs)]
                # computing ES using the average bootstrap ES
                self.es = np.mean(bootstrap_ESs) * - self.position

            # for negative positions
            elif self.position < 0:
                # determining the (1-alpha) percentile for each bootstrap sample
                bootstrap_VaRs = np.percentile(bootstrap_samples, 100*(1-self.alpha), axis=1)
                # calculating the average of returns greater than or equal to the bootstrap VaR for each sample
                bootstrap_ESs = [np.mean(sample[sample >= VaR]) for sample, VaR in zip(bootstrap_samples, bootstrap_VaRs)]
                # computing ES using the average bootstrap ES
                self.es = np.mean(bootstrap_ESs) * - self.position

            # for a zero position
            elif self.position == 0:
                self.es = 0

        # Methods ---------------------------------------------------------------------------

        # VaR
        self.var = max(0,self.var) ; self.var = round(self.var, 4)
        # Es
        self.es = max(0,self.es) ; self.es = round(self.es, 4)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # SUMMARY CALCULATOR <<<<<<<<<<<<<<<

    def summary(self):

        """
        Main
        ----
        Calculates and returns a summary of various risk measures for the given asset.
        This method calculates the maximum loss, maximum excess loss (over VaR), maximum loss over VaR, 
        and the Expected Shortfall over VaR. These measures are used to give an overall picture of the risk 
        associated with the asset. All these measures are rounded off to 4 decimal places for readability.
        
        Returns
        -------
        - summary : dict : A dictionary containing the following key-value pairs:
            - 'var': The Value at Risk (VaR) for the asset.
            - 'maxLoss': The maximum loss for the asset.
            - 'maxExcessLoss': The maximum excess loss (over VaR) for the asset.
            - 'maxExcessLossOverVar': The ratio of the maximum excess loss to VaR.
            - 'es': The Expected Shortfall for the asset.
            - 'esOverVar': The ratio of the Expected Shortfall to VaR.
            
        Note
        ---- 
        The method calculates these measures differently depending on whether the position on the asset is long (positive), 
        short (negative), or there is no position (zero).
        """

        import warnings ; warnings.filterwarnings("ignore")

        # evaluating the maximum potential loss based on the position for positive positions
        if self.position > 0:
            # calculating the maximum potential loss by taking the minimum return and the position
            Max_loss = (np.min(self.returns) * self.position) * -1
        # for negative positions
        elif self.position < 0:
            # calculating the maximum potential loss by taking the maximum return and the position
            Max_loss = np.max(self.returns) * -self.position
        # for neutral positions
        elif self.position == 0:
            Max_loss = 0 

        # ensuring the Max_loss is non-negative
        Max_loss = max(0, Max_loss)

        # computing the maximum excess loss beyond the Value at Risk (VaR)
        Max_Excess_loss = Max_loss - self.var

        # if VaR is zero, setting related values to zero
        if self.var == 0: 
            Max_loss_over_VaR = 0
            Expected_shortfall_over_VaR = 0
        else:
            # computing the ratio of the maximum excess loss to VaR
            Max_loss_over_VaR = Max_Excess_loss / self.var
            # computing the ratio of expected shortfall to VaR
            Expected_shortfall_over_VaR = self.es / self.var

        summary = {"var" : self.var,
                   "maxLoss" : round(Max_loss,4),
                   "maxExcessLoss" : round(Max_Excess_loss,4),
                   "maxExcessLossOverVar" : round(Max_loss_over_VaR,4),
                   "es" : self.es,
                   "esOverVar" : round(Expected_shortfall_over_VaR,4)}
        
        return summary

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # EXTREME VALUE THEORY <<<<<<<<<<<<<<<

    def evt(self, alpha: float = 0.025, quantile_threshold: float = 0.95):

        """
        Main
        ----
        Performs Extreme Value Theory (EVT) to estimate risk measures such as VaR (Value at Risk) 
        and ES (Expected Shortfall). The EVT estimates are calculated using the Generalized Pareto Distribution (GPD)
        and maximum likelihood estimation (MLE).
        EVT is a method used to assess the risk of extreme events. It is particularly useful for estimating the risk 
        of rare events that are not well-represented in the available data.

        Parameters
        -------------
        - alpha: The significance level for the VaR and ES calculation. Default is 0.025.
        - quantile_threshold: The quantile threshold used for the EVT. Default is 0.95.

        Returns
        -------
        - VaR_ES : dict : A dictionary containing the calculated risk measures:
            - 'conf_level': Confidence level which is 1-alpha.
            - 'method': The method used to calculate the risk measures ("quantile" or "bootstrap").
            - 'var': The Value at Risk (VaR).
            - 'es': The Expected Shortfall (ES).
            - 'xi': The shape parameter of the GPD.
            - 'beta': The scale parameter of the GPD.

        Notes
        -----
        The method calculates these measures differently depending on whether the position on the asset is long (positive), 
        short (negative), or there is no position (zero). Also, calculations are done differently based on the method ("quantile" or "bootstrap").
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.optimize import minimize ; import numpy as np   

        from pyriskmgmt.inputsControlFunctions import (check_alpha, check_quantile_threshold)
        
        # check procedure
        alpha = check_alpha(alpha) ; quantile_threshold = check_quantile_threshold(quantile_threshold)

        # importing from pyriskmgmt.SupportFunctions the gpd_log_likelihood
        from pyriskmgmt.SupportFunctions import gpd_log_likelihood

        # checking if the chosen method is 'quantile'
        if self.method == "quantile": 

            # handling the case for a positive position
            if self.position > 0:
                # sorting returns in descending order for positive positions
                sorted_returns = np.sort(self.returns)[::-1] * -1 
                # determining the threshold based on the given quantile
                u = np.quantile(sorted_returns, quantile_threshold) 
                u = u * self.position
                # calculating losses based on the position
                sorted_losses = sorted_returns * self.position
                n = len(sorted_losses)
                # identifying losses that exceed the threshold
                right_hand_tail = sorted_losses[sorted_losses > u]
                nu = len(right_hand_tail)

                # setting the initial parameter estimates for the generalized pareto distribution
                init_params = [0.1, 0.1]  
                # establishing constraints to ensure the parameters are positive
                bounds = [(0.001, None), (0.001, None)] 
                # optimizing the log likelihood for the generalized pareto distribution
                result = minimize(gpd_log_likelihood, init_params, args=(right_hand_tail, u), bounds=bounds)
                # retrieving the optimized parameters
                ξ_hat, β_hat = result.x

                # computing Value-at-Risk (VaR) and Expected Shortfall (ES) using the optimized parameters
                VaR = u  + (β_hat / ξ_hat) * ((((n / nu) * (1 - (1-alpha))) ** -ξ_hat) - 1)
                ES = (VaR + β_hat - ξ_hat * u) / (1- ξ_hat)

            # handling the case for a negative position
            elif self.position < 0:
                # sorting returns in ascending order for negative positions
                sorted_returns = np.sort(self.returns)
                # similar steps as above, tailored for a negative position
                u = np.quantile(sorted_returns, quantile_threshold) 
                u = u * - self.position
                sorted_losses = sorted_returns * self.position 
                n = len(sorted_losses)
                right_hand_tail = sorted_losses[sorted_losses > u]
                nu = len(right_hand_tail)

                # performing the same optimization process as above for the negative position
                init_params = [0.1, 0.1]  
                bounds = [(0.001, None), (0.001, None)] 
                result = minimize(gpd_log_likelihood, init_params, args=(right_hand_tail, u), bounds=bounds)
                ξ_hat, β_hat = result.x

                # calculating VaR and ES for the negative position
                VaR = u  + (β_hat / ξ_hat) * ((((n / nu) * (1 - (1-alpha))) ** -ξ_hat) - 1)
                ES = (VaR + β_hat - ξ_hat * u) / (1- ξ_hat)

            # handling the neutral position scenario
            elif self.position == 0:
                VaR = 0; ES = 0 ; ξ_hat = 0 ; β_hat = 0

        # checking if the chosen method is 'bootstrap'
        if self.method == "bootstrap":

            # handling the case for a positive position
            if self.position > 0:
                # generating bootstrap samples
                bootstrap_samples = np.random.choice(self.returns, (self.n_bootstrap_samples, len(self.returns)), replace=True)

                # initializing lists to store bootstrap results
                VaRs = [] ; ESs = [] ; ξ_hats = [] ; β_hats = []

                for row in range(bootstrap_samples.shape[0]):
                    # sorting and processing bootstrap samples similar to the 'quantile' method for positive positions
                    sorted_returns = np.sort(bootstrap_samples[row,:])[::-1] * -1 
                    u = np.quantile(sorted_returns, quantile_threshold) 
                    u = u * self.position
                    sorted_losses = sorted_returns * self.position 
                    n = len(sorted_losses)
                    right_hand_tail = sorted_losses[sorted_losses >= u]
                    nu = len(right_hand_tail)

                    # estimating parameters using the generalized pareto distribution
                    init_params = [0.1, 0.1]  
                    bounds = [(0.001, None), (0.001, None)] 
                    result = minimize(gpd_log_likelihood, init_params, args=(right_hand_tail, u), bounds=bounds)
                    ξ_hat_th, β_hat_th = result.x

                    # calculating VaR and ES for each bootstrap sample
                    VaR_th = u  + (β_hat_th / ξ_hat_th) * ((((n / nu) * (1 - (1-alpha))) ** -ξ_hat_th) - 1)
                    ES_th = (VaR_th + β_hat_th - ξ_hat_th * u) / (1- ξ_hat_th)
                    VaRs.append(VaR_th) ; ESs.append(ES_th) ; ξ_hats.append(ξ_hat_th) ; β_hats.append(β_hat_th)

                # taking the average of bootstrap results
                VaR = np.mean(VaRs) ; ES = np.mean(ESs) ; ξ_hat = np.mean(ξ_hats) ; β_hat = np.mean(β_hats)

            # handling the case for a negative position
            elif self.position < 0:
                # generating bootstrap samples
                bootstrap_samples = np.random.choice(self.returns, (self.n_bootstrap_samples, len(self.returns)), replace=True)
                # initializing lists to store bootstrap results
                VaRs = [] ; ESs = [] ; ξ_hats = [] ; β_hats = []

                for row in range(bootstrap_samples.shape[0]):
                    # processing bootstrap samples similar to the 'quantile' method for negative positions
                    sorted_returns = np.sort(bootstrap_samples[row,:])
                    u = np.quantile(sorted_returns, quantile_threshold) 
                    u = u * - self.position
                    sorted_losses = sorted_returns * self.position 
                    n = len(sorted_losses)
                    right_hand_tail = sorted_losses[sorted_losses >= u]
                    nu = len(right_hand_tail)

                    # estimating parameters using the generalized pareto distribution
                    init_params = [0.1, 0.1]  
                    bounds = [(0.001, None), (0.001, None)] 
                    result = minimize(gpd_log_likelihood, init_params, args=(right_hand_tail, u), bounds=bounds)
                    ξ_hat_th, β_hat_th = result.x

                    # calculating VaR and ES for each bootstrap sample
                    VaR_th = u  + (β_hat_th / ξ_hat_th) * ((((n / nu) * (1 - (1-alpha))) ** -ξ_hat_th) - 1)
                    ES_th = (VaR_th + β_hat_th - ξ_hat_th * u) / (1- ξ_hat_th)
                    VaRs.append(VaR_th) ; ESs.append(ES_th) ; ξ_hats.append(ξ_hat_th) ; β_hats.append(β_hat_th)

                # averaging bootstrap results for negative positions
                VaR = np.mean(VaRs) ; ES = np.mean(ESs) ; ξ_hat = np.mean(ξ_hats) ; β_hat = np.mean(β_hats)

            # handling the neutral position scenario
            elif self.position == 0:
                VaR = 0 ; ES = 0 ; ξ_hat = 0 ; β_hat = 0

        # storing the results in a dictionary
        VaR_ES = {"conf_level" : 1-alpha,
                    "method" : self.method,
                    "var" : VaR,
                    "es" : ES,
                    "xi" : ξ_hat,
                    "beta" : β_hat}

        if VaR != 0 and ES != 0 and ξ_hat != 0 and β_hat != 0:
            return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in VaR_ES.items()}
        else:
            return VaR_ES
        
###############################################################################################################################################################################
###############################################################################################################################################################################
## 2. EquityNprmPort ###############################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

class EquityNprmPort:

    """
    Main
    ----
    The `EquityNprmPort` class is a non-parametric risk management class designed to provide risk measures for a portfolio of assets.
    The non-parametric approach does not rely on any assumptions regarding the underlying distribution of asset returns. 

    The interval of the VaR/Es calculation follows the frequency of the provided returns.

    Initial Parameters
    ------------------
    - returns: A numpy array of asset returns.
    - positions: A list of asset positions.
      - For each asset, the position you possess can be determined by multiplying the asset's present-day price by the quantity you hold.
    - alpha: The significance level for VaR and ES calculations. Default is 0.05.
    - method: A method to be used for the VaR and ES calculations. The options are "quantile" for quantile-based VaR and ES calculation or "bootstrap" for a 
    bootstrap-based calculation. Default is "quantile".
    - n_bootstrap_samples: The number of bootstrap samples to be used when the bootstrap method is selected. This is ignored when the quantile method is selected. Default is 1000.

    Methods
    --------
    - summary(): Provides a summary of the risk measures, including VaR, Expected Shortfall (ES), Maximum Loss, Maximum Excess Loss, and ratios of these measures.
    - evt(): Provides risk measures using Extreme Value Theory (EVT). This is a semi-parametric approach that fits a Generalized Pareto Distribution (GPD) to the tail 
    of the loss distribution.
    - MargVars(): Provides the marginal VaRs for each asset in the portfolio. This is calculated as the change in portfolio VaR resulting from a small 
    change in the position of a specific asset. Only available if self.method = "quantile".
    
    Attributes
    ---------
    - var: The calculated Value at Risk (VaR) for the portfolio.
    - es: The calculated Expected Shortfall (ES) for the portfolio. 
    - port_returns: Cumulative returns of the portfolio based on the asset returns and positions. 

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, returns: np.ndarray,
                 positions: List[Union[int, float]],
                 alpha: Union[int, float] = 0.05, 
                 method: str = "quantile", 
                 n_bootstrap_samples: int = 10000):

        from pyriskmgmt.inputsControlFunctions import (validate_returns_port, check_positions_port, check_alpha, 
                                                       check_method, check_n_bootstrap_samples)
        
        # check procedure
        self.returns = validate_returns_port(returns) ; self.positions = check_positions_port(positions)

        if len(self.positions) != self.returns.shape[1]:
            raise ValueError("The number of assets != the number of positions")

        self.alpha = check_alpha(alpha) ;  self.method = check_method(method) ; self.n_bootstrap_samples = check_n_bootstrap_samples(n_bootstrap_samples)

        self.var = None ; self.es = None 

        self.fit()  

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        # calculating value at Risk  ----------------------------------

        # copying the returns to a new variable 'portfolio'
        portfolio = np.copy(self.returns)
        # converting all positions to float for arithmetic operations
        new_positions = [float(num) for num in self.positions]

        # iterating to calculate cumulative returns for the portfolio
        for i in range(portfolio.shape[0]):
            # multiplying the returns by the position
            portfolio[i, :] *= new_positions 
            # updating the positions with the new returns 
            new_positions += portfolio[i, :]   

        # saving the total portfolio returns for further use
        self.port_returns = portfolio.sum(axis=1)

        # checking method type for quantile

        if self.method == "quantile":
            # calculating VaR using quantile method
            self.var = np.quantile(self.port_returns, self.alpha) * -1 

        # checking method type for bootstrap

        elif self.method == "bootstrap":
            # drawing samples for bootstrap method
            bootstrap_samples = np.random.choice(self.port_returns, (self.n_bootstrap_samples, len(self.port_returns)), replace=True)
            # calculating VaR for each bootstrap sample
            bootstrap_VaRs = np.percentile(bootstrap_samples, 100*self.alpha, axis=1)
            # averaging out all the VaR values
            self.var = np.mean(bootstrap_VaRs) * -1 

        # calculating Expected Shortfall ----------------------------------

        # checking method type for quantile

        if self.method == "quantile":
            # calculating expected shortfall using quantile method
            self.es =  np.mean(self.port_returns[self.port_returns < - self.var]) * -1 

        # checking method type for bootstrap

        elif self.method == "bootstrap":
            # drawing samples for bootstrap method again
            bootstrap_samples = np.random.choice(self.port_returns, (self.n_bootstrap_samples, len(self.port_returns)), replace=True)
            # calculating VaR for each bootstrap sample
            bootstrap_VaRs = np.percentile(bootstrap_samples, 100*self.alpha, axis=1)
            bootstrap_Es = []
            for row in range(bootstrap_samples.shape[0]):
                # calculating the expected shortfall for the current sample
                Es_th = np.mean(bootstrap_samples[row,:][bootstrap_samples[row,:] <= bootstrap_VaRs[row]]) * -1 
                bootstrap_Es.append(Es_th)
            # averaging out all the expected shortfall values
            self.es = np.mean(bootstrap_Es)

        # Methods ---------------------------------------------------------------------------

        # for later usage
        self.port_returns = self.port_returns

        # VaR
        self.var = max(0,self.var) ; self.var = round(self.var, 4)
        # Es
        self.es = max(0,self.es) ; self.es = round(self.es, 4)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # SUMMARY CALCULATOR <<<<<<<<<<<<<<<

    def summary(self):

        """
        Main
        ----
        Computes various risk metrics for the portfolio and returns them in a dictionary.
        
        Risk Metrics
        ------------
        The metrics included are:
        - Value at Risk (VaR): The maximum loss not exceeded with a certain confidence level. VaR gives an idea of how much we might expect to lose with a certain level of confidence.
        - Maximum Loss: The largest loss observed in the portfolio returns.
        - Maximum Excess Loss: The amount by which the Maximum Loss exceeds VaR.
        - Maximum Loss over VaR: The ratio of Maximum Excess Loss to VaR, indicating how much the maximum loss exceeds the estimated VaR.
        - Expected Shortfall (ES): The average loss expected in the worst case scenarios. ES, also known as Conditional VaR (CVaR), provides an idea of the severity of losses when extreme events occur.
        - ES over VaR: The ratio of ES to VaR, which indicates the severity of losses when extreme events occur relative to the estimated VaR.

        Returns
        -------
        - summary: A dictionary containing the computed risk metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")
                
        # calculating the maximum loss 
        Max_loss = np.min(self.port_returns) * -1 
        # computing the maximum excess loss
        Max_Excess_loss = Max_loss - self.var

        # checking if VaR is not zero for avoiding division by zero
        if self.var != 0: 
            # calculating the ratio of maximum excess loss to VaR
            Max_loss_over_VaR = Max_Excess_loss/self.var
        else: 
            # setting the ratio to zero if VaR is zero
            Max_loss_over_VaR = 0

        # checking if VaR is not zero for computing the ratio
        if self.var != 0: 
            # calculating the ratio of expected shortfall to VaR
            Expected_shortfall_over_VaR = self.es / self.var
        else: 
            # setting the ratio to zero if VaR is zero
            Expected_shortfall_over_VaR = 0

        # handling edge cases where values are very close to zero
        if Max_loss == -0 and Max_Excess_loss == -0.0: 
            # setting values explicitly to zero
            Max_loss = 0 
            Max_Excess_loss = 0

        summary = {"var" : self.var,
                  "maxLoss" : round(Max_loss,4),
                  "maxExcessLoss" : round(Max_Excess_loss,4),
                  "maxExcessLossOverVar" : round(Max_loss_over_VaR,4),
                  "es": self.es,
                  "esOverVar" : round(Expected_shortfall_over_VaR,4)}
        
        return summary

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # EXTREME VALUE THEORY <<<<<<<<<<<<<<<

    def evt(self, alpha: float = 0.025, quantile_threshold: float = 0.95):

        """
        Main
        ----
        Estimates the Value at Risk (VaR) and Expected Shortfall (ES) using the Extreme Value Theory (EVT).
        EVT is used for assessing risk for extreme events. This function implements both quantile-based and bootstrap-based methods for EVT.

        Parameters
        ----------
        - alpha: Significance level for VaR and ES. The default is 0.025.
        - quantile_threshold: Threshold for separating the tail from the rest of the data in the GPD fitting. The default is 0.95.

        Returns
        -------
        - VaR_ES: A dictionary containing the estimated VaR, ES, shape parameter (xi), and scale parameter (beta), as well as the method used and confidence level. 
        Values are rounded to four decimal places. 

        Raises
        ------
        ValueError: If the provided alpha or quantile_threshold are outside their valid ranges.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_alpha, check_quantile_threshold)

        # check procedure
        alpha = check_alpha(alpha) ; quantile_threshold = check_quantile_threshold(quantile_threshold)

        # importing from pyriskmgmt.SupportFunctions the gpd_log_likelihood
        from pyriskmgmt.SupportFunctions import gpd_log_likelihood

        from scipy.optimize import minimize ; import numpy as np   

        # checking if method is Quantile 
        if self.method == "quantile": 
                
                # sorting returns in ascending order
                sorted_losses = np.sort(self.port_returns)[::-1] * -1 
                u = np.quantile(sorted_losses, quantile_threshold)
                n = len(sorted_losses)
                right_hand_tail = sorted_losses[sorted_losses > u]
                nu = len(right_hand_tail)

        # using Max Likelihood Estimation

                # initializing guess for ξ and β
                init_params = [0.1, 0.1]  
                # defining bounds for ξ and β to ensure they are > 0
                bounds = [(0.001, None), (0.001, None)] 
                # optimizing using scipy's minimize function
                result = minimize(gpd_log_likelihood, init_params, args=(right_hand_tail, u), bounds=bounds)
                ξ_hat, β_hat = result.x

        # computing VaR and ES for Quantile method

                VaR = u  + (β_hat / ξ_hat) * ((((n / nu) * (1 - (1-alpha))) ** -ξ_hat) - 1) 
                ES = (VaR + β_hat - ξ_hat * u) / (1- ξ_hat)

        # checking if method is Bootstrap 

        if self.method == "bootstrap":

                bootstrap_samples = np.random.choice(self.port_returns, (self.n_bootstrap_samples, len(self.port_returns)), replace=True)

                VaRs = [] ; ESs = [] ; ξ_hats = [] ; β_hats = []

                for row in range(bootstrap_samples.shape[0]):
                    # sorting sampled losses in ascending order
                    sorted_losses = np.sort(bootstrap_samples[row,:])[::-1] * -1 
                    u = np.quantile(sorted_losses, quantile_threshold) 
                    n = len(sorted_losses)
                    right_hand_tail = sorted_losses[sorted_losses > u]
                    nu = len(right_hand_tail)

            # using Max Likelihood Estimation for bootstrap samples

                    # initializing guess for ξ and β
                    init_params = [0.1, 0.1]  
                    # defining bounds for ξ and β to ensure they are > 0
                    bounds = [(0.001, None), (0.001, None)] 
                    # optimizing using scipy's minimize function for bootstrap samples
                    result = minimize(gpd_log_likelihood, init_params, args=(right_hand_tail, u), bounds=bounds)
                    ξ_hat_th, β_hat_th = result.x

            # computing VaR and ES for each bootstrap sample

                    VaR_th = u  + (β_hat_th / ξ_hat_th) * ((((n / nu) * (1 - (1-alpha))) ** -ξ_hat_th) - 1)
                    ES_th = (VaR_th + β_hat_th - ξ_hat_th * u) / (1- ξ_hat_th)
                    VaRs.append(VaR_th) ; ESs.append(ES_th) ; ξ_hats.append(ξ_hat_th) ; β_hats.append(β_hat_th)
                
                # calculating mean VaR, ES, ξ and β from bootstrap samples
                VaR = np.mean(VaRs)  ; ES = np.mean(ESs) ; ξ_hat = np.mean(ξ_hats) ; β_hat = np.mean(β_hats)

        VaR_ES = {"conf_level" : 1-alpha,
                  "method" : self.method,
                  "var" : VaR,
                  "es" : ES,
                  "xi" : ξ_hat,
                  "beta" : β_hat}

        if VaR != 0 and ES != 0 and ξ_hat != 0 and β_hat != 0:
            return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in VaR_ES.items()}
        else:
            return VaR_ES 

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # MARGINAL VARS <<<<<<<<<<<<<<<
            
    def MargVars(self, scale_factor: float = 0.1):

        """
        Main
        ----
        Computes the Marginal Value at Risk (MVaR) for each asset in the portfolio using a non-parametric approach.
        MVaR measures the rate of change in the portfolio's VaR with respect to a small change in the position of a specific asset. 
        This method works by perturbing each asset's position by a small amount in the same direction (proportional to a scale_factor), calculating the new VaR,
        and measuring the difference from the original VaR. The result is an array of MVaRs, one for each asset.
        - New position = old position +/- (old position * scale_factor), for each position.

        Parameters
        ----------
        - scale_factor: The scale factor used to adjust the asset positions for MVaR calculations. Default is 0.1.

        Returns
        -------
        - A dictonary with: 
          - The calculated VaR
          - A numpy array representing the MVaR for each asset in the portfolio. MVaRs are rounded to four decimal places.

        Note
        -----
        This method can only be used when the 'method' attribute of the class instance is set to 'quantile'. For the 'bootstrap' method, it raises a ValueError.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np

        if self.method == "bootstrap": 
            raise ValueError("The 'MargVars' method can only be used when the 'self.method', in the `EquityNprmPort` class, is set to 'quantile'.")

        from pyriskmgmt.inputsControlFunctions import check_scale_factor

        # check procedure
        scale_factor = check_scale_factor(scale_factor)

        # VaR Calculator -------------------------------------------------------------------------------------

        def Var_Calculator(returns, positions, alpha):
            portfolio = np.copy(returns)
            new_positions = [float(num) for num in positions]
            for i in range(portfolio.shape[0]):
                portfolio[i, :] *= new_positions ; new_positions += portfolio[i, :]   
            port_returns = portfolio.sum(axis=1) ; VAR = np.quantile(port_returns, alpha) * -1
            return VAR 
        
        # -----------------------------------------------------------------------------------------------------

        # The Nonparametric VaR is affected by changes when each position is increased by an amount proportional the scale_factor,
        # in accordance with its corresponding sign

        NpVaR_changes = np.zeros(len(self.positions))

        # creating a copy of positions to work with
        positions = self.positions.copy()

        # -----------------------------------------------------------------------------------

        # calculating the initial Value at Risk (VaR) using the Var_Calculator
        prev_var = Var_Calculator(self.returns, self.positions, self.alpha)

        # iterating through each position in the 'positions' list
        for index in range(len(positions)):

            # defining the absolute increment, defined by scale_factor
            abs_increment = (abs(positions[index]) * scale_factor) 

            # checking if the current position is negative
            if positions[index] < 0:
                # decreasing the negative position  
                positions[index] = positions[index] -  abs_increment
            # checking if the current position is positive
            elif positions[index] > 0:
                # increasing the positive position 
                positions[index] = positions[index] +  abs_increment

            # calculating the new VaR after modifying the position
            new_var = Var_Calculator(self.returns, positions, self.alpha)
            
            # appending the change in VaR (difference between new and initial VaR) to the NpVaR_changes array
            NpVaR_changes[index] = (new_var-prev_var)

            # reverting the position back to its initial value

            # if the position is negative after increasing
            if positions[index] < 0:
                # reverting the increased negative position back to its original value
                positions[index] = positions[index] +  abs_increment
            # if the position is positive after decreasing
            elif positions[index] > 0:
                # reverting the decreased positive position back to its original value
                positions[index] = positions[index] -  abs_increment

      # -----------------------------------------------------------------------------------

        rounded_changes = np.round(NpVaR_changes, 4)

        np.set_printoptions(suppress=True)

        MargVars = {"var" : self.var,
                    "MargVars" : rounded_changes}

        return MargVars
            
"""####################################################################### PARAMETRIC APPROACH ##############################################################################"""

##############################################################################################################################################################################
##############################################################################################################################################################################
## 3. EquityPrmSingle ########################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################

class EquityPrmSingle:

    """
    Main
    -----
    The EquityPrmSingle class provides functionality for risk management of single assets in a portfolio.
    The class allows for the calculation and forecasting Value-at-Risk (VaR) and Expected Shortfall (ES),
    based on different modeling approaches such as simple historical volatility, GARCH, EWMA, and Monte Carlo simulation.

    Initial Parameters
    ------------------
    - returns: An array of historical return data.
    - position: The current position of the asset (how much of the asset is held in monetary term).
    - interval: The interval over which the risk is assessed (in days, assuming 252 traing days in a year) if the frequency of the returns is
    daily, otherwise it follows the time-aggregation of the returns. Default is 1.
      - For example, if you provide weekly returns and the interval is set to 5, the method will calculate the VaR and Es for a period of 5 weeks 
      ahead.
    - alpha: The level of significance for the VaR and ES. Default is 0.05.

    Methods
    -------
    - garch_model(): Calculates the VaR and ES using a GARCH model for volatility.
    - ewma_model(): Calculates the VaR and ES using an EWMA model for volatility.
    - mcModelDrawReturns(): Uses a Monte Carlo simulation to draw from the return distribution to calculate VaR and ES.
    - mcModelGbm(): Simulates a Geometric Brownian Motion (GBM) model to calculate VaR and ES. Allows different volatility and mean return models.
    - kupiec_test(): Performs the Kupiec test (unconditional coverage test) on the VaR model.
    - christoffersen_test(): Performs the Christoffersen test (conditional coverage test) on the VaR model.
    - combined_test(): Performs both the Kupiec and Christoffersen tests on the VaR model and returns the combined results.

    Attributes
    ----------
    - simple_var: The calculated Value at Risk (VaR) for the asset based on simple historical volatility. 
    - simple_es: The calculated Expected Shortfall (ES) for the asset based on simple historical volatility. 
    - simple_sigma: The standard deviation of returns based on historical returns, scaled by the square root of the interval.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, returns: np.ndarray,
                    position: Union[int, float], 
                    interval: int = 1,
                    alpha: Union[int, float] = 0.05):
        
        from pyriskmgmt.inputsControlFunctions import (validate_returns_single, check_position_single, check_interval, check_alpha)
        
        # check procedure
        self.returns = validate_returns_single(returns) ; self.position = check_position_single(position) ; self.interval = check_interval(interval)
        self.alpha = check_alpha(alpha)

        self.simple_var = None ; self.simple_es = None ; self.simple_sigma = None 

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import norm ; import numpy as np

        sigma = np.std(self.returns)

        # Value at Risk ----------------------------------

        q = norm.ppf(1-self.alpha, loc=0, scale=1) ; VaR = q * abs(self.position) * sigma * np.sqrt(self.interval)

        self.simple_var =  max(0,VaR)

        # Expected Shortfall ---------------------------

        q_es = ((np.exp( - (q**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-self.alpha)))) ; ES = q_es * abs(self.position) * sigma * np.sqrt(self.interval)

        self.simple_es = max(0,ES)

        # Methods --------------

        # VaR
        self.simple_var = round(self.simple_var, 4)
        # Es
        self.simple_es = round(self.simple_es, 4)
        # sigma 
        self.simple_sigma = round(sigma * np.sqrt(self.interval), 5)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # GARCH MODEL <<<<<<<<<<<<<<<

    def garch_model(self, p: int = 1, q: int = 1):

        """
        Main
        ----
        Fits a GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model to the volatility of an equity position. This method 
        measures and forecasts volatility and calculates Value at Risk (VaR) and Expected Shortfall (ES) for a given level of significance (alpha).

        Parameters
        ----------
        - p: The order of the GARCH model's autoregressive term. Default is 1.
        - q: The order of the GARCH model's moving average term. Default is 1.

        Returns
        -------
        - dict: A dictionary containing the following keys: 
            - 'var': The Value at Risk (VaR) calculated using the GARCH model.
            - 'es': The Expected Shortfall (ES) calculated using the GARCH model.
            - 'T': The investment horizon over which the VaR and ES are calculated.
            - 'q': The order of the moving average term of the GARCH model.
            - 'p': The order of the autoregressive term of the GARCH model.
            - 'cum_sigma': The cumulative standard deviation forecasted by the GARCH model, in % terms.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

        # check procedure
        p = check_p(p) ; q = check_q(q)

        # to ignore warnings
        import os ; import sys ; import numpy as np ; from scipy.stats import norm ; from arch import arch_model

        # fitting the garch model
        model = arch_model(self.returns, vol="GARCH", p=p, q=q, power= 2.0, dist= "normal") 

        # redirecting stderr to devnull
        stderr = sys.stderr ; sys.stderr = open(os.devnull, 'w')

        try:
            model_fit = model.fit(disp='off')
        except Exception as e:
            print("Garch fitting did not converge:", str(e))
            return
        finally:
            # Reset stderr
            sys.stderr = stderr

        # horizon is 1 --- always better to stay at the highest frequency and then aggregate
        horizon = 1

        # forecasting
        forecasts = model_fit.forecast(start=0, horizon=horizon)

        # getting the variance forecasts
        variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

        # ------------------------------------------------------------------------------------------------------------------------
        
        # 'interval' handler
        if self.interval == 1:
            sigma = np.sqrt(variance_forecasts)
        elif self.interval > 1:
            cumulative_variance = variance_forecasts * self.interval # assuming normality i.i.d for the future
            sigma = np.sqrt(cumulative_variance)

        # ------------------------------------------------------------------------------------------------------------------------

        # Value at Risk ----------------------------------

        quantile = norm.ppf(1-self.alpha, loc=0, scale=1) ; VaR = quantile * abs(self.position) * sigma

        # Expected Shortfall ---------------------------

        q_es = ( ( np.exp( - (quantile**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-self.alpha)))) ; ES = q_es * abs(self.position) * sigma 

        GARCH_MODEL = {"var" : round(VaR, 4),
                       "es" : round(ES, 4),
                        "T" : self.interval,
                        "q" : q,
                        "p" : p,
                        "cum_sigma" : round(sigma, 4)}

        return GARCH_MODEL
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # EWMA MODEL <<<<<<<<<<<<<<<

    def ewma_model(self, lambda_ewma: float = 0.94):

        """
        Main
        ----
        Fits an Exponential Weighted Moving Average (EWMA) model to the volatility of an equity position. The EWMA model provides
        a method of estimating volatility where the weights decrease exponentially for older observations, giving
        more relevance to recent observations. It then calculates Value at Risk (VaR) and Expected Shortfall (ES) 
        for a given level of significance (alpha).

        Parameters
        ----------
        - lambda_ewma: The decay factor for the EWMA model. This parameter determines the speed at which
        the weights decrease. The closer lambda_ewma is to 1, the slower the weights decrease. Default is 0.94.

        Returns
        -------
        - dict: A dictionary containing the following keys: 
            - 'var': The Value at Risk (VaR) calculated using the EWMA model.
            - 'es': The Expected Shortfall (ES) calculated using the EWMA model.
            - 'T': The investment horizon over which the VaR and ES are calculated.
            - 'lambda_ewma': The decay factor used in the EWMA model.
            - 'cum_sigma': The cumulative standard deviation forecasted by the EWMA model, in % terms.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_lambda_ewma)

        # check procedure
        lambda_ewma = check_lambda_ewma(lambda_ewma)
        
        import numpy as np ; from scipy.stats import norm

        # creatting the fitted time series of the ewma(lambda_ewma) variances ------------

        n = len(self.returns)

        # fitting the ewma model

        variance_time_series = np.repeat(self.returns[0]**2, n)

        for t in range(1,n):
            variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * self.returns[t-1]**2

        variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * self.returns[-1]**2

        # ------------------------------------------------------------------------------------------------------------------------

        if self.interval == 1: #########

            sigma = np.sqrt(variance_t_plus1)    

        if self.interval >1 : #########

            sigma = np.sqrt(variance_t_plus1) * np.sqrt(self.interval) # assuming normality i.i.d for the future

        # ---------------------------------------------------------------------------------------------------------------

        # Value at Risk ----------------------------------

        quantile = norm.ppf(1-self.alpha, loc=0, scale=1) ; VaR_ewma = quantile * abs(self.position) * sigma

        # Expected Shortfall ---------------------------

        q_es = ( ( np.exp( - (quantile**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-self.alpha)))) ; Es_ewma = q_es * abs(self.position) * sigma

        VaR_ewma = {"var" : round(VaR_ewma, 4),
                    "es" : round(Es_ewma, 4),
                    "T" : self.interval,
                    "lambda_ewma" : lambda_ewma,
                    "cum_sigma" :round(sigma, 4)}
        
        return VaR_ewma
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # MONTECARLO MODEL [RETURNS - DISTIBUTION] <<<<<<<<<<<<
    
    def mcModelDrawReturns(self, vol: str = "simple", mu: str = "moving_average", num_sims: int = 10000,
                              p: int = 1, q: int = 1, lambda_ewma: float = 0.94,
                              ma_window: int = 15, return_mc_returns: bool = False):

        """
        Main
        -----
        Runs Monte Carlo simulations of future equity returns based on different types of volatilities: 
        simple historical volatility, GARCH model volatility, or Exponential Weighted Moving Average (EWMA) volatility.
        It uses the simulations to estimate Value at Risk (VaR) and Expected Shortfall (ES). 

        Parameters
        ----------
        - vol: Specifies the type of volatility to use in the simulations. Options are "simple", "garch", "ewma". Default is "simple".
        - mu: Specifies the type of mean to use in the simulations. Options are "zero", "constant", "moving_average". Default is "moving_average".
        - num_sims: The number of Monte Carlo simulations to run. Default is 10000.
        - p: The number of lag return terms to include in the GARCH model (if vol="garch"). Default is 1.
        - q: The number of lag variance terms to include in the GARCH model (if vol="garch"). Default is 1.
        - lambda_ewma: The decay factor for the EWMA model (if vol="ewma"). Default is 0.94.
        - ma_window: The window size to use for calculating moving averages. Default is 15.
        - return_mc_returns: if to return the Monte Carlo simulated returns. Default is False.

        Returns
        -------
        - dict: A dictionary containing the following keys: 
            - 'var': The Value at Risk (VaR) estimated from the Monte Carlo simulations.
            - 'es': The Expected Shortfall (ES) estimated from the Monte Carlo simulations.
            - 'T': The investment horizon over which the VaR and ES are calculated.
            - 'mu': The mean of returns used in the simulations,, in % terms.
            - 'cum_sigma': The cumulative standard deviation forecasted by the EWMA model, in % terms.

        Notes
        ------   
        Additional keys depending on the type of volatility used:
            - 'lambda_ewma' (if vol="ewma"): The decay factor used in the EWMA model.
            - 'p' and 'q' (if vol="garch"): The order parameters of the GARCH model.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import warnings ; import numpy as np ; import pandas as pd ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_vol,check_mu,check_number_simulations, check_return_mc_returns)
        
        # check procedure
        vol = check_vol(vol) ; mu = check_mu(mu) ;  num_sims = check_number_simulations(num_sims) ; check_return_mc_returns(return_mc_returns)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # calculating the mean of the distribution
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # checking if interval is set to 1
        if self.interval == 1:

            # setting mean to zero if mu is specified as "zero"
            if mu == "zero":
                mean = 0

            # calculating constant mean from returns if mu is specified as "constant"
            if mu == "constant":
                mean = np.mean(self.returns)

            # calculating moving average if mu is specified as "moving_average"
            if mu == "moving_average":
                from pyriskmgmt.inputsControlFunctions import check_ma_window

                # validating the moving average window
                ma_window = check_ma_window(ma_window)

                # ensuring the moving average window is not greater than the returns length
                if ma_window > len(self.returns):
                    raise ValueError("The ma_window parameter is greater than the number of returns, it is impossible to use the MA approach")
                returns = pd.Series(self.returns)
                sma_ma_window = returns.rolling(window=ma_window).mean()
                mean = sma_ma_window.tail(1).values[0]

        # checking if interval is greater than 1
        if self.interval > 1:

            # trimming returns for reshaping
            num_periods = len(self.returns) // self.interval * self.interval
            trimmed_returns = self.returns[ len(self.returns) - num_periods : ]

            # reshaping the returns for aggregation
            reshaped_returns = trimmed_returns.reshape((-1, self.interval))

            # compounding the returns based on interval
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1

            # setting mean to zero if mu is "zero"
            if mu == "zero":
                mean = 0

            # calculating constant mean from aggregated returns if mu is "constant"
            if mu == "constant":
                mean = np.mean(aggregated_returns)

            # calculating moving average from aggregated returns if mu is "moving_average"
            if mu == "moving_average":
                from pyriskmgmt.inputsControlFunctions import check_ma_window

                # validating the moving average window
                ma_window = check_ma_window(ma_window)

                # ensuring the moving average window is not too large
                if ma_window > len(aggregated_returns):
                    raise ValueError("The ma_window parameter is greater than the number of aggregated_returns, please increase the number of returns or decrease the ma_window parameter ")
                returns = pd.Series(aggregated_returns)
                sma_ma_window = returns.rolling(window=ma_window).mean()
                mean = sma_ma_window.tail(1).values[0]

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # computing the simple historical volatility
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # checking if volatility estimation method is "simple"
        if vol == "simple":

            # calculating volatility for interval of 1
            if self.interval == 1:
                sigma = np.std(self.returns)

            # adjusting volatility for intervals greater than 1
            elif self.interval > 1:
                sigma = np.std(self.returns) * np.sqrt(self.interval)

            # performing Monte Carlo simulations to generate returns
            montecarlo_returns = np.random.normal(loc=mean, scale=sigma, size=num_sims)

            # determining Value at Risk (VaR) and Expected Shortfall (ES) based on position
            if self.position > 0:
                quantile_alpha = np.quantile(montecarlo_returns, self.alpha)
                VaR = self.position * - quantile_alpha 
                ES = np.mean(montecarlo_returns[montecarlo_returns < quantile_alpha]) * - self.position
            elif self.position < 0:
                quantile_alpha = np.quantile(montecarlo_returns, 1 - self.alpha)
                VaR = - self.position * quantile_alpha 
                ES = np.mean(montecarlo_returns[montecarlo_returns > quantile_alpha]) * - self.position
            elif self.position == 0:
                VaR = 0 ; ES = 0

            # packaging results into a dictionary
            MC_DIST = {"var" : round(VaR, 4),
                        "es" : round(ES, 4),
                        "T" : self.interval,
                        "mu" : round(mean,4),
                        "cum_sigma" : round(sigma,4)}
           
        # calculating Garch(p,q) volatility
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # checking if the selected volatility method is GARCH
        if vol == "garch":

            from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

            # validating the values of p and q for GARCH model
            p = check_p(p) ; q = check_q(q)

            # importing necessary libraries for GARCH model
            import os ; import sys ; from arch import arch_model

            # initializing the GARCH model with the given parameters
            model = arch_model(self.returns, vol="GARCH", p=p, q=q, power=2.0, dist="normal") 

            # suppressing stderr output
            stderr = sys.stderr 
            sys.stderr = open(os.devnull, 'w')

            # fitting the GARCH model and handling potential errors
            try:
                model_fit = model.fit(disp='off')
            except Exception as e:
                print("Garch fitting did not converge:", str(e))
                return 
            finally:
                # restoring stderr output
                sys.stderr = stderr

            # setting the forecast horizon to 1
            horizon = 1

            # forecasting using the fitted GARCH model
            forecasts = model_fit.forecast(start=0, horizon=horizon)

            # extracting the variance forecast from the results
            variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

            # calculating sigma based on the interval and variance forecast
            if self.interval == 1:
                sigma = np.sqrt(variance_forecasts)
            elif self.interval > 1:
                cumulative_variance = variance_forecasts * self.interval # assuming future returns follow normal distribution
                sigma = np.sqrt(cumulative_variance)

            # generating Monte Carlo simulations using the estimated mean and sigma
            montecarlo_returns = np.random.normal(loc=mean, scale=sigma, size=num_sims)

            # determining Value at Risk (VaR) and Expected Shortfall (ES) based on the position
            if self.position > 0:
                quantile_alpha = np.quantile(montecarlo_returns, self.alpha)
                VaR = self.position * - quantile_alpha 
                ES = np.mean(montecarlo_returns[montecarlo_returns < quantile_alpha]) * - self.position
            elif self.position < 0:
                quantile_alpha = np.quantile(montecarlo_returns, 1 - self.alpha)
                VaR = - self.position * quantile_alpha 
                ES = np.mean(montecarlo_returns[montecarlo_returns > quantile_alpha]) * - self.position
            elif self.position == 0:
                VaR = 0 ; ES = 0

            # storing the results in a dictionary
            MC_DIST = {"var" : round(VaR, 4),
                       "es" : round(ES, 4),
                       "T" : self.interval,
                       "q" : q,
                       "p" : p,
                       "mu" : round(mean,4),
                       "cum_sigma" : round(sigma,4)}

        # computing EWMA (lambda = lambda_ewma) volatility
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # checking if the selected volatility method is EWMA
        if vol == "ewma":

            from pyriskmgmt.inputsControlFunctions import check_lambda_ewma

            # validating the lambda parameter for EWMA
            lambda_ewma = check_lambda_ewma(lambda_ewma)

            # initializing variance time series based on returns length
            n = len(self.returns)
            variance_time_series = np.repeat(self.returns[0]**2, n)

            # calculating variance using EWMA formula
            for t in range(1,n):
                variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * self.returns[t-1]**2

            # estimating next variance value
            variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * self.returns[-1]**2

            # determining sigma based on the interval and variance
            if self.interval == 1:
                sigma = np.sqrt(variance_t_plus1)
            if self.interval > 1:
                sigma = np.sqrt(variance_t_plus1) * np.sqrt(self.interval)

            # generating Monte Carlo simulations using the estimated mean and sigma
            montecarlo_returns = np.random.normal(loc=mean, scale=sigma, size=num_sims)

            # determining Value at Risk (VaR) and Expected Shortfall (ES) based on the position
            if self.position > 0:
                quantile_alpha = np.quantile(montecarlo_returns, self.alpha)
                VaR = self.position * - quantile_alpha 
                ES = np.mean(montecarlo_returns[montecarlo_returns < quantile_alpha]) * - self.position
            elif self.position < 0:
                quantile_alpha = np.quantile(montecarlo_returns, 1 - self.alpha)
                VaR = - self.position * quantile_alpha 
                ES = np.mean(montecarlo_returns[montecarlo_returns > quantile_alpha]) * - self.position
            elif self.position == 0:
                VaR = 0 ; ES = 0

            # storing the results in a dictionary
            MC_DIST = {"var" : round(VaR, 4),
                       "es" : round(ES, 4),
                       "T" : self.interval,
                       "lambda_ewma" : lambda_ewma,
                       "mu" : round(mean, 4),
                       "cum_sigma" : round(sigma, 4)}

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # deciding what to return based on the value of return_mc_returns
        if return_mc_returns == False:
            return MC_DIST
        else: 
            return MC_DIST, montecarlo_returns

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # MONTECARLO MODEL [STOCK PRICE - GEOMETRIC BROWNIAM MOTION] <<<<<<<<<<<<
    
    def mcModelGbm(self, S0: float = 100.00, vol: str = "simple", mu: str = "moving_average", num_sims: int = 10000,
                              p: int = 1, q: int = 1, lambda_ewma: float = 0.94,
                              ma_window: int = 15, return_gbm_path: bool = False):

        """
        Main
        ----- 
        Runs Monte Carlo simulations of future equity prices using a Geometric Brownian Motion (GBM) model. 
        The model parameters are estimated based on different types of volatilities: 
        simple historical volatility, GARCH model volatility, or Exponential Weighted Moving Average (EWMA) volatility.
        It uses the simulations to estimate Value at Risk (VaR) and Expected Shortfall (ES). 

        Parameters
        ----------
        - S0: The initial equity price - the defalut value is 100. It can be left to 100 since the method will ultimately retrive returns.
        - vol: Specifies the type of volatility to use in the simulations. Options are "simple", "garch", "ewma". Default is "simple".
        - mu: Specifies the type of mean to use in the simulations. Options are "zero", "constant", "moving_average". Default is "moving_average".
        - num_sims: The number of Monte Carlo simulations to run. Default is 10000.
        - p: The number of lag return terms to include in the GARCH model (if vol="garch"). Default is 1.
        - q: The number of lag variance terms to include in the GARCH model (if vol="garch"). Default is 1.
        - lambda_ewma: The decay factor for the EWMA model (if vol="ewma"). Default is 0.94.
        - ma_window: The window size to use for calculating moving averages. Default is 15.
        - return_gbm_path: if to return the GBM path. Default is set to False.

        Returns
        -------
        - dict: A dictionary containing the following keys: 
            - 'var': The Value at Risk (VaR) estimated from the Monte Carlo simulations.
            - 'es': The Expected Shortfall (ES) estimated from the Monte Carlo simulations.
            - 'T': The investment horizon over which the VaR and ES are calculated.
            - 'GbmDrift': The drift of returns used in the simulations - always one perdiod drift
            - 'GbmSigma': The standard deviation of returns used in the simulations - - always one perdiod sigma
        
        Notes
        ------
        Additional keys depending on the type of volatility used:
            - 'lambda_ewma' (if vol="ewma"): The decay factor used in the EWMA model.
            - 'p' and 'q' (if vol="garch"): The order parameters of the GARCH model.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_S0, check_return_gbm_path)
        
        # check procedure
        S0 = check_S0(S0) ; check_return_gbm_path(return_gbm_path)

        # here the mc_model_draw_returns is using to retrieve the drift and sigma from the output dictonary 
        # interval is set to 1 ---- the frequency of the GBM simulation is always 1... then the gbm path is calculated based on self.interval

        from pyriskmgmt.equity_models import EquityPrmSingle 

        # initializing an EquityPrmSingle model
        model = EquityPrmSingle(returns=self.returns, position=self.position, interval=1, alpha=self.alpha) # always using an interval of 1

        # obtaining Monte Carlo model results
        result = model.mcModelDrawReturns(vol=vol, mu=mu, num_sims=num_sims,
                                          p=p, q=q, lambda_ewma=lambda_ewma, ma_window=ma_window)

        # extracting the drift and volatility values from the results
        drift = result["mu"] # always 1 period drift 
        sigma = result["cum_sigma"] # always 1 period sigma 

        # starting the Geometric Brownian Motion (GBM) simulation
        # --------------------------------------------------------------------------------------------------------

        # defining parameters and initial settings
        dt = 1  # always using a time increment of 1

        # simulating GBM paths based on the defined interval
        if self.interval == 1:
            W = np.random.standard_normal(num_sims)
            S = S0 + ((drift * S0 * dt) + (sigma * S0 * np.sqrt(dt) * W))
        elif self.interval > 1:
            W = np.random.standard_normal((num_sims, self.interval))
            GBM_MATRIX = np.empty((num_sims, self.interval + 1))
            GBM_MATRIX[:, 0] = S0
            for t in range(1, self.interval + 1):
                GBM_MATRIX[:, t] = GBM_MATRIX[:, t - 1] * (1 + drift * dt + sigma * W[:, t - 1] * np.sqrt(dt))
            S = GBM_MATRIX[:, -1]

        # computing returns from the GBM paths
        S0 = np.full((num_sims, 1), S0)
        gmb_rets = (S / S0) - 1

        # determining Value at Risk (VaR) and Expected Shortfall (ES) based on the position
        if self.position > 0:
            quantile_alpha = np.quantile(gmb_rets, self.alpha)
            VaR = self.position * - quantile_alpha 
            ES = np.mean(gmb_rets[gmb_rets < quantile_alpha]) * - self.position
        elif self.position < 0:
            quantile_alpha = np.quantile(gmb_rets, 1 - self.alpha)
            VaR = - self.position * quantile_alpha 
            ES = np.mean(gmb_rets[gmb_rets > quantile_alpha]) * - self.position
        elif self.position == 0:
            VaR = 0

        # storing the results in a dictionary
        MC_GBM = {"var": round(VaR, 4),
                  "es" : round(ES, 4),
                  "lambda_ewma" : lambda_ewma,
                  "T" : self.interval,
                  "GbmDrift" : round(drift, 4),
                  "GbmSigma" : round(sigma, 4)}

        # deciding what to return based on the value of return_gbm_path
        if return_gbm_path == False:
            return MC_GBM
        else:
            return MC_GBM, S

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_stocks: Union[int, float], S0_initial: float, test_returns: np.ndarray, var: float, alpha_test: float = 0.05):

        """
        Main
        ----
        Performs the Kupiec test of unconditional coverage to verify the accuracy of the VaR model
        The Kupiec test compares the number of exceptions (instances where the losses exceed the VaR estimate)
        with the expected number of exceptions. Under the null hypothesis, the number of exceptions is consistent
        with the VaR confidence level.

        Parameters
        ----------
        - num_stocks: A float or int with the total number of stocks held (it can be positive for long positions or negative for short ones)
        - S0_initial: Initial price of the stock. This must be the price of the stock at the beginning of the 'test_returns' array.
        - test_returns : The returns to use for testing. This array should have the same frequency than the returns array.
        - var : The Value-at-Risk level to be tested (in value).
        - alpha_test: The significance level of the test. Default is 0.05.

        Returns
        -------
        - dict: A dictionary containing:
            - 'test': The name of the test ('Kupiec').
            - 'var': The VaR level that was tested.
            - 'nObs': The total number of observations.
            - 'nExc': The number of exceptions observed.
            - 'alpha': The theoretical percentage of exceptions.
            - 'actualExc%': The actual percentage of exceptions.
            - 'LR': The likelihood ratio of the test.
            - 'cVal': The critical value of the test at the given significance level.
            - 'rejNull': A boolean indicating whether to reject the null hypothesis (true if the null hypothesis is rejected).
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        from pyriskmgmt.inputsControlFunctions import (check_num_stocks, check_S0_initial, validate_returns_single,check_var, check_alpha_test)

        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_stocks = check_num_stocks(num_stocks); S0_initial = check_S0_initial(S0_initial)
        test_returns = validate_returns_single(test_returns) ; var = check_var(var) ; alpha_test = check_alpha_test(alpha_test)

        if self.interval == 1: ############
            test_returns = test_returns

        elif self.interval > 1: ###########

            # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

            def reshape_and_aggregate_returns(returns, interval):
                num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
                reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                return aggregated_returns
            
            test_returns = reshape_and_aggregate_returns(test_returns, self.interval)

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        # constructing the path of the stock price, starting from S0_initial
        stockPricePath_without_initial = S0_initial * np.cumprod(1 + test_returns) # test_returns here follows the self.interval
        
        # inserting the initial stock price to the beginning of the path
        stockPricePath = np.insert(stockPricePath_without_initial, 0, S0_initial)

        # computing the absolute profit or loss by finding the difference between consecutive stock prices, 
        # and then multiplying by the number of stocks
        PL = np.diff(stockPricePath) * num_stocks # monetary PL

        # checking if the trading position is positive
        if self.position > 0: 
            # extracting negative values (losses) from the PL array
            losses = PL[PL < 0]
            # counting the number of times the losses exceed the provided VaR value
            num_exceptions = np.sum(losses < - VaR) 

        # checking if the trading position is negative
        elif self.position < 0: 
            # extracting positive values (gains, in case of short positions) from the PL array
            gains = PL[PL > 0] 
            # counting the number of times the losses (due to price increase in short positions) exceed the provided VaR value
            num_exceptions = np.sum(gains > VaR)

        # handling scenarios where the trading position is neutral (neither bought nor sold)
        else:
            raise ValueError ("Position is set to 0: impossible to perform the Kupiec unconditional coverage test")

        # getting the total number of observations in the test_returns array
        num_obs = len(PL)

        # % exceptions
        per_exceptions = round(num_exceptions/num_obs,4)

        p = self.alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : self.alpha,                   # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # CHRISTOFFESEN TEST <<<<<<<<<<<<

    def christoffersen_test(self, num_stocks: Union[int, float], S0_initial: float, test_returns: np.ndarray = None, var: float = None, alpha_test: float = 0.05):

        """
        Main
        -----
        Performs the Christoffersen test of independence to verify the accuracy of the VaR model.
        The Christoffersen test assesses the independence of exceptions. Under the null hypothesis, exceptions
        (instances where the losses exceed the VaR estimate) are independently distributed over time.

        Parameters
        ----------
        - num_stocks: A float or int with the total number of stocks held (it can be positive for long positions or negative for short ones)
        - S0_initial: Initial price of the stock. This must be the price of the stock at the beginning of the 'test_returns' array.
        - test_returns : The returns to use for testing. This array should have the same frequency than the returns array.
        - var: The Value-at-Risk level to be tested (in value).
        - alpha_test: The significance level of the test. Default is 0.05.

        Returns
        --------
        - dict: A dictionary containing:
            - 'test': The name of the test ('Christoffersen').
            - 'var': The VaR level that was tested.
            - 'LR': The likelihood ratio of the test.
            - 'cVal': The critical value of the test at the given significance level.
            - 'rejNull': A boolean indicating whether to reject the null hypothesis (true if the null hypothesis is rejected).
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        from pyriskmgmt.inputsControlFunctions import (check_num_stocks, check_S0_initial, validate_returns_single,check_var, check_alpha_test)

        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_stocks = check_num_stocks(num_stocks); S0_initial = check_S0_initial(S0_initial)
        test_returns = validate_returns_single(test_returns) ; var = check_var(var) ; alpha_test = check_alpha_test(alpha_test)

        if self.interval == 1: #############
            test_returns = test_returns

        elif self.interval > 1: ###########

            # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

            def reshape_and_aggregate_returns(returns, interval):
                num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
                reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                return aggregated_returns
            
            test_returns = reshape_and_aggregate_returns(test_returns, self.interval)

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)
 
        # constructing the path of the stock price, starting from S0_initial
        stockPricePath_without_initial = S0_initial * np.cumprod(1 + test_returns) # test_returns here follows the self.interval
        
        # inserting the initial stock price to the beginning of the path
        stockPricePath = np.insert(stockPricePath_without_initial, 0, S0_initial)

        # computing the absolute profit or loss by finding the difference between consecutive stock prices, 
        # and then multiplying by the number of stocks
        PL = np.diff(stockPricePath) * num_stocks # monetary PL

        # checking if the position is positive
        if self.position > 0: 
            exceptions = PL < - VaR

        # checking if the position is negative
        elif self.position < 0:
            exceptions = PL > VaR

        # handling the case where the position is neither positive nor negative
        else:
            raise ValueError ("Position is set to 0: impossible to perform Christoffersen test of independence")

        # shifting exceptions to get previous day's exceptions
        prev_exceptions = np.roll(exceptions, shift=1)

        # dropping the first row (which is NaN because of the shift)
        exceptions = exceptions[1:] ; prev_exceptions = prev_exceptions[1:]

        # calculating transition counts
        T_00, T_01 = np.sum((prev_exceptions == 0) & (exceptions == 0)), np.sum((prev_exceptions == 0) & (exceptions == 1))
        T_10, T_11 = np.sum((prev_exceptions == 1) & (exceptions == 0)), np.sum((prev_exceptions == 1) & (exceptions == 1))

        # estimating conditional probabilities
        p_b0, p_b1 = T_01 / (T_00 + T_01), T_11 / (T_10 + T_11)

        # calculating likelihood ratio
        p = self.alpha
        likelihood_ratio = -2 * (np.log((1 - p) ** (T_00 + T_10) * p ** (T_01 + T_11)) - np.log((1 - p_b0) ** T_00 * p_b0 ** T_01 * (1 - p_b1) ** T_10 * p_b1 ** T_11))

        # critical value from chi-squared distribution
        critical_value = chi2.ppf(1 - alpha_test, df=1)

        # reject the null hypothesis if likelihood ratio is higher than the critical value
        reject_null = likelihood_ratio > critical_value

        return {"test" : "Christoffersen",
                "var" : var,
                "LR": round(likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # COMBINED TEST <<<<<<<<<<<<

    def combined_test(self, num_stocks: Union[int, float], S0_initial: float, test_returns: np.ndarray = None, var: float = None, alpha_test: float = 0.05):

        """
        Main
        -----
        Performs a combined Kupiec and Christoffersen test to assess the accuracy of the VaR model.
        The combined test applies both the Kupiec test of unconditional coverage and the Christoffersen test of independence to the exceptions. 
        The null hypothesis is that the VaR model is accurate.

        Parameters
        ----------
        - num_stocks: A float or int with the total number of stocks held (it can be positive for long positions or negative for short ones)
        - S0_initial: Initial price of the stock. This must be the price of the stock at the beginning of the 'test_returns' array.
        - test_returns: The returns to use for testing. This array should have the same frequency than the returns array.
        - var: The Value-at-Risk level to be tested (in value).
        - alpha_test: The significance level of the test. Default is 0.05.

        Returns
        -------
        - dict: A dictionary containing:
            - 'test': The name of the test ('Kupiec_Christoffersen').
            - 'var': The VaR level that was tested.
            - 'LR': The combined likelihood ratio of the test.
            - 'cVal': The critical value of the test at the given significance level.
            - 'rejNull': A boolean indicating whether to reject the null hypothesis (true if the null hypothesis is rejected).
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import chi2

        # now we can proceed ---------------------------------------------------------------------------------------

        VaR = round(var,4)

        # performing individual tests - no need for retuns aggregation here ---- it will be handled by both kupiec_test and christoffersen_test

        kupiec_results = self.kupiec_test(num_stocks = num_stocks, S0_initial= S0_initial, test_returns = test_returns, 
                                          var = VaR,alpha_test=alpha_test)
        christoffersen_results = self.christoffersen_test(num_stocks = num_stocks, S0_initial= S0_initial, 
                                                          test_returns = test_returns, var = VaR,alpha_test=alpha_test)

        # getting likelihood ratios
        kupiec_lr = kupiec_results["LR"] ; christoffersen_lr = christoffersen_results["LR"]

        # calculating combined likelihood ratio
        combined_likelihood_ratio = kupiec_lr + christoffersen_lr

        # critical value from chi-squared distribution with 2 degrees of freedom
        critical_value = chi2.ppf(1 - alpha_test, df=2)

        # if the combined likelihood ratio is greater than the critical value, then we reject the null hypothesis
        reject_null = combined_likelihood_ratio > critical_value

        return {"test" : "Kupiec_Christoffersen",
                "var" : VaR,
                "LR": round(combined_likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}

##########################################################################################################################################################################
##########################################################################################################################################################################
## 4. EquityPrmPort ######################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EquityPrmPort:

    """
    Main
    ----
    The EquityPrmPort class serves as a specialized toolkit designed for undertaking risk management and assessment tasks across a portfolio comprising multiple assets.
    This class enables users to calculate and forecast Value at Risk (VaR) and Expected Shortfall (ES) using various modeling approaches,
    which include advanced multivariate methods like DCC GARCH, EWMA, and Monte Carlo simulations.

    Initial Parameters
    ------------------
    - returns: An array containing the historical returns data for each asset (same frequency).
    - positions : A list or array of current positions for each asset.
    - interval: The interval over which the risk is assessed (in days, assuming 252 traing days in a year) if the frequency of the returns is
    daily, otherwise it follows the time-aggregation of the returns. Default is 1.
      - For example, if you provide weekly returns and the interval is set to 5, the method will calculate the VaR and Es for a period of 5 weeks 
      ahead.
    - alpha: The significance level for VaR and ES calculations. Default is 0.05.

    Methods
    -------
    - garch_model(): Utilizes a GARCH model for VaR and ES calculation.
    - ewma_model(): Implements an EWMA model to calculate VaR and ES.
    - ff3(): Retrieves a dataframe with the Fama French 3 factors (market, size, value) over a specified date range and frequency.
    - sharpeDiag(): Implements Sharpe's diagonal model using the given set of returns.
    - mcModelDrawPortRets(): Carries out a Monte Carlo simulation to generate new returns for estimating VaR and ES for a portfolio.
    - mcModelRetsSimu(): Conducts a Monte Carlo simulation and performs Cholesky decomposition to estimate VaR and ES for a portfolio.
    - MargVars(): Calculates the marginal VaR for each asset in the portfolio.
    - RelCompVars(): Calculates the relative component VaR for each asset in the portfolio.
    - CompVars(): Determines the component VaR for each asset in the portfolio, indicating the contribution of each asset to the total portfolio VaR.
    - VarEsMinimizer(): Optimizes portfolio positions to minimize VaR and ES.
    - kupiec_test(): Executes the Kupiec test (unconditional coverage test) for the VaR model.
    - christoffersen_test(): Conducts the Christoffersen test (conditional coverage test) for the VaR model.
    - combined_test(): Carries out both Kupiec and Christoffersen tests on the VaR model, yielding combined results that follow a chi-square distribution with two degrees of freedom.

    Attributes
    ----------
    - simple_var: The calculated Value at Risk (VaR) for the entire portfolio based on its assets' historical volatilities. None until the `fit` method is called.
    - simple_es: The calculated Expected Shortfall (ES) for the entire portfolio based on its assets' historical volatilities. None until the `fit` method is called.
    - simple_sigmaPi: The standard deviation of the portfolio returns based on historical returns of the assets, scaled by the square root of the interval.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, returns: np.ndarray,
                positions: List[Union[int, float]],
                interval: int = 1,
                alpha: Union[int, float] = 0.05):
        
        from pyriskmgmt.inputsControlFunctions import (validate_returns_port, check_positions_port, check_interval, check_alpha)
        
        # check procedure
        self.returns = validate_returns_port(returns) ; self.positions = check_positions_port(positions)
            
        if len(self.positions) != self.returns.shape[1]:
            raise ValueError("The number of assets != the number of positions")

        self.alpha = check_alpha(alpha) ;  self.interval = check_interval(interval) ; self.alpha = check_alpha(alpha)

        self.simple_var = None ; self.simple_es = None ; self.simple_sigmaPi = None ; self.totPortPos = None 

        self.garch_model_var = None ; self.ewma_model_var = None # for later usage

        self.fit()  

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import norm ; import numpy as np ; import pandas as pd

        # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

        def reshape_and_aggregate_returns(returns, interval):
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        # -----------------------------------------------------------------------------------------------------------------------------------------

        if self.interval > 1:

            # reshape_and_aggregate_returns
            def reshape_and_aggregate_returns(returns, interval):
                returns = returns.values ; num_periods = len(returns) // interval * interval 
                trimmed_returns = returns[len(returns) - num_periods:] ; reshaped_returns = trimmed_returns.reshape((-1, interval)) 
                aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                return aggregated_returns
            
            # converting self.returns back to a DataFrame to avoid a for loop
            rets_df = pd.DataFrame(self.returns)
            # applying the reshape_and_aggregate_returns function to each column
            returns = rets_df.apply(reshape_and_aggregate_returns, args=(self.interval,)).values 

        else:
            returns = self.returns

        # computing the variance-covariance matrix for the returns
        variance_cov_matrix = np.cov(returns, rowvar=False)

        # calculating Value at Risk using the formula
        var_port = np.dot(np.dot(self.positions,variance_cov_matrix),self.positions.T) 
        sigma = np.sqrt(var_port) 
        q = norm.ppf(1-self.alpha, loc=0, scale=1) 
        VaR = q * sigma 

        # ensuring VaR is non-negative
        self.simple_var =  max(0,VaR)

        # calculating Expected Shortfall using the formula
        q_es = ( ( np.exp( - (q**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-self.alpha)))) 
        ES = q_es * sigma 

        # ensuring ES is non-negative
        self.simple_es = max(0,ES)

        # rounding off the calculated VaR for better readability
        self.simple_var = round(self.simple_var, 4)

        # rounding off the calculated ES for better readability
        self.simple_es = round(self.simple_es, 4)

        # calculating portfolio sigma
        simple_sigmaPi = sigma 
        self.simple_sigmaPi = round(simple_sigmaPi, 5)

        # calculating total portfolio position
        self.totPortPos = round(np.sum(self.positions),5)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # GARCH MODEL (p,q) <<<<<<<<<<<<<<<

    def garch_model(self, p: int = 1, q: int = 1):

        """
        Main
        ----
        The 'garch_model' function melds Python and R for efficiency, particularly due to the DCC GARCH models in R.
        If these are not set up, you can download and install R, and then run 'install.packages("rmgarch")' within your R environment.
        This method employs a DCC GARCH (Dynamic Conditional Correlation Generalized Autoregressive Conditional Heteroskedasticity) model to determine the Value at Risk (VaR),
        Expected Shortfall (ES), and the standard deviation of a portfolio. 
        The DCC GARCH model is used for modeling the variance of portfolio returns, taking into account the dynamic volatility that is characteristic of financial time series.
        The execution of this method requires an active R environment with the 'rmgarch' package installed, which is used to perform the GARCH model calculations.

        Parameters
        ----------
        - p: Specifies the order of the GARCH model. Defaults is 1.
        - q: Defines the order of the moving average term in the GARCH model. Defaults is 1.

        Returns
        -------
        - dict: A dictionary encompassing the calculated VaR, ES, portfolio standard deviation (sigmaPi), p, q, and the time interval (T).
        
        Notes
        -----
        - The VaR and ES computations are predicated on a normal distribution assumption.
        - The variance covariance matrix of the portfolio is calculated using the 'rmgarch' package in R.
        - Portfolio positions are taken into account for the calculation of portfolio variance.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np
        
        from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

        # check procedure
        p = check_p(p) ; q = check_q(q)

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing the supporting function `dccCondCovGarch` from the `SupportFunctions` module
        from pyriskmgmt.SupportFunctions import dccCondCovGarch

        # computing the variance-covariance matrix using the DCC-GARCH model
        variance_cov_matrix = dccCondCovGarch(returns=self.returns, interval=self.interval, p=p, q=q)[0]

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing required statistical tools from scipy
        from scipy.stats import norm

        # calculating Value at Risk (VaR) for the portfolio
        # ------------------------------------------------------------------------------

        # computing portfolio's variance and then its volatility (sigma)
        var_port = np.dot(np.dot(self.positions, variance_cov_matrix), self.positions.T)
        sigma = np.sqrt(var_port)

        # determining the z-score corresponding to the given confidence level `alpha`
        quantile = norm.ppf(1-self.alpha, loc=0, scale=1)

        # calculating VaR and ensuring its non-negativity
        VaR = quantile * sigma
        VaR = max(0, VaR)

        # calculating Expected Shortfall (ES) for the portfolio
        # ------------------------------------------------------------------------------

        # computing ES using the quantile and ensuring its non-negativity
        q_es = (np.exp(- (quantile**2) / 2)) / ((np.sqrt(2 * np.pi)) * (1 - (1-self.alpha)))
        ES = q_es * sigma
        ES = max(0, ES)

        # storing the VaR value computed using the GARCH model in the class attribute
        self.garch_model_var = VaR

        # preparing the results dictionary
        GARCH_MODEL = {"var" : round(VaR, 4),
                       "es:" : round(ES, 4),
                       "T" : self.interval,
                       "p" : p,
                       "q" : q,
                       "sigmaPi" : round(sigma, 4)}

        return GARCH_MODEL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # EWMA MODEL <<<<<<<<<<<<<<<

    def ewma_model(self, lambda_ewma: float = 0.94):

        """
        Main
        ----
        This method applies the Exponential Weighted Moving Average (EWMA) model to compute the Value at Risk (VaR) and 
        Expected Shortfall (ES) for a portfolio of assets. It also computes the portfolio standard deviation.
        The EWMA model is used to calculate a variance-covariance matrix of portfolio returns, giving more weight to recent observations.
        This method can be particularly useful in financial markets that are experiencing significant changes or are highly volatile.

        Parameters
        ----------
        - lambda_ewma: The decay factor for the EWMA model, which determines how quickly the weights decrease for older observations. Defaults to 0.94.

        Returns
        --------
        - dict: A dictionary containing the computed VaR, ES, portfolio standard deviation (sigmaPi), lambda_ewma, and the time interval (T). 
        
        Notes
        ------
        - The final computation of VaR and ES are based on a normal distribution assumption.
        - The portfolio positions are used for the computation of the portfolio variance.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import check_lambda_ewma

        # check procedure
        lambda_ewma = check_lambda_ewma(lambda_ewma)

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing the supporting function `ewmaCondCov` from the `SupportFunctions` module
        from pyriskmgmt.SupportFunctions import ewmaCondCov

        # computing the variance-covariance matrix using the EWMA approach
        variance_cov_matrix = ewmaCondCov(returns=self.returns, interval=self.interval, lambda_ewma=lambda_ewma)

        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # importing required statistical tools from scipy
        from scipy.stats import norm

        # calculating Value at Risk (VaR) for the portfolio
        # ------------------------------------------------------------------------------

        # computing portfolio's variance and then its volatility (sigma)
        var_port = np.dot(np.dot(self.positions, variance_cov_matrix), self.positions.T)
        sigma = np.sqrt(var_port)

        # determining the z-score corresponding to the given confidence level `alpha`
        quantile = norm.ppf(1-self.alpha, loc=0, scale=1)

        # calculating VaR and ensuring its non-negativity
        VaR = quantile * sigma
        VaR = max(0, VaR)

        # calculating Expected Shortfall (ES) for the portfolio
        # ------------------------------------------------------------------------------

        # computing ES using the quantile and ensuring its non-negativity
        q_es = (np.exp(- (quantile**2) / 2)) / ((np.sqrt(2 * np.pi)) * (1 - (1-self.alpha)))
        ES = q_es * sigma
        ES = max(0, ES)

        # storing the VaR value computed using the EWMA model in the class attribute
        self.ewma_model_var = VaR

        # preparing the results dictionary
        EWMA_MODEL = { "var" : round(VaR, 4),
                       "es" : round(ES, 4),
                       "T" : self.interval,
                       "lamda_ewma" : lambda_ewma,
                       "sigmaPi" : round(sigma, 5)}

        return EWMA_MODEL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************  
     
    # [FAMA FRENCH 3] MKT, HML and SMB <<<<<<<<<<<<<<<<<< from 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html'

    def ff3(self, start_date: str, end_date: str, freq: str = "daily"):

        """
        Fama-French Three-Factor Model Data Retrieval
        ---------------------------------------------
        Fetches data for the Fama-French three-factor (FF3) model which includes: 
        - Market return (MKT)
        - Size (SMB: Small Minus Big)
        - Value (HML: High Minus Low)

        The data is sourced from Ken French's Data Library.

        Parameters
        ----------
        - start_date: The start date for the data in 'YYYY-MM-DD' format.
        - end_date: The end date for the data in 'YYYY-MM-DD' format.
        - freq: Frequency of the data. Can be "daily", "monthly", etc. Default is "daily".

        Returns
        -------
        - DataFrame: A pandas DataFrame containing the FF3 data for the specified date range and frequency.
        """

        import warnings ; warnings.filterwarnings("ignore")

        # all the inputs will be controlled in the ff3Downloader function from pyriskmgmt.SupportFunctions

        from pyriskmgmt.SupportFunctions import ff3Downloader

        df = ff3Downloader(start_date, end_date, freq)

        return df

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************   

    # SHARPE DIAGONAL APPROACH <<<<<<<<<<<<<<<<<<

    def sharpeDiag(self, mappingRets: np.ndarray = None, vol: str = "simple", p: int = 1, q: int = 1,
                         lambda_ewma: float = 0.94, warning: bool = True):
        
        """
        Main
        -----
        Implements Sharpe's diagonal model, a prominent strategy for asset allocation, using the provided returns. 
        This method calculates factor exposures (betas), which may be conditional for GARCH and EWMA models, 
        and uses them to transform asset positions directly into factor positions. 
        Unlike mcModelDrawPortRets(), this does not estimate the variance-covariance matrix derived from the mapping procedure. (idiosyncratic variances are ignored here)

        Parameters
        ----------
        - mappingRets: The return series of a single or multiple factors (no limits). If not provided, an exception will be raised.
        Make sure that "mappingRets" is consistent with the frequency of the "returns" array and alligned in time-wise manner.
        - vol: Specifies the volatility model to be used. Options include "simple", "garch", or "ewma". Default is "simple".
        - p: Specifies the order of the GARCH model. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model. Default is 1.
        - lambda_ewma: Specifies the decay factor for the EWMA model, if used. Determines how quickly the weights decrease for older observations. Default is 0.94.
        - warning: A flag indicating whether a warning message should be printed. Default is True.

        Returns
        -------
        - SHRP_DIAG : dict: The output from either sharpeDiagSingle or sharpeDiagMulti based on the dimensionality of the input returns.
        
        Notes
        ------
        - If 'mappingRets' is a one-dimensional array or series, Sharpe's single-index model is used
        - If 'mappingRets' is a multi-dimensional array or dataframe, Sharpe's multi-index model is used.
        - It raises a Warning if the number of assetes are less than 30: 
            - This is significant because of the diversification principle: when a portfolio contains a large number of assets (e.g., more than 30),
            individual asset (idiosyncratic) risks tend to cancel each other out, leading to a reduction of the portfolio's overall risk. 
            This is often referred to as "non-systematic risk reduction".
            - However, in a portfolio with fewer assets, this idiosyncratic or asset-specific risk is not marginal and could influence the portfolio's performance significantly.
            As a result, risk assessment and allocation strategies such as the Sharpe's Diagonal Model may provide misleading results, 
            since these strategies often assume a certain level of diversification.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd ; import numpy as np

        # prelimiry check on mappingRets
        if mappingRets is None:
            raise Exception("Return series, array or df 'mappingRets' not provided.")
        
        if mappingRets is not None:
            if not isinstance(mappingRets,(np.ndarray, pd.Series, pd.DataFrame)):
                raise Exception("'mappingRets' must be provided in the form of: [np.ndarray, pd.Series, pd.DataFrame]")
            
        if warning:
            print("WARNING:")
            print("The frequency of self.returns and the mappingRets (if sharpeDiag) must correspond to the same one single period of analysis.")
            print("This method was written to aggregate both self.returns and the mapping factor/s returns (if selected) by itself if the interval is > 1.")
            print("The method ff3() supports [daily, weekly and monthly] frequency.")
            print("To mute this warning, add 'warning = False' as an argument of the sharpeDiag() function.")

            
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if len(mappingRets.shape) == 1: # mappingRets is a one dimensional array

            from pyriskmgmt.SupportFunctions import sharpeDiagSingle

            SHRP_DIAG = sharpeDiagSingle(self.returns, self.positions, self.interval, self.alpha,
                                      factor_rets = mappingRets, vol = vol, p = p, q = q, lambda_ewma = lambda_ewma, warning = warning)
            
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        else:                           # mappingRets is a multi dimensional array

            from pyriskmgmt.SupportFunctions import sharpeDiagMulti

            SHRP_DIAG = sharpeDiagMulti(self.returns, self.positions, self.interval, self.alpha,
                                      factors_rets = mappingRets, vol = vol, p = p, q = q, lambda_ewma = lambda_ewma, warning = warning)
            
        return SHRP_DIAG

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************    
        
    # MONTECARLO MODEL [RETURNS - DISTIBUTION] <<<<<<<<<<<< 
    
    def mcModelDrawPortRets(self, sharpeDiag: bool = False, mappingRets: np.ndarray = None, 
                                  vol: str = "simple", mu: str = "moving_average", num_sims: int = 10000,
                                  p: int = 1, q: int = 1, lambda_ewma: float = 0.94,
                                  ma_window: int = 10, warning: bool = True, VCV: bool = False):
        """
        Main
        ----
        Conducts Monte Carlo simulations for portfolio returns. If the 'moving_average' method is selected for expected return calculation, 
        it draws returns from a normal distribution based on that mean.
        The function supports several ways to estimate sigma: using the Sharpe Diagonal method (which estimates the variance-covariance matrix mapping 
        to either one or more factors) with volatility models "simple", "GARCH", or "EWMA". Alternatively, if 'sharpeDiag' is False, the estimation relies on the returns themselves.

        Parameters
        ----------
        - sharpeDiag: Determines whether Sharpe-style decomposition is performed. If False, the estimation relies on the returns themselves. Default is False.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on the same frequency than the "returns" array.
        Also align "mappingRets" with "returns" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility model to be used. Options include "simple", "garch", or "ewma". Default is "simple".
        - mu: Specifies the method to calculate expected return. Options include "moving_average", "zero", or "constant". Default is "moving_average".
        - num_sims: Specifies the number of Monte Carlo simulations to run. Default is 10000.
        - p: Specifies the order of the GARCH model. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model. Default is 1.
        - lambda_ewma: Specifies the decay factor for the EWMA model, if used. Determines how quickly the weights decrease for older observations. Default is 0.94.
        - ma_window: Defines the "moving_average" length. Default is 10.
        - warning: A flag indicating whether a warning message should be printed. Default is True.
        - VCV: If True, returns the Variance-Covariance Matrix (VCV). Default is False.

        Returns
        -------
        - MC_DIST: dict: A dictionary containing the portfolio risk metrics including Value at Risk (VaR), Expected Shortfall (ES), and others.

        Note
        ----
        - If 'mappingRets' is a one-dimensional array or series, the Sharpe's single-index model is used,
        and the function returns a dictionary with portfolio weights, variance, standard deviation, and other metrics.
        - If 'mappingRets' is a multi-dimensional array or dataframe, Sharpe's multi-index model is used, 
        and the function returns a DataFrame with weights for each asset in the portfolio and each factor.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; import pandas as pd

        from pyriskmgmt.inputsControlFunctions import (check_sharpeDiag, validate_returns_single, validate_returns_port ,check_vol, check_mu, check_number_simulations,
                                                       check_warning, check_VCV)
        
        # check procedure
        check_sharpeDiag(sharpeDiag); vol = check_vol(vol) ; mu = check_mu(mu) ; num_sims = check_number_simulations(num_sims) ; check_warning(warning) ; check_VCV(VCV)

        if warning:
            print("WARNING:")
            print("The frequency of self.returns and the mappingRets (if sharpeDiag) must correspond to the same one single period of analysis.")
            print("This method was written to aggregate both self.returns and the mapping factor/s returns (if selected) by itself if the interval is > 1.")
            print("The method ff3() supports [daily, weekly and monthly] frequency.")
            print("To mute this warning, add 'warning = False' as an argument of the mcModelRetsSimu() function.")


        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # checking if the user has requested the 'sharpeDiag' method but hasn't provided 'mappingRets'.
        if sharpeDiag and mappingRets is None:
            raise Exception("Error: 'mappingRets' is None and you've selected the 'sharpeDiag' method ...")
        
        # checking if the user has selected 'sharpeDiag' and provided 'mappingRets'.
        if sharpeDiag and mappingRets is not None:
                
                # validating the data type of 'mappingRets'.
                if not isinstance(mappingRets, (np.ndarray, pd.Series, pd.DataFrame)):
                    raise Exception("'mappingRets' must be provided in the form of: [np.ndarray, pd.Series, pd.DataFrame]")
                
                # handling the case where 'mappingRets' is a one-dimensional array.
                if len(mappingRets.shape) == 1:
                    mappingRets = validate_returns_single(mappingRets)
                    # checking if all columns in 'self.returns' are equal to 'mappingRets'.
                    if all(np.array_equal(self.returns[:,i], mappingRets) for i in range(self.returns.shape[1])):
                        raise ValueError("All columns in self.returns are equal to mappingRets...")

                # handling the case where 'mappingRets' is multi-dimensional.
                else:
                    mappingRets = validate_returns_port(mappingRets)
                    # looping through 'mappingRets' and checking against 'self.returns'.
                    for j in range(mappingRets.shape[1]):
                        if all(np.array_equal(self.returns[:,i], mappingRets[:j]) for i in range(mappingRets.shape[1])):
                            raise ValueError(f"All columns in self.returns are equal to the column {j} of the mappingRets array...")
                        
                # ensuring that 'mappingRets' and 'self.returns' have the same row count.
                if mappingRets.shape[0] != self.returns.shape[0]:
                    raise ValueError("Input 'mappingRets' and 'self.returns' should have the same number of rows.")
            
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Before we begin: for cleaner and more efficient code, we should aggregate self.returns (needed if self.interval is > 1)

        if self.interval > 1:

            # reshape_and_aggregate_returns
            def reshape_and_aggregate_returns(returns, interval):
                returns = returns.values ; num_periods = len(returns) // interval * interval 
                trimmed_returns = returns[len(returns) - num_periods:] ; reshaped_returns = trimmed_returns.reshape((-1, interval)) 
                aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                return aggregated_returns
            
            # converting self.returns back to a DataFrame to avoid a for loop
            rets_df = pd.DataFrame(self.returns)
            # applying the reshape_and_aggregate_returns function to each column
            aggregated_returns = rets_df.apply(reshape_and_aggregate_returns, args=(self.interval,)).values 

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # calculating the mean of the distribution *****************************************************************************

        # representing the average value the portfolio would have held if it maintained its initial stock composition throughout
        # the duration of self.return. 

        if self.interval == 1: ##########

            # updating mu -------------

            portfolio = np.copy(self.returns)
            new_positions = [float(num) for num in self.positions]
            for i in range(portfolio.shape[0]):
                portfolio[i, :] *= new_positions
                new_positions += portfolio[i, :]   
            port_returns = portfolio.sum(axis=1)
            
            # setting mean to zero if mu is "zero"
            if mu == "zero": mean = 0                                       

            # calculating the constant mean if mu is "constant"
            if mu == "constant": mean = np.mean(port_returns)               

            # calculating the moving average if mu is "moving_average"
            if mu == "moving_average":
                from pyriskmgmt.inputsControlFunctions import check_ma_window

                # checking the ma_window procedure
                ma_window = check_ma_window(ma_window)

                # raising an error if ma_window is greater than the number of returns
                if ma_window > len(port_returns):
                    raise ValueError("The ma_window parameter is greater than the number of returns...")
                returns = pd.Series(port_returns)
                sma_ma_window = returns.rolling(window=ma_window).mean()
                
                # updating mean with the last value from the moving average
                mean = sma_ma_window.tail(1).values[0]                      

        if self.interval > 1: ##########

            # updating mu -------------

            portfolio = np.copy(aggregated_returns)
            new_positions = [float(num) for num in self.positions]
            for i in range(portfolio.shape[0]):
                portfolio[i, :] *= new_positions
                new_positions += portfolio[i, :]   
            port_returns = portfolio.sum(axis=1)

            # setting mean to zero if mu is "zero"
            if mu == "zero": mean = 0                                       

            # calculating the constant mean if mu is "constant"
            if mu == "constant": mean = np.mean(port_returns)               

            # calculating the moving average if mu is "moving_average"
            if mu == "moving_average":
                from pyriskmgmt.inputsControlFunctions import check_ma_window

                # checking the ma_window procedure
                ma_window = check_ma_window(ma_window)

                # raising an error if ma_window is greater than the number of aggregated returns
                if ma_window > len(port_returns):
                    raise ValueError("The ma_window parameter is greater than the number of the aggregated returns...")
                returns = pd.Series(port_returns)
                sma_ma_window = returns.rolling(window=ma_window).mean() 
                
                # updating mean with the last value from the moving average
                mean = sma_ma_window.tail(1).values[0]                      

        # Sigma of the distribution **********************************************************************************************************************************

        # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
         
        # >>>>>>>>>>>>>>> From the mean part I already have 2 types of return: self.returns and aggregated_returns (the latter array already follows the interval)
        # aggregated_returns is needed and created only when self.interval is > 1 <<<<<<<<<<<<<<<<<<<<<<<

        if vol == "simple":

            if not sharpeDiag: # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if self.interval == 1: ##################
                    variance_covariance_matrix = np.cov(self.returns, rowvar=False)

                if self.interval > 1 : ################
                    variance_covariance_matrix = np.cov(aggregated_returns, rowvar=False)                  # aggregated_returns
        
            if sharpeDiag:   # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
                if len(mappingRets.shape) == 1: # mappingRets is a one dimensional array -------------------------------------------------------------------

                    # Before we begin: for cleaner and more efficient code, we should aggregate mappingRets (needed if self.interval is > 1)

                    if self.interval > 1:

                        num_periods = len(mappingRets) // self.interval * self.interval
                        trimmed_returns = mappingRets[len(mappingRets) - num_periods:]
                        reshaped_returns = trimmed_returns.reshape((-1, self.interval))
                        aggregated_factor_returns = np.product(1 + reshaped_returns, axis=1) - 1
                        
                    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    if self.interval == 1: ##########

                        factor_variance_betas = np.var(mappingRets, ddof=1)

                        # calculating the beta for each stock -------------------------------------------------------------------------------------------------
                        betas = []

                        idiosyncratic_variances = np.zeros(self.returns.shape[1])
                        for i in range(self.returns.shape[1]): # loop over each column
                            stock_rets = self.returns[:, i]
                            covariance = np.cov(stock_rets, mappingRets, ddof=1)[0, 1]
                            beta = covariance / factor_variance_betas
                            residuals = stock_rets - beta * mappingRets
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)
                            betas.append(beta)

                    elif self.interval > 1: ###########

                        factor_variance_betas = np.var(aggregated_factor_returns, ddof=1) 

                        # calculating the beta for each stock ----------------------------------------------------------------------
                        betas = []
                        idiosyncratic_variances = np.zeros(aggregated_returns.shape[1])
                        for i in range(aggregated_returns.shape[1]): # loop over each column
                            stock_rets = aggregated_returns[:,i]  # aggregated_returns
                            covariance = np.cov(stock_rets, aggregated_factor_returns, ddof = 1)[0, 1]   # aggregated_factor_returns
                            beta = covariance / factor_variance_betas
                            residuals = stock_rets - beta * aggregated_factor_returns
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)
                            betas.append(beta)
                    
                    # betas to an array 
                    betas = np.array(betas)
                    
                    # variance_covariance_matrix ----- Σ = β_1 β_1′ σ_1^2 + D_ε
                    variance_covariance_matrix = np.outer(betas , betas) * factor_variance_betas 

                    # the full variance-covariance matrix is then the sum of the 'factor' variance-covariance matrix and a diagonal matrix with the idiosyncratic variances
                    variance_covariance_matrix += np.diagflat(idiosyncratic_variances)

                else :  # mappingRets is a multi dimensional array ----------------------------------------------------------------------------

                    # Before we begin: for cleaner and more efficient code, we should aggregate mappingRets (needed if self.interval is > 1)

                    if self.interval > 1:
                        # converting self.returns back to a DataFrame to avoid a for loop
                        rets_df = pd.DataFrame(mappingRets)
                        # applying the reshape_and_aggregate_returns function to each column
                        aggregated_factor_rets = rets_df.apply(reshape_and_aggregate_returns, args=(self.interval,)).values

                    # /////////////////////////////////////////////////////////////////////////////////////////////////

                    if self.interval == 1: ###########

                        factor_variances = np.var(mappingRets, axis=0, ddof=1)

                        # calculating the beta for each stock ----------------------------------------------------------------------
                        betas_factors = []
                        for j in range(mappingRets.shape[1]):
                            betas = []
                            for i in range(self.returns.shape[1]):
                                beta = np.cov(mappingRets[:,j],self.returns[:,i])[0,1] / factor_variances[j]
                                betas.append(beta)
                            betas_factors.append(betas)

                        betas_factors = np.array(betas_factors).T
                        idiosyncratic_variances = np.zeros(self.returns.shape[1])

                        for i in range(betas_factors.shape[0]):
                            residuals = self.returns[:,i] - mappingRets @ betas_factors[i,:]
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)

                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    elif self.interval > 1: #########

                        factor_variances = np.var(aggregated_factor_rets, axis=0, ddof=1) 

                        # calculating the beta for each stock ##################################################################
                        
                        betas_factors = []
                        for j in range(aggregated_factor_rets.shape[1]):
                            betas = []
                            for i in range(aggregated_returns.shape[1]):
                                beta = np.cov(aggregated_factor_rets[:,j],aggregated_returns[:,i])[0,1] / factor_variances[j]
                                betas.append(beta)
                            betas_factors.append(betas)

                        betas_factors = np.array(betas_factors).T
                        idiosyncratic_variances = np.zeros(self.returns.shape[1])

                        for i in range(betas_factors.shape[0]):
                            residuals = aggregated_returns[:,i] - aggregated_factor_rets @ betas_factors[i,:]
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)

                    # # variance_covariance_matrix -----  Σ = β_1 β_1′ σ_1^2 + β_2 β_2′ σ_2^2 + ... + β_K β_K′ σ_K^2 + D_ε
                    variance_covariance_matrix = np.zeros((self.returns.shape[1], self.returns.shape[1]))

                    for i in range(mappingRets.shape[1]):
                        variance_covariance_matrix = variance_covariance_matrix + np.outer(betas_factors[:,i],betas_factors[:,i]) * factor_variances[i]

                    variance_covariance_matrix += np.diagflat(idiosyncratic_variances)

            # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            var_port = np.dot(np.dot(self.positions,variance_covariance_matrix),self.positions.T)
            sigma = np.sqrt(var_port)

            # ----- monte carlo simulations -------------------------------------------------

            montecarlo_returns = np.random.normal(loc=mean, scale=sigma, size=num_sims)

            # Value at Risk ----------------------------------

            VaR = - np.quantile(montecarlo_returns, self.alpha) 

            # Expected Shortfall ---------------------------
            
            ES = - np.mean(montecarlo_returns[montecarlo_returns < - VaR])

            VaR = max(VaR,0) ; ES = max(ES,0)

            if not VCV: 
                MC_DIST = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                            "T" : self.interval,
                            "mu" : round(mean,4),
                            "sigmaPi" : round(sigma,4)}
            else: 
                MC_DIST = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : self.interval,
                           "mu" : round(mean,4),
                           "sigmaPi" : round(sigma,4),
                           "VCV" : variance_covariance_matrix}
       
        # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        if vol == "garch":

            from pyriskmgmt.inputsControlFunctions import (check_p, check_q) 

            # check procedure
            p = check_p(p) ; q = check_q(q)

            # In this case the self.interval will be taken into cosideration as an argument of the dccCondCovGarch,dccCondCovGarch_Factors function

            # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            # importing the support function dccCondCovGarch(),dccCondCovGarch_Factors(), from the SupportFunctions module

            from pyriskmgmt.SupportFunctions import (dccCondCovGarch, dccCondCovGarch_Factors)

            # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            if not sharpeDiag: # //////////////////
                    
                DDC_GARCH = dccCondCovGarch

                variance_covariance_matrix = DDC_GARCH(returns = self.returns, interval = self.interval, p=p, q=q)[0]

            if sharpeDiag: # //////////////////////

                DDC_GARCH = dccCondCovGarch_Factors
                    
                if len(mappingRets.shape) == 1: # mappingRets is a one dimensional array ------------------------------------------------------------------------------------------

                    # computing conditional betas and variances ------------------------------

                    cond_betas = np.zeros(self.returns.shape[1]) ; idiosyncratic_variances = np.zeros(self.returns.shape[1]) ; total_loops = self.returns.shape[1]
                    
                    for i in range(total_loops):
                        # displaying  the progress
                        print(f'\rR language: DCC-GARCH ({p},{q}) Model - Mapping Procedure -> 1 Factor: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end=" " * 50)
                        if np.array_equal(self.returns[:, i], mappingRets):
                            cond_betas[i] = 1.0 # beta with itself is 1
                            idiosyncratic_variances[i] = 0
                        else:
                            rets_stock_factor = np.column_stack((self.returns[:, i], mappingRets))
                            GARCH_PROCESS =  DDC_GARCH(rets_stock_factor, interval = self.interval, p=p, q=q)
                            cond_cov_rets_stock_factor = GARCH_PROCESS[0]
                            cond_covariance = cond_cov_rets_stock_factor[0,1]
                            cond_var_factor = cond_cov_rets_stock_factor[1,1]
                            cond_betas[i] = cond_covariance / cond_var_factor
                            residuals = GARCH_PROCESS[1][:,0]  # residuals here are extracted from the DCC GARCH process... those are the ith stock returns that the model cannot explain
                            # when applying the DDC garch approach 
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)       

                    # variance_covariance_matrix ------- # Σ = β_1 β_1′ σ_1^2 + D_ε
                    variance_covariance_matrix = np.outer(cond_betas,cond_betas) * cond_var_factor # this is the garch(p,q) 'factor' variance_covariance_matrix

                    # the full variance-covariance matrix is then the sum of the 'factor' variance-covariance matrix and a diagonal matrix with the idiosyncratic variances
                    variance_covariance_matrix += np.diagflat(idiosyncratic_variances)

                else: # mappingRets is a multi dimensional array ------------------------------------------------------------------------------------------------------------------

                    # computing conditional betas and variances ------------------------------

                    betas_factors = np.zeros((self.returns.shape[1], mappingRets.shape[1])) ; idiosyncratic_variances = np.zeros(self.returns.shape[1]) 
                    total_loops = self.returns.shape[1]
                    
                    for i in range(total_loops):
                        # displaying  the progress
                        print(f'\rR language: DCC-GARCH ({p},{q}) Model - Mapping Procedure -> {mappingRets.shape[1]} Factors: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end=" " * 50)
                        rets_stock_factors = np.column_stack((self.returns[:, i], mappingRets))
                        # flagging for columns to drop in case of duplicates 
                        cols_to_drop = []
                        # checking for duplicates
                        for col in range(rets_stock_factors.shape[1] -1):
                            if np.array_equal(rets_stock_factors[:, 0], rets_stock_factors[:, col + 1]):
                                cols_to_drop.append(col + 1) 
                        rets_stock_factors_checked = np.delete(rets_stock_factors, cols_to_drop, axis=1)

                        if rets_stock_factors_checked.shape[1] == rets_stock_factors.shape[1]: # no columns have been dropped
                            GARCH_PROCESS = DDC_GARCH(rets_stock_factors_checked, interval = self.interval, p=p, q=q)
                            cond_var_cov_matrix_inner = GARCH_PROCESS[0]
                            cond_variances_factors = np.diag(cond_var_cov_matrix_inner)[1:]
                            inner_betas = np.zeros(mappingRets.shape[1])

                            for j in range(1, cond_var_cov_matrix_inner.shape[1]):
                                beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j]
                                inner_betas[j - 1] = beta
                            betas_factors[i,:] = inner_betas

                            # residuals calculation ///////////////////////////////////////////////////////////////////////////////////////////////

                            residuals = GARCH_PROCESS[1][:,0]  # residuals here are extracted from the DCC GARCH process... those are the ith stock returns that the model cannot explain
                            # when applying the DDC garch approach --- in this case a multi factors model
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)

                        elif rets_stock_factors_checked.shape[1] != rets_stock_factors.shape[1]: # the cols_to_drop column has been dropped
                            # fitting the GARCH_PROCESS on n-1 columns
                            GARCH_PROCESS = DDC_GARCH(rets_stock_factors_checked, interval = self.interval, p=p, q=q)
                            cond_var_cov_matrix_inner = GARCH_PROCESS[0]
                            cond_variances_factors = np.diag(cond_var_cov_matrix_inner)[1:]

                            if cols_to_drop[0] > rets_stock_factors_checked.shape[1] -1:
                                cond_variances_factors = np.append(cond_variances_factors, cond_var_cov_matrix_inner[0,0])
                            else:
                                cond_variances_factors = np.insert(cond_variances_factors, cols_to_drop[0], cond_var_cov_matrix_inner[0,0])
                            inner_betas = np.zeros(mappingRets.shape[1]-1)

                            for j in range(1, cond_var_cov_matrix_inner.shape[1]):
                                beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j]
                                inner_betas[j - 1] = beta

                            if cols_to_drop[0] > rets_stock_factors_checked.shape[1]-1:
                                inner_betas = np.append(inner_betas, 1)
                            else:
                                inner_betas = np.insert(inner_betas, cols_to_drop[0], 1)
                            betas_factors[i,:] = inner_betas

                            # residuals calculation ///////////////////////////////////////////////////////////////////////////////////////////////

                            residuals = GARCH_PROCESS[1][:,0]  # residuals here are extracted from the DCC GARCH process... those are the ith stock returns that the model cannot explain
                            # when applying the DDC garch approach --- in this case a multi factor model
                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)

                    # # variance_covariance_matrix ------ Σ = β_1 β_1′ σ_1^2 + β_2 β_2′ σ_2^2 + ... + β_K β_K′ σ_K^2 + D_ε
                    variance_covariance_matrix = np.zeros((self.returns.shape[1], self.returns.shape[1]))

                    for i in range(mappingRets.shape[1]):
                        variance_covariance_matrix += np.outer(betas_factors[:,i],betas_factors[:,i]) * cond_variances_factors[i]

                    variance_covariance_matrix += np.diagflat(idiosyncratic_variances)

            # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            var_port = np.dot(np.dot(self.positions,variance_covariance_matrix),self.positions.T)
            sigma = np.sqrt(var_port)

            # ----- monte carlo simulations -------------------------------------------------

            montecarlo_returns = np.random.normal(loc=mean, scale=sigma, size=num_sims)

            # Value at Risk ----------------------------------
        
            VaR = - np.quantile(montecarlo_returns, self.alpha) 

            # Expected Shortfall ---------------------------
            
            ES = - np.mean(montecarlo_returns[montecarlo_returns < - VaR])

            VaR = max(VaR,0) ; ES = max(ES,0)

            if not VCV: 
                MC_DIST = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : self.interval,
                           "p" : p,
                           "q" : q,
                           "mu" : round(mean,4),
                           "sigmaPi" : round(sigma,4)} 
            else: 
                MC_DIST = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : self.interval,
                           "p" : p,
                           "q" : q,
                           "mu": round(mean,4),
                           "sigmaPi" : round(sigma,4),
                           "VCV" : variance_covariance_matrix}
                     
         # EWMA (lamdba = lambda_ewma) Volatility ------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        if vol == "ewma":

            from pyriskmgmt.inputsControlFunctions import (check_lambda_ewma)

            # check procedure
            lambda_ewma = check_lambda_ewma(lambda_ewma)

            # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            # importing the support function ewmaCondCov(),ewmaCondCov__Factors(), from the SupportFunctions module

            from pyriskmgmt.SupportFunctions import (ewmaCondCov, ewmaCondCov__Factors)

            # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            if not sharpeDiag: # //////////////////
                    
                    EWMA_COV = ewmaCondCov

                    variance_covariance_matrix = EWMA_COV(returns = self.returns, interval = self.interval, lambda_ewma = lambda_ewma)

            if sharpeDiag: # //////////////////

                EWMA_COV = ewmaCondCov__Factors
                    
                if len(mappingRets.shape) == 1: # mappingRets is a one dimensional array ------------------------------------------------------------------------------------------

                    # computing conditional betas and variances ------------------------------

                    cond_betas = np.zeros(self.returns.shape[1])
                    cond_variance_factor = np.zeros(self.returns.shape[1]) 
                    idiosyncratic_variances = np.zeros(self.returns.shape[1])
                    total_loops = self.returns.shape[1]

                    for i in range(total_loops):
                        # Display the progress
                        print(f'\rEWMA MODEL ({lambda_ewma}) - Mapping Procedure -> 1 Factor: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end=" " * 50)
                        if np.array_equal(self.returns[:, i], mappingRets):
                            cond_betas[i] = 1.0 # beta with itself is 1
                            idiosyncratic_variances[i] = 0
                        else:
                            rets_stock_factor = np.column_stack((self.returns[:, i], mappingRets))
                            cond_cov_rets_stock_factor = EWMA_COV(rets_stock_factor, interval = self.interval, lambda_ewma = lambda_ewma)
                            cond_covariance = cond_cov_rets_stock_factor[0,1]
                            cond_var_factor = cond_cov_rets_stock_factor[1,1]
                            cond_variance_factor[i] = cond_var_factor 
                            cond_betas[i] = cond_covariance / cond_var_factor

                            # residuals calculation ///////////////////////////////////////////////////////////////////////////////////////////////

                            if self.interval == 1: 
                                residual_rets = rets_stock_factor
                            else: 
                                num_periods = len(mappingRets) // self.interval * self.interval  
                                trimmed_returns = mappingRets[len(mappingRets) - num_periods:]
                                reshaped_returns = trimmed_returns.reshape((-1, self.interval))  
                                aggregated_factor_returns = np.product(1 + reshaped_returns, axis=1) - 1
                                residual_rets = np.column_stack((aggregated_returns[:, i], aggregated_factor_returns))

                                # *********************************

                            residuals = np.zeros(residual_rets.shape[0] - ma_window)
                            for t in range(residual_rets.shape[0] - ma_window):  
                                moving_rets = residual_rets[t : t + ma_window,:]  
                                cond_cov_rets_stock_factor_inner = EWMA_COV(moving_rets, interval = 1, lambda_ewma = lambda_ewma)
                                cond_covariance_inner = cond_cov_rets_stock_factor_inner[0,1] 
                                cond_variance_inner = cond_cov_rets_stock_factor_inner[1,1]
                                cond_beta = cond_covariance_inner / cond_variance_inner
                                residuals[t] = residual_rets[t + ma_window ,0] - cond_beta * residual_rets[t + ma_window ,1]

                            idiosyncratic_variances[i] = np.var(residuals, ddof=1)  

                    # variance_covariance_matrix 
                    variance_covariance_matrix = np.outer(cond_betas,cond_betas) * cond_variance_factor[-1] # this is the ewma(lambda_ewma)'factor' variance_covariance_matrix

                    # The full variance-covariance matrix is then the sum of the 'factor' variance-covariance matrix and a diagonal matrix with the idiosyncratic variances
                    variance_covariance_matrix += np.diagflat(idiosyncratic_variances)

                else: # mappingRets is a multi dimensional array ------------------------------------------------------------------------------------------------------------------

                    # computing conditional betas and variances ------------------------------

                    betas_factors = np.zeros((self.returns.shape[1], mappingRets.shape[1])) 
                    idiosyncratic_variances = np.zeros(self.returns.shape[1]) 
                    total_loops = self.returns.shape[1]

                    for i in range(total_loops):
                        # displaying  the progress
                        print(f'\rEWMA MODEL ({lambda_ewma}) - Mapping Procedure -> {mappingRets.shape[1]} Factors: {i+1}/{total_loops} ({((i+1)/total_loops)*100:.2f}%)', end=" " * 60)
                        rets_stock_factors = np.column_stack((self.returns[:, i], mappingRets))
                        cond_var_cov_matrix_inner = EWMA_COV(rets_stock_factors, interval = self.interval, lambda_ewma = lambda_ewma) 
                        cond_variances_factors = np.diag(cond_var_cov_matrix_inner)[1:]
                        inner_betas = np.zeros(mappingRets.shape[1])
                        for j in range(1, cond_var_cov_matrix_inner.shape[1]):
                            beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j] 
                            inner_betas[j - 1] = beta
                        betas_factors[i,:] = inner_betas

                        # residuals calculation ///////////////////////////////////////////////////////////////////////////////////////////////

                        if self.interval == 1: 
                            residual_rets =  rets_stock_factors
                        else: 
                            residual_rets = pd.DataFrame(rets_stock_factors)
                            # applying the reshape_and_aggregate_returns function to each column
                            residual_rets = residual_rets.apply(reshape_and_aggregate_returns, args=(self.interval,)).values 

                        # ************************************************************************

                        residuals = np.zeros(residual_rets.shape[0] - ma_window)

                        for t in range(residual_rets.shape[0] - ma_window): 
                            moving_rets = residual_rets[t : t + ma_window, : ]  
                            cond_cov_rets_stock_factor_inner = EWMA_COV(moving_rets, interval = 1, lambda_ewma = lambda_ewma)
                            inner_betas_single = np.zeros(mappingRets.shape[1])

                            for j in range(1, cond_cov_rets_stock_factor_inner.shape[1]):
                                beta = cond_var_cov_matrix_inner[0,j] / cond_var_cov_matrix_inner[j,j] 
                    
                            inner_betas_single[j - 1] = beta
                            residuals[t] = residual_rets[t + ma_window ,0] - np.dot(inner_betas_single, residual_rets[t + ma_window ,1:])

                        idiosyncratic_variances[i] = np.var(residuals, ddof=1)  

                    # # variance_covariance_matrix ------ Σ = β_1 β_1′ σ_1^2 + β_2 β_2′ σ_2^2 + ... + β_K β_K′ σ_K^2 + D_ε
                    variance_covariance_matrix = np.zeros((self.returns.shape[1], self.returns.shape[1]))

                    for i in range(mappingRets.shape[1]):
                        variance_covariance_matrix += np.outer(betas_factors[:,i],betas_factors[:,i]) * cond_variances_factors[i]

                    variance_covariance_matrix += np.diagflat(idiosyncratic_variances)

            # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            var_port = np.dot(np.dot(self.positions,variance_covariance_matrix),self.positions.T) ; sigma = np.sqrt(var_port) # sigma here already follows the interval 

            # ----- monte carlo simulations -------------------------------------------------

            montecarlo_returns = np.random.normal(loc=mean, scale=sigma, size=num_sims)

            # Value at Risk ----------------------------------
        
            VaR = - np.quantile(montecarlo_returns, self.alpha) 

            # Expected Shortfall ---------------------------
            
            ES = - np.mean(montecarlo_returns[montecarlo_returns < - VaR])

            VaR = max(VaR,0) ; ES = max(ES,0)

            if not VCV: 
                MC_DIST = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : self.interval,
                           "lambda_ewma" : lambda_ewma,
                           "mu" : round(mean,4),
                           "sigmaPi" : round(sigma,4)} 
            else: 
                MC_DIST = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : self.interval,
                           "lambda_ewma" : lambda_ewma,
                           "mu" : round(mean,4),
                           "sigmaPi" : round(sigma,4),
                           "VCV" : variance_covariance_matrix} 
                
        # -----------------------------------------------------------------------------------------------------

        if mappingRets is not None and vol == "ewma":
            print("")
        if mappingRets is not None and vol == "garch":
            print("")

        return MC_DIST 

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************    
        
    # MONTECARLO MODEL [RETURNS SIMULATION] <<<<<<<<<<<<
    
    def mcModelRetsSimu(self, sharpeDiag: bool = False , mappingRets: np.ndarray = None, vol: str = "simple", num_sims: int = 10000,
                         p: int = 1, q: int = 1, lambda_ewma: float = 0.94, warning: bool = True, return_simulated_rets: bool = False):
        """
        Main
        ----
        This method performs Monte Carlo simulations on portfolio returns using a specified model.
        The simulations can be based either on the Sharpe Diagonal model or directly on the portfolio returns.
        This method utilizes Cholesky Decomposition for correlated returns between different asset returns in the portfolio.
        When 'sharpeDiag' is True, the function adopts a slightly different approach from the sharpeDiag() method. 
        It calculates the variance-covariance matrix resulting from the mapping procedure,
        incorporating the estimation of idiosyncratic variance to ensure the matrix is positive and semi-definite and numerical stability.
        Furthermore, here the Value at Risk (VaR) under the 'sharpeDiag' approach is calculated through a Monte Carlo simulation, assuming a mean of zero.
        This has been shown to yield a VaR nearly identical to the pure sharpeDiag() approach, particularly for larger portfolios.

        
        Parameters
        ----------
        - sharpeDiag: Determines whether Sharpe-style decomposition is performed. If False, the estimation relies on the returns themselves. Default is False.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on the same frequency than the "returns" array.
        Also align "mappingRets" with "returns" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility model to be used. Options include "simple", "garch", or "ewma". Default is "simple".
        - num_sims: The number of Monte Carlo simulations to run. Default is 10000.
        - p: The order of the GARCH model, if selected. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model, if selected. Default is 1.
        - lambda_ewma: The decay factor for the EWMA model, if used. Determines the rate of weight decrease for older observations. Default is 0.94.
        - warning: A flag indicating whether a warning message should be printed. Default is True.
        - return_simulated_rets: if True the simulated returns are returned. Default is False.

        Returns
        -------
        - MC_DIST: dict: A dictionary containing the portfolio risk metrics including Value at Risk (VaR), Expected Shortfall (ES), and the chosen interval. 

        Notes
        -----
        - Additional metrics specific to the selected volatility model (sigmaPi for "simple", p, q, and sigmaPi for "garch", or lambda_ewma and sigmaPi for "ewma") are also included.
        - The progress of the simulation is printed on the console. 
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; import pandas as pd ; from scipy.stats import norm ; from joblib import Parallel, delayed

        from pyriskmgmt.inputsControlFunctions import check_return_simulated_rets

        #check procedure 
        check_return_simulated_rets(return_simulated_rets)

        if warning:
            print("WARNING:")
            print("The frequency of self.returns and the mappingRets (if sharpeDiag) must correspond to the same one single period of analysis.")
            print("This method was written to aggregate both self.returns and the mapping factor/s returns ")
            print("(if selected) by itself if the interval is > 1.")
            print("The method ff3() supports [daily, weekly and monthly] frequency.")
            print("To mute this warning, add 'warning = False' as an argument of the mcModelRetsSimu() function.")

        if vol == "garch":
            
            if warning and self.returns.shape[1] > 50 and not sharpeDiag:
                print("IMPORTANT:")
                print("The 'garch' option for the 'vol' parameter is currently in use, with an asset count exceeding 50.")
                print("This configuration can lead to slower processing and inefficient resource usage, due to the complexity ")
                print("of estimating the full variance-covariance matrix (VCV) in high-dimensional space.")
                print("It's strongly advised to employ a mapping approach instead. This approach maps the multiple assets to one or more factors,")
                print("effectively reducing the dimensionality of the problem.")
                print("This is not only more computationally efficient, but it also avoids the 'curse of dimensionality',")
                print("as it reduces the number of parameters needed to be estimated. The mapping approach aids in providing more stable and reliable estimations,")
                print("especially in large, complex systems print.")

        warning = False

        # ----------------------------------------------------------------------
       
        # Using mcModelDrawPortRets to calculate the variance_cov_matrix 

        # all the inputs will be checked here

        variance_covariance_matrix = self.mcModelDrawPortRets(sharpeDiag = sharpeDiag, mappingRets =  mappingRets,  vol = vol, num_sims = num_sims,
                                                       p = p, q=q, lambda_ewma = lambda_ewma, warning = warning, VCV = True )["VCV"]
        
        var_port = np.dot(np.dot(self.positions,variance_covariance_matrix),self.positions.T) ; sigma = np.sqrt(var_port) # sigma here already follows the interval 
        

        # computing Cholesky decomposition of the variance covariance matrix ----------------------------------------------------------------------------------

        if self.interval == 1:
            number_of_observations = self.returns.shape[0]
        else: 
            number_of_observations = int((len(self.returns) // self.interval * self.interval) / self.interval)

        try:
            cholesky_matrix = np.linalg.cholesky(variance_covariance_matrix)
        except np.linalg.LinAlgError:
            if self.interval == 1:
                print("WARNING:")
                print("Failed to decompose the Variance-Covariance Matrix using the Cholesky decomposition.")
                print(f"There are {self.returns.shape[1]} number of assests and {number_of_observations} number of observations...")
                print("Please consider:")
                print("- widening your data range")
                print("- reducing the interval")
                print("- checking the quality of your input data")
                print("- using one or more mapping factors")
                print("However, keep in mind that the less data you use, the less precise the estimation.")
            else:
                print("WARNING:")
                print("Failed to decompose the Variance-Covariance Matrix using the Cholesky decomposition.")
                print("- After the Aggregation Process -")
                print(f"There are {self.returns.shape[1]} number of assests and {number_of_observations} number of observations...")
                print("Please consider:")
                print("- widening your data range")
                print("- reducing the interval")
                print("- checking the quality of your input data")
                print("- using one or more mapping factors")
                print("However, keep in mind that the less data you use, the less precise the estimation.")

        def run_simulation(simulation):
            # generating random numbers using np.random.rand
            random_numbers = norm.ppf(np.random.rand(self.returns.shape[0], len(self.positions)))

            # multipling the Cholesky matrix with the random numbers
            simulated_returns = pd.DataFrame(np.matmul(cholesky_matrix, random_numbers.T).T)

            PL = simulated_returns @ np.transpose(self.positions)

            VaR = - np.quantile(PL, self.alpha)
            ES = - np.mean(PL[PL < - VaR])

            # returning Value at Risk, Expected Shortfall and simulated returns
            return VaR, ES, simulated_returns.values

        # -------------------------------------------------------------------------------------------------------------------------------------------------------

        simulated_returns_3D = np.zeros((self.returns.shape[0], len(self.positions), num_sims))

        MC_VARS = np.zeros(num_sims) ; MC_ES = np.zeros(num_sims)

        # just some aesthetics
        from pyriskmgmt.SupportFunctions import DotPrinter ; import time

        # starting the animation
        my_dot_printer = DotPrinter(f"\r {num_sims} Correlated Mc Simulations using Cholesky Decomposition - joblib - Parallel Computation") ; my_dot_printer.start()

        # performing num_sims simulations
        results = Parallel(n_jobs=-1)(delayed(run_simulation)(simulation) for simulation in range(num_sims))

        # stopping the animation when minimization is done
        my_dot_printer.stop()
        print(f"\rCholesky Decomposition - {num_sims} Simulations --->  Done", end=" " * 90)
        time.sleep(0.3)
        print("")

        for i, (VaR, ES, simulated_returns) in enumerate(results):
            MC_VARS[i] = VaR
            MC_ES[i] = ES
            simulated_returns_3D[:,:,i] = simulated_returns

        VaR = np.mean(MC_VARS) ; ES = np.mean(MC_ES) ; VaR = max(VaR,0) ; ES = max(ES,0)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        # compiling and updating the dictinary with the results
        MC_DIST = {"var" : round(VaR, 4),
                   "es" : round(ES, 4),
                   "T" : self.interval}

        if vol == "simple":
            MC_DIST.update({"sigmaPi" : round(sigma,4)})

        if vol == "garch":
            MC_DIST.update({"p" : p,
                            "q" : q,
                            "sigmaPi" : round(sigma,4)})

        elif vol == "ewma":
            MC_DIST.update({"lambda_ewma" : lambda_ewma,
                            "sigmaPi" : round(sigma,4)})

        if return_simulated_rets:
            return MC_DIST, simulated_returns_3D
        else:
            return MC_DIST
    
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 

    # MARGINAL VARS <<<<<<<<<<<<<<<
            
    def MargVars(self, sharpeDiag: bool = False , mappingRets: np.ndarray = None, vol: str = "simple", num_sims: int = 10000,
                  p: int = 1, q: int = 1, lambda_ewma: float = 0.94):
        
        """
        Main
        ----
        Implements the calculation of Marginal Value at Risk (MVaR) for individual assets in the portfolio using either Sharpe's Diagonal model or the portfolio returns directly.
        This methos accommodates different volatility models, including "simple", "garch", and "ewma".
        The Marginal VaR measures the change in the VaR of a portfolio due to an increase of 1 unit in a specific asset, keeping all other positions constant. 
        When 'sharpeDiag' is True, the function adopts a slightly different approach from the sharpeDiag() method. 
        It calculates the variance-covariance matrix resulting from the mapping procedure,
        incorporating the estimation of idiosyncratic variance to ensure that the matrix is positive and semi-definite, for numerical stability.
        Furthermore, here the Value at Risk (VaR) under the 'sharpeDiag' approach is calculated through a Monte Carlo simulation, assuming a mean of zero.
        This has been shown to yield a VaR nearly identical to the pure sharpeDiag() approach, particularly for larger portfolios.

        Parameters
        ----------
        - sharpeDiag: Determines whether Sharpe-style decomposition is performed. If False, the estimation relies on the returns themselves. Default is False.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on the same frequency than the "returns" array.
        Also align "mappingRets" with "returns" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility model to be used. Options include "simple", "garch", or "ewma". Default is "simple".
        - num_sims: Specifies the number of Monte Carlo simulations to run. Default is 10000.
        - p: Specifies the order of the GARCH model. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model. Default is 1.
        - lambda_ewma: Specifies the decay factor for the EWMA model, if used. Determines how quickly the weights decrease for older observations. Default is 0.94.
        
        Returns
        - MVars : dict: A dictionary containing the Marginal VaR for each asset in the portfolio.

        Notes
        -----
        - This function integrates Python and R for efficiency when 'vol = "garch', particularly due to the utilization of DCC GARCH models in R. 
        - Ensure that an active R environment and the 'rmgarch' package are available for this function to operate correctly.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np 
                    
        warning = False

        # ------------------------------------------------------------------------------------------------------------------------
       
        # Using mcModelDrawPortRets to calculate the variance_cov_matrix, setting mu = 0.

        # all the inputs will be checked here, ma_window (needed for the residuls in the case of the ewma approach) is set to be 15 by default

        VAR_COV_PROCESS_VaR = self.mcModelDrawPortRets(sharpeDiag = sharpeDiag,  mu = "zero", mappingRets =  mappingRets,  vol = vol, num_sims = num_sims,
                                                       p = p, q=q, lambda_ewma = lambda_ewma, warning = warning, VCV = True)
        
        # extracting the variance-covariance matrix and portfolio variance from the results
        variance_covariance_matrix = VAR_COV_PROCESS_VaR["VCV"] 
        var_port = VAR_COV_PROCESS_VaR["sigmaPi"] ** 2

        # checking for the sharpeDiag method and assigning appropriate VaR value
        if sharpeDiag:
            VaR = VAR_COV_PROCESS_VaR["var"]
        else:
            # assigning the VaR based on the volatility estimation method
            if vol == "simple":
                VaR = self.simple_var
            elif vol == "garch":
                # retrieving the garch model VaR, or calculating it if it's not available
                VaR = self.garch_model_var if self.garch_model_var is not None else self.garch_model(p = p, q = q)["var"]
            elif vol == "ewma":
                # retrieving the ewma model VaR, or calculating it if it's not available
                VaR = self.ewma_model_var if self.ewma_model_var is not None else self.ewma_model(lambda_ewma = lambda_ewma)["var"]

        # calculating the marginalVaRs using the variance-covariance matrix, positions, and VaR value
        marginalVaRs =  ((np.dot(variance_covariance_matrix, self.positions.T)) / var_port) * VaR

        marginalVaRs = np.round(marginalVaRs, 4)

        return {"var" : round(VaR,4),
                "vol" : vol,
                "MargVars" : marginalVaRs} 

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # RELATIVE COMPONENTS VARS <<<<<<<<<<<<<<<
            
    def RelCompVars(self, sharpeDiag: bool = False , mappingRets: np.ndarray = None,  vol: str = "simple", num_sims: int = 10000,
                     p: int = 1, q: int = 1, lambda_ewma: float = 0.94):
        """
        Main
        ----
        Implements the calculation of Component Value at Risk relative to the total VaR of the portfolio, using either Sharpe's Diagonal model or the portfolio returns directly.
        This method supports various volatility models, including "simple", "garch", and "ewma".
        The Relative Components VaRs provide insight into the proportion (%) of total risk that each asset contributes.
        When 'sharpeDiag' is True, the function adopts a slightly different approach from the sharpeDiag() method.
        It calculates the variance-covariance matrix resulting from the mapping procedure,
        incorporating the estimation of idiosyncratic variance to ensure that the matrix is positive and semi-definite, for numerical stability.
        Here, the Value at Risk (VaR) under the 'sharpeDiag' approach is calculated through a Monte Carlo simulation, assuming a mean of zero.
        This method has been demonstrated to yield a VaR nearly identical to the pure sharpeDiag() approach, especially for larger portfolios.

        Parameters
        ----------
        - sharpeDiag: Determines whether Sharpe-style decomposition is performed. If False, the estimation relies on the returns themselves. Default is False.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on the same frequency than the "returns" array.
        Also align "mappingRets" with "returns" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility model to be used. Options include "simple", "garch", or "ewma". Default is "simple".
        - num_sims: Specifies the number of Monte Carlo simulations to run. Default is 10000.
        - p: Specifies the order of the GARCH model. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model. Default is 1.
        - lambda_ewma: Specifies the decay factor for the EWMA model, if used. Determines how quickly the weights decrease for older observations. Default is 0.94.
        
        Returns
        -------
        - RelComVars : dict: A dictionary containing the relative Component VaR for each asset in the portfolio.

        Notes
        ------
        - This function integrates Python and R for efficiency when 'vol = "garch', particularly due to the utilization of DCC GARCH models in R.
        - Ensure that an active R environment and the 'rmgarch' package are available for this function to operate correctly.
        """

        import warnings ; warnings.filterwarnings("ignore")
                
        # calculating marginal VaRs using the MargVars function
        MARGINAL_VARS_PROCESS = self.MargVars(sharpeDiag = sharpeDiag, mappingRets =  mappingRets,  vol = vol, num_sims = num_sims,
                                              p = p, q = q, lambda_ewma = lambda_ewma)
        
        # extracting marginal VaRs and VaR from the results
        marginal_VaRs =  MARGINAL_VARS_PROCESS["MargVars"] 
        VaR = MARGINAL_VARS_PROCESS["var"]

        # computing the components VaRs using the extracted marginal VaRs and positions
        components_VaRs = marginal_VaRs.T * self.positions 
        
        # calculating the relative components VaRs by dividing with VaR
        relcomponentsVaRs = components_VaRs / VaR

        relcomponentsVaRs = np.round(relcomponentsVaRs, 4)

        relcomponents_VaRs = {"var" : round(VaR,4),
                              "vol" : vol,
                              "RelComVars" : relcomponentsVaRs}

        return relcomponents_VaRs
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # COMPONENTS VARS <<<<<<<<<<<<<<<
            
    def CompVars(self, sharpeDiag: bool = False , mappingRets: np.ndarray = None, vol: str = "simple", num_sims: int = 10000,
                  p: int = 1, q: int = 1, lambda_ewma: float = 0.94):
        """
        Main
        ----
        Implements the calculation of Component Value at Risk, using either Sharpe's Diagonal model or the portfolio returns directly.
        This method supports various volatility models, including "simple", "garch", and "ewma".
        The Components VaRs provides insight into the proportion of total risk (currency) that each asset contributes.
        When 'sharpeDiag' is True, the function adopts a slightly different approach from the sharpeDiag() method.
        It calculates the variance-covariance matrix resulting from the mapping procedure,
        incorporating the estimation of idiosyncratic variance to ensure that the matrix is positive and semi-definite, for numerical stability.
        Here, the Value at Risk (VaR) under the 'sharpeDiag' approach is calculated through a Monte Carlo simulation, assuming a mean of zero.
        This method has been demonstrated to yield a VaR nearly identical to the pure sharpeDiag() approach, especially for larger portfolios.

        Parameters
        ----------
        - sharpeDiag: Determines whether Sharpe-style decomposition is performed. If False, the estimation relies on the returns themselves. Default is False.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on the same frequency than the "returns" array.
        Also align "mappingRets" with "returns" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility model to be used. Options include "simple", "garch", or "ewma". Default is "simple".
        - num_sims: Specifies the number of Monte Carlo simulations to run. Default is 10000.
        - p: Specifies the order of the GARCH model. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model. Default is 1.
        - lambda_ewma: Specifies the decay factor for the EWMA model, if used. Determines how quickly the weights decrease for older observations. Default is 0.94.
        
        Returns
        -------
        - RelComVars : dict: A dictionary containing the relative Component VaR for each asset in the portfolio.

        Notes
        ------
        - This function integrates Python and R for efficiency when 'vol = "garch', particularly due to the utilization of DCC GARCH models in R.
        - Ensure that an active R environment and the 'rmgarch' package are available for this function to operate correctly.
        """

        import warnings ; warnings.filterwarnings("ignore")

        # invoking the MargVars function to get the process details
        MARGINAL_VARS_PROCESS = self.MargVars(sharpeDiag = sharpeDiag, mappingRets =  mappingRets,  vol = vol, num_sims = num_sims,
                                              p = p, q = q, lambda_ewma = lambda_ewma)
        
        # extracting the marginal VaRs and VaR from the result
        marginal_VaRs =  MARGINAL_VARS_PROCESS["MargVars"] 
        VaR =  MARGINAL_VARS_PROCESS["var"]

        # calculating the components of VaRs using the marginal VaRs and positions
        componentsVaRs = marginal_VaRs.T * self.positions

        componentsVaRs = np.round(componentsVaRs, 4)

        components_VaRs = {"var" : round(VaR,4),
                           "vol" : vol,
                          "ComVars" : componentsVaRs}

        return components_VaRs
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    #  VAR and ES MINIMAZER  <<<<<<<<<<<<<<<

    def VarEsMinimizer(self, sharpeDiag: bool = False, mappingRets: np.ndarray = None, vol: str = "simple",
                        p: int = 1, q: int = 1, lambda_ewma: float = 0.94, allow_short: bool = True):

        """
        Main
        ----
        This method carries out an optimization of portfolio positions with the goal of minimizing the portfolio's Value at Risk (VaR) and Expected Shortfall (ES).
        It accommodates a range of volatility models ("simple", "garch", or "ewma") and can use either the Sharpe's Diagonal model or the portfolio returns directly
        for the optimization process. 
        When 'sharpeDiag' is enabled, the function calculates the variance-covariance matrix, which includes the idiosyncratic variance, from the mapping procedure. 
        This ensures the matrix is positive semi-definite, thereby enhancing numerical stability.
        The VaR is then computed via a Monte Carlo simulation with a mean of zero, offering a result closely mirroring the outcome of the pure sharpeDiag() approach,
        particularly for larger portfolios. This function allows the option to include or exclude short positions in the optimization process.

        Parameters
        ----------
        - sharpeDiag: Determines whether Sharpe-style decomposition is performed. If False, the estimation relies on the returns themselves. Default is False.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on the same frequency than the "returns" array.
        Also align "mappingRets" with "returns" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility model used. Options are "simple", "garch", or "ewma". Default is "simple".
        - p: Specifies the order of the GARCH model. Default is 1.
        - q: Defines the order of the moving average term in the GARCH model. Default is 1.
        - lambda_ewma: Specifies the decay factor for the EWMA model, if used. Determines how quickly the weights decrease for older observations. Default is 0.94.
        - allow_short: Determines if short positions are allowed in the optimization process. Default is True.
        
        Returns
        --------
        - MIN_PROCESS : dict: A dictionary containing the minimized VaR, ES, and other risk metrics along with the old and new portfolio positions.

        Notes
        ------
        - This function integrates Python and R when 'vol = "garch', particularly due to the utilization of DCC GARCH models in R.
        - Ensure that an active R environment and the 'rmgarch' package are available for this function to operate correctly.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.stats import norm

        from pyriskmgmt.inputsControlFunctions import (check_allow_short)

        # check procedure
        check_allow_short(allow_short)
                
        warning = False

        # using mcModelDrawPortRets to calculate the variance_cov_matrix 
        variance_covariance_matrix = self.mcModelDrawPortRets(sharpeDiag = sharpeDiag, mu = "zero", mappingRets =  mappingRets,  vol = vol,
                                                       p = p, q=q, lambda_ewma = lambda_ewma, warning = warning, VCV = True )["VCV"]

        # defining the function to be minimized
        def Var_Calculator(positions, Variance_Cov_Matrix):    
            sigma = np.sqrt(positions.T @ Variance_Cov_Matrix @ positions) 
            q = norm.ppf(1-self.alpha, loc=0, scale=1) 
            VaR = q * sigma 
            return VaR
        
        # importing necessary libraries for optimization
        from scipy.optimize import minimize

        # initializing variables for optimization
        n = len(self.positions)
        init_guess = np.repeat(np.mean(self.positions),n)

        # setting bounds based on short position allowance
        bounds = ((None, None),) * n if allow_short else ((0, None),) * n     

        # ensuring that the sum of positions remains constant
        positions_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - np.sum(self.positions)}

        # creating a loading animation for aesthetics
        from pyriskmgmt.SupportFunctions import DotPrinter
        my_dot_printer = DotPrinter("Minimization Process") ; my_dot_printer.start()

        # if short selling is not allowed and self.totPortPos <= 0
        if not allow_short and self.totPortPos <= 0:
            my_dot_printer.stop()

            print(f"\rMinimization Process: >>>> Convergence Unsuccessful <<<<", end=" " * 90)
            raise Exception(f"""
            The initial total position of the Portfolio stands at {self.totPortPos}, which is negative. 
            This method tries to keep (more or less) the same total position of the Porfolio.
            When short selling is prohibited, maintaining a negative position for a Portfolio becomes challenging.
            Furthermore, the solver is prone to defaulting to the simplest solution, which is zero for all positions.
            """)

        # performing the optimization
        optimized = minimize(Var_Calculator, init_guess, args=(variance_covariance_matrix), method='SLSQP',
                            options={'disp': False}, constraints=(positions_sum), bounds=bounds)
        
        # ending the loading animation after optimization
        my_dot_printer.stop()
        print("\rMinimization Process --->  Done - Minimum Found", end=" " * 80)
        print("")
        
        # calculating Value at Risk
        var_port = np.dot(np.dot(optimized.x,variance_covariance_matrix),optimized.x.T)
        sigma = np.sqrt(var_port) 
        quantile = norm.ppf(1-self.alpha, loc=0, scale=1) 
        VaR = quantile * sigma 
        VaR =  max(0,VaR)

        # calculating Expected Shortfall
        q_es = ((np.exp( - (quantile**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-self.alpha)))) 
        ES = q_es * sigma 
        ES = max(0,ES)

        # rounding positions for better presentation
        old_pos = np.around(self.positions,4) 
        new_pos =  np.around(optimized.x, 4) 
        Diff = new_pos - old_pos 

        # suppressing scientific notation for print
        np.set_printoptions(suppress=True)

        # preparing output dictionary based on the volatility model
        MIN_PROCESS = {"var" : round(VaR, 4),
                       "es" : round(ES, 4),
                       "T" : self.interval}
        
        if vol == "simple":
            MIN_PROCESS.update({"OLDPOS" : old_pos,
                                "NEWPOS" : new_pos,
                                "DIFFER" : Diff })
        elif vol == "garch":
            MIN_PROCESS.update({"p" : p,
                                "q" : q, 
                                "OLDPOS" : old_pos,
                                "NEWPOS" : new_pos,
                                "DIFFER" : Diff})
        else:  # for "ewma"
            MIN_PROCESS.update({"lambda_ewma": lambda_ewma,
                                "OLDPOS" : old_pos,
                                "NEWPOS" : new_pos,
                                "DIFFER" : Diff})

        return MIN_PROCESS

    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_stocks_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray, var: float, alpha_test: float =0.05):

        """
        Main
        ----
        Performs the Kupiec test of unconditional coverage to verify the accuracy of the VaR model.
        The Kupiec test compares the number of exceptions (instances where the losses exceed the VaR estimate)
        with the expected number of exceptions. Under the null hypothesis, the number of exceptions is consistent
        with the VaR confidence level.

        Parameters
        ----------
        - num_stocks_s: An array indicating the quantity of each security held.
        - S0_initial_s: The starting prices of the securities at the onset of the test_return array.
        - test_returns: The returns to use for testing. This must have the same frequency of the returns array and correspond to one
        period of analysis.
        - var: The Value-at-Risk level to test (in value).
        - alpha_test: The significance level of the test. Default is 0.05.

        Returns
        -------
        - dict: A dictionary containing:
            - 'test': The name of the test ('Kupiec').
            - 'var': The VaR level that was tested.
            - 'nObs': The total number of observations.
            - 'nExc': The number of exceptions observed.
            - 'alpha': The theoretical percentage of exceptions.
            - 'actualExc%': The actual percentage of exceptions.
            - 'LR': The likelihood ratio of the test.
            - 'cVal': The critical value of the test at the given significance level.
            - 'rejNull': A boolean indicating whether to reject the null hypothesis (true if the null hypothesis is rejected).
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        import pandas as pd ; import numpy as np ; from scipy.stats import chi2
    
        from pyriskmgmt.inputsControlFunctions import (check_num_stocks_s, check_S0_initial_s, validate_returns_port,check_var,
                                                       check_alpha_test)

        # check procedure
        num_stocks_s = check_num_stocks_s(num_stocks_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s)
        test_returns = validate_returns_port(test_returns) ; var = check_var(var) ; alpha_test = check_alpha_test(alpha_test)

        if self.interval == 1: 

            test_returns = test_returns

        if self.interval > 1: 

            test_returns = pd.DataFrame(test_returns)

            # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

            def reshape_and_aggregate_returns(returns, interval):
                returns = returns.values
                num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
                reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                
                return aggregated_returns
            
            test_returns = test_returns.apply(lambda x: reshape_and_aggregate_returns(x, self.interval)) ; test_returns = test_returns.values

        # now we can proceed ------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        # creating a copy of the test_returns array
        portfolio = np.copy(test_returns)

        # calculating initial portfolio positions by multiplying the number of each stock with its initial price
        starting_positions = num_stocks_s * S0_initial_s

        # converting the starting positions into a list of floats
        new_positions = [float(num) for num in starting_positions]

        # iterating through each time period (row) in the portfolio
        for i in range(portfolio.shape[0]):
            # multiplying the returns for each security by its current position
            portfolio[i, :] *= new_positions
            # updating the position for the next period by adding the gains/losses
            new_positions += portfolio[i, :]

        # summing up the portfolio returns for each period
        port_returns = portfolio.sum(axis=1)

        # filtering out the negative returns (losses) from the portfolio returns
        losses = port_returns[port_returns < 0]

        # counting the number of times the loss exceeded the negative VaR value
        num_exceptions = np.sum(losses < - VaR)

        # total number of observations -----------------------------------------------------------------------------------

        num_obs = len(port_returns)

        # % exceptions -------------------------------------------------------------------------------------------------------

        per_exceptions = round(num_exceptions/num_obs,4)

        p = self.alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : self.alpha,                   # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # CHRISTOFFESEN TEST <<<<<<<<<<<<

    def christoffersen_test(self,num_stocks_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray, var: float, alpha_test: float =0.05):

        """
        Main
        -----
        Performs the Christoffersen test of independence to verify the accuracy of the VaR model.
        The Christoffersen test assesses the independence of exceptions. Under the null hypothesis, exceptions
        (instances where the losses exceed the VaR estimate) are independently distributed over time.

        Parameters
        ----------
        - num_stocks_s: An array indicating the quantity of each security held.
        - S0_initial_s: The starting prices of the securities at the onset of the test_return array.
        - test_returns: The returns to use for testing. This must have the same frequency of the returns array and correspond to one
        period of analysis.
        - var: The Value-at-Risk level to test (in value).
        - alpha_test: The significance level of the test. Default is 0.05.

        Returns
        --------
        - dict: A dictionary containing:
            - 'test': The name of the test ('Christoffersen').
            - 'var': The VaR level that was tested.
            - 'LR': The likelihood ratio of the test.
            - 'cVal': The critical value of the test at the given significance level.
            - 'rejNull': A boolean indicating whether to reject the null hypothesis (true if the null hypothesis is rejected).
        """

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd ; import numpy as np ; from scipy.stats import chi2
    
        from pyriskmgmt.inputsControlFunctions import (check_num_stocks_s, check_S0_initial_s, validate_returns_port,check_var,
                                                       check_alpha_test)

        # check procedure
        num_stocks_s = check_num_stocks_s(num_stocks_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s)
        test_returns = validate_returns_port(test_returns) ; var = check_var(var) ; alpha_test = check_alpha_test(alpha_test)

        if self.interval == 1: ##########

            test_returns = test_returns

        if self.interval > 1: ##########

            test_returns = pd.DataFrame(test_returns)

            # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

            def reshape_and_aggregate_returns(returns, interval):
                returns = returns.values
                num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
                reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
                
                return aggregated_returns
            
            test_returns = test_returns.apply(lambda x: reshape_and_aggregate_returns(x, self.interval)) ; test_returns = test_returns.values

        # now we can proceed ------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        # creating a copy of the test_returns array
        portfolio = np.copy(test_returns)

        # calculating initial portfolio positions by multiplying the number of each stock with its initial price
        starting_positions = num_stocks_s * S0_initial_s

        # converting the starting positions into a list of floats
        new_positions = [float(num) for num in starting_positions]

        # iterating through each time period (row) in the portfolio
        for i in range(portfolio.shape[0]):
            # multiplying the returns for each security by its current position
            portfolio[i, :] *= new_positions
            # updating the position for the next period by adding the gains/losses
            new_positions += portfolio[i, :]

        # summing up the portfolio returns for each period
        port_returns = portfolio.sum(axis=1)
        
        exceptions = port_returns < - VaR

        # shifting exceptions to get previous day's exceptions
        prev_exceptions = np.roll(exceptions, shift=1)

        # dropping the first row (which is NaN because of the shift)
        exceptions = exceptions[1:] ; prev_exceptions = prev_exceptions[1:]

        # calculating transition counts
        T_00, T_01 = np.sum((prev_exceptions == 0) & (exceptions == 0)), np.sum((prev_exceptions == 0) & (exceptions == 1))
        T_10, T_11 = np.sum((prev_exceptions == 1) & (exceptions == 0)), np.sum((prev_exceptions == 1) & (exceptions == 1))

        # estimating conditional probabilities
        p_b0, p_b1 = T_01 / (T_00 + T_01), T_11 / (T_10 + T_11)

        # calculating likelihood ratio
        p = self.alpha
        likelihood_ratio = -2 * (np.log((1 - p) ** (T_00 + T_10) * p ** (T_01 + T_11)) - np.log((1 - p_b0) ** T_00 * p_b0 ** T_01 * (1 - p_b1) ** T_10 * p_b1 ** T_11))

        # critical value from chi-squared distribution
        critical_value = chi2.ppf(1 - alpha_test, df=1)

        # rejecting the null hypothesis if likelihood ratio is higher than the critical value
        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Christoffersen",
            "var" : var,
            "LR" : round(likelihood_ratio, 5),
            "cVal" : round(critical_value, 5),
            "rejNull" : reject_null
        }

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # COMBINED TEST <<<<<<<<<<<<

    def combined_test(self,num_stocks_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray, var: float, alpha_test: float =0.05):

        """
        Main
        -----
        Performs a combined Kupiec and Christoffersen test to assess the accuracy of the VaR model.
        The combined test applies both the Kupiec test of unconditional coverage and the Christoffersen test of independence to the exceptions. 
        The null hypothesis is that the VaR model is accurate.

        Parameters
        ----------
        - num_stocks_s: An array indicating the quantity of each security held.
        - S0_initial_s: The starting prices of the securities at the onset of the test_return array.
        - test_returns: The returns to use for testing. This must have the same frequency of the returns array and correspond to one
        period of analysis.
        - var: The Value-at-Risk level to test (in value).
        - alpha_test: The significance level of the test. Default is 0.05.

        Returns
        -------
        - dict: A dictionary containing:
            - 'test': The name of the test ('Kupiec_Christoffersen').
            - 'var': The VaR level that was tested.
            - 'LR': The combined likelihood ratio of the test.
            - 'cVal': The critical value of the test at the given significance level.
            - 'rejNull': A boolean indicating whether to reject the null hypothesis (true if the null hypothesis is rejected).
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import chi2

        # now we can proceed ------------------------------------------------------------------------------------------------------------------
        
        VaR = round(var,4)

        # perform individual tests - no need for retuns aggregation here ---- it will be handled by both kupiec_test and christoffersen_test

        kupiec_results = self.kupiec_test(num_stocks_s = num_stocks_s, S0_initial_s = S0_initial_s, test_returns = test_returns,
                                           var = VaR,alpha_test=alpha_test)
        christoffersen_results = self.christoffersen_test(num_stocks_s = num_stocks_s, S0_initial_s = S0_initial_s,
                                                           test_returns = test_returns, var = VaR,alpha_test=alpha_test)

        # getting likelihood ratios
        kupiec_lr = kupiec_results["LR"] ; christoffersen_lr = christoffersen_results["LR"]

        # calculating combined likelihood ratio
        combined_likelihood_ratio = kupiec_lr + christoffersen_lr

        # critical value from chi-squared distribution with 2 degrees of freedom
        critical_value = chi2.ppf(1 - alpha_test, df=2)

        # if the combined likelihood ratio is greater than the critical value, then we reject the null hypothesis
        reject_null = combined_likelihood_ratio > critical_value

        return {
            "test" : "Kupiec_Christoffersen",
            "var" : VaR,
            "LR" : round(combined_likelihood_ratio, 5),
            "cVal" : round(critical_value, 5),
            "rejNull" : reject_null
        }

    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 
























