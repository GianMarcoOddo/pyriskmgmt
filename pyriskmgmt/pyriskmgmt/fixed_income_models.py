
# ------------------------------------------------------------------------------------------
# Author: GIAN MARCO ODDO
# Module: fixed_income_models
#     - Part of the pyriskmgmt package.
# Contact Information:
#   - Email: gian.marco.oddo@usi.ch
#   - LinkedIn: https://www.linkedin.com/in/gian-marco-oddo-8a6b4b207/
#   - GitHub: https://github.com/GianMarcoOddo
# Feel free to reach out for any questions or further clarification on this code.
# ------------------------------------------------------------------------------------------

from typing import Union, List ; import numpy as np ; import pandas as pd

"""
####################################################################### NON PARAMETRIC APPROACH ##############################################################################
"""

##########################################################################################################################################################################
##########################################################################################################################################################################
## 1. FixedIncomeNprmSingle ###################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class FixedIncomeNprmSingle:

    """
    Main
    ----
    This class is designed to model Non-Parametric Risk Measures for a single fixed-income position. The class performs calculations
    for the Value at Risk (VaR) and Expected Shortfall (ES) using either the quantile or bootstrap method.
    Furthermore, this class also allows for fitting the generalized Pareto distribution (GPD) to the right-hand tail 
    of the loss distribution to perform an Extreme Vlue Theory (EVT) analysis.

    The interval of the VaR/Es calculation follows the frequency of the provided returns.
    
    Initial Parameters
    ------------------
    - returns: Array with returns for the single fixed-income position.
    - position: The position on the single fixed-income security. A positive value indicates a long position, a negative value indicates a short position.  
      - The position you possess can be determined by multiplying today's price of the fixed-income security by the quantity you hold.
    - alpha: The significance level for the risk measure calculations (VaR and ES). Alpha is the probability of the
    occurrence of a loss greater than or equal to VaR or ES. Default is 0.05.
    - method: The method to calculate the VaR and ES. The possible values are "quantile" or "bootstrap". Default is "quantile".
    - n_bootstrap_samples: The number of bootstrap samples to use when method is set to "bootstrap". Default is 10000.

    Methods
    --------
    - summary(): Returns a dictionary containing the summary of the calculated risk measures.
    - evt(): Performs an Extreme Value Theory (EVT) analysis by fitting the Generalized Pareto Distribution 
    to the right-hand tail of the loss distribution. Returns the calculated VaR and ES under EVT along with the shape and scale parameters of the fitted GPD.

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

    # this class replicates the EquityNprmSingle class from the pyriskmgmt.equity_models module
    # it is included here for potential custom developments 

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
        Calculates and returns a summary of various risk measures for a given single fixed-income position.
        This method calculates the maximum loss, maximum excess loss (over VaR), maximum loss over VaR, 
        and the Expected Shortfall over VaR. These measures are used to give an overall picture of the risk 
        associated with the bond. All these measures are rounded off to 4 decimal places for readability.
        
        Returns
        -------
        - summary : dict : A dictionary containing the following key-value pairs:
            - 'var': The Value at Risk (VaR) for the fixed-income position.
            - 'maxLoss': The maximum loss for the fixed-income position.
            - 'maxExcessLoss': The maximum excess loss (over VaR) for the fixed-income position.
            - 'maxExcessLossOverVar': The ratio of the maximum excess loss to VaR.
            - 'es': The Expected Shortfall for the fixed-income position.
            - 'esOverVar': The ratio of the Expected Shortfall to VaR.
            
        Note
        ---- 
        The method calculates these measures differently depending on whether the position on the bond is long (positive), 
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
        The method calculates these measures differently depending on whether the position on the fixed-income security is long (positive), 
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
## 2. FixedIncomeNprmPort ###############################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

class FixedIncomeNprmPort:

    """
    Main
    ----
    The `FixedIncomeNprmPort` class is a non-parametric risk management class designed to provide risk measures for a portfolio of fixed-income positions.
    The non-parametric approach does not rely on any assumptions regarding the underlying distribution of the fixed-income securities returns. 

    The interval of the VaR/Es calculation follows the frequency of the provided returns.

    Initial Parameters
    ------------------
    - returns: A numpy array of fixed-income positions returns.
    - positions: A list of fixed-income positions.
      - For each bond: the position you possess can be determined by multiplying the bond's present-day price by the quantitiy you hold.
    - alpha: The significance level for VaR and ES calculations. Default is 0.05.
    - method: A method to be used for the VaR and ES calculations. The options are "quantile" for quantile-based VaR and ES calculation or "bootstrap" for a 
    bootstrap-based calculation. Default is "quantile".
    - n_bootstrap_samples: The number of bootstrap samples to be used when the bootstrap method is selected. This is ignored when the quantile method is selected.
    Default is 10000.

    Methods
    --------
    - summary(): Provides a summary of the risk measures, including VaR, Expected Shortfall (ES), Maximum Loss, Maximum Excess Loss, and ratios of these measures.
    - evt(): Provides risk measures using Extreme Value Theory (EVT). This is a semi-parametric approach that fits a Generalized Pareto Distribution (GPD) to the tail 
    of the loss distribution.
    - MargVars(): Provides the marginal VaRs for each bond position in the portfolio. This is calculated as the change in portfolio VaR resulting from a small 
    change in the position of a specific fixed-income security.
    
    Attributes
    ---------
    - var: The calculated Value at Risk (VaR) for the portfolio.
    - es: The calculated Expected Shortfall (ES) for the portfolio.
    - port_returns: Cumulative returns of the portfolio based on the fixed-income positions returns. 

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # this class replicates the EquityNprmPort class from the pyriskmgmt.equity_models module
    # it is included here for potential custom developments 

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
            raise ValueError("The number of derivatives != the number of positions")

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
        Computes various risk metrics for the portfolio of bonds and returns them in a dictionary.
        
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
        Estimates the Value at Risk (VaR) and Expected Shortfall (ES) using the Extreme Value Theory (EVT).
        EVT is used for assessing risk for extreme events. This function implements both quantile-based and bootstrap-based methods for EVT.

        Parameters
        ----------
        - alpha: Significance level for VaR and ES. The default is 0.025.
        - quantile_threshold: Threshold for separating the tail from the rest of the data in the GPD fitting. The default is 0.95.

        Returns
        -------
        - VaR_ES (dict): A dictionary containing the estimated VaR, ES, shape parameter (xi), and scale parameter (beta), as well as the method used and confidence level. 
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
        Computes the Marginal Value at Risk (MVaR) for each bond position in the portfolio using a non-parametric approach.
        MVaR measures the rate of change in the portfolio's VaR with respect to a small change in the position of a specific fixed-income security. 
        This method works by perturbing each bond position by a small amount in the same direction (proportional to a scale_factor), calculating the new VaR,
        and measuring the difference from the original VaR. The result is an array of MVaRs, one for each fixed-income position.
        - New position = old position +/- (old position * scale_factor) for each position.

        Parameters
        ----------
        - scale_factor: The scale factor used to adjust the bond positions for MVaR calculations. Default is 0.1.

        Returns
        -------
        - A dictonary with: 
          - The calculated VaR
          - A numpy array representing the MVaR for each fixed-income position in the portfolio. MVaRs are rounded to four decimal places.

        Note
        -----
        This method can only be used when the 'method' attribute of the class instance is set to 'quantile'. For the 'bootstrap' method, it raises a ValueError.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np

        if self.method == "bootstrap": 
            raise ValueError("The 'MargVars' method can only be used when the 'self.method', in the `FixedIncomeNprmPort` class, is set to 'quantile'.")

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

"""
####################################################################### MODEL APPROACH #################################################################################
"""

###############################################################################################################################################################################
###############################################################################################################################################################################
## 3. YieldCurve #############################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

class YieldCurve:

    """
    Main
    ----
    This class provides functionalities for fetching, plotting, and interpolating the U.S. Treasury yield curve data. 
    The yield curve represents the relationship between interest rates and the time to maturity of a debt for a borrower in a given currency.
    The curve is a critical indicator in financial markets, often used as a benchmark for other interest rates, and provides insights into future economic conditions.

    - https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value

    - The downloaded yield curve will have a DataFrame structure, where the columns are time to maturity in months and the values are interest rates in percentages. 
    For example, a value of 5.37 represents an interest rate of 5.37%.

    ```
    | Date      |   1  |   2  |   3  |  6   |  12  |  24  |  36  |  60  |  84  |  120 |  240 |  360 |
    |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
    | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
    | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
    ...
    | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
    | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

    ```
    Zero-Coupon Bond (ZCB) Pricing Example
    ----------------------------------------
    One application of this yield curve data is in pricing zero-coupon bonds (ZCBs). 
    - Let fix the yield for a 24-month maturity at 4.87%, and the face value of the ZCB at $1000.
    - The price of a ZCB, calculated using the well-known formula, would be:
      - Price = Face Value / (1 + (Yield / 100))^Time to Maturity = $1000 / (1 + (4.87 / 100))^2 = $909.28
    
    Initial Parameters
    ------------------
    - years_back : Number of years back for which to fetch yield curve data. Default is set to 2.

    Methods:
    --------
    - PlotYieldCurve(): Plots the yield curve using seaborn and matplotlib. Accepts optional figsize argument to adjust the size of the plot.
    - InterpolateYieldCurve(): Interpolates the yield curve using either linear interpolation or cubic spline. 
    Optionally only interpolates the yield curve for today.

    Attributes
    ----------
    yield_curve_df : DataFrame containing fetched yield curve data.
    today : Today's date in MM/DD/YYYY format.
    yield_curve_today : Series containing yield curve data for today.
    
    Example:
    --------
    >>> from pyriskmgmt.fixed_income_models import YieldCurve
    >>> y = YieldCurve(years_back=2)
    >>> y.PlotYieldCurve(figsize = (12,3))
    >>> interpolated_data = y.InterpolateYieldCurve(method="cubic_spline", yield_curve_today = True)

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, years_back: int = 2):

        from pyriskmgmt.inputsControlFunctions import check_years_back

        # check procedure 
        self.years_back = check_years_back(years_back)

        self.yield_curve_df = None ; self.today = None ; self.yield_curve_today = None 

        self.fit()

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")
        
        # importing required libraries
        from datetime import datetime, timedelta ; import requests ; import pandas as pd ; from bs4 import BeautifulSoup
        
        # -------------------------------------------------------------------------------------------------------------------------

        # defining a function to fetch yield curve data for a given year
        def fetch_yield_curve_for_year(year):
            # constructing the URL for the web request
            url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}'
            
            try:
                # sending a GET request to the URL
                response = requests.get(url)
                # checking if the request was successful
                response.raise_for_status()
            except requests.RequestException as e:
                # throwing an exception if the request was not successful
                raise Exception("Failed to download data. Please reduce the self.years_back parameter or check your internet connection") from e

            # parsing the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # finding the table element in the HTML
            table = soup.find('table')
            if table is None:
                # throwing an exception if the table was not found
                raise Exception("Failed to find data table. Please reduce the self.years_back parameter.")
            
            # converting the table to a pandas DataFrame
            df = pd.read_html(str(table))[0]
            df = pd.concat([df.iloc[:, :1], df.iloc[:, -13:]], axis=1)
            
            # dropping the "4 Mo" column and setting "Date" as the index
            df.drop("4 Mo", axis=1, inplace=True)
            df.set_index("Date", inplace=True)
            
            return df
        
        # -------------------------------------------------------------------------------------------------------------------------
        
        # importing custom DotPrinter class for progress animation
        from pyriskmgmt.SupportFunctions import DotPrinter
        
        # initializing the DotPrinter class and starting the animation
        my_dot_printer = DotPrinter(f"Fetching the last {self.years_back}-year Yield Curve from treasury.gov")
        my_dot_printer.start()

        # initializing variables to hold the DataFrames and calculate the number of months
        dfs = []
        current_datetime = datetime.now() 
        
        # extracting the year from the current date
        year = current_datetime.year

        # looping for self.years_back years to fetch data
        for i in range(self.years_back):
            # fetching the yield curve data for the current year
            df = fetch_yield_curve_for_year(year)
            # sorting the DataFrame by index
            df.sort_index(inplace=True, ascending=True)
            # prepending the DataFrame to the list of DataFrames
            dfs.insert(0, df)
            
            year -= 1

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
        # concatenating all DataFrames into one DataFrame, dropping any duplicates, and then dropping columns with NaN
        self.yield_curve_df = pd.concat(dfs).drop_duplicates().dropna(axis=1, how='any')

        # Transform column names
        transformed_columns = []
        for col in self.yield_curve_df.columns:
            if "Mo" in col:
                # Extract the numeric part and divide by 12
                num = int(col.split(" ")[0])
                transformed_columns.append(num)
            elif "Yr" in col:
                # Extract the numeric part
                num = int(col.split(" ")[0])
                transformed_columns.append(num * 12)
            else:
                transformed_columns.append(col)

        self.yield_curve_df.columns = transformed_columns

        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # stopping the animation
        my_dot_printer.stop()
        print("\rFetching Process --->  Done", end=" " * 100)
        
        # getting today's date in MM/DD/YYYY format
        self.today = datetime.now().strftime("%m/%d/%Y")
        
        # initializing the date object for finding today's yield curve
        date_object = datetime.strptime(self.today, '%m/%d/%Y')

        # looping until today's yield curve is found
        while True:
            # converting the date object to string format
            date_str = date_object.strftime('%m/%d/%Y')
            
            # checking if the yield curve for today exists
            if date_str in self.yield_curve_df.index:
                self.yield_curve_today = self.yield_curve_df.loc[date_str]
                break
            # checking if the yield curve data is unavailable for today and any previous day
            elif date_object < datetime.strptime(self.yield_curve_df.index.min(), '%m/%d/%Y'):
                raise Exception("No available yield curve data for today or any prior date.")
            else:
                # decrementing the date by one day for the next iteration
                date_object -= timedelta(days=1)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def PlotYieldCurve(self, figsize: tuple = (12, 3)):

        """
        Main
        -----
        Plots the yield curve data for today using seaborn and matplotlib.
        This method takes the yield_curve_today Series attribute to plot a line graph, representing the yield curve for the 
        specified date. The x-axis indicates the time to maturity in months, and the y-axis indicates the yield in percentage.
        
        Parameters
        ----------
        figsize : A tuple specifying the width and height in inches of the plot. The default is (12, 3), but can be changed to adjust the size.
            
        Returns
        -------
        None: Displays the yield curve plot for today but does not return any value.
        
        Example
        -------
        >>> from pyriskmgmt.fixed_income_models import YieldCurve
        >>> y = YieldCurve(years_back=2)
        >>> y.PlotYieldCurve(figsize=(15, 4))
        
        This will display a plot of the yield curve for today with the dimensions of 15x4 inches.
        """

        import warnings ; warnings.filterwarnings("ignore")

        # importing required libraries for plotting
        import seaborn as sns ; import matplotlib.pyplot as plt

        # initializing the figure and setting its size
        plt.figure(figsize=figsize)
        sns.set_style("whitegrid")

        # creating the line plot using Seaborn
        sns.lineplot(x = self.yield_curve_today.index, y = self.yield_curve_today.values)

        # adding labels and title
        plt.title(f'Yield Curve for {self.today} - from treasury.gov')
        plt.xlabel('Months')
        plt.ylabel('Yield (%)')

        # rotating x-axis labels for better readability
        plt.xticks(rotation=45)

        # displaying the plot
        plt.show()

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def InterpolateYieldCurve(self, method: str = "cubic_spline", yield_curve_today: bool = False):

        """
        Main
        ----
        Interpolates the yield curve data to provide more granular time-to-maturity points. 
        Interpolation can be performed either for the yield curve of today or for all historical yield curve data in the DataFrame.
        The method offers two interpolation techniques: linear interpolation and cubic spline interpolation.
        
        Cubic Spline Framework
        -----------------------
        When the 'cubic_spline' method is chosen, the interpolation is done using cubic splines. Cubic spline is a piecewise function defined by 
        cubic polynomials, which are smooth and continuous. Note that while cubic spline interpolation tends to produce smoother curves, it's 
        important to remember that these are still approximations and may not perfectly represent the true underlying yield curve.

        Use-Case in Risk Management
        ---------------------------
        The primary purpose of this class and its methods is for risk management calculations like Value at Risk (VaR) and Expected Shortfall (ES).
        For these purposes, the approximation produced by cubic spline or linear interpolation is generally acceptable. 
        However, it's worth mentioning that other classes within this risk management package do accept manually-provided monthly yield curves as an 
        argument. This provides the users with the flexibility to use a more accurate yield curve if they have one. This package is intended for risk 
        management, not for arbitrage strategies.

        Parameters
        ----------
        - method: The interpolation method to use. Can be either "linear" or "cubic_spline". Default is "cubic_spline".
        - yield_curve_today: Whether to interpolate only today's yield curve. Default is False, which interpolates the entire yield curve DataFrame.
            
        Returns
        -------
        interpolated_df: A DataFrame containing the interpolated yield curve. If 'yield_curve_today' is True, the DataFrame will contain 
        only today's interpolated yield curve. Otherwise, it will contain interpolated yield curves for all dates.
        
        Example
        -------
        >>> from pyriskmgmt.fixed_income_models import YieldCurve
        >>> y = YieldCurve(years_back=2)
        >>> interpolated_data = y.InterpolateYieldCurve(method="cubic_spline", yield_curve_today=True)
        This will return a DataFrame containing the interpolated yield curve for today, using cubic spline interpolation.
        
        Note
        ----
        The generated yield curve is a monthly yield curve, meaning that yield values are provided for each month from 1 to 360. 
        """

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd ; import numpy as np ; from scipy.interpolate import CubicSpline

        from pyriskmgmt.inputsControlFunctions import (check_method_yield_curve, check_yield_curve_today_bool)

        # check procedure
        method = check_method_yield_curve(method) ; check_yield_curve_today_bool(yield_curve_today)

        # defining new columns for the interpolated DataFrame
        new_columns = np.arange(1, 361, 1) # every month --- this package needs the yield curve with a montly frequency

        # yield_curve_today //////////////////////////////////////////////////////////////////////////////////////////

        if yield_curve_today:
            # initializing the DataFrame with new columns
            interpolated_df = pd.DataFrame(columns=new_columns)
            
            # getting the yield curve data for the most recent date
            row = self.yield_curve_df.iloc[-1]
            
            # converting index and values to lists
            existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()

            # applying linear or cubic spline interpolation based on the method chosen
            if method == "linear":
                interpolated_yields = np.interp(new_columns, existing_terms, existing_yields)
            elif method == "cubic_spline":
                cs = CubicSpline(existing_terms, existing_yields) ; interpolated_yields = cs(new_columns)

            # adding the interpolated yields to the DataFrame and converting the DataFrame to float dtype
            interpolated_df.loc[row.name] = interpolated_yields ; interpolated_df = interpolated_df.astype(float)
            
            return interpolated_df
        
        # not yield_curve_today //////////////////////////////////////////////////////////////////////////////////////////

        else:
            # initializing DataFrame for interpolating yield curves for all dates
            interpolated_df = pd.DataFrame(index=self.yield_curve_df.index, columns=new_columns)

            # looping through all rows of the existing yield curve DataFrame
            for i, row in self.yield_curve_df.iterrows():
                # converting index and values to lists
                existing_terms = row.index.astype(int).tolist()
                existing_yields = row.values.tolist()

                # applying linear or cubic spline interpolation based on the method chosen
                if method == "linear":
                    interpolated_yields = np.interp(new_columns, existing_terms, existing_yields)
                elif method == "cubic_spline":
                    cs = CubicSpline(existing_terms, existing_yields)
                    interpolated_yields = cs(new_columns)

                # adding the interpolated yields to the DataFrame
                interpolated_df.loc[i] = interpolated_yields

            # converting the DataFrame to float dtype
            interpolated_df = interpolated_df.astype(float)

            return interpolated_df
        
##########################################################################################################################################################################
##########################################################################################################################################################################
## 4. FixedIncomePrmSingle ###############################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class FixedIncomePrmSingle:
    
    """
    Main
    ----
    This class provides the functionalities to price fixed-income securities, specifically bonds, and calculate their 
    associated risk metrics such as duration, yield-to-maturity, Value-at-Risk (VaR), and Expected Shortfall (ES). 
    
    Important
    ---------
    - Due to the constraint in how the 'maturity' parameter is designed, it can only accept either an integer value representing full years or a decimal value
    like 0.5 to indicate semi-annual periods. 
    - As a result of this constraint, the class is designed to assist in evaluating the risk and return of a bond position at the time of the bond's issuance or
    just after a coupon payment in the case of a coupon-bearing bond. For a zero-coupon bond, it is meant to be used either at the time of issuance or
    every six months thereafter.
    The class supports both zero-coupon bonds (ZCB) and coupon-bearing bonds, with the option for annual or semi-annual payments.

    ************************************************************************************************************
    For other types of Bond this module offers a NON-PARAMETRIC APPROACH: FixedIncomeNprmSingle() 
    ************************************************************************************************************

    Purpose
    -------
    This class aims to provide a solid foundation for understanding and analyzing fixed-income securities. It is designed to serve two main audiences:

    1. Those who are looking to further develop or integrate more complex calculations and risk management strategies. 
    This class offers a reliable basis upon which additional features and functionalities can be built.
    
    2. Those who intend to use the class within its given boundaries for straightforward pricing and risk assessment of fixed-income securities.
    For this audience, the class provides ready-to-use methods for calculating key metrics like bond price, duration, yield-to-maturity, Value-at-Risk (VaR), and Expected Shortfall (ES).

    - By catering to both of these needs, the class offers a flexible and robust starting point for a range of fixed-income analysis and risk management tasks.

    Initial Parameters
    ------------------
    - yield_curve_today : The DataFrame containing the current yield curve, with terms in months as columns and the yield rates as row values.
    The yield rates should be represented as a percentage, e.g., 3% should be given as 3.

    ```
    | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
    |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
    | 08/29/2023| 5.54 | 5.53 | 5.52 | 5.51 | 5.51 | 4.87 | 4.56 | 4.26 | 4.21 | 4.19 | 4.18 | 4.19 |
    ```

    - maturity: The time to maturity of the bond in years. Must be either an integer value for full years or a decimal value like 0.5 to indicate semi-annual periods.
    Default is 1.
    - num_assets: Number of identical assets/bonds for the calculation.
    Default is 1.
    - face_value: The face value of the bond.
    Default is 1000.
    - is_zcb: Indicator to specify whether the bond is a zero-coupon bond. Default is True.
    - coupon_rate: The annual coupon rate of the bond represented as a percentage (e.g., 3% should be given as 3) (required if is_zcb is set to False).
    Default is None.
    - semi_annual_payment: Indicator to specify if the bond pays coupons semi-annually (required if is_zcb is set to False).
    Default is None.

    Methods
    -------
    - DurationNormal(): Calculates the Value-at-Risk (VaR) and Expected Shortfall (ES) of a bond position using a normal distribution model.
    This method assumes a normal distribution of yield changes and uses the modified duration of the bond
    to approximate the change in bond price for a given change in yield.
    - HistoricalSimulation(): In this approach, historical data is used to generate a distribution of potential outcomes,
    which can be more realistic compared to model-based approaches like the Duration-Normal method.

    Attributes
    ----------
    - bond_price: The price of the bond calculated based on either zero-coupon or coupon-bearing.
    - duration: The Macaulay duration of the bond (in years).
    - modified_duration: The modified duration of the bond (in years).
    - yield_to_maturity: Yield to maturity of the bond (yearly).

    Exceptions
    ----------
    - Raises an exception if the coupon_rate and semi_annual_payment are not provided when is_zcb is set to False.
    - Raises an exception if the maturity provided is beyond the highest maturity available in the yield_curve_today DataFrame.
    - Raises an exception if the maturity provided ends with .5, is_zcb is False and semi_annual_payment is not True.

    Note
    ----
    The yield curve provided should be a monthly yield curve, with maturity ranging from 1 to n months.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************
    
    def __init__(self, yield_curve_today : pd.DataFrame,              # not in decimal point. For instance: for 3% it should be 3.
                       maturity: int = 1,                             # in years
                       num_assets: int = 1,
                       face_value: int = 1000,
                       is_zcb: bool = True,     
                       coupon_rate: int = None,                       # not in decimal point. For instance: for 3% it should be 3.
                       semi_annual_payment: bool = None):
        

        from pyriskmgmt.inputsControlFunctions import (check_yield_curve_today, check_maturity, check_num_assets,check_face_value, check_is_zcb,
                                                       check_coupon_rate, check_semi_annual_payment)

        # check procedure
        self.yield_curve_today = check_yield_curve_today(yield_curve_today) ; self.maturity = check_maturity(maturity) ; self.num_assets = check_num_assets(num_assets)
        self.face_value = check_face_value(face_value) ; check_is_zcb(is_zcb) ; self.is_zcb = is_zcb

        # if we have a coupound bond /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if is_zcb == False:
            if coupon_rate is None and semi_annual_payment is None: 
                raise Exception("Both coupon_rate and semi_annual_payment must be provided when is_zcb is False.")
            if coupon_rate is None or semi_annual_payment is None: 
                raise Exception("Either coupon_rate or semi_annual_payment is missing. Both are required when is_zcb is False.")
            if coupon_rate is not None and semi_annual_payment is not None: 
                # check procedure
                self.coupon_rate = check_coupon_rate(coupon_rate) ; check_semi_annual_payment(semi_annual_payment) ; self.semi_annual_payment = semi_annual_payment
            if (self.maturity * 2) % 2 != 0 and not self.semi_annual_payment:
                raise Exception("If maturity is in fractions of .5 years, then semi_annual_payment must be set to True.")
    
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (self.maturity * 12) > self.yield_curve_today.columns[-1]:
            raise Exception(f"""
            The provided maturity for the bond is {maturity} year(s), which is equivalent to {maturity * 12} months.
            This method requires the user to provide a 'yield_curve_today' with monthly frequency.
            The highest maturity available in the 'yield_curve_today' DataFrame is {self.yield_curve_today.columns[-1]} months.
            Therefore, the method does not have the required yield for the given maturity.
            """)

        self.bond_price = None  ; self.duration = None ; self.modified_duration = None ; self.yield_to_maturity = None

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.optimize import newton ; import pandas as pd

        # -----------------------------------------------------------------------------------------------------------
        # Note:
        # - The yield curve should be a pandas DataFrame with the following characteristics:
        # - The index should represent today's date or the last available date.
        # - Columns should start from 1 and go onward, representing the terms in months.
        # - The DataFrame should contain a single row with interest rates, corresponding element-wise to the columns.
        # - Interest rates should be in the format like 5.56, 5.22, etc., without decimal points.
        # ------------------------------------------------------------------------------------------------------------

        # checking if the bond is a zero-coupon bond
        if self.is_zcb == True:

            # fetching the last yield value from the DataFrame for the given maturity
            exact_yield = self.yield_curve_today.loc[self.yield_curve_today.index[-1], self.maturity * 12]

            # calculating the bond's price using the zero-coupon bond formula
            self.bond_price = self.face_value / ( (1 + (exact_yield/100))  ** self.maturity)

            # setting duration equal to the bond's maturity for a zero-coupon bond
            self.duration = self.maturity 

            # storing the bond's yield to maturity
            self.yield_to_maturity = round(exact_yield/100,6)

            # calculating and storing the modified duration
            self.modified_duration = round(self.duration / (1 + self.yield_to_maturity), 4)

        # checking if the bond is not a zero-coupon bond
        if self.is_zcb == False:

            # checking if the bond pays coupons annually
            if not self.semi_annual_payment:
                paymentSchedules = np.arange(1,(self.maturity + 1),1) * 12 

            # checking if the bond pays coupons semi-annually
            if self.semi_annual_payment:

                # setting the payment schedule for semi-annual payments
                paymentSchedules = np.arange(0.5,(self.maturity + 0.5),0.5) * 12 

                # adjusting the coupon rate for semi-annual payments
                self.coupon_rate = self.coupon_rate / 2 

            # initializing an array to store bond cash flows
            cashFlows = np.zeros(len(paymentSchedules))

            # computing bond cash flows
            for i in range(len(cashFlows)):
                if i != (len(cashFlows)-1):
                    cashFlows[i] = self.face_value * (self.coupon_rate/100)
                else:
                    cashFlows[i] = (self.face_value * (self.coupon_rate/100)) + self.face_value

            # initializing an array to store present values of cash flows
            pvCashFlow = np.zeros(len(paymentSchedules))

            # computing the present value of cash flows
            for index, value in enumerate(paymentSchedules):
                exact_yield = self.yield_curve_today.loc[self.yield_curve_today.index[-1], value]
                pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield/100)))**(value/12)) 

            # calculating and storing the bond's price
            self.bond_price = np.sum(pvCashFlow)

            # defining a function to compute the difference between NPV and bond price
            def npv_minus_price(irr): 
                cashFlows = np.zeros(len(paymentSchedules))
                for i in range(len(cashFlows)):
                    if i != (len(cashFlows)-1):
                        cashFlows[i] = self.face_value * (self.coupon_rate/100)
                    else:
                        cashFlows[i] = (self.face_value * (self.coupon_rate/100)) + self.face_value
                pvCashFlow = np.zeros(len(paymentSchedules))
                for index, value in enumerate(paymentSchedules):
                    pvCashFlow[index] = cashFlows[index] /  ((1+((irr/100)))**(value/12)) 
                sumPvCashFlow = np.sum(pvCashFlow)
                return sumPvCashFlow - self.bond_price

            # setting an initial guess for the internal rate of return (IRR)
            initial_guess = 5.0

            # computing the IRR using the newton method
            irr_result = newton(npv_minus_price, initial_guess)

            # storing the computed yield to maturity
            self.yield_to_maturity = round(irr_result/100, 6)

            # computing bond's duration
            durationWeights = pvCashFlow / self.bond_price

            # initializing an array to store the product of weights and time
            Weights_times_t = np.zeros(len(durationWeights))

            # computing the product of weights and time
            for index, value in enumerate(durationWeights):
                Weights_times_t[index] = durationWeights[index] * (paymentSchedules[index] / 12)

            # calculating and storing the bond's duration
            self.duration = round(np.sum(Weights_times_t), 4)

            # determining the payment frequency
            if self.semi_annual_payment:
                frequency = 2  
            else:
                frequency = 1  

            # calculating and storing the modified duration
            self.modified_duration = self.duration / (1 + (self.yield_to_maturity / frequency))
            self.modified_duration = round(self.modified_duration, 4)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # DURATION METHOD <<<<<<<<<<<<<<<

    def DurationNormal(self , yield_curve : pd.DataFrame, vol: str = "simple", interval: int = 1, alpha: float = 0.05, p: int = 1, q: int = 1, lambda_ewma: float = 0.94):

        """
        Main
        -----
        This method calculates the Value at Risk (VaR) and Expected Shortfall (ES) for a given bond position using the Modified Duration mapping approach.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns (int) and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - vol: Type of volatility to use ("simple", "garch", or "ewma"). Default is "simple".
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall). Default is 1.
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha: Significance level for VaR/ES. Default is 0.05.
        - p: The 'p' parameter for the GARCH model. Default is 1.
        - q: The 'q' parameter for the GARCH model. Default is 1.
        - lambda_ewma: The lambda parameter for Exponentially Weighted Moving Average (EWMA) model. Default is 0.94.

        Methodology:
        ------------
        - Input Checks: Validates the types and values of input parameters.
        - Interval Constraints: Throws errors if the interval is inappropriate relative to the bond's maturity.
        - Interpolation:
            - The 'target_month' is calculated based on the Modified Duration of the bond, scaled to a monthly time frame (Modified Duration * 12).
            - Cubic Spline Interpolation is employed to get a smoother yield curve and extract yields for target months, IF NOT directly available in the original yield curve.
            - A window of months around the 'target_month' is selected to perform the cubic spline interpolation.
            Edge cases are handled by ensuring the interpolation window fits within the available yield curve data.
            - Once the interpolated yield curve is available, the yield for the 'target_month' is extracted. 
        - Volatility Estimation: Depending on the selected model ("simple", "garch", "ewma"), calculates the volatility of yield changes.
        - VaR and ES Calculation: Applies the Modified Duration mapping approach to calculate VaR and ES.

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import norm ; import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd
        
        from pyriskmgmt.inputsControlFunctions import (check_yield_curve_df, check_vol, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; vol = check_vol(vol) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        if interval >= (self.maturity * 12 * 25):
            raise ValueError(f"""
            The maturity of the bond is {self.maturity} years.
            Which is equivalent to {self.maturity * 12} months or {self.maturity * 12 * 25} days.
            The 'interval' for the VaR/ES (Value at Risk/Expected Shortfall) calculation cannot exceed, or be equal to, {self.maturity * 12 * 25} days.
            Right now, it is set to {interval}!
            The 'interval' is meant to represent a time period over which you're assessing the risk (VaR/ES) of the bond. 
            If the 'interval' were to match or exceed the bond's maturity, the risk metrics would no longer be meaningful. 
            """)
        
        limit = 25*6
        
        if interval > (limit):
            raise Exception(f"""
            The 'interval' parameter is intended to specify a time frame for assessing the risk metrics VaR/ES 
            (Value at Risk/Expected Shortfall) of the bond. 
            Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed due to the following reasons:

            1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods,
            making the resulting risk metrics unreliable.
            
            2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc.,
            are likely to significantly affect the bond's price. These factors are often not captured accurately in the models,
            leading to overly simplistic and potentially misleading results.
            
            Therefore, the maximum allowed "interval" for risk metric calculation is set to be {limit} days, which is equivalent to 25 trading days * 6 months.
            """)
                
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # Note:
        # Modified Duration (MD) serves as a scaling factor or "mapping multiplier," analogous to beta and delta in other financial instruments.
        # By analyzing the bond in the FixedIncomePrmSingle class through the lens of its Modified Duration and Price, we can effectively treat it as if 
        # it were a Zero-Coupon Bond (ZCB) with a maturity equal to the bond's Modified Duration (in years).

        # To obtain the yield for this "equivalent ZCB," we use cubic spline interpolation to find the yield for a ZCB with the same maturity as 
        # the Modified Duration from the existing (monthly) yield curve.
        # ----------------------------------------------------------------------------------------------------------------------------------------

        target_month = self.modified_duration * 12

        # initializing an empty DataFrame to store the interpolated yields
        interpolated_df = pd.DataFrame()

        # checking if target_month is already in the columns of the yield_curve
        if target_month in yield_curve.columns: 
            y = yield_curve.loc[:, target_month].values
            y = y / 100

        else:                                   
            # handling edge cases
            if np.floor(target_month) < 3: 
                left = 1
                right = left + 6
            elif np.ceil(target_month) > max(yield_curve.columns) - 3:
                right = max(yield_curve.columns)
                left = right - 6
            else:
                left = np.floor(target_month) - 3
                right = np.ceil(target_month) + 3

            # defining new columns for the interpolated DataFrame
            new_columns = np.arange(left, right, 0.1) 

            interpolated_df = pd.DataFrame(index=yield_curve.index, columns=new_columns)

            # looping through all rows of the existing yield curve DataFrame
            for i, row in yield_curve.iterrows():
                # converting index and values to lists
                existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
                cs = CubicSpline(existing_terms, existing_yields)
                interpolated_yields = cs(new_columns)
                # adding the interpolated yields to the DataFrame
                interpolated_df.loc[i] = interpolated_yields

            # converting the DataFrame to float dtype
            interpolated_df = interpolated_df.astype(float)

            # ////////////////////////////////////////////////////////////////////////////////////////////////

            closest_column = min(interpolated_df.columns, key = lambda x: abs(x - target_month))
            y = (interpolated_df[closest_column].values) / 100

        delta_y = np.diff(y)

        # Sigma of the delta_y **********************************************************************************************************************************

        # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        if vol == "simple":
            if interval == 1:
                sigma_delta_y = np.std(delta_y)
            elif interval > 1:
                sigma_delta_y = np.std(delta_y) * np.sqrt(interval)

        # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "garch":

            from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

            # check procedure
            p = check_p(p) ; q = check_q(q) 

            if p > 1 or q > 1:
                raise Exception("p and q are limited to 1 here in order to ensure numerical stability")

            # to ignore warnings
            import os ; import sys ; import numpy as np ; from scipy.stats import norm ; from arch import arch_model

            model = arch_model(delta_y, vol="GARCH", p=p, q=q, power= 2.0, dist = "normal") 

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

            # horizon is 1 --- better always stay at the highest frequency and then aggregate
            horizon = 1

            # forecasting
            forecasts = model_fit.forecast(start=0, horizon=horizon)

            # getting the variance forecasts
            variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1:
                sigma_delta_y = np.sqrt(variance_forecasts)
            elif interval > 1:
                cumulative_variance = variance_forecasts * interval # assuming normality i.i.d for the future
                sigma_delta_y = np.sqrt(cumulative_variance)


         # EWMA (lamdba = lambda_ewma) Volatility ------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "ewma":

            from pyriskmgmt.inputsControlFunctions import check_lambda_ewma

            # check procedure
            lambda_ewma = check_lambda_ewma(lambda_ewma) 

            # creatting the fitted time series of the ewma(lambda_ewma) variances ------------

            n = len(delta_y)

            variance_time_series = np.repeat(delta_y[0]**2, n)

            for t in range(1,n):
                variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * delta_y[t-1]**2

            variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * delta_y[-1]**2

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1: #########

                sigma_delta_y = np.sqrt(variance_t_plus1)    

            if interval >1 : #########

                sigma_delta_y = np.sqrt(variance_t_plus1) * np.sqrt(interval) # assuming normality i.i.d for the future

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # ----------------------------------------------------------------------------------------------------------------------------------

        absOverAllPos = abs(self.bond_price * self.num_assets * self.modified_duration)

        quantile = norm.ppf(1-alpha, loc=0, scale=1) 
        q_es = ( ( np.exp( - (quantile**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-alpha))))

        # Value at Risk and Expected Shortfall 
        VaR = quantile * absOverAllPos * sigma_delta_y ; ES = q_es * absOverAllPos * sigma_delta_y

        # consolidating results into a dictionary to return
        DURATION_NORMAL = {"var" : round(VaR, 4),
                           "es:" : round(ES, 4),
                           "T" : interval,
                           "BondPod" : round(self.bond_price * self.num_assets,4),
                           "Maturity" : self.maturity, 
                           "Duration" : self.duration,
                           "ModifiedDuration" : self.modified_duration,
                           "MappedBondPos" : round(absOverAllPos,4),
                           "BondPrice" : round(self.bond_price, 4),
                           "SigmaDeltaY" : round(sigma_delta_y,5)}

        return DURATION_NORMAL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self , yield_curve: pd.DataFrame, interval: int = 1, alpha: float = 0.05):

        """
        Main
        -----
        In this approach, historical data is used to generate a distribution of potential outcomes, which can be more realistic compared to model-based approaches like 
        the Duration-Normal method.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall).
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed. Default is 1.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha : Significance level for VaR/ES. Default is 0.05.

        Methodology:
        ------------
        1. Initialization and Input Checks:
        - Validate yield_curve, interval, and alpha.
        - Check maturity and interval constraints.

        2. Yield Curve Interpolation:
        - Convert yield rates to decimals.
        - Apply cubic spline interpolation.

        3. Data Preprocessing:
        - Round column names for precision.
        - Calculate yield curve differences at specified interval.
        - Generate simulated yield curves.

        4. Calculating Profit/Loss:
        - For Zero-Coupon Bonds:
            - Calculate new maturity date.
            - Calculate time-effect only price.
            - Calculate yield-shift effect and Profit/Loss.
        - For Non-Zero Coupon Bonds:
            - Prepare payment schedule.
            - Calculate time-effect only price.
            - Calculate yield-shift effect and Profit/Loss.

        5. Risk Metrics Calculation:
        - Generate a losses array.
        - Calculate Value at Risk (VaR).
        - Calculate Expected Shortfall (ES).

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd
        
        from pyriskmgmt.inputsControlFunctions import (check_yield_curve_df, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        if interval >= (self.maturity * 12 * 25):
            raise ValueError(f"""
            The maturity of the bond is {self.maturity} years.
            Which is equivalent to {self.maturity * 12} months or {self.maturity * 12 * 25} days.
            The 'interval' for the VaR/ES calculation cannot exceed, or be equal to, {self.maturity * 12 * 25} days.
            Right now, it is set to {interval}!
            The 'interval' is meant to represent a time period over which you're assessing the risk (VaR/ES) of the bond. 
            If the 'interval' were to match or exceed the bond's maturity, the risk metrics would no longer be meaningful. 
            """)
        
        limit = 25*6
        
        if interval > (limit):
            raise Exception(f"""
            The 'interval' parameter is intended to specify a time frame for assessing the risk metrics VaR/ES 
            (Value at Risk/Expected Shortfall) of the bond. 
            Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed due to the following reasons:

            1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods,
            making the resulting risk metrics unreliable.
            
            2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc.,
            are likely to significantly affect the bond's price. These factors are often not captured accurately in the models,
            leading to overly simplistic and potentially misleading results.
            
            Therefore, the maximum allowed "interval" for risk metric calculation is set to be {limit} days, which is equivalent to 25 trading days * 6 months.
            """)

        # dividing each entry by 100 to convert percentages to decimals
        yield_curve = yield_curve / 100

        # creating an array for new columns; each step is 0.04 to represent a day (1/25 month)
        new_columns = np.arange(0.04, yield_curve.shape[1] + 0.04 , 0.04)

        # initializing an empty DataFrame with the same row index as 'yield_curve' and columns as 'new_columns'
        interpolated_df = pd.DataFrame(index = yield_curve.index, columns=new_columns)

        # iterating through each row in the 'yield_curve' DataFrame
        for i, row in yield_curve.iterrows():
            # converting the index and row values to lists for cubic spline interpolation
            existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
            # applying cubic spline interpolation
            cs = CubicSpline(existing_terms, existing_yields)
            # obtaining new interpolated yields
            interpolated_yields = cs(new_columns)
            # storing the interpolated yields into the new DataFrame
            interpolated_df.loc[i] = interpolated_yields

        # converting the data type of all elements in the DataFrame to float
        interpolated_df = interpolated_df.astype(float)

        # initializing an empty array to store rounded column names
        columns_rounded = np.zeros(len(interpolated_df.columns))

        # iterating through each column and rounding the name to 2 decimal places
        for i, col in enumerate(interpolated_df.columns):
            columns_rounded[i] = round(col, 2)

        # updating the column names with the rounded values
        interpolated_df.columns = columns_rounded

        # calculating the difference between rows at a given interval, then removing NA rows
        DiffIntervalYieldCurve = interpolated_df.diff(periods = interval, axis=0).dropna()

        # calculating the new maturity by deducting the interval and scaling it down to months
        NewMaturity = round(((self.maturity * 12 * 25) - interval) / 25, 2)

        # determining the maximum number of possible simulations based on the number of rows
        NumSims = DiffIntervalYieldCurve.shape[0]

        # creating a DataFrame of yield curves for simulation, based on the differences and the last yield curve
        SimuYieldCurve = DiffIntervalYieldCurve + interpolated_df.iloc[-1]

        # Time Decay and Yield Curve Shifts
        # ---------------------------------
        # This method will consider both the natural time decay of the bond (i.e., approaching maturity) and potential shifts in the yield curve
        # ------------------------------------------------------------------------------------------------------------------------------------------

        PL = np.zeros(NumSims)

        # if self.is_zcb == True ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if self.is_zcb == True:

            # looping through all rows in the simulated yield curve. Each row represents a different scenario.
            for i in range(SimuYieldCurve.shape[0]):

                # calculating the new maturity date, considering the interval step. 
                # the original maturity of the bond is given by 'self.maturity' (in years), 
                # which is converted to days and then adjusted by the 'interval'.
                # it's then rounded to 2 decimal places and converted back to years.
                NewMaturity = round(((self.maturity * 12 * 25) - interval) / 25, 2)

                # Part 1: Price of the bond without any change (time effect only)
                # ----------------------------------------------------------------
                # this is done to isolate the time effect on the bond price, holding everything else constant.
                # using today's yield curve (the last row of the interpolated_df), find the yield for the NewMaturity.
                exact_yield = interpolated_df.iloc[-1].loc[NewMaturity]
                
                # using this yield, calculate what the bond's fair price would be at the NewMaturity date, 
                # without considering any shift in the yield curve.
                fair_price_at_new_maturity = round(self.face_value / ((1 + (exact_yield)) ** (NewMaturity / 12)), 4)
                
                # Part 2: Price of the bond with a change in the yield curve
                # -----------------------------------------------------------
                # this accounts for any shifts in the yield curve.
                # retrieving the yield for the bond at its NewMaturity under the current simulation scenario.
                exact_yield = SimuYieldCurve.iloc[i].loc[NewMaturity]
                
                # calculate the new price of the bond under the current simulation scenario.
                NewPrice = round(self.face_value / ((1 + (exact_yield)) ** (NewMaturity / 12)), 4)

                # computing the Profit/Loss only due to the simulated shifts in the interpolated yield curve
                # -------------------------------------------------------------------------------------------
                # PL[i] is the difference between the new price (NewPrice) and the 'time effect only' price (fair_price_at_new_maturity),
                # both scaled by the number of assets. This captures the effect of the shift in the yield curve.
                PL[i] = (NewPrice * self.num_assets) - (fair_price_at_new_maturity * self.num_assets)

        # if self.is_zcb == False ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if self.is_zcb == False:

            # Part 1: Calculating the bond's price accounting for time effects only
            # -----------------------------------------------------------------------
            
            # determining if payments are yearly
            if not self.semi_annual_payment:
                # calculating the payment schedule in terms of months, adjusting for the interval
                paymentSchedules = (np.arange(1 ,(self.maturity + 1), 1) * 12) - round(interval / 25,2)

            # determining if payments are semi-annual
            if self.semi_annual_payment:
                # calculating the semi-annual payment schedule in terms of months, adjusting for the interval
                paymentSchedules = (np.arange(0.5, (self.maturity + 0.5), 0.5) * 12) - round(interval / 25,2)

            # rounding each value in the paymentSchedules array to two decimal places for precision
            for i in range(len(paymentSchedules)):
                paymentSchedules[i] = round(paymentSchedules[i], 2)

            # filtering the paymentSchedules array to keep only values that are strictly greater than zero
            # if a value is less or equal to 0 in the paymentSchedules, it indicates that the bond has already paid the coupons for those periods.
            paymentSchedules = [x for x in paymentSchedules if x > 0]

            # initializing a zero array to hold the bond's cash flows
            cashFlows = np.zeros(len(paymentSchedules))

            # iterating through each payment schedule to set the cash flows
            for z in range(len(cashFlows)):
                # assigning coupon payments for all but the last payment
                if z != (len(cashFlows) - 1):
                    cashFlows[z] = self.face_value * (self.coupon_rate / 100)
                # assigning coupon plus face value for the last payment
                else:
                    cashFlows[z] = (self.face_value * (self.coupon_rate / 100)) + self.face_value

            # initializing a zero array to hold the present value of cash flows
            pvCashFlow = np.zeros(len(paymentSchedules))

            # iterating through each payment schedule to calculate the present value of cash flows
            for idx, vle in enumerate(paymentSchedules):
                # extracting the yield corresponding to the new maturity
                exact_yield = interpolated_df.iloc[-1].loc[vle]
                # calculating the present value of each cash flow
                pvCashFlow[idx] = cashFlows[idx] / ((1 + (exact_yield)) ** (vle / 12))

            # calculating the bond's fair price at the new maturity, rounding to 4 decimal places
            fair_price_at_new_maturity = round(np.sum(pvCashFlow), 4)

            # Part 2: Calculating the bond's price accounting for changes in the yield curve
            # ------------------------------------------------------------------------------
            
            # iterating through each simulated yield curve
            for i in range(SimuYieldCurve.shape[0]):
                # re-initializing the zero array to hold the present value of cash flows
                pvCashFlow = np.zeros(len(paymentSchedules))

                # iterating through each payment schedule to calculate the present value of cash flows
                for idx, vle in enumerate(paymentSchedules):
                    # extracting the simulated yield corresponding to the new maturity
                    exact_yield = SimuYieldCurve.iloc[i].loc[vle]
                    # calculating the present value of each cash flow using the simulated yield
                    pvCashFlow[idx] = cashFlows[idx] / ((1 + (exact_yield)) ** (vle / 12))

                # calculating the bond's new price based on the simulated yield, rounding to 4 decimal places
                NewPrice = round(np.sum(pvCashFlow), 4)

                # computing the profit/loss due to the change in yield, scaled by the number of assets
                PL[i] = (NewPrice * self.num_assets) - (fair_price_at_new_maturity * self.num_assets)

        # -------------------------------------------------------------------------------------------------------------------------

        # losses
        losses = PL[PL<0]

        if len(losses) == 0:
            raise Exception("""
            No losses were generated in the simulation based on the current data and 'interval' settings.            
            Consider doing one or more of the following:

            1. Review the quality of your input data to ensure it's suitable for simulation.
            2. Slightly adjust the 'interval' parameter to potentially produce a different simulation outcome.
            """)

        NewMaturity = round(((self.maturity * 12 * 25) - interval) / 25, 2)

        # Value at Risk and Expected Shortfall 
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < - VaR]) * -1

        # consolidating results into a dictionary to return
        HS = {"var" : round(VaR, 4),
              "es" : round(ES, 4),
              "T" : interval,
              "BondPod" : round(self.bond_price * self.num_assets,4),
              "Maturity" : self.maturity,
              "NewMaturity" : round(NewMaturity/12,4), 
              "BondPrice" : round(self.bond_price, 4),
              "NumSims" : len(PL)}

        return HS

##########################################################################################################################################################################
##########################################################################################################################################################################
## 5. FixedIncomePrmPort  ################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class FixedIncomePrmPort:
    
    """
    Main
    ----
    This class offers the functionalities for pricing and risk assessment of a portfolio of bond securities. 
    It not only facilitates the calculation of portfolio-level risk metrics such as duration, yield-to-maturity, Value-at-Risk (VaR), and Expected Shortfall (ES),
    but also provides the same metrics for each individual bond within the portfolio.

    Important Features and Constraints
    ----------------------------------
    1. `maturity_s` Array Parameter:
    - Accepts either integer values to represent full years or decimals like 0.5 for semi-annual periods.
    - Optimized for risk and return evaluation at specific time-points: either at the time of issuance or 
      just after a coupon payment for coupon-bearing bonds.
    
    2. Alignment Within Maturities:
    - The class is designed to account for alignment within the maturities of the bonds in the portfolio.
    - The system in place requires every bond to either mature or have its subsequent payment due in intervals that are multiples of six months.
    - While this is a limitation since it doesn't account for bonds with non-standard payment schedules or already issued ones, it provides a base that future developers can build upon to create more advanced evaluation mechanisms.
    - Therefore, optimal usage is at the time of issuance of all bonds bond or every six months thereafter.
    
    3. Bond Compatibility:
    - Compatible with portfolios that include both zero-coupon bonds and coupon-bearing bonds.
    - Provides options for annual or semi-annual coupon payments for coupon-bearing bonds.
  

    ******************************************************************************************************************
    For other types of Bond Portofolio this module offers a NON-PARAMETRIC APPROACH: FixedIncomeNprmPort() 
    ******************************************************************************************************************

    Purpose
    -------
    This class aims to provide a comprehensive toolkit for understanding, analyzing, and managing portfolios of fixed-income securities. 
    It is tailored to serve two main audiences:

    1. Portfolio Managers and Quantitative Analysts: 
        - For professionals who aim to develop or integrate more sophisticated calculations and risk management strategies, 
        this class serves as a robust foundation. It offers a reliable basis upon which additional features, functionalities, and asset classes can be incorporated.

    2. Individual Investors and Educators: 
        - For those who intend to use the class for straightforward portfolio pricing and risk assessment, the class offers an array of ready-to-use methods. 
        These methods enable the calculation of key portfolio-level metrics such as aggregate bond price, portfolio duration, portfolio yield-to-maturity,
        Value-at-Risk (VaR), and Expected Shortfall (ES). 

    In addition, this class also allows users to drill down into the specifics of each bond within the portfolio, offering metrics like individual bond price, duration,
    and yield-to-maturity.
    By catering to both of these needs, the class offers a flexible and robust starting point for a range of fixed-income analysis and risk management tasks.

    Initial Parameters
    ------------------
    - yield_curve_today : The DataFrame containing the current yield curve, with terms in months as columns and the yield rates as row values.
    The yield rates should be represented as a percentage, e.g., 3% should be given as 3.

    ```
    | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
    |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
    | 08/29/2023| 5.54 | 5.53 | 5.52 | 5.51 | 5.51 | 4.87 | 4.56 | 4.26 | 4.21 | 4.19 | 4.18 | 4.19 |
    ```

    - maturity_s: Array of times to maturity for the bonds in the portfolio, in years. Each value must be either an integer for full years or a decimal like 0.5
    to indicate semi-annual periods.
    - num_assets_s: Array representing the number of identical assets/bonds for each bond in the portfolio.
    - face_value_s: Array containing the face values of the bonds in the portfolio.
    - is_zcb_s: Array of boolean indicators to specify whether each bond in the portfolio is a zero-coupon bond.
    - coupon_rate_s: Array of the annual coupon rates for the bonds in the portfolio, represented as percentages 
    (e.g., 3% should be entered as 3). This array is required if `is_zcb` contains any `False` values. Default is None.
    - semi_annual_payment_s: Array of boolean indicators to specify whether each bond in the portfolio pays coupons semi-annually.
    This array is required if `is_zcb` contains any `False` values. Default is None.

    Example of Initial Parameters
    -----------------------------
    - maturity_s = array([4, 1, 2.5, 2, 4.5])
    - num_assets_s = array([ 100,  40, -35,  25, -75])
    - face_value_s = array([1000, 1000, 1000, 1000, 1000])
    - is_zcb_s = array([ True,  True, False, False, False])
    - coupon_rate_s = array([0, 0, 2, 2, 3])
    - semi_annual_payment_s = array([False, False,  True, False,  True])

    Methods
    -------
    - DurationNormal(): Calculates the Value-at-Risk (VaR) and Expected Shortfall (ES) of a portfolio of bond using a normal distribution model.
    This method assumes a normal distribution of yield changes and uses the modified duration of the portfolio
    to approximate the change in bond price for a given change in yield.
    - HistoricalSimulation(): In this approach, historical data is used to generate a distribution of potential outcomes,
    which can be more realistic compared to model-based approaches like the Duration-Normal method.

    Attributes
    ----------
    - bond_prices: The price of each bond within the portfolio, calculated based on whether it is a zero-coupon or coupon-bearing bond.
    - durations: The Macaulay duration (in years) for each bond within the portfolio.
    - modified_durations: The modified duration (in years) for each bond within the portfolio.
    - yield_to_maturities: The yield to maturity (annualized) for each bond within the portfolio.
    - initial_position: Initial financial position of the portfolio, considering the characteristics of each bond.
    - tot_future_payments: Total future payments to be received from all bonds within the portfolio.
    - summary: A comprehensive summary of the portfolio's attributes.

    Exceptions
    ----------
    - Raises an exception if the coupon_rate and semi_annual_payment are not provided when is_zcb is set to False.
    - Raises an exception if the maturity provided is beyond the highest maturity available in the yield_curve_today DataFrame.

    Note
    ----
    The yield curve provided should be a monthly yield curve, with maturity ranging from 1 to n months.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore") 

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, yield_curve_today: pd.DataFrame,               # not in decimal point. For instance: for 3% it should be 3.
                       maturity_s: np.ndarray,                        # array - in years
                       num_assets_s: np.ndarray, 
                       face_value_s: np.ndarray, 
                       is_zcb_s: np.ndarray, 
                       coupon_rate_s: np.ndarray = None,              # array - not in decimal point. For instance: for 3% it should be 3.
                       semi_annual_payment_s: np.ndarray = None):
        

        from pyriskmgmt.inputsControlFunctions import (check_yield_curve_today, check_maturity_s, check_num_assets_s,check_face_value_s, check_is_zcb_s,
                                                       check_coupon_rate_s, check_semi_annual_payment_s)

        # check procedure
        self.yield_curve_today = check_yield_curve_today(yield_curve_today) ; self.maturity_s = check_maturity_s(maturity_s) ; self.num_assets_s = check_num_assets_s(num_assets_s)
        self.face_value_s = check_face_value_s(face_value_s) ; self.is_zcb_s = check_is_zcb_s(is_zcb_s) # here it returns the value

        # -----------------------------------------------------------------------------------------------------
        
        # first checking
        # checking the same length for all the imputs *args

        def check_same_length(*args):
            """
            Checks if all given numpy arrays have the same length.
            """
            it = iter(args)
            the_len = len(next(it))

            if not all(len(l) == the_len for l in it):
                raise ValueError("Not all the input arrays in the FixedIncomePrmPort class have the same length!")
            
        # Here we are ensuring that all arrays have the same length
        check_same_length(maturity_s, num_assets_s, face_value_s, is_zcb_s)

        # ------------------------------------------------------------------------------------------------

        # checking if at least one value in is_zcb_s the is False
        if not all(is_zcb_s):

            if coupon_rate_s is None and semi_annual_payment_s is None: 
                raise Exception("Both coupon_rate_s and semi_annual_payment_s must be provided when at least one element in the is_zcb_s is False.")
            if coupon_rate_s is None or semi_annual_payment_s is None: 
                raise Exception("Either coupon_rate_s or semi_annual_payment_s is missing. Both are required when at least one element in the is_zcb_s is False.")
            
            if coupon_rate_s is not None and semi_annual_payment_s is not None: 

                # check procedure
                self.coupon_rate_s = check_coupon_rate_s(coupon_rate_s) ; self.semi_annual_payment_s = check_semi_annual_payment_s(semi_annual_payment_s)

               # second checking
                # Here we are ensuring that all arrays have the same length
                check_same_length(maturity_s, num_assets_s, face_value_s, is_zcb_s,coupon_rate_s,semi_annual_payment_s)

            for index, (mat, semi) in enumerate(zip(self.maturity_s, self.semi_annual_payment_s)):
                if (mat * 2) % 2 != 0 and not semi:
                    raise Exception(f"""
                    If the maturity of a bond is in fractions of .5 years, then element-wise semi_annual_payment must be set to True. 
                                    
                    This class checks every bond and for those that have a "half-year" maturity makes sure that they have semi-annual payments.
                    This ensures that the payment schedule aligns with the bond's maturity period, preventing logical inconsistencies.
                                    
                    Issue at index {index}.
                      
                    More precisely, the following bond:
                    - maturity: {self.maturity_s[index]}, 
                    - semi_annual_payment value set to: {self.semi_annual_payment_s[index]},
                    - num_asset: {self.num_assets_s[index]},
                    - face_value: {self.face_value_s[index]}
                    """)
                
    # *******************************************************************************************************************************************
        
        for index in range(len(self.num_assets_s)):

            if (self.maturity_s[index] * 12) > self.yield_curve_today.columns[-1]:
                raise Exception(f"""
                Issue at index {index}.
                More precisely, the following bond:
                - maturity: {self.maturity_s[index]}, or {self.maturity_s[index] * 12} months,
                - semi_annual_payment value set to: {self.semi_annual_payment_s[index]},
                - num_asset: {self.num_assets_s[index]},
                - face_value: {self.face_value_s[index]}
                                
                The highest maturity available in the 'yield_curve_today' DataFrame is {self.yield_curve_today.columns[-1]} months.
                Therefore, the method does not have the required yield for the maturity of the bond at index {index}.
                """)

        self.bond_prices = None  ; self.durations = None ; self.modified_durations = None ; self.yield_to_maturities = None
        self.initial_position = None ; self.tot_future_payments = None ; self.summary = None

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.optimize import newton ; import pandas as pd
 
        # *****************************************************************************************************************************************************

        def AttributesCalculator(yield_curve_today, maturity_s, num_assets_s, face_value_s, is_zcb_s, coupon_rate_s, semi_annual_payment_s):
 
            # initializing arrays to store bond attributes
            bond_prices = np.zeros(len(num_assets_s)) ; durations = np.zeros(len(num_assets_s)) ; modified_durations = np.zeros(len(num_assets_s))
            yield_to_maturities = np.zeros(len(num_assets_s)) ; tot_future_payments = np.zeros(len(num_assets_s))

            # ------------------------------------------------------------------------------------------------------------------

            # iterating over each bond
            for bond_num in range(len(num_assets_s)):

                # if is_zcb_s[bond_num] == True /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == True:
  
                    # calculating the exact yield and bond price for zero-coupon bonds
                    maturity = maturity_s[bond_num] * 12 ; exact_yield = yield_curve_today.iloc[-1].loc[maturity]

                    # calculating the bond price using the zero-coupon bond pricing formula: P = F / (1 + r)^n
                    bond_prices[bond_num] = round(face_value_s[bond_num] / ( (1 + (exact_yield/100))  ** maturity_s[bond_num]),4)
 
                    # assigning duration and yield to maturity
                    durations[bond_num] = maturity_s[bond_num] ; yield_to_maturities[bond_num] = round(exact_yield/100,6)

                    # calculating modified duration and setting total future payments as 1 for zero-coupon bonds
                    modified_durations[bond_num] = round(durations[bond_num] / (1 + (yield_to_maturities[bond_num])), 4) ; tot_future_payments[bond_num] = 1

                # if is_zcb_s[bond_num] == False ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == False:

                    # determining the payment schedules based on the type of payments (yearly or semi-annual)
                    # if yearly payments -------------------------------------------------------------------------------------
                    if not semi_annual_payment_s[bond_num]: 
                        paymentSchedules = np.arange(1,(maturity_s[bond_num] + 1),1) * 12 
                    # if semi annual payment payments ------------------------------------------------------------------------
                    if semi_annual_payment_s[bond_num]:
                        paymentSchedules = np.arange(0.5,(maturity_s[bond_num] + 0.5),0.5) * 12 
                        coupon_rate_s[bond_num] = coupon_rate_s[bond_num] / 2 # PIVOTAL
                    # --------------------------------------------------------------------------------------------------------

                    # calculating cash flows and their present values
                    cashFlows = np.zeros(len(paymentSchedules))

                    for i in range(len(cashFlows)):
                        if i != (len(cashFlows)-1):
                            cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                        else:
                            cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]

                    pvCashFlow = np.zeros(len(paymentSchedules))

                    for index, value in enumerate(paymentSchedules):
                        exact_yield = yield_curve_today.iloc[-1].loc[value]
                        pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield/100)))**(value/12)) 
                    
                    # calculating the bond price by summing up the present values and setting total future payments for the cb
                    bond_prices[bond_num] = round(np.sum(pvCashFlow),4) ; tot_future_payments[bond_num] = len(paymentSchedules)

                    # finding yield to maturity by minimizing the function npv_minus_price ----------------------------------------

                    def npv_minus_price(irr): #irr here
                        cashFlows = np.zeros(len(paymentSchedules))
                        for i in range(len(cashFlows)):
                            if i != (len(cashFlows)-1):
                                cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                            else:
                                cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]
                        pvCashFlow = np.zeros(len(paymentSchedules))
                        for index, value in enumerate(paymentSchedules):
                            pvCashFlow[index] = cashFlows[index] /  ((1+((irr/100)))**(value/12)) # #irr here
                        sumPvCashFlow = np.sum(pvCashFlow)
                        return sumPvCashFlow - bond_prices[bond_num]
                    
                    # --------------------------------------------------------------------------------------------------------------------
                    
                    # assuming the initial guess for IRR is 5%
                    initial_guess = 5.0 ; irr_result = newton(npv_minus_price, initial_guess)

                    # storing the result
                    yield_to_maturities[bond_num] = round(irr_result/100, 6)

                    # calculating duration and modified duration 
                    durationWeights = pvCashFlow / bond_prices[bond_num] ; Weights_times_t = np.zeros(len(durationWeights))

                    for index, value in enumerate(durationWeights):
                        Weights_times_t[index] = durationWeights[index] * (paymentSchedules[index] / 12)

                    durations[bond_num] = round(np.sum(Weights_times_t), 4)

                    if semi_annual_payment_s[bond_num]:
                        frequency = 2  # payments are semi-annual
                    else:
                        frequency = 1  # payments are annual

                    modified_durations[bond_num] = durations[bond_num] / (1 + (yield_to_maturities[bond_num] / frequency))
                    modified_durations[bond_num] = round(modified_durations[bond_num], 4)

            # ----------------------------------------------------------------------------------------------------------------
            
            # calculating the initial portfolio position
            initial_position = round(np.dot(num_assets_s, bond_prices),4)

            return bond_prices, durations, modified_durations, yield_to_maturities, initial_position, tot_future_payments
        
        # *****************************************************************************************************************************************************

        # calling the AttributesCalculator function and storing the results
        self.bond_prices, self.durations, self.modified_durations, self.yield_to_maturities, self.initial_position, self.tot_future_payments = AttributesCalculator(
            self.yield_curve_today, self.maturity_s, self.num_assets_s, self.face_value_s, self.is_zcb_s, self.coupon_rate_s, self.semi_annual_payment_s)

        # creating a summary DataFrame
        self.summary = pd.DataFrame({"ZCB" : self.is_zcb_s, "Semi" : self.semi_annual_payment_s, "BondPrice" : self.bond_prices, "FaceValue" : self.face_value_s,
            "Maturity" : self.maturity_s, "TotFutPay" : self.tot_future_payments, "NumAsset" : self.num_assets_s, "Pos" : self.bond_prices * self.num_assets_s, 
            "Duration" : self.durations, "MD" : self.modified_durations, "YTM" : self.yield_to_maturities, "CouponRate" : self.coupon_rate_s / 100
        })
        
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # DURATION METHOD <<<<<<<<<<<<<<<

    def DurationNormal(self, yield_curve: pd.DataFrame, vol: str = "simple", interval: int = 1, alpha: float = 0.05, p: int = 1, q: int = 1, lambda_ewma: float = 0.94):

        """
        Main
        -----
        This method calculates the Value at Risk (VaR) and Expected Shortfall (ES) for a given bond portfolio using the Modified Duration mapping approach.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - vol : Type of volatility to use ("simple", "garch", or "ewma").
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall). Default is 1.
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha : Significance level for VaR/ES. Default is 0.05.
        - p: The 'p' parameter for the GARCH model. Default is 1.
        - q: The 'q' parameter for the GARCH model. Default is 1.
        - lambda_ewma: The lambda parameter for Exponentially Weighted Moving Average (EWMA) model. Default is 0.94.

        Methodology:
        ------------
        - Input Checks: Validates the types and values of input parameters.
        - Validates the 'interval' parameter against two key constraints:
            - a) Max Possible Interval: Calculated based on the bond with the shortest maturity.
            - b) Predefined Limit: Set to a fixed number of days (150 days, equivalent to 6 months).
        - Raises exceptions if the provided 'interval' exceeds either constraint, offering reasons and suggestions for adjustment.
        - Interpolation:
            - The 'target_month' is calculated based on the Modified Duration of the bond, scaled to a monthly time frame (Modified Duration * 12).
            - Cubic Spline Interpolation is employed to get a smoother yield curve and extract yields for target months, IF NOT directly available in the original yield curve.
            - A window of months around the 'target_month' is selected to perform the cubic spline interpolation.
            Edge cases are handled by ensuring the interpolation window fits within the available yield curve data.
            - Once the interpolated yield curve is available, the yield for the 'target_month' is extracted. 
        - Volatility Estimation: Depending on the selected model ("simple", "garch", "ewma"), calculates the volatility of yield changes.
        - VaR and ES Calculation: Applies the Modified Duration mapping approach to calculate VaR and ES.

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import norm ; import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd
        
        from pyriskmgmt.inputsControlFunctions import (check_yield_curve_df, check_vol, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; vol = check_vol(vol) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        # calculating the maximum possible interval based on the bond with the shortest maturity
        max_possible_interval = min(self.maturity_s) * 12 * 25  # assuming 25 trading days in a month
        limit = 25 * 6  # a predefined limit set to 150 days (equivalent to 6 months)

        # checking if the provided 'interval' exceeds either of the two constraints: max_possible_interval or limit
        if interval >= max_possible_interval or interval >= limit:
            if interval >= max_possible_interval:
                reason1 = f"""
                The bond with the shortest maturity in the portfolio has a remaining time of {min(self.maturity_s)} years,
                equivalent to {min(self.maturity_s) * 12} months or {max_possible_interval} days."""
                reason2 = f"""
                The provided 'interval' of {interval} days exceeds or matches this limit, making the VaR/ES metrics unreliable."""
                suggestion1 = f"""
                Please choose an 'interval' less than {max_possible_interval} days.
                """
            else:
                reason1 = f"""
                Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed for the following reasons:"""
                reason2 = """
                1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods.
                2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc., are likely to 
                significantly affect the bond's price."""
                suggestion1 = f"""
                Therefore, the maximum allowed 'interval' for risk metric calculation is set to be {limit} days.
                """

            raise Exception(f"""
                Invalid 'interval' for VaR/ES (Value at Risk/Expected Shortfall) calculation.
            {reason1}
            {reason2}
            {suggestion1}
            """)
        
        # >>>>>> although the handler for interval >= max_possible_interval appears redundant, it's retained for future development <<<<<<<<<

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Note:
        # When dealing with a portfolio of bonds, the concept of Modified Duration extends to the portfolio level as the weighted average of the Modified Durations
        # of the individual assets in the portfolio. This portfolio-level Modified Duration serves as a composite "mapping multiplier," 
        # similar to how beta and delta function in other financial contexts.

        # By analyzing the portfolio's Modified Duration and Price, one can treat the entire portfolio as if it were a single Zero-Coupon Bond (ZCB) 
        # with a maturity equal to the portfolio's weighted average Modified Duration (in years).

        # To estimate the yield for this "equivalent ZCB," cubic spline interpolation can be applied to the existing (monthly) yield curve. 
        # The aim is to find the yield associated with a ZCB that has the same maturity as the portfolio's weighted average Modified Duration.
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------

        # calculating the absolute values of the initial positions for each asset (here represented by the product of bond prices and the number of such assets)
        self.abs_initial_position = [abs(x) for x in (self.bond_prices * self.num_assets_s)]

        # summing up all the absolute values to get the total absolute position of the portfolio
        AbsPosPort = np.sum(self.abs_initial_position)

        # calculating the weights of each asset in the portfolio by dividing its absolute value by the total absolute position
        weights = self.abs_initial_position / AbsPosPort

        # calculating the weighted average duration of the portfolio using dot product
        duration_port = np.dot(weights, self.durations)

        # calculating the weighted average modified duration of the portfolio using dot product
        modified_duration_port = np.dot(weights, self.modified_durations)

        # setting the target_month value to be the weighted average modified duration of the portfolio
        target_month = modified_duration_port

        # initializing an empty DataFrame to store the interpolated yields
        interpolated_df = pd.DataFrame()

        # if target_month is already in the yield_curve.columns
        if target_month in yield_curve.columns: 
            y = yield_curve.loc[:, target_month].values
            y = y / 100

        else:                                   
            # handling edge cases
            if np.floor(target_month) < 3: 
                left = 1
                right = left + 6
            elif np.ceil(target_month) > max(yield_curve.columns) - 3:
                right = max(yield_curve.columns)
                left = right - 6
            else:
                left = np.floor(target_month) - 3
                right = np.ceil(target_month) + 3

            # defining new columns for the interpolated DataFrame
            new_columns = np.arange(left, right, 0.1) 

            interpolated_df = pd.DataFrame(index=yield_curve.index, columns=new_columns)

            # looping through all rows of the existing yield curve DataFrame
            for i, row in yield_curve.iterrows():
                # converting index and values to lists
                existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
                cs = CubicSpline(existing_terms, existing_yields)
                interpolated_yields = cs(new_columns)
                # adding the interpolated yields to the DataFrame
                interpolated_df.loc[i] = interpolated_yields

            # converting the DataFrame to float dtype
            interpolated_df = interpolated_df.astype(float)

            # ////////////////////////////////////////////////////////////////////////////////////////////////

            closest_column = min(interpolated_df.columns, key = lambda x: abs(x - target_month))
            y = (interpolated_df[closest_column].values) / 100

        delta_y = np.diff(y)

        # Sigma of the delta_y **********************************************************************************************************************************

        # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        if vol == "simple":
            if interval == 1:
                sigma_delta_y = np.std(delta_y)
            elif interval > 1:
                sigma_delta_y = np.std(delta_y) * np.sqrt(interval)

        # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "garch":

            from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

            # check procedure
            p = check_p(p) ; q = check_q(q) 

            if p > 1 or q > 1:
                raise Exception("p and q are limited to 1 here in order to ensure numerical stability")

            # to ignore warnings
            import os ; import sys ; import numpy as np ; from scipy.stats import norm ; from arch import arch_model

            model = arch_model(delta_y, vol="GARCH", p=p, q=q, power= 2.0, dist = "normal") 

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

            # horizon is 1 --- better always stay at the highest frequency and then aggregate
            horizon = 1

            # forecasting
            forecasts = model_fit.forecast(start=0, horizon=horizon)

            # getting the variance forecasts
            variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1:
                sigma_delta_y = np.sqrt(variance_forecasts)
            elif interval > 1:
                cumulative_variance = variance_forecasts * interval # assuming normality i.i.d for the future
                sigma_delta_y = np.sqrt(cumulative_variance)


         # EWMA (lamdba = lambda_ewma) Volatility ------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "ewma":

            from pyriskmgmt.inputsControlFunctions import check_lambda_ewma

            # check procedure
            lambda_ewma = check_lambda_ewma(lambda_ewma) 

            # creatting the fitted time series of the ewma(lambda_ewma) variances ------------

            n = len(delta_y)

            variance_time_series = np.repeat(delta_y[0]**2, n)

            for t in range(1,n):
                variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * delta_y[t-1]**2

            variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * delta_y[-1]**2

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1: #########

                sigma_delta_y = np.sqrt(variance_t_plus1)    

            if interval >1 : #########

                sigma_delta_y = np.sqrt(variance_t_plus1) * np.sqrt(interval) # assuming normality i.i.d for the future

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # ----------------------------------------------------------------------------------------------------------------------------------

        absOverAllPos = np.sum(self.abs_initial_position) * modified_duration_port

        quantile = norm.ppf(1-alpha, loc=0, scale=1) 
        q_es = ((np.exp(-(quantile**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha))))

        # Value at Risk and Expected Shortfall 
        VaR = quantile * absOverAllPos * sigma_delta_y ; ES = q_es * absOverAllPos * sigma_delta_y

        # consolidating results into a dictionary to return
        DURATION_NORMAL = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : interval,
                           "DurationPort" : round(duration_port,4),
                           "ModifiedDurationPort" : round(modified_duration_port,4),
                           "MappedBondPos" : round(absOverAllPos,4), 
                           "OrigPos" : self.initial_position,
                           "SigmaDeltaY" : round(sigma_delta_y,5)}

        return DURATION_NORMAL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, yield_curve: pd.DataFrame, interval: int = 1, alpha: float = 0.05):

        """
        Main
        -----
        In this approach, historical data is used to generate a distribution of potential outcomes, which can be more realistic compared to model-based approaches like 
        the Duration-Normal method.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall). Default is 1.
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha : Significance level for VaR/ES. Default is 0.05.

        Methodology:
        ------------
        1. Initialization and Input Checks:
        - Validate yield_curve, interval, and alpha.
        - Check maturity and interval constraints.

        2. Yield Curve Interpolation:
        - Convert yield rates to decimals.
        - Apply cubic spline interpolation.

        3. Data Preprocessing:
        - Round column names for precision.
        - Calculate yield curve differences at specified interval.
        - Generate simulated yield curves.
        
        4. Calculating Profit/Loss:
            - For each bond and position in the portfolio:
                - For Zero-Coupon Bonds:
                    - Calculate new maturity date.
                    - Calculate time-effect only price.
                    - Calculate yield-shift effect and Profit/Loss.
                - For Non-Zero Coupon Bonds:
                    - Prepare payment schedule.
                    - Calculate time-effect only price.
                    - Calculate yield-shift effect and Profit/Loss.
            - Finally, the aggregate value of the portfolio for each simulation is contrasted with the 
            value that arises solely from the passage of time, in order to compute the Profit/Loss.

        5. Risk Metrics Calculation:
        - Generate a losses array.
        - Calculate Value at Risk (VaR).
        - Calculate Expected Shortfall (ES).

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd 
        
        from pyriskmgmt.inputsControlFunctions import (check_yield_curve_df, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        # calculating the maximum possible interval based on the bond with the shortest maturity
        max_possible_interval = min(self.maturity_s) * 12 * 25  # assuming 25 trading days in a month
        limit = 25 * 6  # a predefined limit set to 150 days (equivalent to 6 months)

        # checking if the provided 'interval' exceeds either of the two constraints: max_possible_interval or limit
        if interval >= max_possible_interval or interval >= limit:
            if interval >= max_possible_interval:
                reason1 = f"""
                The bond with the shortest maturity in the portfolio has a remaining time of {min(self.maturity_s)} years,
                equivalent to {min(self.maturity_s) * 12} months or {max_possible_interval} days."""
                reason2 = f"""
                The provided 'interval' of {interval} days exceeds or matches this limit, making the VaR/ES metrics unreliable."""
                suggestion1 = f"""
                Please choose an 'interval' less than {max_possible_interval} days.
                """
            else:
                reason1 = f"""
                Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed for the following reasons:"""
                reason2 = """
                1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods.
                2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc., are likely to 
                significantly affect the bond's price."""
                suggestion1 = f"""
                Therefore, the maximum allowed 'interval' for risk metric calculation is set to be {limit} days.
                """

            raise Exception(f"""
                Invalid 'interval' for VaR/ES (Value at Risk/Expected Shortfall) calculation.
            {reason1}
            {reason2}
            {suggestion1}
            """)
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        # >>>>>> although the handler for interval >= max_possible_interval appears redundant, it's retained for future development <<<<<<<<<
        # -------------------------------------------------------------------------------------------------------------------------------------

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # dividing each entry by 100 to convert percentages to decimals
        yield_curve = yield_curve / 100

        # creating an array for new columns; each step is 0.04 to represent a day (1/25 month)
        new_columns = np.arange(0.04, yield_curve.shape[1] + 0.04 , 0.04)

        # initializing an empty DataFrame with the same row index as 'yield_curve' and columns as 'new_columns'
        interpolated_df = pd.DataFrame(index = yield_curve.index, columns=new_columns)

        # iterating through each row in the 'yield_curve' DataFrame
        for i, row in yield_curve.iterrows():
            # converting the index and row values to lists for cubic spline interpolation
            existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
            # applying cubic spline interpolation
            cs = CubicSpline(existing_terms, existing_yields)
            # obtaining new interpolated yields
            interpolated_yields = cs(new_columns)
            # storing the interpolated yields into the new DataFrame
            interpolated_df.loc[i] = interpolated_yields

        # converting the data type of all elements in the DataFrame to float
        interpolated_df = interpolated_df.astype(float)

        # initializing an empty array to store rounded column names
        columns_rounded = np.zeros(len(interpolated_df.columns))

        # iterating through each column and rounding the name to 2 decimal places
        for i, col in enumerate(interpolated_df.columns):
            columns_rounded[i] = round(col, 2)

        # updating the column names with the rounded values
        interpolated_df.columns = columns_rounded

        # calculating the difference between rows at a given interval, then removing NA rows
        DiffIntervalYieldCurve = interpolated_df.diff(periods = interval, axis=0).dropna()

        # creating a DataFrame of yield curves for simulation, based on the differences and the last yield curve
        SimuYieldCurve = DiffIntervalYieldCurve + interpolated_df.iloc[-1]

        # Time Decay and Yield Curve Shifts
        # ---------------------------------
        # This method will consider both the natural time decay of each bond (i.e., approaching maturity) and potential shifts in the yield curve
        # ---------------------------------------------------------------------------------------------------------------------------------------

        # PL calculator ***************************************************************************************************************************************
        
        def PlCalculator(SimuYieldCurve,interpolated_df, interval, num_assets_s, is_zcb_s,face_value_s,maturity_s, semi_annual_payment_s, coupon_rate_s):
            
            # 1.FAIR PRICES

            # initializing arrays to store bond attributes
            bond_prices_fair = np.zeros(len(num_assets_s))

            # ------------------------------------------------------------------------------------------------------------------

            # iterating over each bond
            for bond_num in range(len(num_assets_s)):

                # if is_zcb_s[bond_num] == True /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == True:

                    # calculating the exact yield and bond price for zero-coupon bonds
                    NewMaturity =  round(((maturity_s[bond_num] * 12 * 25) - interval) / 25,2)
                    exact_yield = interpolated_df.iloc[-1].loc[NewMaturity]

                    # calculating the bond price using the zero-coupon bond pricing formula: P = F / (1 + r)^n
                    bond_prices_fair[bond_num] = face_value_s[bond_num] / ( (1 + (exact_yield))  ** (NewMaturity/12))

                # if is_zcb_s[bond_num] == False ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == False:

                    # determining the payment schedules based on the type of payments (yearly or semi-annual)

                    # if yearly payments -------------------------------------------------------------------------------------
                    if not semi_annual_payment_s[bond_num]: 
                        paymentSchedules = (np.arange(1 ,(maturity_s[bond_num] + 1), 1) * 12) - (interval / 25)
                    # if semi annual payment payments ------------------------------------------------------------------------
                    if semi_annual_payment_s[bond_num]:
                        paymentSchedules = (np.arange(0.5 ,(maturity_s[bond_num] + 0.5), 0.5) * 12) - (interval / 25)
                        coupon_rate_s[bond_num] = coupon_rate_s[bond_num] / 2 # PIVOTAL
                    # --------------------------------------------------------------------------------------------------------

                    # rounding each value in the paymentSchedules array to two decimal places for precision
                    for u in range(len(paymentSchedules)):
                        paymentSchedules[u] = round(paymentSchedules[u], 2)

                    # filtering the paymentSchedules array to keep only values that are strictly greater than zero
                    # if a value is less or equal to 0 in the paymentSchedules, it indicates that the bond has already paid the coupons for those periods.
                    # FOR FUTURE DEVELOPEMENT */*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
                    paymentSchedules = [x for x in paymentSchedules if x > 0]

                    # calculating cash flows and their present values
                    cashFlows = np.zeros(len(paymentSchedules))

                    for i in range(len(cashFlows)):
                        if i != (len(cashFlows)-1):
                            cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                        else:
                            cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]

                    pvCashFlow = np.zeros(len(paymentSchedules))
                    for index, value in enumerate(paymentSchedules):
                        exact_yield = interpolated_df.iloc[-1].loc[value]
                        pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield)))**(value/12)) 
                    
                    # calculating the bond price by summing up the present values and setting total future payments for the cb
                    bond_prices_fair[bond_num] = np.sum(pvCashFlow)
            
            # *************************************************************************

            fair_positions_at_new_interval = np.dot(num_assets_s,bond_prices_fair)

            # --------------------------------------------------------------------------------------------------------------------------------

            # 1.SIMULATED PRICES

            PL = np.zeros(SimuYieldCurve.shape[0])
            for j in range(SimuYieldCurve.shape[0]):

                # initializing arrays to store bond attributes
                bond_prices = np.zeros(len(num_assets_s))

                # ------------------------------------------------------------------------------------------------------------------

                # iterating over each bond
                for bond_num in range(len(num_assets_s)):

                    # if is_zcb_s[bond_num] == True /////////////////////////////////////////////////////////////////////////////////////////////////////////

                    if is_zcb_s[bond_num] == True:

                        # calculating the exact yield and bond price for zero-coupon bonds
                        NewMaturity =  round(((maturity_s[bond_num] * 12 * 25) - interval) / 25,2)
                        exact_yield = SimuYieldCurve.iloc[j].loc[NewMaturity]
                        # calculating the bond price using the zero-coupon bond pricing formula: P = F / (1 + r)^n
                        bond_prices[bond_num] = face_value_s[bond_num] / ( (1 + (exact_yield))  ** (NewMaturity/12))

                    # if is_zcb_s[bond_num] == False ////////////////////////////////////////////////////////////////////////////////////////////////////////

                    if is_zcb_s[bond_num] == False:

                        # determining the payment schedules based on the type of payments (yearly or semi-annual)
                        # if yearly payments -------------------------------------------------------------------------------------
                        if not semi_annual_payment_s[bond_num]: 
                            paymentSchedules = (np.arange(1 ,(maturity_s[bond_num] + 1), 1) * 12) - (interval / 25)
                        # if semi annual payment payments ------------------------------------------------------------------------
                        if semi_annual_payment_s[bond_num]:
                            paymentSchedules = (np.arange(0.5 ,(maturity_s[bond_num] + 0.5), 0.5) * 12) - (interval / 25)
                            coupon_rate_s[bond_num] = coupon_rate_s[bond_num] / 2 # PIVOTAL
                        # --------------------------------------------------------------------------------------------------------

                        # rounding each value in the paymentSchedules array to two decimal places for precision
                        for u in range(len(paymentSchedules)):
                            paymentSchedules[u] = round(paymentSchedules[u], 2)

                        # filtering the paymentSchedules array to keep only values that are strictly greater than zero
                        # if a value is less or equal to 0 in the paymentSchedules, it indicates that the bond has already paid the coupons for those periods.
                        paymentSchedules = [x for x in paymentSchedules if x > 0]

                        # calculating cash flows and their present values
                        cashFlows = np.zeros(len(paymentSchedules))

                        for i in range(len(cashFlows)):
                            if i != (len(cashFlows)-1):
                                cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                            else:
                                cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]

                        pvCashFlow = np.zeros(len(paymentSchedules))

                        for index, value in enumerate(paymentSchedules):
                            exact_yield = SimuYieldCurve.iloc[j].loc[value]
                            pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield)))**(value/12)) 
                        
                        # calculating the bond price by summing up the present values and setting total future payments for the cb
                        bond_prices[bond_num] = np.sum(pvCashFlow)

            # *************************************************************************

                PL[j] = np.dot(num_assets_s,bond_prices) - fair_positions_at_new_interval

            return PL
            
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        print(f"\rHistorical Simulation for {len(self.maturity_s)} Bonds ----> In progress ...", end=" " * 90)

        PL = PlCalculator(SimuYieldCurve, interpolated_df, interval, self.num_assets_s, self.is_zcb_s,self.face_value_s,self.maturity_s, self.semi_annual_payment_s, self.coupon_rate_s)
        
        print("\rHistorical Simulation --->  Done", end=" " * 90)

        # losses
        losses = PL[PL<0]

        if len(losses) == 0:
            raise Exception("""
            No losses were generated in the simulation based on the current data and 'interval' settings.            
            Consider doing one or more of the following:

            1. Review the quality of your input data to ensure it's suitable for simulation.
            2. Slightly adjust the 'interval' parameter to potentially produce a different simulation outcome.
            """)

        # Value at Risk and Expected Shortfall 
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < - VaR]) * -1

        # consolidating results into a dictionary to return
        HS = {"var" : round(VaR, 4),
              "es:" : round(ES, 4),
              "T" : interval,
              "OrigPos" : self.initial_position}

        return HS
    
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 









        