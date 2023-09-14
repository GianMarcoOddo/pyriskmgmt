
# ------------------------------------------------------------------------------------------
# Author: GIAN MARCO ODDO
# Module: derivative_models
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
## 1. DerivaNprmSingle ###################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class DerivaNprmSingle:

    """
    Main
    ----
    This class is designed to model Non-Parametric Risk Measures for a single derivative position. The class performs calculations
    for the Value at Risk (VaR) and Expected Shortfall (ES) using either the quantile or bootstrap method.
    Furthermore, this class also allows for fitting the generalized Pareto distribution (GPD) to the right-hand tail 
    of the loss distribution to perform an Extreme Vlue Theory (EVT) analysis.

    The interval of the VaR/Es calculation follows the frequency of the provided returns.
    
    Initial Parameters
    ------------------
    - returns: Array with returns for the single derivative position.
    - position: The position on the single derivative. A positive value indicates a long position, a negative value indicates a short position.  
      - The position you possess can be determined by multiplying the single derivative's present-day price by the quantity you hold.
    - alpha: The significance level for the risk measure calculations (VaR and ES). Alpha is the probability of the
    occurrence of a loss greater than or equal to VaR or ES. Default is 0.05.
    - method: The method to calculate the VaR and ES. The possible values are "quantile" or "bootstrap". Default is "quantile".
    - n_bootstrap_samples: (default=10000) The number of bootstrap samples to use when method is set to "bootstrap". Default is 1000.

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
        Calculates and returns a summary of various risk measures for the given derivative position.
        This method calculates the maximum loss, maximum excess loss (over VaR), maximum loss over VaR, 
        and the Expected Shortfall over VaR. These measures are used to give an overall picture of the risk 
        associated with the derivative. All these measures are rounded off to 4 decimal places for readability.
        
        Returns
        -------
        - summary : dict : A dictionary containing the following key-value pairs:
            - 'var': The Value at Risk (VaR) for the derivative.
            - 'maxLoss': The maximum loss for the derivative.
            - 'maxExcessLoss': The maximum excess loss (over VaR) for the derivative.
            - 'maxExcessLossOverVar': The ratio of the maximum excess loss to VaR.
            - 'es': The Expected Shortfall for the derivative.
            - 'esOverVar': The ratio of the Expected Shortfall to VaR.
            
        Note
        ---- 
        The method calculates these measures differently depending on whether the position on the derivative is long (positive), 
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
        The method calculates these measures differently depending on whether the position on the derivative is long (positive), 
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
## 2. DerivaNprmPort ###############################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

class DerivaNprmPort:

    """
    Main
    ----
    The `DerivaNprmPort` class is a non-parametric risk management class designed to provide risk measures for a portfolio of derivatives' positions.
    The non-parametric approach does not rely on any assumptions regarding the underlying distribution of derivatives returns. 

    The interval of the VaR/Es calculation follows the frequency of the provided returns.

    Initial Parameters
    ------------------
    - returns: A numpy array of derivatives returns.
    - positions: A list of derivatives positions.
      - For each derivative: the position you possess can be determined by multiplying the derivatives's present-day price by the quantitiy you hold.
    - alpha: The significance level for VaR and ES calculations. Default is 0.05.
    - method: A method to be used for the VaR and ES calculations. The options are "quantile" for quantile-based VaR and ES calculation or "bootstrap" for a 
    bootstrap-based calculation. Default is "quantile".
    - n_bootstrap_samples: The number of bootstrap samples to be used when the bootstrap method is selected. This is ignored when the quantile method is selected. Default is 1000.

    Methods
    --------
    - summary(): Provides a summary of the risk measures, including VaR, Expected Shortfall (ES), Maximum Loss, Maximum Excess Loss, and ratios of these measures.
    - evt(): Provides risk measures using Extreme Value Theory (EVT). This is a semi-parametric approach that fits a Generalized Pareto Distribution (GPD) to the tail 
    of the loss distribution.
    - MargVars(): Provides the marginal VaRs for each derivative position in the portfolio. This is calculated as the change in portfolio VaR resulting from a small 
    change in the position of a specific derivative.
    
    Attributes
    ---------
    - var: The calculated Value at Risk (VaR) for the portfolio.
    - es: The calculated Expected Shortfall (ES) for the portfolio.
    - port_returns: Cumulative returns of the portfolio based on the derivatives returns and positions. 

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
        Computes various risk metrics for the portfolio of derivatives and returns them in a dictionary.
        
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
        Computes the Marginal Value at Risk (MVaR) for each derivative in the portfolio using a non-parametric approach.
        MVaR measures the rate of change in the portfolio's VaR with respect to a small change in the position of a specific derivative. 
        This method works by perturbing each derivative's position by a small amount in the same direction (proportional to a scale_factor), calculating the new VaR,
        and measuring the difference from the original VaR. The result is an array of MVaRs, one for each derivative.
        - New position = old position +/- (old position * scale_factor) for each position.

        Parameters
        ----------
        - scale_factor: The scale factor used to adjust the derivative positions for MVaR calculations. Default is 0.1.

        Returns
        -------
        - A dictonary with: 
          - The calculated VaR
          - A numpy array representing the MVaR for each derivative in the portfolio. MVaRs are rounded to four decimal places.

        Note
        -----
        This method can only be used when the 'method' attribute of the class instance is set to 'quantile'. For the 'bootstrap' method, it raises a ValueError.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np

        if self.method == "bootstrap": 
            raise ValueError("The 'MargVars' method can only be used when the 'self.method', in the `DerivaNprmPort` class, is set to 'quantile'.")

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

##########################################################################################################################################################################
##########################################################################################################################################################################
## 3. EuOptionRmSingle ###################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EuOptionRmSingle:

    """
    Main
    ----
    EuOptionRmSingle provides a comprehensive framework for assessing the risk of European options using the Black-Scholes methodology.
    This class facilitates the calculation of both call and put option prices, their respective Greeks, and a suite of popular risk measures via various methods.

    Initial Parameters
    ------------------
    - K: Strike price of the option.
    - T: Time to expiration (in years). For instance: for a 8-day contract use 8/252
    - S0: Initial stock price at the time t = 0.    
    - r: Risk-free interest rate (annualized).
    - sigma: Volatility of the stock price (annualized).
    - option_type: Type of the option. Acceptable values are 'call' for call options and 'put' for put options.

    Methods
    -------
    - DeltaNormal(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) using the Delta-Normal approach.
    - HistoricalSimulation(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) using historical simulations based on past return scenarios.
    - mcModelDrawReturns(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) by simulating future stock returns using a Monte Carlo method based on various 
        statistical models for volatility and mean returns.
    - mcModelGbm(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) using a Monte Carlo simulation based on the Geometric Brownian Motion (GBM) model.
    - kupiec_test(): Conducts the Kupiec backtest for Value at Risk (VaR). The Kupiec test is a likelihood ratio test used to validate the accuracy of VaR models.
    - christoffersen_test(): Conducts the Christoffersen test for Value at Risk (VaR). The test assesses the accuracy of VaR models by evaluating the clustering of violations, 
    where a violation occurs when the actual return is less than the predicted VaR.
    - combined_test(): Combines the Kupiec and Christoffersen tests to provide a comprehensive evaluation of Value at Risk (VaR) model accuracy.

    Attributes
    ----------
    - d1, d2: These are intermediate calculated values used in the Black-Scholes formula.
    - bs_price: The Black-Scholes option price. This is the theoretical price of the option.
    - greeks: A dictionary containing the option's Greeks (delta, gamma, vega, theta, rho). These are metrics that describe the sensitivity of the option's price to various factors.
        - delta: Measures the rate of change of the option value with respect to changes in the underlying asset's price.
        - gamma: Measures the rate of change in the delta with respect to changes in the underlying price.
        - vega: Measures sensitivity to volatility.
        - theta: Measures the sensitivity of the option price to changes in time to expiry.
        - rho: Measures the sensitivity of the option price to changes in the interest rate.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************
    
    def __init__(self, K: Union[int,float],
                       T: Union[int,float],
                       S0: Union[int,float],
                       r: float,
                       sigma: float,
                       option_type: str):
        
        from pyriskmgmt.inputsControlFunctions import (check_K, check_T,check_S0, check_r, check_sigma_deri, check_option_type)

        # check procedure
        self.K = check_K(K) ; self.T = check_T(T) ; self.S0 = check_S0(S0) ; self.r = check_r(r) ; self.sigma = check_sigma_deri(sigma) 
        self.option_type = check_option_type(option_type)

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.stats import norm

        # calculation of d1 and d2 parameters
        self.d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T)) ; self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

        # calculation of the option price
        if self.option_type == 'call':
            bs_price = self.S0 * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2) ; self.bs_price = round(bs_price, 4)

        elif self.option_type == 'put':
            bs_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S0 * norm.cdf(-self.d1) ; self.bs_price = round(bs_price, 4)

        # calculation of the Greeks

        # The most common of the Greeks are the first order derivatives: delta, vega, theta and rho as well as gamma, a second-order derivative of the value function
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        # delta measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.
        # gamma measures the rate of change in the delta with respect to changes in the underlying price
        # vega  measures sensitivity to volatility. Vega is the derivative of the option value with respect to the volatility of the underlying asset.
        # theta measures the sensitivity of the option price to changes in time to expiry.
        # rho measures the sensitivity of the option price to changes in the interest rate.
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------

        if self.option_type == 'call':
            self.greeks = {
                'delta' : round(norm.cdf(self.d1), 6),
                'gamma' : round(norm.pdf(self.d1) / (self.S0 * self.sigma * np.sqrt(self.T)), 6),
                'vega' : round(self.S0 * norm.pdf(self.d1) * np.sqrt(self.T), 6),
                'theta' : round(-((self.S0 * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))) - (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)), 6),
                'rho' : round(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2), 6)
            }

        elif self.option_type == 'put':
            self.greeks = {
                'delta' : round(-norm.cdf(-self.d1), 4),
                'gamma' : round(norm.pdf(self.d1) / (self.S0 * self.sigma * np.sqrt(self.T)), 6),
                'vega' : round(self.S0 * norm.pdf(self.d1) * np.sqrt(self.T), 6),
                'theta' : round(-((self.S0 * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))) + (self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)), 6),
                'rho' : round(-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2), 6)
            }

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # DELTA NORMAL VAR <<<<<<<<<<<<<<<

    def DeltaNormal(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, vol: str = "simple",
                    alpha: float = 0.05, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, warning: bool = True):
        
        """
        Main
        ----
        DeltaNormal computes the Value at Risk (VaR) and Expected Shortfall (ES) for a European option, using the Delta-Normal method, 
        based on the provided input parameters. This method supports various volatility estimation techniques, 
        such as simple historical volatility, GARCH (p, q), and Exponentially Weighted Moving Average (EWMA).

        Parameters
        ----------
        - num_options : The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets : Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval : The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - vol : method for volatility estimation. Acceptable values are 'simple' for simple historical volatility, 'garch' for GARCH (p, q), and 'ewma' for EWMA. (default is "simple")
        - alpha : The significance level for VaR and ES computations. Default is 0.05.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The oorder of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma : The decay factor for the 'ewma' volatility method. Default is 0.94.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - DELTA_NORMAL : A dictionary containing the computed VaR, ES, time horizon, delta, underlying asset position, option position, and the Black-Scholes option price.

        Notes
        --------
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, validate_returns_single , check_interval, check_vol,
                                                       check_alpha, check_warning)
        
        from scipy.stats import norm ; import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; vol = check_vol(vol) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the DeltaNormal() method.")

        # determining the volatility of the underlying asset *************************************************************************************************

        # calculating simple historical volatility -----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        if vol == "simple":
            # checking if the interval is one day and using the standard deviation of the returns
            if interval == 1:
                sigma_stock = np.std(underlying_rets)
            # adjusting the standard deviation by the square root of the interval for intervals greater than one day
            elif interval > 1:
                sigma_stock = np.std(underlying_rets) * np.sqrt(interval) # assuming future returns are i.i.d with normal distribution

        # estimating GARCH (p,q) volatility ------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        elif vol == "garch":
            # importing necessary functions for parameter validation
            from pyriskmgmt.inputsControlFunctions import (check_p, check_q)

            # validating the p and q values
            p = check_p(p) ; q = check_q(q) 

            # importing required libraries for GARCH modeling
            import os ; import sys ; import numpy as np ; from scipy.stats import norm ; from arch import arch_model

            # defining the GARCH model with specific parameters
            model = arch_model(underlying_rets, vol="GARCH", p=p, q=q, power= 2.0, dist = "normal") 

            # silencing any warnings by redirecting stderr to devnull
            stderr = sys.stderr ; sys.stderr = open(os.devnull, 'w')

            # fitting the GARCH model to the data
            try:
                model_fit = model.fit(disp='off')
            except Exception as e:
                # displaying an error message if the GARCH fitting process fails
                print("Garch fitting did not converge:", str(e))
                return
            finally:
                # restoring the standard error output
                sys.stderr = stderr

            # setting the forecasting horizon to one day
            horizon = 1

            # forecasting future variances using the GARCH model
            forecasts = model_fit.forecast(start=0, horizon=horizon)

            # extracting the variance forecast
            variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

            # adjusting the variance based on the interval 
            if interval == 1:
                sigma_stock = np.sqrt(variance_forecasts)
            elif interval > 1:
                cumulative_variance = variance_forecasts * interval # assuming future returns are i.i.d with normal distribution
                sigma_stock = np.sqrt(cumulative_variance)

        # calculating EWMA (lambda = lambda_ewma) volatility -------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        elif vol == "ewma":
            # importing the necessary function for parameter validation
            from pyriskmgmt.inputsControlFunctions import check_lambda_ewma

            # validating the lambda_ewma value
            lambda_ewma = check_lambda_ewma(lambda_ewma) 

            # initializing a variance time series based on the first return in the dataset
            n = len(underlying_rets)
            variance_time_series = np.repeat(underlying_rets[0]**2, n)

            # calculating the EWMA variance for the entire dataset
            for t in range(1,n):
                variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * underlying_rets[t-1]**2

            # projecting the variance for the next period
            variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * underlying_rets[-1]**2

            # adjusting the projected variance based on the interval 
            if interval == 1: 
                sigma_stock = np.sqrt(variance_t_plus1)    
            if interval >1 : 
                sigma_stock = np.sqrt(variance_t_plus1) * np.sqrt(interval) # assuming future returns are i.i.d with normal distribution

        # continuing with the option's delta normal approach *****************************************************************************

        # Note: for a single option position, we use the absolute delta irrespective of the option type (call or put)
        delta = abs(self.greeks["delta"])

        # computing the effective stock position using the option's delta, the number of options, and the contract size
        stockPosition = delta * abs(num_options) * contract_size * self.S0

        # determining the quantile and q_es values for risk metrics calculations
        quantile = norm.ppf(1-alpha, loc=0, scale=1) 
        q_es = (( np.exp( - (quantile**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha))))

        # computing Value at Risk and Expected Shortfall using the delta-normal method
        VaR = quantile * stockPosition * sigma_stock 
        ES = q_es * stockPosition * sigma_stock 

        # packaging all results into a dictionary and returning it
        DELTA_NORMAL = {"var" : round(VaR, 4),
                        "es:" :round(ES, 4),
                        "T" : interval,
                        "ExpInDays" : int(self.T * 252),
                        "AbsDelta" : delta,
                        "UnderPos" : round(stockPosition,4),
                        "OptPos" : round(num_options * contract_size *  self.bs_price,4),
                        "BsPrice" : self.bs_price}

        return DELTA_NORMAL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05, warning: bool = True):

        """
        Main
        ----
        HistoricalSimulation computes the Value at Risk (VaR) and Expected Shortfall (ES) for a European option using the historical simulation method, 
        based on the provided input parameters. 
        The method leverages past return scenarios to simulate future option positions, then derives risk measures from these simulated outcomes.

        Parameters
        ----------
        - num_options : The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets : Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval : The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - alpha: The significance level for VaR and ES computations. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - HS: A dictionary containing the computed VaR, ES, time horizon, number of simulations, option position at time t=0, and the Black-Scholes option price.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, check_interval, check_alpha,
                                                    validate_returns_single, check_warning)

        import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha); check_warning(warning) 

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the HistoricalSimulation() method.")
            
        # ---------------------------------------------------------------------------------------------------------------------------------------------------

        # calculating the number of complete intervals in the underlying returns data
        num_periods = len(underlying_rets) // interval * interval  
        # trimming the underlying returns to fit the complete intervals
        trimmed_returns = underlying_rets[len(underlying_rets) - num_periods:]
        # reshaping the trimmed returns data to form a matrix of return intervals
        reshaped_returns = trimmed_returns.reshape((-1, interval))  
        # computing the product of returns for each interval and adjusting it to represent return rate
        underlying_rets = np.product(1 + reshaped_returns, axis=1) - 1

        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # initializing an array to store simulated stock prices
        SimuStockPrices = np.zeros(underlying_rets.shape[0]) 

        # iterating through the reshaped returns to calculate the simulated stock prices
        for t in range(underlying_rets.shape[0]):
            SimuStockPrices[t] = self.S0 * (1 + underlying_rets[t])

        # importing the BlackAndScholesPrice function for option pricing
        from pyriskmgmt.SupportFunctions import BlackAndScholesPrice # BlackAndScholesPrice(St, T, K, sigma, r, option_type)

        # adjusting the time to maturity based on the length of the interval
        NewT = ((252 * self.T) -  interval) / 252

        # computing the initial option position value
        OptionPos0 = self.bs_price * num_options * contract_size 

        # initializing an array to store simulated option positions
        SimuOptionPos = np.zeros(underlying_rets.shape[0])

        # iterating through the simulated stock prices to compute the simulated option positions
        for index, value in enumerate(SimuStockPrices):
            OptionPrice = BlackAndScholesPrice(St = value, T = NewT, K = self.K, sigma = self.sigma, r = self.r, option_type = self.option_type)
            SimuOptionPos[index] = OptionPrice * num_options * contract_size

        # initializing an array to store the profit or loss of each option position
        PL = np.zeros(len(SimuOptionPos))

        # iterating through the simulated option positions to calculate the profit or loss
        for index, value in enumerate(SimuOptionPos):
            PL[index] = value - OptionPos0

        # computing the Value-at-Risk (VaR) for the specified alpha level
        VaR = np.quantile(PL, alpha) * -1 
        # computing the Expected Shortfall (ES) based on VaR
        ES = np.mean(PL[PL < - VaR]) * -1 

        # preparing the results in a dictionary
        HS = {"var" : round(VaR, 4),
              "es:" : round(ES, 4),
              "T" : interval,
              "ExpInDays" : int(self.T * 252),
              "NumSims" : len(underlying_rets),
              "OptPos0" : round(num_options * contract_size *  self.bs_price,4),
              "BsPrice" : self.bs_price}

        return HS

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Monte Carlo Simulation <<<<<<<<<<<<<<<

    def mcModelDrawReturns(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05,
                           vol: str = "simple", mu: str = "moving_average", num_sims: int = 1000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, 
                           ma_window: int = 10,  warning: bool = True):
        
        """
        Main
        ----
        mcModelDrawReturns computes the Value at Risk (VaR) and Expected Shortfall (ES) for a European option using a Monte Carlo simulation method based on 
        historical returns and user-specified parameters for volatility and mean calculations. The method generates a series of possible future returns 
        using a Monte Carlo approach, simulates the option's behavior under these return scenarios, and then extracts risk measures from these simulations.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets: Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval: The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - alpha: The significance level for VaR and ES computations (default is 0.05).
        - vol: Method for volatility estimation. Can be 'simple' for simple historical volatility, 'garch' for GARCH (p, q), and 'ewma' for EWMA. Default is "simple".
        - mu: Method for mean returns calculation. Can be 'moving_average' (default) or other techniques based on user specifications.
        - num_sims: Number of Monte Carlo simulations to be performed. Default is 1000.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The order of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma: The decay factor for the 'ewma' volatility method. Default is 0.94.
        - ma_window: Window for the 'moving_average' mean calculation. Default is 10.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - MC_DIST: A dictionary containing the computed VaR, ES, time horizon, number of simulations, option position at time t=0, the Black-Scholes option 
        price, mean return of the underlying, and cumulative sigma of the underlying.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, validate_returns_single,  check_interval, check_alpha, 
                                                      check_warning)
        
        import numpy as np 

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning) 

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the mcModelDrawReturns() method.")

        warning = False
        
        # this method relies upon the EquityPrmSingle class from the pyriskmgmt.equity_models module
        from pyriskmgmt.equity_models import EquityPrmSingle

        # initializing the single equity risk management model from the provided returns
        equity_rm = EquityPrmSingle(returns=underlying_rets, position=1, interval=interval, alpha=alpha)  # note: position doesn't matter here

        # generating a Monte Carlo model for the equity and drawing returns from it
        model = equity_rm.mcModelDrawReturns(vol=vol, mu=mu, num_sims=num_sims, p=p, q=q, lambda_ewma=lambda_ewma, ma_window=ma_window,
                                            return_mc_returns=True)

        # extracting Monte Carlo simulated returns and updated mu and sigma values
        MonteCarloRets = model[1]
        mu = model[0]["mu"]
        sigma = model[0]["cum_sigma"]

        # initializing an array for simulated stock prices
        SimuStockPrices = np.zeros(MonteCarloRets.shape[0])

        # simulating stock prices using Monte Carlo returns
        for t in range(MonteCarloRets.shape[0]):
            SimuStockPrices[t] = self.S0 * (1 + MonteCarloRets[t])

        from pyriskmgmt.SupportFunctions import BlackAndScholesPrice  # importing Black-Scholes pricing function

        # recalculating the time to expiration for the option after adjusting for the interval
        NewT = ((252 * self.T) - interval) / 252

        # computing the initial option position value
        OptionPos0 = self.bs_price * num_options * contract_size

        # initializing an array for simulated option positions
        SimuOptionPos = np.zeros(MonteCarloRets.shape[0])

        # simulating option positions based on simulated stock prices
        for index, value in enumerate(SimuStockPrices):
            OptionPrice = BlackAndScholesPrice(St=value, T=NewT, K=self.K, sigma=self.sigma, r=self.r, option_type=self.option_type)
            SimuOptionPos[index] = OptionPrice * num_options * contract_size

        # initializing an array for profit and loss calculations
        PL = np.zeros(len(SimuOptionPos))

        # calculating profit and loss for each simulated option position
        for index, value in enumerate(SimuOptionPos):
            PL[index] = value - OptionPos0

        # calculating Value at Risk (VaR) and Expected Shortfall (ES) from the simulated profit and loss
        VaR = np.quantile(PL, alpha) * -1
        ES = np.mean(PL[PL < -VaR]) * -1

        # consolidating the results into a dictionary
        MC_DIST = {"var" : round(VaR, 4),
                   "es:" : round(ES, 4),
                   "T" : interval,
                   "ExpInDays" : int(self.T * 252),
                   "NumSims" : len(MonteCarloRets),
                   "OptPos0" : round(num_options * contract_size * self.bs_price, 4),
                   "BsPrice" : self.bs_price,
                   "muUnder" : mu,
                   "SigmaCumUnder" : sigma}

        return MC_DIST

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Monte Carlo Geometric Brownian Motion <<<<<<<<<<<<<<<

    def mcModelGbm(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05,
                           vol: str = "simple", mu: str = "moving_average", num_sims: int = 1000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, 
                           ma_window: int = 10,  warning: bool = True):
        
        """
        Main
        ----
        The mcModelGbm method computes the Value at Risk (VaR) and Expected Shortfall (ES) for a European option using the Geometric Brownian Motion (GBM) 
        Monte Carlo simulation model. The method projects future stock prices with GBM, simulates option prices based on these stock prices, 
        and then derives risk measures (VaR and ES) from the simulated option prices.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets : Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval : The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - alpha: The significance level for VaR and ES computations (default is 0.05).
        - vol: Method for volatility estimation. Can be 'simple' for simple historical volatility, 'garch' for GARCH (p, q), and 'ewma' for EWMA. Default is "simple".
        - mu: Method for mean returns calculation. Can be 'moving_average' (default) or other techniques based on user specifications.
        - num_sims: Number of Monte Carlo simulations to be performed. Default is 1000.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The order of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma: The decay factor for the 'ewma' volatility method. Default is 0.94.
        - ma_window: Window for the 'moving_average' mean calculation. Default is 10.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - MC_GBM: Dictionary containing computed VaR, ES, time horizon, number of simulations, option position at t=0, and the Black-Scholes option price.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.

        Remember that the GBM model assumes constant volatility and normally distributed returns, which might be more applicable for shorter time horizons.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, validate_returns_single,  check_interval, check_alpha,check_warning)
        
        import numpy as np 

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("Moreover, models like the Geometric Brownian Motion (GBM) can be effective at modeling short-term price dynamics,")
            print("since they assume constant volatility and normally distributed returns. ")
            print("Over short periods, these assumptions may hold reasonably well, leading to similar VaR estimates as models that")
            print("do not make these simplifications.")
            print("To mute this warning, add 'warning = False' as an argument of the mcModelGbm() function")

        warning = False

        # this method relies upon the EquityPrmSingle class from the pyriskmgmt.equity_models module
        from pyriskmgmt.equity_models import EquityPrmSingle

        # initializing the single equity risk management model with the provided returns
        equity_rm = EquityPrmSingle(returns=underlying_rets, position=1, interval=interval, alpha=alpha)

        # generating a Geometric Brownian Motion (GBM) model for the equity
        GBM_PATH = equity_rm.mcModelGbm(S0=self.S0, vol=vol, mu=mu, num_sims=num_sims, p=p, q=q, lambda_ewma=lambda_ewma, ma_window=ma_window,
                                        return_gbm_path=True)

        # extracting stock prices from the GBM model path
        SimuStockPrices = GBM_PATH[1]

        # extracting drift and sigma values from the GBM model
        drift = GBM_PATH[0]["GbmDrift"] ; sigma = GBM_PATH[0]["GbmSigma"]

        from pyriskmgmt.SupportFunctions import BlackAndScholesPrice  # importing Black-Scholes pricing function

        # recalculating the time to expiration for the option after adjusting for the interval
        NewT = ((252 * self.T) - interval) / 252

        # computing the initial option position value
        OptionPos0 = self.bs_price * num_options * contract_size

        # initializing an array for simulated option positions
        SimuOptionPos = np.zeros(SimuStockPrices.shape[0])

        # simulating option positions based on the GBM stock prices
        for index, value in enumerate(SimuStockPrices):
            OptionPrice = BlackAndScholesPrice(St=value, T=NewT, K=self.K, sigma=self.sigma, r=self.r, option_type=self.option_type)
            SimuOptionPos[index] = OptionPrice * num_options * contract_size

        # initializing an array for profit and loss calculations
        PL = np.zeros(len(SimuOptionPos))

        # calculating profit and loss for each simulated option position
        for index, value in enumerate(SimuOptionPos):
            PL[index] = value - OptionPos0

        # calculating Value at Risk (VaR) and Expected Shortfall (ES) from the simulated profit and loss
        VaR = np.quantile(PL, alpha) * -1
        ES = np.mean(PL[PL < -VaR]) * -1

        # consolidating the results into a dictionary
        MC_GBM = {"var" : round(VaR, 4),
                  "es:" : round(ES, 4),
                  "T" : interval,
                  "ExpInDays" : int(self.T * 252),
                  "NumSims" : len(SimuStockPrices),
                  "OptPos0" : round(num_options * contract_size * self.bs_price, 4),
                  "BsPrice" : self.bs_price,
                  "GbmDrift" : round(drift, 4),
                  "GbmSigma" : round(sigma, 4)}

        return MC_GBM
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_options: int, contract_size: int, S0_initial: float, test_returns: np.ndarray, var: float,interval: int = 1, alpha: float = 0.05,
                    alpha_test: float = 0.05, warning = True):
        
        """
        Main
        ----
        kupiec_test performs the Kupiec's Proportional Of Failures (POF) test for backtesting the accuracy of VaR predictions. The test assesses if the number of 
        times the losses exceed the VaR prediction is consistent with the confidence level used to compute the VaR. If the number of exceptions is too high, 
        it indicates that the VaR measure may be understating the risk.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - S0_initial: Initial stock price or the starting value of the underlying asset at the beginning of the "test_returns" array.
        - test_returns: Array containing the returns of the underlying asset used for the backtesting. Ensure that the frequency of the "test_returns" array is daily.
        - var: The Value at Risk (VaR) measure being tested.
        - interval: The time horizon (in days, assuming 252 trading days) over which the risk measure was computed. Default is 1.
        - alpha: The significance level used in the VaR calculation. Default is 0.05.
        - alpha_test: The significance level for the chi-square test. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'test_returns' frequency for the analysis. Default is True.

        Returns
        -------
        - KUPIEC_TEST: A dictionary containing the test's name, computed VaR, time horizon, number of observations, number of exceptions, theoretical % of 
        exceptions based on alpha, actual % of exceptions, likelihood ratio, critical value for the test, and a boolean indicating if the null hypothesis 
        should be rejected.

        Notes
        -----
        Please ensure that the 'test_returns' frequency is 'daily'.
        For instance:
        - for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.
        - for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. 
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, check_S0_initial, validate_returns_single,check_var,
                                                       check_alpha, check_alpha_test, check_warning, check_interval)
        
        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; test_returns = validate_returns_single(test_returns) 
        var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) 
        S0_initial = check_S0_initial(S0_initial) ; check_warning(warning) 

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to backtest the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' frequency is 'daily'.")
            print("For instance:")
            print("- for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.")
            print("- for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. ")
            print("To hide this warning, use 'warning=False' when calling the kupiec_test() function.")

        # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------
        
        def reshape_and_aggregate_returns(returns, interval):
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        test_returns = reshape_and_aggregate_returns(test_returns,interval)

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        from pyriskmgmt.SupportFunctions import BlackAndScholesPrice  # Importing the Black-Scholes pricing function

        # computing the stock price path using the initial stock price (S0_initial) and the given returns
        stockPricePath_without_initial = S0_initial * np.cumprod(1 + test_returns)
        # inserting the initial stock price at the beginning of the path
        stockPricePath = np.insert(stockPricePath_without_initial, 0, S0_initial)

        # initializing an array to store option positions at each time t
        OptionPositionTimet = np.zeros(len(stockPricePath))

        # calculating the option position value for each stock price in the path
        for t in range(stockPricePath.shape[0]):
            OptionPositionTimet[t] = BlackAndScholesPrice(St=stockPricePath[t], T=self.T, K=self.K, sigma=self.sigma, r=self.r,
                                                        option_type=self.option_type) * num_options * contract_size

        # computing the profit and loss by finding the difference in consecutive option positions
        PL = np.diff(OptionPositionTimet)
        # counting the number of instances where the profit and loss is less than the negative Value at Risk (VaR)
        num_exceptions = np.sum(PL < -VaR)

        # total number of observations and % exceptions
        num_obs = len(PL) ; per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "T" : interval,                         # interval
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # CHRISTOFFESEN TEST <<<<<<<<<<<<

    def christoffersen_test(self, num_options: int, contract_size: int, S0_initial: float, test_returns: np.ndarray, var: float, interval: int = 1, alpha: float = 0.05,
                    alpha_test: float = 0.05, warning = True):
        
        """
        Main
        ----
        The christoffersen_test method conducts Christoffersen's Conditional Coverage test for backtesting the accuracy of VaR predictions. The test examines both the 
        unconditional coverage (like Kupiec's POF) and the independence of exceptions. The test checks if exceptions are clustering or if they occur independently 
        over time. It uses a likelihood ratio method to compare the likelihood of observing the given sequence of exceptions under the assumption of independent 
        exceptions versus the likelihood under the observed conditional probabilities.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - S0_initial: Initial stock price or the starting value of the underlying asset at the beginning of the "test_returns" array.
        - test_returns: Array containing the returns of the underlying asset used for the backtesting. Ensure that the frequency of the "test_returns" array is daily.
        - var: The Value at Risk (VaR) measure being tested.
        - interval: The time horizon (in days, assuming 252 trading days) over which the risk measure was computed. Default is 1.
        - alpha: The significance level used in the VaR calculation. Default is 0.05.
        - alpha_test: The significance level for the chi-square test. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'test_returns' frequency for the analysis. Default is True.

        Returns
        -------
        - CHRISTOFFERSEN_TEST: A dictionary containing the test's name, computed VaR, likelihood ratio, critical value for the test, and a boolean indicating 
        if the null hypothesis should be rejected.

        Notes
        -----
        Please ensure that the 'test_returns' frequency is 'daily'.
        For instance:
        - for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.
        - for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. 
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, check_S0_initial, validate_returns_single,check_var,
                                                       check_alpha, check_alpha_test, check_warning, check_interval)

        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; test_returns = validate_returns_single(test_returns) 
        var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) 
        S0_initial = check_S0_initial(S0_initial) ; check_warning(warning) 


        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to backtest the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' frequency is 'daily'.")
            print("For instance:")
            print("- for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.")
            print("- for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. ")
            print("To hide this warning, use 'warning=False' when calling the christoffersen_test() function.")


        # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

        def reshape_and_aggregate_returns(returns, interval):
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        test_returns = reshape_and_aggregate_returns(test_returns,interval)

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        from pyriskmgmt.SupportFunctions import BlackAndScholesPrice  # importing the Black-Scholes pricing function

        # calculating the stock price path without the initial value by multiplying the initial stock price with the cumulative product of returns
        stockPricePath_without_initial = S0_initial * np.cumprod(1 + test_returns)
        # inserting the initial stock price at the beginning of the path to get the full path
        stockPricePath = np.insert(stockPricePath_without_initial, 0, S0_initial)

        # initializing an array to store the value of the option position at each time 't'
        OptionPositionTimet = np.zeros(len(stockPricePath))
                        
        # looping through each stock price in the path
        for t in range(stockPricePath.shape[0]):
            # calculating and storing the value of the option position at time 't' using the Black-Scholes pricing formula
            OptionPositionTimet[t] = BlackAndScholesPrice(St = stockPricePath[t], T = self.T, K = self.K, sigma = self.sigma, r = self.r,
                                                        option_type = self.option_type) * num_options * contract_size 

        # computing the daily changes in the option position's value (Profit/Loss)
        PL = np.diff(OptionPositionTimet)
        # determining which days had losses greater than the given Value at Risk (VaR)
        exceptions = PL < - VaR

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
        p = alpha
        likelihood_ratio = -2 * (np.log((1 - p) ** (T_00 + T_10) * p ** (T_01 + T_11)) - np.log((1 - p_b0) ** T_00 * p_b0 ** T_01 * (1 - p_b1) ** T_10 * p_b1 ** T_11))

        # critical value from chi-squared distribution
        critical_value = chi2.ppf(1 - alpha_test, df=1)

        # reject the null hypothesis if likelihood ratio is higher than the critical value
        reject_null = likelihood_ratio > critical_value

        return {"test" : "Christoffersen",
                "var" : var,
                "LR" : round(likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # COMBINED TEST <<<<<<<<<<<<

    def combined_test(self, num_options: int, contract_size: int, S0_initial: float, test_returns: np.ndarray, var: float, interval: int = 1, alpha: float = 0.05,
                    alpha_test: float = 0.05, warning = True):
        
        """
        Main
        ----
        The combined_test method conducts a combined backtesting test using both Kupiec's POF and Christoffersen's Conditional Coverage tests. This test leverages both 
        the POF test which tests for correct unconditional coverage and the conditional coverage test that checks for the independence of exceptions. By 
        combining the results of these two tests, the combined_test provides a more robust way to assess the accuracy of VaR predictions.
        The combined_test calls both the kupiec_test and christoffersen_test methods and aggregates their results. The combined likelihood ratio is derived 
        from summing the individual likelihood ratios from both tests. The chi-square test's critical value used here has 2 degrees of freedom, reflecting 
        the combination of the two individual tests.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - S0_initial: Initial stock price or the starting value of the underlying asset at the beginning of the "test_returns" array.
        - test_returns: Array containing the returns of the underlying asset used for the backtesting. Ensure that the frequency of the "test_returns" array is daily.
        - var: The Value at Risk (VaR) measure being tested.
        - interval: The time horizon (in days, assuming 252 trading days) over which the risk measure was computed. Default is 1.
        - alpha: The significance level used in the VaR calculation. Default is 0.05.
        - alpha_test: The significance level for the chi-square test. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'test_returns' frequency for the analysis. Default is True.

        Returns
        -------
        - COMBINED_TEST: A dictionary containing the test's name, computed VaR, combined likelihood ratio, critical value for the test, and a boolean indicating 
        if the null hypothesis should be rejected.

        Notes
        -----
        Please ensure that the 'test_returns' frequency is 'daily'.
        For instance:
        - for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.
        - for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. 
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        from scipy.stats import chi2

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to backtest the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' frequency is 'daily'.")
            print("For instance:")
            print("- for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.")
            print("- for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. ")
            print("To hide this warning, use 'warning=False' when calling the combined_test() function.")

        warning = False

        # no need for retuns aggregation here ---- it will be handled by both kupiec_test and christoffersen_test

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        kupiec_results = self.kupiec_test(num_options = num_options, contract_size = contract_size, S0_initial = S0_initial,
                                        test_returns = test_returns, var = VaR, interval = interval, alpha = alpha, alpha_test = alpha_test, 
                                        warning = warning)
        
        christoffersen_results = self.christoffersen_test(num_options = num_options, contract_size = contract_size, S0_initial = S0_initial,
                                        test_returns = test_returns, var = VaR, interval = interval, alpha = alpha, alpha_test = alpha_test, 
                                        warning = warning)
        
        # getting likelihood ratios
        kupiec_lr = kupiec_results["LR"] ; christoffersen_lr = christoffersen_results["LR"]

        # calculating combined likelihood ratio
        combined_likelihood_ratio = kupiec_lr + christoffersen_lr

        # critical value from chi-squared distribution with [[[2]]] degrees of freedom
        critical_value = chi2.ppf(1 - alpha_test, df=2)

        # if the combined likelihood ratio is greater than the critical value, then we reject the null hypothesis
        reject_null = combined_likelihood_ratio > critical_value

        return {"test" : "Kupiec_Christoffersen",
                "var" : VaR,
                "LR" : round(combined_likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}
    
##########################################################################################################################################################################
##########################################################################################################################################################################
## 4. AmOptionRmSingle ###################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class AmOptionRmSingle:

    """
    Main
    ----
    The AmOptionRmSingle class is designed to calculate the price and several Risk Measures of an American option using the binomial model. 
    
    The class handles both call and put options. If the computed price from the binomial model is lower than the European 
    option price (calculated using the Black-Scholes model), the class uses the European option price instead. This decision 
    was made necessary due to the computational cost of using a large-scale binomial model.
     
    This accounts for the fact that American option prices cannot be less than the prices of 
    their European counterparts due to the additional early exercise feature of American options.
    
    >>>> This is not a high-frequency trading (HFT) package.
    Therefore, approximations (TRADE-OFF) to the order of '0.00x' are considered acceptable for its applications <<<<<<

    Initial Parameters
    ------------------
    - K: Strike price of the option.
    - T: Time to expiration (in years assuming 252 trading days). For instance: for a 8-day contract use 8/252
    - S0: Initial stock price at the time t = 0.    
    - r: Risk-free interest rate (annualized).
    - sigma: Volatility of the stock price (annualized).
    - option_type: Type of the option. Acceptable values are 'call' for call options and 'put' for put options.

    Methods
    -------
    - HistoricalSimulation(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) using historical simulations based on past return scenarios.
    - mcModelDrawReturns(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) by simulating future stock returns using a Monte Carlo method based on various 
      statistical models for volatility and mean returns.
    - mcModelGbm(): Computes the Value at Risk (VaR) and Expected Shortfall (ES) using a Monte Carlo simulation based on the Geometric Brownian Motion (GBM) model.
    - kupiec_test(): Conducts the Kupiec backtest for Value at Risk (VaR). The Kupiec test is a likelihood ratio test used to validate the accuracy of VaR models.
    - christoffersen_test(): Conducts the Christoffersen test for Value at Risk (VaR). The test assesses the accuracy of VaR models by evaluating the clustering of violations, 
      where a violation occurs when the actual return is less than the predicted VaR.
    - combined_test(): Combines the Kupiec and Christoffersen tests to provide a comprehensive evaluation of Value at Risk (VaR) model accuracy.

    Attributes
    ----------
    - bi_price : The calculated option price using the binomial model. If this price is less than the Black-Scholes price, then the Black-Scholes price is used as the option price.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, K: Union[int,float],
                  T: Union[int,float],
                  S0: Union[int,float],
                  r: float, sigma: float,
                  option_type: str):

        
        from pyriskmgmt.inputsControlFunctions import (check_K, check_T, check_S0, check_r, check_sigma_deri, check_option_type)

        self.K = check_K(K) ; self.T = check_T(T) ; self.S0 = check_S0(S0) ; self.r = check_r(r) ; self.sigma = check_sigma_deri(sigma)
        self.option_type = check_option_type(option_type)

        self.fit()

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice

        steps = int( 252 * self.T ) 

        price = BiAmericanOptionPrice(K = self.K, T = self.T, S0 = self.S0, r = self.r, sigma = self.sigma, option_type = self.option_type, steps = steps)

        # Connecting BS to BI in the case that the BiAmericanOptionPrice is lower than the BlackAndScholesPrice --------------------------------------

        from pyriskmgmt.derivative_models import EuOptionRmSingle

        model = EuOptionRmSingle(K = self.K, T = self.T, S0 = self.S0, r = self.r, sigma = self.sigma, option_type = self.option_type)

        bsprice = model.bs_price 

        if price < bsprice: price = bsprice 
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------------

        self.bi_price = round(price, 4)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05, warning: bool = True):

        """
        Main
        ----
        HistoricalSimulation computes the Value at Risk (VaR) and Expected Shortfall (ES) for a American option using the historical simulation method, 
        based on the provided input parameters. The method leverages past return scenarios to simulate future option positions, then derives risk measures 
        from these simulated outcomes.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets : Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval : The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - alpha: The significance level for VaR and ES computations. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - HS: A dictionary containing the computed VaR, ES, time horizon, number of simulations, option position at time t=0, and the Binomial Model option price.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, check_interval, check_alpha,
                                                    validate_returns_single, check_warning)
        
        import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha); check_warning(warning)

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the HistoricalSimulation() method.")
    
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # calculating the number of complete intervals in the underlying returns data
        num_periods = len(underlying_rets) // interval * interval
        # trimming the underlying returns to fit the complete intervals
        trimmed_returns = underlying_rets[len(underlying_rets) - num_periods:]
        # reshaping the trimmed returns data to form a matrix of return intervals
        reshaped_returns = trimmed_returns.reshape((-1, interval))
        # computing the product of returns for each interval and adjusting it to represent return rate
        underlying_rets = np.product(1 + reshaped_returns, axis=1) - 1

        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # initializing a zeros array for simulated stock prices
        SimuStockPrices = np.zeros(underlying_rets.shape[0])

        # simulating stock prices based on the returns data
        for t in range(underlying_rets.shape[0]):
            SimuStockPrices[t] = self.S0 * (1 + underlying_rets[t])

        # importing the biamerican option pricing function
        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice

        # adjusting the time parameter based on intervals passed
        NewT = ((252 * self.T) - interval) / 252

        # calculating the initial option position value
        OptionPos0 = self.bi_price * num_options * contract_size

        # initializing a zeros array for simulated option positions
        SimuOptionPos = np.zeros(underlying_rets.shape[0])

        # calculating the number of steps for the biamerican option pricing function
        steps = int((252 * self.T)) - interval

        # computing the option prices for simulated stock prices
        for index, value in enumerate(SimuStockPrices):
            OptionPrice = BiAmericanOptionPrice(K=self.K, T=NewT, S0=value, r=self.r, sigma=self.sigma,
                                                 option_type=self.option_type, steps=steps)
            SimuOptionPos[index] = OptionPrice * num_options * contract_size

        # initializing an array to store profit and loss values
        PL = np.zeros(len(SimuOptionPos))

        # calculating profit and loss values for the simulated option positions
        for index, value in enumerate(SimuOptionPos):
            PL[index] = value - OptionPos0

        losses = PL[PL < 0]

        # computing the Value at Risk (VaR) from the profit and loss array
        VaR = np.quantile(losses, alpha) * -1
        # computing the Expected Shortfall (ES) from the profit and loss array
        ES = np.mean(losses[losses < - VaR]) * -1

        # compiling the results into a dictionary
        HS = {"var" : round(VaR, 4),
              "es:" : round(ES, 4),
              "T" : interval,
              "ExpInDays" : int(self.T * 252),
              "NumSims" : len(underlying_rets),
              "OptPos0" : round(num_options * contract_size * self.bi_price, 4),
              "BiPrice" : self.bi_price}

        return HS

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Monte Carlo Simulation <<<<<<<<<<<<<<<

    def mcModelDrawReturns(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05,
                           vol: str = "simple", mu: str = "moving_average", num_sims: int = 1000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, 
                           ma_window: int = 10,  warning: bool = True):
        
        """
        Main
        ----
        mcModelDrawReturns computes the Value at Risk (VaR) and Expected Shortfall (ES) for a American option using a Monte Carlo simulation method based on 
        historical returns and user-specified parameters for volatility and mean calculations. The method generates a series of possible future returns 
        using a Monte Carlo approach, simulates the option's behavior under these return scenarios, and then extracts risk measures from these simulations.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets : Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval : The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - alpha: The significance level for VaR and ES computations. Default is 0.05.
        - vol: Method for volatility estimation. Can be 'simple' for simple historical volatility, 'garch' for GARCH (p, q), and 'ewma' for EWMA. Default is "simple".
        - mu: Method for mean returns calculation. Can be 'moving_average' or "zero"/"constant" based on user specifications. Default is "moving_average".
        - num_sims: Number of Monte Carlo simulations to be performed. Default is 1000.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The order of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma: The decay factor for the 'ewma' volatility method. Default is 0.94.
        - ma_window: Window for the 'moving_average' mean calculation. Default is 10.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - MC_DIST: A dictionary containing the computed VaR, ES, time horizon, number of simulations, option position at time t=0, the Black-Scholes option 
        price, mean return of the underlying, and cumulative sigma of the underlying.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, validate_returns_single,  check_interval, check_alpha, 
                                                      check_warning)
        
        import numpy as np 

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning) 

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the mcModelDrawReturns() method.")

        warning = False

        # this method relies upon the EquityPrmSingle class from the pyriskmgmt.equity_models module
        from pyriskmgmt.equity_models import EquityPrmSingle

        # initializing the EquityPrmSingle class with given parameters
        # note: the position parameter does not have an impact on the result in this case
        equity_rm = EquityPrmSingle(returns=underlying_rets, position=1, interval=interval, alpha=alpha)

        # running the Monte Carlo simulation to model returns
        model = equity_rm.mcModelDrawReturns(vol=vol, mu=mu, num_sims=num_sims, p=p, q=q, 
                                             lambda_ewma=lambda_ewma, ma_window=ma_window, return_mc_returns=True)

        # extracting Monte Carlo simulated returns and statistics from the model
        MonteCarloRets = model[1] ; mu = model[0]["mu"] ; sigma = model[0]["cum_sigma"]

        # initiating the simulation for stock prices
        SimuStockPrices = np.zeros(MonteCarloRets.shape[0])

        # calculating simulated stock prices based on the Monte Carlo returns
        for t in range(MonteCarloRets.shape[0]):
            SimuStockPrices[t] = self.S0 * (1 + MonteCarloRets[t])

        # importing the BiAmericanOptionPrice function for option pricing
        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice

        # adjusting the time to expiration based on given parameters
        NewT = ((252 * self.T) - interval) / 252

        # calculating the initial option position
        OptionPos0 = self.bi_price * num_options * contract_size

        # initiating an array to hold simulated option positions
        SimuOptionPos = np.zeros(MonteCarloRets.shape[0])

        # defining the number of steps for the binomial option pricing model
        steps = int((252 * self.T)) - interval

        # simulating option positions based on the simulated stock prices
        for index, value in enumerate(SimuStockPrices):
            OptionPrice = BiAmericanOptionPrice(K=self.K, T=NewT, S0=value, r=self.r, sigma=self.sigma, option_type=self.option_type, steps=steps)
            SimuOptionPos[index] = OptionPrice * num_options * contract_size

        # initiating an array to hold profit/loss values
        PL = np.zeros(len(SimuOptionPos))

        # calculating profit/loss for each simulated option position
        for index, value in enumerate(SimuOptionPos):
            PL[index] = value - OptionPos0

        # calculating Value at Risk (VaR) and Expected Shortfall (ES) for the simulated profit/loss
        VaR = np.quantile(PL, alpha) * -1
        ES = np.mean(PL[PL < -VaR]) * -1

        # compiling all the results into a dictionary
        MC_DIST = {"var" : round(VaR, 4),
                   "es:" : round(ES, 4),
                   "T" : interval,
                   "ExpInDays" : int(self.T * 252),
                   "NumSims" : len(MonteCarloRets),
                   "OptPos0" : round(num_options * contract_size *  self.bi_price, 4),
                   "BiPrice" : self.bi_price,
                   "muUnder" : mu,
                   "SigmaCumUnder" : sigma}
        
        return MC_DIST
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Monte Carlo Geometric Brownian Motion <<<<<<<<<<<<<<<

    def mcModelGbm(self, num_options: int, contract_size: int, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05,
                           vol: str = "simple", mu: str = "moving_average", num_sims: int = 1000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, 
                           ma_window: int = 10,  warning: bool = True):
        
        """
        Main
        ----
        The mcModelGbm method computes the Value at Risk (VaR) and Expected Shortfall (ES) for a American option using the Geometric Brownian Motion (GBM) 
        Monte Carlo simulation model. The method projects future stock prices with GBM, simulates option prices based on these stock prices, 
        and then derives risk measures (VaR and ES) from the simulated option prices.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - underlying_rets : Array containing historical returns of the underlying asset. Ensure that the 'underlying_rets' frequency is 'daily'.
        - interval : The time in days (assuming 252 trading days in a year) over which the risk measure is to be computed. Default is 1.
        - alpha: The significance level for VaR and ES computations. Default is 0.05.
        - vol: Method for volatility estimation. Can be 'simple' for simple historical volatility, 'garch' for GARCH (p, q), and 'ewma' for EWMA. Default is "simple".
        - mu: Method for mean returns calculation. Can be 'moving_average' or "zero"/"constant" based on user specifications. Default is "moving_average".
        - num_sims: Number of Monte Carlo simulations to be performed. Default is 1000.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The order of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma: The decay factor for the 'ewma' volatility method. Default is 0.94.
        - ma_window: Window for the 'moving_average' mean calculation. Default is 10.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - MC_GBM: Dictionary containing computed VaR, ES, time horizon, number of simulations, option position at t=0, and the Binomial Model option price.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, validate_returns_single,  check_interval, check_alpha, 
                                                      check_warning)
        
        import numpy as np 

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; underlying_rets = validate_returns_single(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to calculate the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("Moreover, models like the Geometric Brownian Motion (GBM) can be effective at modeling short-term price dynamics,")
            print("since they assume constant volatility and normally distributed returns. ")
            print("Over short periods, these assumptions may hold reasonably well, leading to similar VaR estimates as models that")
            print("do not make these simplifications.")
            print("To mute this warning, add 'warning = False' as an argument of the mcModelGbm() function")

        warning = False

        # this method relies upon the EquityPrmSingle class from the pyriskmgmt.equity_models module
        from pyriskmgmt.equity_models import EquityPrmSingle

        # initializing the equity risk management model with specified parameters
        equity_rm = EquityPrmSingle(returns = underlying_rets, position = 1, interval = interval , alpha = alpha)

        # generating Geometric Brownian Motion (GBM) paths for the equity risk model with specified parameters
        GBM_PATH = equity_rm.mcModelGbm( S0 = self.S0, vol = vol, mu = mu, num_sims = num_sims, p = p, q = q, lambda_ewma = lambda_ewma, ma_window = ma_window,
                                                  return_gbm_path = True)
        
        # extracting simulated stock prices from the GBM path
        SimuStockPrices = GBM_PATH[1]

        # extracting drift and sigma (volatility) from the GBM path
        drift = GBM_PATH[0]["GbmDrift"] ; sigma = GBM_PATH[0]["GbmSigma"]

        # importing the function to compute the bivariate American option price
        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice

        # recalculating the time to maturity, accounting for the interval
        NewT = ( ( 252 * self.T ) -  interval ) / 252

        # computing the initial option position value
        OptionPos0 = self.bi_price * num_options * contract_size
        
        # initializing an array to store simulated option position values
        SimuOptionPos = np.zeros(SimuStockPrices.shape[0])

        # computing the number of steps to use in the option pricing function
        steps = int(( 252 * self.T )) - interval

        # simulating option prices for each stock price and storing the results in SimuOptionPos
        for index, value in enumerate(SimuStockPrices):
            OptionPrice = BiAmericanOptionPrice(K = self.K, T = NewT, S0 = value,r = self.r, sigma = self.sigma, option_type = self.option_type, steps = steps)
            SimuOptionPos[index] = OptionPrice * num_options * contract_size

        # initializing an array to store profit and loss (P&L) values
        PL = np.zeros(len(SimuOptionPos))

        # computing the P&L for each simulated option position
        for index, value in enumerate(SimuOptionPos):
            PL[index] = value - OptionPos0

        # calculating the Value at Risk (VaR) and Expected Shortfall (ES) for the simulated P&L
        VaR = np.quantile(PL, alpha) * -1 ; ES = np.mean(PL[PL < - VaR]) * -1 

        # organizing the results into a dictionary to return
        MC_GBM = {"var" : round(VaR, 4),
                  "es:" : round(ES, 4),
                  "T" : interval,
                  "ExpInDays" : int(self.T * 252),
                  "NumSims" : len(SimuStockPrices), 
                  "OptPos0" : round(num_options * contract_size *  self.bi_price,4),
                  "BiPrice" : self.bi_price,
                  "GbmDrift" : round(drift,4),
                  "GbmSigma" : round(sigma,4)}
        
        return MC_GBM
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_options: int, contract_size: int, S0_initial: float, test_returns: np.ndarray, var: float, interval: int = 1, alpha: float = 0.05,
                    alpha_test: float = 0.05, warning = True):
        
        """
        Main
        ----
        kupiec_test performs the Kupiec's Proportional Of Failures (POF) test for backtesting the accuracy of VaR predictions. The test assesses if the number of 
        times the losses exceed the VaR prediction is consistent with the confidence level used to compute the VaR. If the number of exceptions is too high, 
        it indicates that the VaR measure may be understating the risk.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - S0_initial: Initial stock price or the starting value of the underlying asset at the beginning of the "test_returns" array.
        - test_returns: Array containing the returns of the underlying asset used for the backtesting. Ensure that the frequency of the "test_returns" array is daily.
        - var: The Value at Risk (VaR) measure being tested.
        - interval: The time horizon (in days, assuming 252 trading days) over which the risk measure was computed. Default is 1.
        - alpha: The significance level used in the VaR calculation. Default is 0.05.
        - alpha_test: The significance level for the chi-square test. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'test_returns' frequency for the analysis. Default is True.

        Returns
        -------
        - KUPIEC_TEST: A dictionary containing the test's name, computed VaR, time horizon, number of observations, number of exceptions, theoretical % of 
        exceptions based on alpha, actual % of exceptions, likelihood ratio, critical value for the test, and a boolean indicating if the null hypothesis 
        should be rejected.

        Notes
        -----
        Please ensure that the 'test_returns' frequency is 'daily'.
        For instance:
        - for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.
        - for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. 
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, check_S0_initial, validate_returns_single,check_var,
                                                       check_alpha, check_alpha_test, check_warning, check_interval)

        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; test_returns = validate_returns_single(test_returns) 
        var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) 
        S0_initial = check_S0_initial(S0_initial) ; check_warning(warning) 

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to backtest the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' frequency is 'daily'.")
            print("For instance:")
            print("- for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.")
            print("- for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. ")
            print("To hide this warning, use 'warning=False' when calling the kupiec_test() function.")

        # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

        def reshape_and_aggregate_returns(returns, interval):
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        test_returns = reshape_and_aggregate_returns(test_returns,interval)

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice #  BiAmericanOptionPrice(K , T, S0, r, sigma, option_type,  steps)

        # computing the stock price path by cumulatively applying returns to the initial stock price S0_initial
        stockPricePath_without_initial = S0_initial * np.cumprod(1 + test_returns) 
        # inserting the initial stock price to the beginning of the computed path
        stockPricePath = np.insert(stockPricePath_without_initial, 0, S0_initial)

        # initializing an array to store option positions for each time step and computing the number of steps for option pricing
        OptionPositionTimet = np.zeros(len(stockPricePath))
        steps = int(( 252 * self.T )) 
                
        # for each stock price in the path, compute the option position value and store it in OptionPositionTimet
        for t in range(stockPricePath.shape[0]):
            OptionPositionTimet[t] = BiAmericanOptionPrice(S0 = stockPricePath[t], T = self.T, K = self.K, sigma = self.sigma, r = self.r,
                                                 option_type = self.option_type, steps = steps) * num_options * contract_size 
            
        # computing the day-to-day changes in option position values to get profit and loss (P&L)
        PL = np.diff(OptionPositionTimet)

        # total number of observations and exceptions
        num_obs = len(PL) 
        num_exceptions = np.sum(PL < - VaR)

        # % exceptions
        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test": "Kupiec",                      # Test
            "var": VaR,                            # Value at Risk
            "T": interval,                         # interval
            "nObs": num_obs,                       # Number of observations
            "nExc": num_exceptions,                # Number of exceptions
            "alpha": alpha,                        # Theoretical % of exceptions
            "actualExc%": per_exceptions,          # Real % of exceptions
            "LR": likelihood_ratio,                # Likelihood ratio
            "cVal": round(critical_value, 5),      # Critical value
            "rejNull": reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # CHRISTOFFESEN TEST <<<<<<<<<<<<

    def christoffersen_test(self, num_options: int, contract_size: int, S0_initial: float, test_returns: np.ndarray, var: float,interval: int = 1, alpha: float = 0.05,
                    alpha_test: float = 0.05, warning = True):
        
        """
        Main
        ----
        The christoffersen_test method conducts Christoffersen's Conditional Coverage test for backtesting the accuracy of VaR predictions. The test examines both the 
        unconditional coverage (like Kupiec's POF) and the independence of exceptions. The test checks if exceptions are clustering or if they occur independently 
        over time. It uses a likelihood ratio method to compare the likelihood of observing the given sequence of exceptions under the assumption of independent 
        exceptions versus the likelihood under the observed conditional probabilities.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - S0_initial: Initial stock price or the starting value of the underlying asset at the beginning of the "test_returns" array.
        - test_returns: Array containing the returns of the underlying asset used for the backtesting. Ensure that the frequency of the "test_returns" array is daily.
        - var: The Value at Risk (VaR) measure being tested.
        - interval: The time horizon (in days, assuming 252 trading days) over which the risk measure was computed. Default is 1.
        - alpha: The significance level used in the VaR calculation. Default is 0.05.
        - alpha_test: The significance level for the chi-square test. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'test_returns' frequency for the analysis. Default is True.

        Returns
        -------
        - CHRISTOFFERSEN_TEST: A dictionary containing the test's name, computed VaR, likelihood ratio, critical value for the test, and a boolean indicating 
        if the null hypothesis should be rejected.

        Notes
        -----
        Please ensure that the 'test_returns' frequency is 'daily'.
        For instance:
        - for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.
        - for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. 
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_options, check_contract_size, check_S0_initial, validate_returns_single,check_var,
                                                       check_alpha, check_alpha_test, check_warning, check_interval)
        
        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_options = check_num_options(num_options) ; contract_size = check_contract_size(contract_size) ; test_returns = validate_returns_single(test_returns) 
        var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) 
        S0_initial = check_S0_initial(S0_initial) ; check_warning(warning)


        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to backtest the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' frequency is 'daily'.")
            print("For instance:")
            print("- for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.")
            print("- for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. ")
            print("To hide this warning, use 'warning=False' when calling the christoffersen_test() function.")

        # reshape_and_aggregate_returns ----------------------------------------------------------------------------------------

        def reshape_and_aggregate_returns(returns, interval):
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        test_returns = reshape_and_aggregate_returns(test_returns,interval)

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice #  BiAmericanOptionPrice(K , T, S0, r, sigma, option_type,  steps)

        # computing the stock price path by cumulatively applying returns to the initial stock price S0_initial
        stockPricePath_without_initial = S0_initial * np.cumprod(1 + test_returns) 
        # inserting the initial stock price to the beginning of the computed path
        stockPricePath = np.insert(stockPricePath_without_initial, 0, S0_initial)

        # initializing an array to store option positions for each time step and computing the number of steps for option pricing
        OptionPositionTimet = np.zeros(len(stockPricePath))
        steps = int(( 252 * self.T )) 
                
        # for each stock price in the path, compute the option position value and store it in OptionPositionTimet
        for t in range(stockPricePath.shape[0]):
            OptionPositionTimet[t] = BiAmericanOptionPrice(S0 = stockPricePath[t], T = self.T, K = self.K, sigma = self.sigma, r = self.r,
                                                 option_type = self.option_type, steps = steps) * num_options * contract_size 
            
        # computing the day-to-day changes in option position values to get profit and loss (P&L)
        PL = np.diff(OptionPositionTimet)

        exceptions = PL < - VaR

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
        p = alpha
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

    def combined_test(self, num_options: int, contract_size: int, S0_initial: float, test_returns: np.ndarray, var: float, interval: int = 1, alpha: float = 0.05,
                    alpha_test = 0.05, warning = True):
        
        """
        Main
        ----
        The combined_test method conducts a combined backtesting test using both Kupiec's POF and Christoffersen's Conditional Coverage tests. This test leverages both 
        the POF test which tests for correct unconditional coverage and the conditional coverage test that checks for the independence of exceptions. By 
        combining the results of these two tests, the combined_test provides a more robust way to assess the accuracy of VaR predictions.
        The combined_test calls both the kupiec_test and christoffersen_test methods and aggregates their results. The combined likelihood ratio is derived 
        from summing the individual likelihood ratios from both tests. The chi-square test's critical value used here has 2 degrees of freedom, reflecting 
        the combination of the two individual tests.

        Parameters
        ----------
        - num_options: The number of option contracts.
        - contract_size: The size of each option contract. How many stocks or underlying positions the option allows you to buy/sell.
        - S0_initial: Initial stock price or the starting value of the underlying asset at the beginning of the "test_returns" array.
        - test_returns: Array containing the returns of the underlying asset used for the backtesting. Ensure that the frequency of the "test_returns" array is daily.
        - var: The Value at Risk (VaR) measure being tested.
        - interval: The time horizon (in days, assuming 252 trading days) over which the risk measure was computed. Default is 1.
        - alpha: The significance level used in the VaR calculation. Default is 0.05.
        - alpha_test: The significance level for the chi-square test. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'test_returns' frequency for the analysis. Default is True.

        Returns
        -------
        - COMBINED_TEST: A dictionary containing the test's name, computed VaR, combined likelihood ratio, critical value for the test, and a boolean indicating 
        if the null hypothesis should be rejected.

        Notes
        -----
        Please ensure that the 'test_returns' frequency is 'daily'.
        For instance:
        - for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.
        - for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. 
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        from scipy.stats import chi2

        if interval >= int(self.T * 252):
            raise Exception(f"""
            The option will expire in {int(self.T * 252)} days. 
            You've selected a VaR interval of {interval} days, which exceeds the option's remaining lifetime.
            It's not possible to backtest the Value-at-Risk (VaR) for an interval that goes beyond the option's expiration date.
            Please choose an interval less to the option's remaining lifetime.
            """)
        
        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' frequency is 'daily'.")
            print("For instance:")
            print("- for the backtesting procedure of a 5-day VaR: provide daily 'test_returns' and set the interval to 5.")
            print("- for the backtesting procedure of a 2-week VaR: provide daily 'test_returns' and set the interval to 10. ")
            print("To hide this warning, use 'warning=False' when calling the combined_test() function.")

        warning = False

        # no need for retuns aggregation here ---- it will be handled by both kupiec_test and christoffersen_test

        # Now we can proceed -------------------------------------------------------------------------------------------------------------------

        VaR = round(var,4)

        kupiec_results = self.kupiec_test(num_options = num_options, contract_size = contract_size, S0_initial = S0_initial,
                                        test_returns = test_returns, var = VaR, interval = interval, alpha = alpha, alpha_test = alpha_test, 
                                        warning = warning)
        
        christoffersen_results = self.christoffersen_test(num_options = num_options, contract_size = contract_size, S0_initial = S0_initial,
                                        test_returns = test_returns, var = VaR, interval = interval, alpha = alpha, alpha_test = alpha_test, 
                                        warning = warning)
        
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
                "LR" : round(combined_likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}
    
##########################################################################################################################################################################
##########################################################################################################################################################################
## 5. EuOptionRmPort #####################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EuOptionRmPort:

    """
    Main
    ----
    The `EuOptionRmPort` class provides a comprehensive risk management tool designed to evaluate the Value at Risk (VaR) 
    and Expected Shortfall (ES) for portfolios consisting of European options. Leveraging the Black-Scholes model, 
    it not only calculates the price for each option in the portfolio but also computes the Greeks, including Delta, Gamma, Vega, Theta, and Rho,
    given specific parameters. A DataFrame is constructed which provides a detailed summary of each option's type, its Black-Scholes price, 
    and associated Greeks. 

    Initial Parameters
    ------------------
    - K_s: An array of strike prices for each option in the portfolio. Represents the price at which the option can be exercised.
    - T_s: An array of time to maturities for each option. Specifies the time remaining until the option contract is due for expiration.
      - For instance: expDays = np.array([5,8,12,15,18]); T_s = expDays/252
    - S0_s: An array of initial spot prices for the underlying asset for each option. It represents the current market value of the asset which the option derives its value from.
    - r_s: An array of risk-free interest rates for each option (annualized). The risk-free rate represents a theoretical rate of return on an investment with zero risk.
    - sigma_s: An array of volatilities for the underlying asset for each option (annualized).
    - option_type_s: An array indicating the type of each option (e.g., 'call' or 'put'). A call option gives the holder the right to buy the underlying asset, 
    while a put option gives the holder the right to sell the underlying asset.

    Methods
    -------
    - HistoricalSimulation(): Calculates the Value at Risk (VaR) and Expected Shortfall (ES) using historical simulations based on past return scenarios.
    - ff3(): Fetches data for the Fama-French three-factor (FF3) model, which includes Market return (MKT), Size (SMB: Small Minus Big), 
    and Value (HML: High Minus Low) from Ken French's Data Library.
    - mcModelDrawReturns(): Determines the Value at Risk (VaR) and Expected Shortfall (ES) by simulating correlated future stock returns using a Monte Carlo method, 
      rooted in various statistical models for volatility and mean returns.
    - VarEsMinimizer(): seeks to minimize the Value at Risk (VaR) for a given portfolio of European options using historical simulations.
    - kupiec_test(): Conducts the Kupiec backtest for Value at Risk (VaR). The Kupiec test employs a likelihood ratio test to validate VaR model accuracy.
    - christoffersen_test(): Implements the Christoffersen test for Value at Risk (VaR), which examines the clustering of violations where actual returns underperform the predicted VaR.
    - combined_test(): Merges the Kupiec and Christoffersen tests, delivering a holistic assessment of Value at Risk (VaR) model precision.

    Attributes
    ----------
    - bs_prices: An array with the calculated Black-Scholes option prices.
    - summary : A pandas DataFrame summarizing each option's type, its Black-Scholes price, and associated Greeks.

    Notes
    -----
    All input arrays (`K_s`, `T_s`, `S0_s`, `r_s`, `sigma_s`, `option_type_s`) must share the same length as they correspond to 
    attributes for each option in the portfolio. The `fit` method is activated by default during the initialization to generate the necessary values and populate the summary DataFrame.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, K_s: np.ndarray,
                 T_s: np.ndarray, 
                 S0_s: np.ndarray,
                 r_s:  np.ndarray,
                 sigma_s: np.ndarray,
                 option_type_s: np.ndarray):

        
        from pyriskmgmt.inputsControlFunctions import (check_K_s, check_T_s, check_S0_s, check_r_s, chek_sigma_deri_s, check_option_type_s)

        self.K_s = check_K_s(K_s) ; self.T_s = check_T_s(T_s) ; self.S0_s = check_S0_s(S0_s) ; self.r_s = check_r_s(r_s) ; self.sigma_s = chek_sigma_deri_s(sigma_s)
        self.option_type_s = check_option_type_s(option_type_s)

        # ------------------------------------------------------------------------------------------------

        # checking the same length for all the imputs *args

        def check_same_length(*args):
            """
            Checks if all given numpy arrays have the same length.
            """
            it = iter(args)
            the_len = len(next(it))

            if not all(len(l) == the_len for l in it):
                raise ValueError("Not all the input arrays in the EuOptionRmPort class have the same length!")
            
        # Here we are ensuring that all arrays have the same length
        check_same_length(K_s, T_s, S0_s, r_s, sigma_s, option_type_s)

        # ------------------------------------------------------------------------------------------------

        self.bs_prices = None 

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd 
        from pyriskmgmt.SupportFunctions import (BlackAndScholesPrice, CalculateGreeks)

        summary = pd.DataFrame()

        # calculating prices and greeks for each set of parameters and add to summary DataFrame
        for S0, T, K, sigma, r, option_type in zip(self.S0_s, self.T_s, self.K_s, self.sigma_s, self.r_s, self.option_type_s):
            price = BlackAndScholesPrice(S0, T, K, sigma, r, option_type)
            delta, gamma, vega, theta, rho = CalculateGreeks(S0, T, K, sigma, r, option_type)
            
            summary = summary._append({'type' : option_type,'bsprice' : price,'delta' : delta,'gamma' : gamma,'vega' : vega,'theta' : theta,'rho' : rho,'K' : K,
                                       'T' : T,'S0' : S0,'r' : r,'sigma' : sigma}, ignore_index=True)

        # rounding all the values to 4 decimal places
        summary = summary.round(4)

        # setting "type" as index
        summary.set_index('type', inplace=True)

        self.summary = summary 
        self.bs_prices = summary["bsprice"].values

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05,
                              warning: bool = True):       

        """
        Main
        ----
        Performs a historical simulation method for the computation of VaR and Expected Shortfall (ES) for a portfolio of European options. 
        The method uses historical returns of the underlying asset to simulate potential future returns and subsequently calculates the VaR and ES based
        on the distribution of the simulated profits and losses.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - underlying_rets: 2D Array containing historical returns of the underlying assets associated with the options in the portfolio. 
        Ensure that the frequency of every column in the "underlying_rets" array is the same and 'daily'.
        - interval: The time horizon over which the risk measure is to be computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - HS: A dictionary containing the computed VaR, Expected Shortfall (ES), time horizon (T), minimum expiration in days (MinExpInDays), 
        maximum expiration in days (MaxExpInDays), number of simulations (NumSimu), and total initial position (TotInPos).

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd ; import numpy as np
        
        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, validate_returns_port, check_interval, check_alpha, check_warning)
        
        # check procedure 

        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; underlying_rets = validate_returns_port(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning)

        # validating that the length of num_options_s matches self.K_s
        if len(num_options_s) != len(self.K_s):
            raise ValueError("Length of num_options_s does not match length of self.K_s")

        # validating that the length of contract_size_s matches self.K_s
        if len(contract_size_s) != len(self.K_s):
            raise ValueError("Length of contract_size_s does not match length of self.K_s")

        # validating that the number of columns in underlying_rets matches self.K_s
        if underlying_rets.shape[1] != len(self.K_s):
            raise ValueError("Number of columns in underlying_rets_s does not match length of self.K_s")
        
        
        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Calculating Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("Ensure also that every column in the 'underlying_rets' has the same frequency (daily).")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the HistoricalSimulation() method.")
                        
        # converting underlying_rets to a DataFrame for easier manipulation
        rets_df = pd.DataFrame(underlying_rets)

        # defining a function to reshape and aggregate returns based on the given interval
        def reshape_and_aggregate_returns(returns, interval):
            # extracting the values from the DataFrame column
            returns = returns.values 
            # computing the number of periods to aggregate
            num_periods = len(returns) // interval * interval 
            # trimming the returns to fit the aggregation period
            trimmed_returns = returns[len(returns) - num_periods:]
            # reshaping the returns for aggregation
            reshaped_returns = trimmed_returns.reshape((-1, interval)) 
            # aggregating the returns by multiplying (compounding) them
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        # applying the reshape and aggregate function to each column of rets_df
        rets_df = rets_df.apply(lambda x: reshape_and_aggregate_returns(x, interval))
        # extracting the aggregated returns from the DataFrame to a numpy array
        underlying_rets = rets_df.values

        # beginning the vectorization process
        # creating 2D versions of various parameter arrays for broadcasting
        T_s_2D = self.T_s[np.newaxis, :] ; K_s_2D = self.K_s[np.newaxis, :]
        sigma_s_2D = self.sigma_s[np.newaxis, :] ; r_s_2D = self.r_s[np.newaxis, :] 
        option_type_s_2D = self.option_type_s[np.newaxis, :]

        # adjusting the shapes of the parameter arrays to match that of underlying_rets
        New_T_s = ((252 * T_s_2D - interval) / 252) * np.ones(underlying_rets.shape)
        New_K_s = K_s_2D * np.ones(underlying_rets.shape)
        New_sigma_s = sigma_s_2D * np.ones(underlying_rets.shape)
        New_r_s = r_s_2D * np.ones(underlying_rets.shape)
        # broadcasting the option type to match the shape of underlying_rets
        New_option_type_s = np.broadcast_to(option_type_s_2D, underlying_rets.shape)

        # importing the function for vectorized Black-Scholes option pricing
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization

        # calculating simulated stock prices using the aggregated returns
        SimuStockPrices = self.S0_s * (1 + underlying_rets)
        # calculating the total initial position value
        total_initial_pos = np.sum(self.bs_prices * num_options_s * contract_size_s)

        # computing option prices using the vectorized Black-Scholes function
        OptionPrices = BlackAndScholesPriceVectorization(SimuStockPrices, New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s)
        # calculating simulated option positions using the option prices
        SimuOptionPos = OptionPrices * num_options_s * contract_size_s
        # computing profit & loss for each simulated option position
        PL = SimuOptionPos - self.bs_prices * num_options_s * contract_size_s
        # aggregating the P&L across all simulations
        total_PL = PL.sum(axis=1)
        # computing the Value at Risk (VaR) and Expected Shortfall (ES) from the total P&L
        losses = total_PL[total_PL<0]
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < - VaR]) * -1 

        # consolidating the results into a dictionary for return
        HS = {"var" : round(VaR, 4),
              "es" : round(ES, 4),
              "T" : interval,
              "MinExpInDays" : int(min(self.T_s)*252),
              "MaxExpInDays" : int(max(self.T_s)*252),
              "NumSimu" : underlying_rets.shape[0],
              "TotInPos" : round(total_initial_pos, 4)}
        
        return HS
    
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
        
    # MONTECARLO MODEL [RETURNS SIMULATION] <<<<<<<<<<<<
    
    def mcModelRetsSimu(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, underlying_rets: np.ndarray, interval: int = 1,
                        alpha: float = 0.05, sharpeDiag: bool = False , mappingRets: np.ndarray = None, vol: str = "simple",
                        num_sims: int = 1000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, warning: bool = True):
        
        """
        Main
        ----
        Simulates returns using Monte Carlo models and calculates the Value-at-Risk (VaR) 
        and Expected Shortfall (ES) based on specified configurations.
        To utilize Sharpe diagnostics, ensure 'sharpeDiag' is set to True and 'mappingRets' is provided.
        When 'sharpeDiag' is enabled, the variance-covariance matrix of underlying returns is calculated using a mapping procedure from n factors,
        incorporating the estimation of idiosyncratic variance to ensure the matrix is positive and semi-definite and numerical stability.
        'num_sims' simulations of joint returns (correlated) will be used to estimate the Risk Measures.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - underlying_rets: 2D Array containing historical returns of the underlying assets associated with the options in the portfolio. 
        Ensure that the frequency of every column in the "underlying_rets" array is the same as the others and all 'daily'.
        - interval: The time horizon over which the risk measure is to be computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The alpha value for VaR calculation. Default is 0.05.
        - sharpeDiag: If enabled, the function employs a Sharpe diagonal approach to map "underlying_rets" to 1 or more factors. Default is None.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on a daily frequency
        and align "mappingRets" with "underlying_rets" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility estimation model to use, either "simple","garch",or "ewma". Default is "simple".
        - num_sims: Number of Monte Carlo simulations to be performed. Default is 1000.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The order of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma: The decay factor for the 'ewma' volatility method. Default is 0.94.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - SimulationResults: A dictionary containing computed values such as VaR, Expected Shortfall (ES), and other relevant metrics.

        Notes
        -----
        Please ensure that the 'underlying_rets' and the 'mappingRets' frequency is 'daily'.
        ['mappingRets' is needed only if 'sharpeDiag' = True].
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and 'mappingRets' (if sharpeDiag) and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        import numpy as np
        
        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, validate_returns_port, check_interval, check_alpha, check_warning)
        
        # check procedure 
        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; underlying_rets = validate_returns_port(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Calculating Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("Ensure also that every column in the 'underlying_rets' has the same frequency (daily).")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the mcModelRetsSimu() method.")

        if vol == "garch":
            
            if warning:
                from pyriskmgmt.SupportFunctions import garch_warnings
                garch_warnings("mcModelRetsSimu()")
                
            if warning and underlying_rets.shape[1] > 50 and not sharpeDiag:
                print("IMPORTANT:")
                print("The 'garch' option for the 'vol' parameter is currently in use, with an asset count exceeding 50.")
                print("This configuration can lead to slower processing and inefficient resource usage, due to the complexity ")
                print("of estimating the full variance-covariance matrix (VCV) in high-dimensional space.")
                print("It's strongly advised to employ a mapping approach instead. This approach maps the multiple assets to one or more factors,")
                print("effectively reducing the dimensionality of the problem.")
                print("This is not only more computationally efficient, but it also avoids the 'curse of dimensionality',")
                print("as it reduces the number of parameters needed to be estimated. ")
                print("The mapping approach aids in providing more stable and reliable estimations, especially in large, complex systems.")
        

        warning = False

        # handling other inputs, specific cases and initializations using the EquityPrmPort class
        from pyriskmgmt.equity_models import EquityPrmPort

        # instantiating the EquityPrmPort model with given parameters
        model = EquityPrmPort(returns=underlying_rets, positions=np.random.uniform(-1, 1, underlying_rets.shape[1]),
                              interval=interval, alpha=alpha)  # note: 'positions' is auxiliary and main use of this instance is to retrieve simulated returns
        
        # retrieving simulated returns using the Monte Carlo model from the instantiated object
        simulated_rets = model.mcModelRetsSimu(sharpeDiag=sharpeDiag, mappingRets=mappingRets, vol=vol, num_sims=num_sims, p=p, q=q,
                                               lambda_ewma=lambda_ewma, warning=warning, return_simulated_rets=True)[1]

        # beginning the vectorization process
        # creating 2D versions of parameters for broadcasting
        T_s_2D = self.T_s[np.newaxis, :] ; K_s_2D = self.K_s[np.newaxis, :]
        sigma_s_2D = self.sigma_s[np.newaxis, :] ; r_s_2D = self.r_s[np.newaxis, :] ; option_type_s_2D = self.option_type_s[np.newaxis, :]

        # adjusting the shapes of the parameters to match that of simulated_rets
        New_T_s = ((252 * T_s_2D - interval) / 252) * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        New_K_s = K_s_2D * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        New_sigma_s = sigma_s_2D * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        New_r_s = r_s_2D * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        
        # extracting a matrix from simulated_rets and broadcasting option_type to match its shape
        one_matrix_simu = simulated_rets[:, :, 0]
        New_option_type_s = np.broadcast_to(option_type_s_2D, one_matrix_simu.shape)

        # importing required functions for the next steps
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization
        from joblib import Parallel, delayed

        # defining a function that calculates simulated VaR and ES for given parameters
        def calculate_simulated_var_es(sim, simulated_rets, S0_s, bs_prices, num_options_s, contract_size_s, 
                                       New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s, alpha):
            
            SimuStockPrices = S0_s * (1 + simulated_rets[:, :, sim])  # computing simulated stock prices
            OptionPrices = BlackAndScholesPriceVectorization(SimuStockPrices, New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s)
            SimuOptionPos = OptionPrices * num_options_s * contract_size_s  # computing option positions for simulated prices
            PL = SimuOptionPos - bs_prices * num_options_s * contract_size_s  # calculating profit & loss
            total_PL = PL.sum(axis=1)  # aggregating the P&L
            VaR = np.quantile(total_PL, alpha) * -1 ; ES = np.mean(total_PL[total_PL < - VaR]) * -1  # calculating VaR and ES
            
            return VaR, ES

        # using joblib for parallel processing of the simulations
        num_sims = simulated_rets.shape[2]

        # setting up a visual aid for the progress
        from pyriskmgmt.SupportFunctions import DotPrinter
        my_dot_printer = DotPrinter(f"\r {num_sims} Scenarios for {one_matrix_simu.shape[1]} European Options - joblib - VaR & ES Parallel Computation")
        my_dot_printer.start()  # starting the visual progress indicator

        # launching parallel computation
        results = Parallel(n_jobs=-1)(delayed(calculate_simulated_var_es)(sim, simulated_rets, self.S0_s, self.bs_prices, num_options_s, contract_size_s, New_T_s,
                                                                          New_K_s, New_sigma_s, New_r_s, New_option_type_s, alpha) for sim in range(num_sims))
        
        # ending the visual progress indicator after computation completes
        my_dot_printer.stop()
        print("\rParallel Computation --->  Done", end=" " * 90)

        # extracting and averaging the VaR and ES results from parallel computations
        VARS, ESs = zip(*results) ; VaR = np.mean(VARS) ; ES = np.mean(ESs)

        # calculating the total initial position
        total_initial_pos = np.sum(self.bs_prices * num_options_s * contract_size_s)
        TotInPos = round(total_initial_pos, 4)

        # consolidating results into a dictionary to return
        MC_DIST = { "var" : round(VaR, 4),
                    "es" : round(ES, 4),
                    "T" : interval,
                    "MinExpInDays" : int(min(self.T_s)*252),
                    "MaxExpInDays" : int(max(self.T_s)*252),
                    "NumSimu" : simulated_rets.shape[0],
                    "McNumSim" : num_sims,
                    "TotInPos" : TotInPos }
        
        return MC_DIST

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    #  VAR ES MINIMAZER <<<<<<<<<<<<<<<

    def VarEsMinimizer(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05, warning: bool = True,
                       allow_short: bool = True):

        """
        Main
        ----
        Optimizes the number of options in a portfolio to minimize the Value-at-Risk (VaR) and Expected Shortfall (ES) 
        using historical simulation over a specified interval.
        The optimization process maintains the overall portfolio position and can allow or disallow short positions based on the 'allow_short' parameter.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - underlying_rets: 2D Array containing historical returns of the underlying assets associated with the options in the portfolio. 
        Ensure that the frequency of every column in the "underlying_rets" array is the same as the others and all 'daily'.
        - interval: The time horizon over which the risk measure is to be computed and minimized (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation and minization. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.
        - allow_short: If True, allows the minimization process to consider short positions. Default is True.

        Returns
        -------
        - MIN_PROCESS: A dictionary containing optimized values such as VaR, ES, initial and final positions, and other relevant metrics.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day VaR minization process: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week VaR minization process: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        import pandas as pd ; import numpy as np
        
        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, validate_returns_port, check_interval, check_alpha, check_warning,
                                                       check_allow_short)
        
        # check procedure 

        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; underlying_rets = validate_returns_port(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; warning = check_warning(warning) ; check_allow_short(allow_short)

        # validating that the length of num_options_s matches self.K_s
        if len(num_options_s) != len(self.K_s):
            raise ValueError("Length of num_options_s does not match length of self.K_s")

        # validating that the length of contract_size_s matches self.K_s
        if len(contract_size_s) != len(self.K_s):
            raise ValueError("Length of contract_size_s does not match length of self.K_s")

        # validating that the number of columns in underlying_rets matches self.K_s
        if underlying_rets.shape[1] != len(self.K_s):
            raise ValueError("Number of columns in underlying_rets_s does not match length of self.K_s")
        
        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Calculating and minimizing Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)
 
        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("Ensure also that every column in the 'underlying_rets' has the same frequency (daily).")
            print("For instance:")
            print("- for the 5-day minimization process: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week minimization process: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the VarEsMinimizer() method.")
            
        # converting the underlying returns into a DataFrame
        rets_df = pd.DataFrame(underlying_rets)

        # defining a function that reshapes and aggregates returns based on a specified interval
        def reshape_and_aggregate_returns(returns, interval):
            returns = returns.values
            num_periods = len(returns) // interval * interval
            trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval))
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns

        # applying the above function to the DataFrame for each column
        rets_df = rets_df.apply(lambda x: reshape_and_aggregate_returns(x, interval)) 
        underlying_rets = rets_df.values

        # vectorizing variables to prepare them for operations with the underlying returns
        T_s_2D = self.T_s[np.newaxis, :]
        K_s_2D = self.K_s[np.newaxis, :]
        sigma_s_2D = self.sigma_s[np.newaxis, :]
        r_s_2D = self.r_s[np.newaxis, :]
        option_type_s_2D = self.option_type_s[np.newaxis, :]  

        # adjusting variables to match the shape of underlying_rets
        New_T_s = ((252 * T_s_2D - interval) / 252) * np.ones(underlying_rets.shape)
        New_K_s = K_s_2D * np.ones(underlying_rets.shape)
        New_sigma_s = sigma_s_2D * np.ones(underlying_rets.shape)
        New_r_s = r_s_2D * np.ones(underlying_rets.shape) 
        New_option_type_s = np.broadcast_to(option_type_s_2D, underlying_rets.shape)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization

        # ----------------------------------------------------------------------------------------------------------------------------------------
       
        # calculate_final_var() function
        def calculate_final_var(underlying_rets, S0_s, bs_prices, num_options_s, contract_size_s, 
                                        New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s, alpha):
            
            SimuStockPrices = S0_s * (1 + underlying_rets)  # Simulated stock prices
            # BlackAndScholesPriceVectorization
            OptionPrices = BlackAndScholesPriceVectorization(SimuStockPrices, New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s)
            SimuOptionPos = OptionPrices * num_options_s * contract_size_s
            # computing Profit & Loss and total P&L
            PL = SimuOptionPos - bs_prices * num_options_s * contract_size_s 
            total_PL = PL.sum(axis=1)  # Total P&L
            # computing Value at Risk (VaR) from total P&L
            VaR = np.quantile(total_PL, alpha) * -1

            return VaR
        
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        # minimization process
        from scipy.optimize import minimize

        # optimizing the final VaR
        def optimize_final_var(num_options_s):
            return calculate_final_var(underlying_rets, self.S0_s, self.bs_prices, num_options_s, contract_size_s, 
                                        New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s, alpha)
            
        # setting up initial conditions and constraints for the optimization
        initial_guess_for_num_options_s = num_options_s
        total_initial_pos = np.sum(self.bs_prices * num_options_s * contract_size_s)
        cons = ({'type': 'eq', 
                'fun' : lambda x: np.sum(self.bs_prices * x * contract_size_s) - total_initial_pos})
        
        # just some aesthetics
        from pyriskmgmt.SupportFunctions import DotPrinter

        # starting the animation
        my_dot_printer = DotPrinter(f"\rPortfolio of {len(self.K_s)} Options - Historical Simulation - Minimization Process") 
        my_dot_printer.start()

        # initializing variables for optimization
        n = len(self.K_s)
        
        # defining bounds for the optimization 
        if allow_short:
            bounds = ((None, None),) * n   # short positions allowed
        else:
            bounds = ((0, None),) * n      # no short positions allowed

        # error handling for situations where short selling is prohibited but the initial position is negative
        total_INITIAL_pos = np.sum(self.bs_prices * num_options_s * contract_size_s)

        if not allow_short and total_initial_pos <= 0:
            my_dot_printer.stop()

            print(f"\rMinimization Process: >>>> Convergence Unsuccessful <<<<", end=" " * 90)
            raise Exception(f""" The initial total position of the Option Portfolio stands at {total_INITIAL_pos}, which is negative. 
            This method tries to keep (more or less) the same total position of the Option Porfolio.
            When short selling is prohibited, maintaining a negative position for an Options Portfolio becomes challenging.
            Furthermore, the solver is prone to defaulting to the simplest solution, which is zero for all positions.
            """)
        
        # performing the optimization
        result = minimize(optimize_final_var, initial_guess_for_num_options_s, constraints=cons, bounds = bounds)

       # stopping the animation when minimization is done
        my_dot_printer.stop()
        print(f"\rMinimization Process --> Done", end=" " * 80)

        # extracting the results and calculating the final VaR and ES
        optimized_num_options_s = result.x 
        optimized_num_options_s = [round(i) for i in optimized_num_options_s] 
        optimized_num_options_s = [int(i) for i in optimized_num_options_s]

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # using HistoricalSimulation to calculate the minimum var and es

        # calculating the minimum var and es using a historical simulation approach
        result = self.HistoricalSimulation(num_options_s = optimized_num_options_s, contract_size_s= contract_size_s, underlying_rets = underlying_rets,
                                          interval = interval, alpha = alpha, warning =  False ) # warning = False here

        # summarizing the results into a dictionary for output
        total_FINAL_pos = np.sum(self.bs_prices * optimized_num_options_s * contract_size_s)
        
        TotInPos = round(total_INITIAL_pos, 4) ; TotFinPos = round(total_FINAL_pos, 4)

        # consolidating results into a dictionary to return
        MIN_PROCESS = {"var" : result["var"],
                       "es:" : result["es"],
                       "T" : interval,
                       "MinExpInDays" : int(min(self.T_s)*252),
                       "MaxExpInDays" : int(max(self.T_s)*252),
                       "TotInPos" : TotInPos,
                       "TotFinPos" : TotFinPos,
                       "OldPos" : num_options_s,
                       "NewPos" : np.array(optimized_num_options_s),
                       "Differ": optimized_num_options_s - num_options_s}
                
        return MIN_PROCESS
    
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray,
                     var: float, interval: int = 1, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):

        """
        Main
        ----
        Backtests the Value-at-Risk (VaR) of a portfolio by applying the Kupiec test, which uses a likelihood ratio to compare the 
        predicted and actual exceptions.
        The Kupiec test uses a likelihood ratio to assess whether the number of actual exceptions is consistent 
        with the VaR model's predictions.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - S0_initial_s: Array indicating the initial stock/underlying prices at the beginning of the "test_returns" array.
        - test_returns: Array containing returns for backtesting. Ensure that the frequency of the "test_returns" is "daily".
        - var: The estimated Value-at-Risk to be tested.     
        - interval: The time horizon over which the risk measure was computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - alpha_test: The alpha value for the Kupiec test. Default is 0.05.
        - warning: If True, displays a warning about the frequency of 'test_returns'. Default is True.

        Returns
        -------
        - TestResults: A dictionary containing results of the Kupiec test such as number of exceptions, likelihood ratio, 
        critical value, and decision regarding null hypothesis.

        Notes
        -----
        Please ensure that the 'test_returns' is 'daily'.
        For instance:
        - for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        import pandas as pd ; import numpy as np ; from scipy.stats import chi2

        # ---------------------------------------------------------------------------------------

        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, check_S0_initial_s, validate_returns_port, check_var,
                                                        check_interval, check_alpha,check_alpha_test, check_warning)
        
        # check procedure
        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s)
        test_returns = validate_returns_port(test_returns) ; var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha)
        alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Backtesting a Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' is 'daily'.")
            print("For instance:")
            print("- for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the kupiec_test() function.")

        # defining the reshape_and_aggregate_returns function to transform and group the return data
        def reshape_and_aggregate_returns(returns, interval):
            # converting DataFrame values to a numpy array
            returns = returns.values
            # determining the number of periods that align with the interval
            num_periods = len(returns) // interval * interval
            # trimming the returns array to exclude any excess values that don't fit the interval 
            trimmed_returns = returns[len(returns) - num_periods:]
            # reshaping the returns data into chunks based on the interval
            reshaped_returns = trimmed_returns.reshape((-1, interval))
            # aggregating reshaped returns to produce net returns for each interval
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            
            # returning the aggregated returns
            return aggregated_returns
                
        # converting test_returns to a DataFrame for easier data manipulation
        test_returns = pd.DataFrame(test_returns)

        # applying the reshape and aggregate function to each column of the test returns DataFrame
        test_returns = test_returns.apply(lambda x: reshape_and_aggregate_returns(x, interval))
        # converting the DataFrame back to a numpy array
        test_returns = test_returns.values

        # rounding the var to 4 decimal places for precision
        VaR = round(var, 4)

        # initializing the stock price path matrix with zeros
        stockPricePathTotal = np.zeros((test_returns.shape[0]+1, len(self.K_s)))

        # setting the first row of stockPricePathTotal matrix with initial stock prices
        stockPricePathTotal[0,:] = S0_initial_s

        # calculating stock price for each simulated path based on the returns data
        for i in range(len(self.K_s)):
            stockPricePathTotal[1:, i] = S0_initial_s[i] * np.cumprod(1 + test_returns[:, i])

        # importing the BlackAndScholesPriceVectorization function for option pricing
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization

        # initializing the portfolio option position array with zeros
        PortOpPosition = np.zeros(len(stockPricePathTotal))

        # calculating option prices and overall portfolio option position for each stock price path
        for row in range(stockPricePathTotal.shape[0]):
            # calculating option prices for each stock price using the Black-Scholes formula
            prices = BlackAndScholesPriceVectorization(St_s = stockPricePathTotal[row,:],
                                                    T_s = self.T_s, K_s = self.K_s, sigma_s = self.sigma_s, r_s = self.r_s, option_type_s = self.option_type_s)
            # computing the portfolio option position for the given stock prices
            PortOpPosition[row] = np.sum(prices * num_options_s * contract_size_s)

        # calculating profit and loss by taking the difference in consecutive portfolio positions
        PL = np.diff(PortOpPosition)
        # counting the number of times the loss exceeded the VaR value
        num_exceptions = np.sum(PL < - VaR)

        # total number of observations -----------------------------------------------------------------------------------
        num_obs = len(PL)

        # % exceptions -------------------------------------------------------------------------------------------------------
        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "T" : interval,                         # interval
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # CHRISTOFFESEN TEST <<<<<<<<<<<<

    def christoffersen_test(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray,
                     var: float, interval: int = 1, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):
        """
        Christoffersen Test
        -------------------
        Backtests the Value-at-Risk (VaR) of a portfolio using the Christoffersen test, a test that focuses on the independence of 
        exceptions over time. It utilizes the likelihood ratio to examine if the clustering of exceptions is statistically significant.
        The Christoffersen test evaluates the clustering of exceptions, providing insight into the independence of VaR breaches.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - S0_initial_s: Array indicating the initial stock/underlying prices at the beginning of the "test_returns" array.
        - test_returns: Array containing returns for backtesting. Ensure that the frequency of the "test_returns" is "daily".
        - var: The estimated Value-at-Risk to be tested.     
        - interval: The time horizon over which the risk measure was computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - alpha_test: The alpha value for the Kupiec test. Default is 0.05.
        - warning: If True, displays a warning about the frequency of 'test_returns'. Default is True.

        Returns
        -------
        - TestResults: A dictionary containing results of the Christoffersen test such as the likelihood ratio, critical value, 
        and decision regarding the null hypothesis.

        Notes
        -----
        Please ensure that the 'test_returns' is 'daily'.
        For instance:
        - for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        import pandas as pd ; import numpy as np ; from scipy.stats import chi2

        # ---------------------------------------------------------------------------------------

        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, check_S0_initial_s, validate_returns_port, check_var,
                                                        check_interval, check_alpha,check_alpha_test, check_warning)

        # check procedure
        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s)
        test_returns = validate_returns_port(test_returns) ; var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha)
        alpha_test = check_alpha_test(alpha_test) ; check_warning(warning)

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Backtesting a Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' is 'daily'.")
            print("For instance:")
            print("- for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the christoffersen_test() function.")

        # defining a function to reshape returns data and aggregate them by given interval
        def reshape_and_aggregate_returns(returns, interval):
            # extracting array values from a DataFrame
            returns = returns.values
            # calculating the number of periods that fit the given interval
            num_periods = len(returns) // interval * interval
            # trimming the returns array to fit the interval
            trimmed_returns = returns[len(returns) - num_periods:]
            # reshaping the returns to create intervals
            reshaped_returns = trimmed_returns.reshape((-1, interval))
            # aggregating returns within each interval
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            
            # returning the aggregated returns
            return aggregated_returns

        # converting test_returns to a DataFrame
        test_returns = pd.DataFrame(test_returns)
        
        # applying the reshaping and aggregation function to the test_returns DataFrame
        test_returns = test_returns.apply(lambda x: reshape_and_aggregate_returns(x, interval))
        # converting the modified DataFrame back to a numpy array
        test_returns = test_returns.values

        # rounding the VaR value for precision
        VaR = round(var, 4)

        # initializing an array for stock prices simulation with zeros
        stockPricePathTotal = np.zeros((test_returns.shape[0]+1, len(self.K_s)))

        # setting the initial stock prices for simulation
        stockPricePathTotal[0,:] = S0_initial_s

        # simulating stock prices for each path using historical returns
        for i in range(len(self.K_s)):
            stockPricePathTotal[1:, i] = S0_initial_s[i] * np.cumprod(1 + test_returns[:, i])

        # importing the BlackAndScholesPriceVectorization function for option pricing
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization

        # initializing an array for the portfolio option positions
        PortOpPosition = np.zeros(len(stockPricePathTotal))

        # calculating the option prices and portfolio option position for each stock price path
        for row in range(stockPricePathTotal.shape[0]):

            # getting option prices using the Black-Scholes formula
            prices = BlackAndScholesPriceVectorization(St_s = stockPricePathTotal[row,:], T_s = self.T_s, K_s = self.K_s, sigma_s = self.sigma_s,
                                                       r_s = self.r_s, option_type_s = self.option_type_s)
            
            # calculating the portfolio option position at current stock prices
            PortOpPosition[row] = np.sum(prices * num_options_s * contract_size_s)
        
        # computing the profit and loss values for the portfolio
        PL = np.diff(PortOpPosition)
        # checking for instances where the loss exceeded the VaR
        exceptions = PL < - VaR

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
        p = alpha
        likelihood_ratio = -2 * (np.log((1 - p) ** (T_00 + T_10) * p ** (T_01 + T_11)) - np.log((1 - p_b0) ** T_00 * p_b0 ** T_01 * (1 - p_b1) ** T_10 * p_b1 ** T_11))

        # critical value from chi-squared distribution
        critical_value = chi2.ppf(1 - alpha_test, df=1)

        # rejecting the null hypothesis if likelihood ratio is higher than the critical value
        reject_null = likelihood_ratio > critical_value

        return {"test" : "Christoffersen",
                "var" : var,
                "LR" : round(likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # COMBINED TEST <<<<<<<<<<<<

    def combined_test(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray,
                     var: float, interval: int = 1, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):
        
        """
        Main
        ----
        Backtests the Value-at-Risk (VaR) of a portfolio by combining results from both the Kupiec and the Christoffersen tests. 
        This combined approach aims to capture the advantages of both individual tests, providing a more comprehensive assessment 
        of VaR performance.
        The test performs the Kupiec test and the Christoffersen test individually and combines their results to offer a more 
        robust backtesting assessment. The combined likelihood ratio from both tests is then used to make a decision regarding 
        the performance of the VaR model.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - S0_initial_s: Array indicating the initial stock/underlying prices at the beginning of the "test_returns" array.
        - test_returns: Array containing returns for backtesting. Ensure that the frequency of the "test_returns" is "daily".
        - var: The estimated Value-at-Risk to be tested.     
        - interval: The time horizon over which the risk measure was computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - alpha_test: The alpha value for the Kupiec test. Default is 0.05.
        - warning: If True, displays a warning about the frequency of 'test_returns'. Default is True.

        Returns
        -------
        - TestResults: A dictionary containing results of the combined test such as the combined likelihood ratio, critical value, 
        and decision regarding the null hypothesis.

        Notes
        -----
        Please ensure that the 'test_returns' is 'daily'.
        For instance:
        - for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and 'mappingRets' (if sharpeDiag) and set the interval to 5.
        - for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        # no need for inputs handler --- the process will be taken care in the kupiec_test and in the christoffersen_test function.
    
        from scipy.stats import chi2

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Backtesting a Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' is 'daily'.")
            print("For instance:")
            print("- for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the combined_test() function.")

        warning = False

        # no need for retuns aggregation here ---- it will be handled by both kupiec_test and christoffersen_test

        # now we can proceed ------------------------------------------------------------------------------------------------

        VaR = round(var, 4)

        # perform individual tests 
        kupiec_results = self.kupiec_test(num_options_s = num_options_s, contract_size_s = contract_size_s, S0_initial_s = S0_initial_s,
                                        test_returns = test_returns, var = var, interval = interval, alpha = alpha,
                                        alpha_test = alpha_test, warning = warning)
        
        christoffersen_results = self.christoffersen_test(num_options_s = num_options_s, contract_size_s = contract_size_s, S0_initial_s = S0_initial_s,
                                        test_returns = test_returns, var = var, interval = interval, alpha = alpha,
                                        alpha_test = alpha_test, warning = warning)
        
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
                "LR" : round(combined_likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}

##########################################################################################################################################################################
##########################################################################################################################################################################
## 6. AmOptionRmPort #####################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class AmOptionRmPort:

    """
    Main
    ----
    The `AmOptionRmPort` class provides a comprehensive risk management tool designed to evaluate the Value at Risk (VaR) 
    and Expected Shortfall (ES) for portfolios consisting of American options, leveraging the Binomial Lattice model.
    At each step, it is assumed that the underlying asset will move either up or down by a certain percentage with a given probability. 
    This results in a binomial tree representation of possible future asset prices.
    A DataFrame is constructed which provides a detailed summary of each option's type and its Binomial-Model price.

    Methodology
    ------------
    1. The function sets up a binomial tree with a user-specified number of time steps (steps) leading up to the option's expiration.

    2. The probability of an upward stock price movement (denoted as 'p') and downward movement (denoted as '1-p') are derived in a risk-neutral framework.
    The derivation depends on the risk-free rate, the volatility of the underlying asset, and the time step duration.

    3. The formulas for these probabilities are:
        - u (up factor) = e^(sigma * sqrt(dt))
        - d (down factor) = 1/u
        - p (probability of an up movement) = (e^(r * dt) - d) / (u - d)
        Where:
            - r = risk-free rate
            - sigma = volatility of the underlying
            - dt = T/steps (duration of a step, with T being the time to expiration)

    4. Starting at the final nodes of the tree (maturity), the option values are computed based on their intrinsic value. For a call option, it's max(S-K, 0) and for a put option, 
    it's max(K-S, 0), where S is the stock price and K is the strike price.

    5. The function then recursively calculates the option price at each prior node by discounting the expected future option values from the two possible succeeding nodes 
    (up and down nodes).

    6. For American options, at each node, the function also checks if exercising the option immediately provides a higher value than holding it.
    If so, the value from immediate exercise is used.

    7. The price of the American option is found at the root of the tree.

    Initial Parameters
    ------------------
    - K_s: An array of strike prices for each option in the portfolio. Represents the price at which the option can be exercised.
    - T_s: An array of time to maturities for each option. Specifies the time remaining until the option contract is due for expiration.
      - For instance: expDays = np.array([5,8,12,15,18]); T_s = expDays/252
    - S0_s: An array of initial spot prices for the underlying asset for each option. It represents the current market value of the asset which the option derives its value from.
    - r_s: An array of risk-free interest rates for each option (annualized). The risk-free rate represents a theoretical rate of return on an investment with zero risk.
    - sigma_s: An array of volatilities for the underlying asset for each option (annualized). 
    - option_type_s: An array indicating the type of each option (e.g., 'call' or 'put'). A call option gives the holder the right to buy the underlying asset, 
    while a put option gives the holder the right to sell the underlying asset.

    Methods
    -------
    - HistoricalSimulation(): Calculates the Value at Risk (VaR) and Expected Shortfall (ES) using historical simulations based on past return scenarios.
    - ff3 (): Fetches data for the Fama-French three-factor (FF3) model, which includes Market return (MKT), Size (SMB: Small Minus Big), 
        and Value (HML: High Minus Low) from Ken French's Data Library.
    - mcModelDrawReturns(): Determines the Value at Risk (VaR) and Expected Shortfall (ES) by simulating correlated future stock returns using a Monte Carlo method, 
      rooted in various statistical models for volatility and mean returns.
    - VarEsMinimizer(): seeks to minimize the Value at Risk (VaR) for a given portfolio of European options using historical simulations.
    - kupiec_test(): Conducts the Kupiec backtest for Value at Risk (VaR). The Kupiec test employs a likelihood ratio test to validate VaR model accuracy.
    - christoffersen_test(): Implements the Christoffersen test for Value at Risk (VaR), which examines the clustering of violations where actual returns underperform the predicted VaR.
    - combined_test(): Merges the Kupiec and Christoffersen tests, delivering a holistic assessment of Value at Risk (VaR) model precision.

    Attributes
    ----------
    - bi_prices : Array of option prices calculated using the binomial lattice model. This attribute is populated after calling the `fit` method.
    - summary : A DataFrame containing a summary of each option's type, binomial model price, strike price, time to maturity, initial spot price, risk-free rate, and volatility. 

    Notes
    -----
    All input arrays (`K_s`, `T_s`, `S0_s`, `r_s`, `sigma_s`, `option_type_s`) must share the same length as they correspond to 
    attributes for each option in the portfolio. 
    The `fit` method is activated by default during the initialization to generate the necessary values and populate the summary DataFrame.

    Additional Notes
    -----------------
    - In some methods of this class, for computational reasons, this class employs the QuantLib library package instead of a custom-built binomial tree. 

    - Utilizing the QuantLib library's well-optimized tools can provide more efficient and accurate results, especially for complex 
    option structures or when high computational performance is needed.

    - In that case: 
        1. The function begins by setting the calculation date to today's date.
        2. The Null calendar (which assumes no holidays) and the Actual365Fixed day count convention are used.
        3. The option type is determined (either call or put).
        4. Using the provided time to maturity, the exercise date is computed.
        5. An American exercise is defined, which signifies that the option can be exercised any time before or on the expiration date.
        6. With all the necessary components (payoff, exercise type, underlying spot, risk-free rate, and volatility), a Black-Scholes process is initiated.
        7. The BinomialVanillaEngine, specifically with the "eqp" method (equal probabilities), is then applied to this process. 
        The number of steps provided defines the tree's granularity.
        8. Once the pricing engine is set, the Net Present Value (NPV) of the option is computed, giving the option's price.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, K_s: np.ndarray,
                 T_s: np.ndarray, 
                 S0_s: np.ndarray,
                 r_s:  np.ndarray,
                 sigma_s: np.ndarray,
                 option_type_s: np.ndarray):

        
        from pyriskmgmt.inputsControlFunctions import (check_K_s, check_T_s, check_S0_s, check_r_s, chek_sigma_deri_s, check_option_type_s)

        self.K_s = check_K_s(K_s) ; self.T_s = check_T_s(T_s) ; self.S0_s = check_S0_s(S0_s) ; self.r_s = check_r_s(r_s) ; self.sigma_s = chek_sigma_deri_s(sigma_s)
        self.option_type_s = check_option_type_s(option_type_s)

        # ------------------------------------------------------------------------------------------------

        # checking the same length for all the imputs *args

        def check_same_length(*args):
            """
            Checks if all given numpy arrays have the same length.
            """
            it = iter(args)
            the_len = len(next(it))

            if not all(len(l) == the_len for l in it):
                raise ValueError("Not all the input arrays in the AmOptionRmPort class have the same length!")
            
        # Here we are ensuring that all arrays have the same length
        check_same_length(K_s, T_s, S0_s, r_s, sigma_s, option_type_s)

        # ------------------------------------------------------------------------------------------------

        self.bi_prices = None 

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd ; import numpy as np

        # importing the BiAmericanOptionPrice function from pyriskmgmt.SupportFunctions
        from pyriskmgmt.SupportFunctions import BiAmericanOptionPrice # (K , T, S0, r, sigma, option_type,  steps):

        prices = np.zeros(len(self.K_s))

        for i in range(len(self.K_s)):
            steps_s = int(self.T_s[i]* 252) # Keeping steps concise has shown to be effective. A good trade-off between speed and precision.
            price = BiAmericanOptionPrice(self.K_s[i], self.T_s[i], self.S0_s[i], self.r_s[i], self.sigma_s[i], self.option_type_s[i], steps_s)
            prices[i] = price

        # Connecting BS to BI in the case that the BiAmericanOptionPrice is lower than the BlackAndScholesPrice --------------------------------------
        from pyriskmgmt.derivative_models import  EuOptionRmPort

        model = EuOptionRmPort(K_s = self.K_s, T_s = self.T_s, S0_s = self.S0_s, r_s = self.r_s, sigma_s = self.sigma_s, option_type_s = self.option_type_s)
        bsprices = model.bs_prices

        for i in range(len(prices)):
            if prices[i] < bsprices[i]:
                prices[i] = bsprices[i]

        # ------------------------------------------------------------------------------------------------------------------------------------------------------

        summary = pd.DataFrame({'type': self.option_type_s, 'biprice': prices, "K" : self.K_s , "T": self.T_s, "S0": self.S0_s, "r": self.r_s, "sigma":  self.sigma_s})

        # rounding all the values to 4 decimal places
        summary = summary.round(4)

        # setting "type" as index
        summary.set_index('type', inplace=True)

        self.summary = summary ; self.bi_prices = summary["biprice"].values

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05, warning: bool = True):
        
        """
        Main
        -----
        Performs a historical simulation method for the computation of VaR and Expected Shortfall (ES) for a portfolio of American options. 
        The method uses historical returns of the underlying asset to simulate potential future returns and subsequently calculates the VaR and ES based on the 
        distribution of the simulated profits and losses.

        The computation of American option prices within the simulation relies on QuantLib's BinomialVanillaEngine, which is based on the binomial tree model.
        This model provides a discrete-time approximation for the option pricing. 

        - The specifics of this method are:
        1. Set the calculation date to today's date.
        2. Use the Null calendar and Actual365Fixed day count convention.
        3. Determine the option type (either call or put).
        4. Compute the exercise date using the provided time to maturity.
        5. Define an American exercise.
        6. Initiate a Black-Scholes process with the required components.
        7. Apply the BinomialVanillaEngine with the "eqp" method and the provided number of steps.
        8. Compute the option's Net Present Value (NPV) for its price.

        Moreover, an essential step in this simulation is a comparison between the price from QuantLib's method and the standard Black and Scholes model:
        - Whenever the QuantLib price is lower than the Black and Scholes price, the latter is used. It is crucial to note this approach: 
        - This difference has been tested, and its magnitude is minor (on average 0.00xxx). However, it is an approximation.
        - The focus of this package is Risk Management, not HFT (High-Frequency Trading), making this a good compromise in terms of computational costs.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - underlying_rets: 2D Array containing historical returns of the underlying assets associated with the options in the portfolio. 
        Ensure that the frequency of every column in the "underlying_rets" array is the same and 'daily'.
        - interval: The time horizon over which the risk measure is to be computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.
       
        Returns
        -------
        - HS: A dictionary containing the computed VaR, Expected Shortfall (ES), time horizon (T), minimum expiration in days (MinExpInDays), 
        maximum expiration in days (MaxExpInDays), number of simulations (NumSimu), and total initial position (TotInPos).

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        import pandas as pd ; import numpy as np

        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, validate_returns_port, check_interval, check_alpha, check_warning)

        # check procedure 

        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; underlying_rets = validate_returns_port(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning) 

        # validating that the length of num_options_s matches self.K_s
        if len(num_options_s) != len(self.K_s):
            raise ValueError("Length of num_options_s does not match length of self.K_s")

        # validating that the length of contract_size_s matches self.K_s
        if len(contract_size_s) != len(self.K_s):
            raise ValueError("Length of contract_size_s does not match length of self.K_s")

        # validating that the number of columns in underlying_rets matches self.K_s
        if underlying_rets.shape[1] != len(self.K_s):
            raise ValueError("Number of columns in underlying_rets_s does not match length of self.K_s")
        
        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Calculating Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("Ensure also that every column in the 'underlying_rets' has the same frequency (daily).")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the HistoricalSimulation() method.")
                        
        # defining the function to reshape and aggregate returns for a given interval
        def reshape_and_aggregate_returns(returns, interval):
            # converting the returns to their values
            returns = returns.values
            # determining the number of periods based on the interval
            num_periods = len(returns) // interval * interval
            # trimming the returns based on the computed number of periods
            trimmed_returns = returns[len(returns) - num_periods:]
            # reshaping the trimmed returns into chunks of the specified interval
            reshaped_returns = trimmed_returns.reshape((-1, interval))
            # aggregating the reshaped returns
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            # returning the aggregated returns
            return aggregated_returns

        # converting the underlying returns to a DataFrame
        rets_df = pd.DataFrame(underlying_rets)
        # applying the reshaping and aggregating function to the returns DataFrame
        rets_df = rets_df.apply(lambda x: reshape_and_aggregate_returns(x, interval))
        # converting the reshaped and aggregated returns back to an array
        underlying_rets = rets_df.values

        # importing the required functions from the pyriskmgmt.SupportFunctions module
        from pyriskmgmt.SupportFunctions import (QuantLibAmericanOptionPriceVectorized, BlackAndScholesPriceVectorization)

        # defining the VaREsCalculator function to compute Value-at-Risk and Expected Shortfall
        def VaREsCalculator(num_options_s, contract_size_s, underlying_rets, K_s, T_s, S0_s, r_s, sigma_s, option_type_s, total_initial_pos):
            # determining the number of options
            n_opts = len(S0_s)
            # computing simulated stock prices
            SimuStockPrices = S0_s * (1 + underlying_rets)
            # initializing the simulated option positions array
            SimuOptionPos = np.zeros((underlying_rets.shape[0], n_opts))
            # initializing the option price array
            OptionPrice = np.zeros((underlying_rets.shape[0], n_opts))
            # updating the time-to-maturity based on the interval
            NewTs = ((365 * T_s) - interval) / 365
            # computing the new number of steps for each option based on the updated time-to-maturity
            new_steps = np.array([int(365 * T - interval) for T in T_s], dtype=np.int64)
            # iterating through the underlying returns to compute option prices and positions
            for t in range(underlying_rets.shape[0]):
                # computing option prices using the QuantLibAmericanOptionPriceVectorized function
                OptionPrice[t, :] = QuantLibAmericanOptionPriceVectorized(K_s, NewTs, SimuStockPrices[t,:], r_s, sigma_s, option_type_s, new_steps)
                # computing option prices using the BlackAndScholesPriceVectorization function
                pricesBS = BlackAndScholesPriceVectorization(SimuStockPrices[t,:], T_s, K_s, sigma_s, r_s, option_type_s)
                # selecting the higher of the two computed option prices
                OptionPrice[t, :] = np.where(OptionPrice[t, :] < pricesBS, pricesBS, OptionPrice[t, :])
                # computing the simulated option position for each return scenario
                SimuOptionPos[t, :] = OptionPrice[t, :] * num_options_s * contract_size_s
            # summing the simulated option positions to get the simulated option portfolio value
            SimuOptionPort = np.sum(SimuOptionPos, axis=1)
            # computing the profit and loss for the portfolio
            Port_PL = SimuOptionPort - total_initial_pos
            # returning the computed profit and loss
            return Port_PL

        # computing the total initial position for the portfolio
        total_initial_pos = np.sum(self.bi_prices * num_options_s * contract_size_s)
        # determining the total number of options in the portfolio
        tot_ops = len(self.K_s)

        # importing the DotPrinter function for aesthetic purposes
        from pyriskmgmt.SupportFunctions import DotPrinter

        # initializing the DotPrinter for animation purposes
        my_dot_printer = DotPrinter(f"\rPortfolio of {tot_ops} American Options - Historical Simulation - VaR & ES Computation")
        # starting the DotPrinter animation
        my_dot_printer.start()

        # computing the profit and loss for the portfolio using the VaREsCalculator function
        Port_PL = VaREsCalculator(num_options_s, contract_size_s, underlying_rets, self.K_s,  self.T_s, self.S0_s, self.r_s, self.sigma_s, self.option_type_s, total_initial_pos)

        # stopping the DotPrinter animation
        my_dot_printer.stop()
        # printing a completion message
        print("\rVaR & ES Computation --->  Done", end=" " * 150)

        # computing the Value-at-Risk for the portfolio at the specified confidence level
        VaR = - np.quantile(Port_PL, alpha)
        # computing the Expected Shortfall for the portfolio
        ES = - np.mean(Port_PL[Port_PL < - VaR])

        # compiling the computed metrics into a dictionary
        HS = {"var" : round(VaR, 4),
              "es" : round(ES, 4),
              "T" : interval,
              "MinExpInDays" : int(min(self.T_s)*252),
              "MaxExpInDays" : int(max(self.T_s)*252),
              "NumSimu" : underlying_rets.shape[0],
              "TotInPos": round(total_initial_pos, 4)}

        return HS

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
        
    # MONTECARLO MODEL [RETURNS SIMULATION] <<<<<<<<<<<<
    
    def mcModelRetsSimu(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, underlying_rets: np.ndarray, interval: int = 1,
                        alpha: float = 0.05, sharpeDiag: bool = False , mappingRets :np.ndarray = None, vol: str = "simple",
                        num_sims: int = 1000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, warning: bool = True):
        
        """
        Main
        ----
        Simulates returns using Monte Carlo models and calculates the Value-at-Risk (VaR) and Expected Shortfall (ES) based on specified configurations. 
        
        A significant feature of this function is the approximation mechanism applied to adjust American option prices.

        - Here's a detailed breakdown:
        1. Utilizing the Binomial and Black-Scholes Models:
        The method takes into account both the Binomial Model prices and the Black-Scholes prices for American options.
        The Black-Scholes model is a foundational tool for derivatives traders, but it doesn't consider early exercise of options. 
        The Binomial model, on the other hand, can handle early exercise and is particularly suited for American options. 
        However, calculating American option prices using the Binomial model can be computationally expensive, especially for large steps in the tree. 
        This leads us to the core idea of the approximation.

        2. Introducing the Approximation Mechanism:
        To optimize speed without compromising too much on accuracy, an approximation ratio (ratio_bi_bs) is calculated from the prices derived from these two models.
        Essentially, for each option, this ratio represents the TODAY's relationship of its Binomial Model price to its Black-Scholes price. 
        This is an innovative way to approximate American option prices, borrowing strengths from both models.

        3. Ensuring Robustness against Zero Division and NaN:
        To ensure that the method remains robust and doesn't falter due to mathematical anomalies, there are layers of checks integrated:
        - First Layer: Direct comparison to zero. If either the Binomial Model price or the Black-Scholes price for an option is zero, the ratio is set to 1. 
        This avoids potential division by zero.
        - Second Layer: After calculating the ratios, any resulting NaN (Not-a-Number) value, which can arise due to divisions like 0/0, is also set to 1.
        NaN values can cause unpredictable behavior and setting them to 1 ensures stability in the calculations.
        The essence of this approximation is to quickly and efficiently estimate the prices of American options while taking into account the variations
        and peculiarities of both the Binomial and Black-Scholes models. This mechanism enables faster risk calculations, making it a valuable asset in time-sensitive financial scenarios.
        
        To utilize Sharpe diagnostics, ensure 'sharpeDiag' is set to True and 'mappingRets' is provided.
        When 'sharpeDiag' is enabled, the variance-covariance matrix of underlying returns is calculated 
        using a mapping procedure from n factors, incorporating the estimation of idiosyncratic variance 
        to ensure the matrix is positive and semi-definite and ensures numerical stability.
        'num_sims' simulations of joint returns (correlated) will be used to estimate the Risk Measures.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - underlying_rets: 2D Array containing historical returns of the underlying assets associated with the options in the portfolio. 
        Ensure that the frequency of every column in the "underlying_rets" array is the same as the others and all 'daily'.
        - interval: The time horizon over which the risk measure is to be computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The alpha value for VaR calculation. Default is 0.05.
        - sharpeDiag: If enabled, the function employs a Sharpe diagonal approach to map "underlying_rets" to 1 or more factors. Default is None.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on a daily frequency
        and align "mappingRets" with "underlying_rets" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Specifies the volatility estimation model to use, either "simple","garch",or "ewma". Default is "simple".
        - num_sims: Number of Monte Carlo simulations to be performed. Default is 1000.
        - p: The order of the autoregressive term for the 'garch' volatility method. Default is 1.
        - q: The order of the moving average term for the 'garch' volatility method. Default is 1.
        - lambda_ewma: The decay factor for the 'ewma' volatility method. Default is 0.94.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.

        Returns
        -------
        - MC_DIST: A dictionary containing computed values such as VaR, Expected Shortfall (ES), 
        and other relevant metrics.

        Notes
        -----
        Please ensure that the 'underlying_rets' and the 'mappingRets' frequency is 'daily'.
        ['mappingRets' is needed only if 'sharpeDiag' = True].
        For instance:
        - for the 5-day ahead VaR: provide daily 'underlying_rets' and 'mappingRets' (if sharpeDiag) and set the interval to 5.
        - for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        import numpy as np ;  from joblib import Parallel, delayed
        
        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, validate_returns_port, check_interval, check_alpha, check_warning)
        
        # check procedure 
        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; underlying_rets = validate_returns_port(underlying_rets)
        interval = check_interval(interval) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Calculating Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("Ensure also that every column in the 'underlying_rets' has the same frequency (daily).")
            print("For instance:")
            print("- for the 5-day ahead VaR: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week ahead VaR: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the mcModelRetsSimu() method.")

        if vol == "garch":
            
            if warning:
                from pyriskmgmt.SupportFunctions import garch_warnings
                garch_warnings("mcModelRetsSimu()")
                
            if warning and underlying_rets.shape[1] > 50 and not sharpeDiag:
                print("IMPORTANT:")
                print("The 'garch' option for the 'vol' parameter is currently in use, with an asset count exceeding 50.")
                print("This configuration can lead to slower processing and inefficient resource usage, due to the complexity ")
                print("of estimating the full variance-covariance matrix (VCV) in high-dimensional space.")
                print("It's strongly advised to employ a mapping approach instead. This approach maps the multiple assets to one or more factors,")
                print("effectively reducing the dimensionality of the problem.")
                print("This is not only more computationally efficient, but it also avoids the 'curse of dimensionality',")
                print("as it reduces the number of parameters needed to be estimated. ")
                print("The mapping approach aids in providing more stable and reliable estimations, especially in large, complex systems.")

        warning = False

        # all the other inputs will be handled here, in the EquityPrmPort class -------------------------------------------------------------------------------------

        # importing the EquityPrmPort class from the pyriskmgmt module
        from pyriskmgmt.equity_models import EquityPrmPort

        # instantiating the EquityPrmPort model with given parameters
        model = EquityPrmPort(returns=underlying_rets, positions=np.random.uniform(-1, 1, underlying_rets.shape[1]),
                            interval=interval, alpha=alpha)

        # generating Monte Carlo simulated returns using the instantiated model
        simulated_rets = model.mcModelRetsSimu(sharpeDiag=sharpeDiag, mappingRets=mappingRets, vol=vol, num_sims=num_sims, p=p, q=q,
                                            lambda_ewma=lambda_ewma, warning=warning, return_simulated_rets=True)[1]

        # vectorizing inputs for further processing

        # creating a 2D version of T_s, K_s, sigma_s, r_s, and option_type_s by adding a new axis
        T_s_2D = self.T_s[np.newaxis, :]
        K_s_2D = self.K_s[np.newaxis, :]
        sigma_s_2D = self.sigma_s[np.newaxis, :]
        r_s_2D = self.r_s[np.newaxis, :]
        option_type_s_2D = self.option_type_s[np.newaxis, :]

        # adjusting the shape of T_s, K_s, sigma_s, r_s arrays to match the shape of simulated_rets
        New_T_s = ((252 * T_s_2D - interval) / 252) * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        New_K_s = K_s_2D * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        New_sigma_s = sigma_s_2D * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))
        New_r_s = r_s_2D * np.ones((simulated_rets.shape[0], simulated_rets.shape[1]))

        # broadcasting option_type_s_2D array to match the shape of one_matrix_simu
        one_matrix_simu = simulated_rets[:, :, 0]
        New_option_type_s = np.broadcast_to(option_type_s_2D, one_matrix_simu.shape)

        # importing the EuOptionRmPort class from the pyriskmgmt module
        from pyriskmgmt.derivative_models import EuOptionRmPort

        # instantiating the EuOptionRmPort model with given parameters
        Eu = EuOptionRmPort(K_s=self.K_s, T_s=self.T_s, S0_s=self.S0_s, r_s=self.r_s, sigma_s=self.sigma_s, option_type_s=self.option_type_s)

        # retrieving Black-Scholes prices from the instantiated model
        bs_prices = Eu.bs_prices

        # initializing an array to store the ratio between bi_prices and bs_prices
        ratio_bi_bs = np.zeros(len(self.bi_prices))

        # handling potential divisions by zero when computing the ratio

        # first layer of handling division by zero
        for index in range(len(ratio_bi_bs)):
            # checking if both bi_prices and bs_prices values are zero
            if self.bi_prices[index] and bs_prices[index] == 0:
                ratio_bi_bs[index] = 1
            else:
                # computing the ratio of bi_prices to bs_prices
                ratio_bi_bs[index] = self.bi_prices[index] / bs_prices[index]

        # second layer of handling division by zero: replacing NaN values with 1
        ratio_bi_bs[np.isnan(ratio_bi_bs)] = 1

        # importing the BlackAndScholesPriceVectorization function for vectorized computation of option prices
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization

        # defining a function to calculate Value-at-Risk (VaR) and Expected Shortfall (ES) based on simulated returns
        def calculate_simulated_var_es(sim, simulated_rets, S0_s, bs_prices, num_options_s, contract_size_s, 
                                        New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s, alpha):
            
            # computing the simulated stock prices based on the initial stock prices and the simulated returns
            SimuStockPrices = S0_s * (1 + simulated_rets[:,:,sim])
            
            # computing option prices using the Black-Scholes model and the simulated stock prices
            OptionPrices = BlackAndScholesPriceVectorization(SimuStockPrices, New_T_s, New_K_s, New_sigma_s, New_r_s, New_option_type_s)
            
            # adjusting or scaling the OptionPrices based on a predefined ratio (ratio_bi_bs)
            for index in range(OptionPrices.shape[0]):
                OptionPrices[index,:] = OptionPrices[index,:] * ratio_bi_bs
            
            # computing the simulated option positions based on the option prices, number of options, and contract sizes
            SimuOptionPos = OptionPrices * num_options_s * contract_size_s
            
            # calculating the Profit & Loss (P&L) and the total P&L
            PL = SimuOptionPos - bs_prices * num_options_s * contract_size_s
            total_PL = PL.sum(axis=1)
            
            # computing Value-at-Risk (VaR) and Expected Shortfall (ES) based on the total P&L
            VaR = np.quantile(total_PL, alpha) * -1
            ES = np.mean(total_PL[total_PL < - VaR]) * -1 

            return VaR, ES

        # retrieving the number of simulations based on the shape of the simulated_rets array
        num_sims = simulated_rets.shape[2]

        # importing the DotPrinter function for a progress animation
        from pyriskmgmt.SupportFunctions import DotPrinter

        # initializing and starting the progress animation
        my_dot_printer = DotPrinter(f"\r {num_sims} Scenarios for {one_matrix_simu.shape[1]} American Options - joblib - VaR & ES Parallel Computation")
        my_dot_printer.start()

        # parallelizing the computation of VaR and ES across multiple simulations
        results = Parallel(n_jobs=-1)(delayed(calculate_simulated_var_es)(sim, simulated_rets, self.S0_s, bs_prices, num_options_s, contract_size_s, New_T_s,
                                                                        New_K_s, New_sigma_s, New_r_s, New_option_type_s, alpha) for sim in range(num_sims))

        # stopping the progress animation after computation
        my_dot_printer.stop()
        print("\rParallel Computation --->  Done", end=" " * 90)

        # extracting the VaR and ES results from the parallel computation and computing their means
        VARS, ESs = zip(*results)
        VaR = np.mean(VARS)
        ES = np.mean(ESs)

        # computing the total initial position of the portfolio
        total_initial_pos = np.sum(self.bi_prices * num_options_s * contract_size_s)
        TotInPos = round(total_initial_pos, 4)

        # compiling the results into a dictionary and rounding them for better presentation
        MC_DIST = {"var": round(VaR, 4),
                   "es" : round(ES, 4),
                   "T": interval,
                   "MinExpInDays" : int(min(self.T_s) * 252),
                   "MaxExpInDays" : int(max(self.T_s) * 252),
                   "NumSimu" : simulated_rets.shape[0],
                   "McNumSim" : num_sims,
                   "TotInPos" : TotInPos}

        return MC_DIST
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    #  VAR ES MINIMAZER <<<<<<<<<<<<<<<

    def VarEsMinimizer(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, underlying_rets: np.ndarray, interval: int = 1, alpha: float = 0.05,
                        warning: bool = True, allow_short: bool = True):
        
        """
        Main
        ----
        Optimizes the number of options in a portfolio to minimize the Value-at-Risk (VaR) and Expected Shortfall (ES) 
        using historical simulation over a specified interval.
        The optimization process maintains the overall portfolio position and can allow or disallow short positions based on the 'allow_short' parameter.
        - The pricing of American options is often close to that of European options, especially when the 
        option is not deeply in or out of the money and when there's a substantial amount of time until expiration.
        - Given this closeness in pricing, and for computational efficiency, we use the European option prices to minimize the Value-at-Risk (VaR). 
        - This allows us to leverage faster techniques available for European options, providing a good approximation for risk minimization of American options.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - underlying_rets: 2D Array containing historical returns of the underlying assets associated with the options in the portfolio. 
        Ensure that the frequency of every column in the "underlying_rets" array is the same as the others and all 'daily'.
        - interval: The time horizon over which the risk measure is to be computed and minimized (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation and minization. Default is 0.05.
        - warning: If True, provides a warning about ensuring the correct 'underlying_rets' frequency for the analysis. Default is True.
        - allow_short: If True, allows the minimization process to consider short positions. Default is True.

        Returns
        -------
        - MIN_PROCESS: A dictionary containing optimized values such as VaR, ES, initial and final positions, and other relevant metrics.

        Notes
        -----
        Please ensure that the 'underlying_rets' frequency is 'daily'.
        For instance:
        - for the 5-day VaR minization process: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week VaR minization process: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np 
        
        # -------------------------------------------------------------------------------------------------------------------------------------------------------
        # Note on this approach: # >>>> Approximation <<<<<
        # The pricing of American options is often close to that of European options, especially when the 
        # option is not deeply in or out of the money and when there's a substantial amount of time until expiration.
        # Given this closeness in pricing, and for computational efficiency, we use the European option prices to minimize the Value-at-Risk (VaR). 
        # This allows us to leverage faster techniques available for European options, providing a good approximation for risk minimization of American options.
        # -------------------------------------------------------------------------------------------------------------------------------------------------------

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Calculating and minimizing Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)
 
        if warning:
            print("WARNING:")
            print("Please ensure that the 'underlying_rets' frequency is 'daily'.")
            print("Ensure also that every column in the 'underlying_rets' has the same frequency (daily).")
            print("For instance:")
            print("- for the 5-day minimization process: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week minimization process: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the VarEsMinimizer() method.")

        warning = False
        
        # All the inputs will be handled here!
        
        from pyriskmgmt.derivative_models import  EuOptionRmPort

        # initializing the EuOptionRmPort class with given parameters
        Eu = EuOptionRmPort(K_s = self.K_s, T_s = self.T_s, S0_s = self.S0_s, r_s = self.r_s, sigma_s = self.sigma_s,
                                option_type_s = self.option_type_s)
        
        # optimizing the number of options based on minimizing Value-at-Risk (VaR) and Expected Shortfall (ES)
        optimized_num_options_s = Eu.VarEsMinimizer(num_options_s = num_options_s, contract_size_s = contract_size_s, interval= interval,
                                                       underlying_rets = underlying_rets, warning = warning, allow_short = allow_short)["NewPos"]
        
        # Now back here to HistoricalSimulation - American Options ---------------------------------------------------------------------------------------------------------

        # optimized_num_options_s from the European Options framework!

        result = self.HistoricalSimulation(num_options_s = optimized_num_options_s, contract_size_s= contract_size_s, underlying_rets = underlying_rets, interval = interval,
                                           alpha = alpha, warning =  False) #  warning = False
        
        # calculating the total initial position based on bi_prices, number of options, and contract size
        total_INITIAL_pos = np.sum(self.bi_prices * num_options_s * contract_size_s)
        total_FINAL_pos = np.sum(self.bi_prices * optimized_num_options_s * contract_size_s)
        
        # rounding the initial and final total positions for better presentation
        TotInPos = round(total_INITIAL_pos, 4) ; TotFinPos = round(total_FINAL_pos, 4)
        
        # compiling the results into a dictionary
        MIN_PROCESS = {"var" : result["var"],
                      "es" : result["es"],
                      "T" : interval,
                      "MinExpInDays" : int(min(self.T_s)*252),
                      "MaxExpInDays" : int(max(self.T_s)*252),
                      "TotInPos" : TotInPos,
                      "TotFinPos" : TotFinPos,
                      "OldPos" : num_options_s,
                      "NewPos" : np.array(optimized_num_options_s),
                      "Differ" : optimized_num_options_s - num_options_s}
                
        return MIN_PROCESS
    
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray,
                     var: float, interval: int = 1, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):

        """
        Main
        ----
        Backtests the Value-at-Risk (VaR) of a portfolio by applying the Kupiec test, which uses a likelihood ratio to compare the 
        predicted and actual exceptions.
        The Kupiec test uses a likelihood ratio to assess whether the number of actual exceptions is consistent 
        with the VaR model's predictions.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - S0_initial_s: Array indicating the initial stock/underlying prices at the beginning of the "test_returns" array.
        - test_returns: Array containing returns for backtesting. Ensure that the frequency of the "test_returns" is "daily".
        - var: The estimated Value-at-Risk to be tested.     
        - interval: The time horizon over which the risk measure was computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - alpha_test: The alpha value for the Kupiec test. Default is 0.05.
        - warning: If True, displays a warning about the frequency of 'test_returns'. Default is True.

        Returns
        -------
        - TestResults: A dictionary containing results of the Kupiec test such as number of exceptions, likelihood ratio, 
        critical value, and decision regarding null hypothesis.

        Notes
        -----
        Please ensure that the 'test_returns' is 'daily'.
        For instance:
        - for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd ; import numpy as np ; from scipy.stats import chi2 ;  from joblib import Parallel, delayed

        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, check_S0_initial_s, validate_returns_port, check_var,
                                                        check_interval, check_alpha,check_alpha_test, check_warning)

        # check procedure
        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s)
        test_returns = validate_returns_port(test_returns) ; var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha)
        alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Backtesting a Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' is 'daily'.")
            print("For instance:")
            print("- for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the kupiec_test() function.")

        # defining a function to reshape and aggregate returns over a given interval
        def reshape_and_aggregate_returns(returns, interval):
            # converting the returns to an array
            returns = returns.values
            # computing the number of periods and trimming returns to be a multiple of the interval
            num_periods = len(returns) // interval * interval
            trimmed_returns = returns[len(returns) - num_periods:]
            # reshaping the returns according to the specified interval
            reshaped_returns = trimmed_returns.reshape((-1, interval))
            # aggregating the returns over the interval
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            
            return aggregated_returns

        # converting test returns to a pandas DataFrame
        test_returns = pd.DataFrame(test_returns)

        # applying the reshape and aggregate function to the test returns for each column (series of returns)
        test_returns = test_returns.apply(lambda x: reshape_and_aggregate_returns(x, interval))
        # converting the DataFrame back to an array
        test_returns = test_returns.values

        # rounding the Value-at-Risk (VaR) to 4 decimal places
        VaR = round(var, 4)

        # initializing an array to store simulated stock price paths
        stockPricePathTotal = np.zeros((test_returns.shape[0]+1, len(self.K_s)))

        # setting the initial stock prices for the simulation
        stockPricePathTotal[0,:] = S0_initial_s

        # simulating stock price paths using the aggregated test returns
        for i in range(len(self.K_s)):
            stockPricePathTotal[1:, i] = S0_initial_s[i] * np.cumprod(1 + test_returns[:, i])

        # importing the necessary functions for option pricing
        from pyriskmgmt.SupportFunctions import ( BiAmericanOptionPriceVectorized, BiAmericanOptionPrice)
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization 

        # defining a function to compute the option portfolio position in parallel for given stock price paths
        def compute_port_op_position(row, T_s, K_s, sigma_s, r_s, option_type_s, steps_s, num_options_s, contract_size_s):
            # computing the Binomial American option prices
            prices = BiAmericanOptionPriceVectorized(S0_s = row, T_s = T_s, K_s = K_s, sigma_s = sigma_s, r_s = r_s, option_type_s = option_type_s, steps_s = steps_s,
                                                    inner_function = BiAmericanOptionPrice)
            # computing the Black-Scholes option prices
            pricesBS = BlackAndScholesPriceVectorization(St_s = row, T_s = T_s, K_s = K_s, sigma_s = sigma_s, r_s = r_s, option_type_s = option_type_s)
            
            # comparing the two sets of prices and using the higher one
            prices = np.where(prices < pricesBS, pricesBS, prices)

            # ----------------------------------------------------------------------------------------------------------------------------------------------
            # Note on this approach:
            # this difference has been tested its magnitude is on average 0.00xxx - however, it is an approximation!
            # this package is about Risk Management, not HFT!
            # ----------------------------------------------------------------------------------------------------------------------------------------------
            
            # returning the aggregated portfolio position for the given stock prices
            return np.sum(prices * num_options_s * contract_size_s)

        # calculating the number of steps based on the time to maturity for each option
        steps_s = [int(252 * T) for T in self.T_s]

        # importing the DotPrinter function for aesthetic purposes
        from pyriskmgmt.SupportFunctions import DotPrinter

        # starting a printing animation to indicate progress
        my_dot_printer = DotPrinter(f"\r {len(self.K_s)} Options - joblib - Parallel Backtesting Procedure - kupiec_test")
        my_dot_printer.start()

        # parallelizing the computation of the portfolio option position for each stock price path
        results = Parallel(n_jobs=-1)(delayed(compute_port_op_position)(row, self.T_s, self.K_s, self.sigma_s, self.r_s, self.option_type_s,
                                                                        steps_s, num_options_s, contract_size_s) for row in stockPricePathTotal)

        # stopping the printing animation
        my_dot_printer.stop()
        print("\rParallel Computation --->  Done", end=" " * 80)

        # extracting the portfolio positions from the results
        PortOpPosition = np.array(results)
        # computing the Profit & Loss (P&L) from the portfolio positions
        PL = np.diff(PortOpPosition)
        # counting the number of exceptions (times when the P&L was worse than the VaR)
        num_exceptions = np.sum(PL < - VaR)
        
        # total number of observations -----------------------------------------------------------------------------------

        num_obs = len(PL)

        # % exceptions -------------------------------------------------------------------------------------------------------

        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "T" : interval,                         # interval
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # CHRISTOFFESEN TEST <<<<<<<<<<<<

    def christoffersen_test(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray,
                     var: float, interval: int = 1, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):

        """
        Christoffersen Test
        -------------------
        Backtests the Value-at-Risk (VaR) of a portfolio using the Christoffersen test, a test that focuses on the independence of 
        exceptions over time. It utilizes the likelihood ratio to examine if the clustering of exceptions is statistically significant.
        The Christoffersen test evaluates the clustering of exceptions, providing insight into the independence of VaR breaches.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - S0_initial_s: Array indicating the initial stock/underlying prices at the beginning of the "test_returns" array.
        - test_returns: Array containing returns for backtesting. Ensure that the frequency of the "test_returns" is "daily".
        - var: The estimated Value-at-Risk to be tested.     
        - interval: The time horizon over which the risk measure was computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - alpha_test: The alpha value for the Kupiec test. Default is 0.05.
        - warning: If True, displays a warning about the frequency of 'test_returns'. Default is True.

        Returns
        -------
        - TestResults: A dictionary containing results of the Christoffersen test such as the likelihood ratio, critical value, 
        and decision regarding the null hypothesis.

        Notes
        -----
        Please ensure that the 'test_returns' is 'daily'.
        For instance:
        - for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.
        - for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        import pandas as pd ; import numpy as np ; from scipy.stats import chi2 ; from joblib import Parallel, delayed

        from pyriskmgmt.inputsControlFunctions import (check_num_options_s, check_contract_size_s, check_S0_initial_s, validate_returns_port, check_var,
                                                        check_interval, check_alpha,check_alpha_test, check_warning)

        # check procedure
        num_options_s = check_num_options_s(num_options_s) ; contract_size_s = check_contract_size_s(contract_size_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s)
        test_returns = validate_returns_port(test_returns) ; var = check_var(var) ; interval = check_interval(interval) ; alpha = check_alpha(alpha)
        alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Backtesting a Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' is 'daily'.")
            print("For instance:")
            print("- for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the christoffersen_test() function.")

        # defining a function to reshape and aggregate returns over a given interval
        def reshape_and_aggregate_returns(returns, interval):
            # converting the returns to an array
            returns = returns.values
            # computing the number of periods and trimming returns to be a multiple of the interval
            num_periods = len(returns) // interval * interval
            trimmed_returns = returns[len(returns) - num_periods:]
            # reshaping the returns according to the specified interval
            reshaped_returns = trimmed_returns.reshape((-1, interval))
            # aggregating the returns over the interval
            aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            
            return aggregated_returns

        # converting test returns to a pandas DataFrame
        test_returns = pd.DataFrame(test_returns)

        # applying the reshape and aggregate function to the test returns for each column (series of returns)
        test_returns = test_returns.apply(lambda x: reshape_and_aggregate_returns(x, interval))
        # converting the DataFrame back to an array
        test_returns = test_returns.values

        # rounding the Value-at-Risk (VaR) to 4 decimal places
        VaR = round(var, 4)

        # initializing an array to store simulated stock price paths
        stockPricePathTotal = np.zeros((test_returns.shape[0]+1, len(self.K_s)))

        # setting the initial stock prices for the simulation
        stockPricePathTotal[0,:] = S0_initial_s

        # simulating stock price paths using the aggregated test returns
        for i in range(len(self.K_s)):
            stockPricePathTotal[1:, i] = S0_initial_s[i] * np.cumprod(1 + test_returns[:, i])

        # importing the necessary functions for option pricing
        from pyriskmgmt.SupportFunctions import ( BiAmericanOptionPriceVectorized, BiAmericanOptionPrice)
        from pyriskmgmt.SupportFunctions import BlackAndScholesPriceVectorization 

        # defining a function to compute the option portfolio position in parallel for given stock price paths
        def compute_port_op_position(row, T_s, K_s, sigma_s, r_s, option_type_s, steps_s, num_options_s, contract_size_s):
            # computing the Binomial American option prices
            prices = BiAmericanOptionPriceVectorized(S0_s = row, T_s = T_s, K_s = K_s, sigma_s = sigma_s, r_s = r_s, option_type_s = option_type_s, steps_s = steps_s,
                                                    inner_function = BiAmericanOptionPrice)
            # computing the Black-Scholes option prices
            pricesBS = BlackAndScholesPriceVectorization(St_s = row, T_s = T_s, K_s = K_s, sigma_s = sigma_s, r_s = r_s, option_type_s = option_type_s)
            
            # comparing the two sets of prices and using the higher one
            prices = np.where(prices < pricesBS, pricesBS, prices)

            # ----------------------------------------------------------------------------------------------------------------------------------------------
            # Note on this approach:
            # this difference has been tested its magnitude is on average 0.00xxx - however, it is an approximation!
            # this package is about Risk Management, not HFT!
            # ----------------------------------------------------------------------------------------------------------------------------------------------
            
            # returning the aggregated portfolio position for the given stock prices
            return np.sum(prices * num_options_s * contract_size_s)

        # calculating the number of steps based on the time to maturity for each option
        steps_s = [int(252 * T) for T in self.T_s]

        # importing the DotPrinter function for aesthetic purposes
        from pyriskmgmt.SupportFunctions import DotPrinter

        # starting a printing animation to indicate progress
        my_dot_printer = DotPrinter(f"\r {len(self.K_s)} Options - joblib - Parallel Backtesting Procedure - christoffersen_test")
        my_dot_printer.start()

        # parallelizing the computation of the portfolio option position for each stock price path
        results = Parallel(n_jobs=-1)(delayed(compute_port_op_position)(row, self.T_s, self.K_s, self.sigma_s, self.r_s, self.option_type_s,
                                                                        steps_s, num_options_s, contract_size_s) for row in stockPricePathTotal)

        # stopping the printing animation
        my_dot_printer.stop()
        print("\rParallel Computation --->  Done", end=" " * 100)

        # extracting the portfolio positions from the results
        PortOpPosition = np.array(results)
        # computing the Profit & Loss (P&L) from the portfolio positions
        PL = np.diff(PortOpPosition)
        exceptions = PL < - VaR

        # shifting exceptions to get previous day's exceptions
        prev_exceptions = np.roll(exceptions, shift=1)

        # dropping the first row (which is NaN because of the shift)
        exceptions = exceptions[1:]
        prev_exceptions = prev_exceptions[1:]

        # calculating transition counts
        T_00, T_01 = np.sum((prev_exceptions == 0) & (exceptions == 0)), np.sum((prev_exceptions == 0) & (exceptions == 1))
        T_10, T_11 = np.sum((prev_exceptions == 1) & (exceptions == 0)), np.sum((prev_exceptions == 1) & (exceptions == 1))

        # estimating conditional probabilities
        p_b0, p_b1 = T_01 / (T_00 + T_01), T_11 / (T_10 + T_11)

        # calculating likelihood ratio
        p = alpha
        likelihood_ratio = -2 * (np.log((1 - p) ** (T_00 + T_10) * p ** (T_01 + T_11)) - np.log((1 - p_b0) ** T_00 * p_b0 ** T_01 * (1 - p_b1) ** T_10 * p_b1 ** T_11))

        # critical value from chi-squared distribution
        critical_value = chi2.ppf(1 - alpha_test, df=1)

        # rejecting the null hypothesis if likelihood ratio is higher than the critical value
        reject_null = likelihood_ratio > critical_value

        return {"test" : "Christoffersen",
                "var" : var,
                "LR" : round(likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # COMBINED TEST <<<<<<<<<<<<

    def combined_test(self, num_options_s: np.ndarray, contract_size_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray,
                     var: float, interval: int = 1, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):
        """
        Main
        ----
        Backtests the Value-at-Risk (VaR) of a portfolio by combining results from both the Kupiec and the Christoffersen tests. 
        This combined approach aims to capture the advantages of both individual tests, providing a more comprehensive assessment 
        of VaR performance.
        The test performs the Kupiec test and the Christoffersen test individually and combines their results to offer a more 
        robust backtesting assessment. The combined likelihood ratio from both tests is then used to make a decision regarding 
        the performance of the VaR model.

        Parameters
        ----------
        - num_options_s: Array indicating the number of option contracts for each option in the portfolio.
        - contract_size_s: Array indicating the size of each option contract in the portfolio. For each option, how many 
        stocks or underlying positions it allows you to buy/sell.
        - S0_initial_s: Array indicating the initial stock/underlying prices at the beginning of the "test_returns" array.
        - test_returns: Array containing returns for backtesting. Ensure that the frequency of the "test_returns" is "daily".
        - var: The estimated Value-at-Risk to be tested.     
        - interval: The time horizon over which the risk measure was computed (days, assuming 252 trading days in a year). Default is 1.
        - alpha: The significance level used for the VaR and ES calculation. Default is 0.05.
        - alpha_test: The alpha value for the Kupiec test. Default is 0.05.
        - warning: If True, displays a warning about the frequency of 'test_returns'. Default is True.

        Returns
        -------
        - TestResults: A dictionary containing results of the combined test such as the combined likelihood ratio, critical value, 
        and decision regarding the null hypothesis.

        Notes
        -----
        Please ensure that the 'test_returns' is 'daily'.
        For instance:
        - for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and 'mappingRets' (if sharpeDiag) and set the interval to 5.
        - for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.
        """

        import warnings ; warnings.filterwarnings("ignore")

        # no need for inputs handler --- the process will be taken care in the kupiec_test and in the christoffersen_test function.
    
        from scipy.stats import chi2

        if interval >= int(min(self.T_s) * 252):
            raise Exception(f"""
            The earliest expiring option in your portfolio has only {int(min(self.T_s) * 252)} days left until expiration. 
            You've chosen a VaR interval of {interval} days, which exceeds this option's remaining duration. 
            Backtesting a Value-at-Risk (VaR) for a period that surpasses an option's expiration is not feasible. 
            Kindly adjust the interval to be shorter than the earliest option's remaining lifetime.
            """)

        if warning:
            print("WARNING:")
            print("Please ensure that the 'test_returns' is 'daily'.")
            print("For instance:")
            print("- for the 5-day VaR backtesting procedure: provide daily 'underlying_rets' and set the interval to 5.")
            print("- for the 2-week VaR backtesting procedure: provide daily 'underlying_rets' and adjust the interval to 10.")
            print("To hide this warning, use 'warning=False' when calling the combined_test() function.")

        warning = False

        # no need for retuns aggregation here ---- it will be handled by both kupiec_test and christoffersen_test

        # now we can proceed ------------------------------------------------------------------------------------------------

        VaR = round(var, 4)

        # perform individual tests ----------------------------

        kupiec_results = self.kupiec_test(num_options_s = num_options_s, contract_size_s = contract_size_s, S0_initial_s = S0_initial_s,
                                        test_returns = test_returns, var = var, interval = interval, alpha = alpha,
                                        alpha_test = alpha_test, warning = warning)
        
        christoffersen_results = self.christoffersen_test(num_options_s = num_options_s, contract_size_s = contract_size_s, S0_initial_s = S0_initial_s,
                                        test_returns = test_returns, var = var, interval = interval, alpha = alpha,
                                        alpha_test = alpha_test, warning = warning)
        
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
                "LR" : round(combined_likelihood_ratio, 5),
                "cVal" : round(critical_value, 5),
                "rejNull" : reject_null}
    
##########################################################################################################################################################################
##########################################################################################################################################################################
## 7. EquityForwardSingle ################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EquityForwardSingle:

    """
    Main
    ----
    This class offers tools to assess and compute the VaR and ES at the maturity of forward contracts, 
    critical in derivative trading and risk management. The two risk assessment techniques 
    include historical simulation and Monte Carlo simulation using the Geometric Brownian Motion (GBM) model. 

    Additionally, a Kupiec backtest is integrated for evaluating the accuracy of the VaR predictions, crucial for regulatory and internal risk assessment purposes.
    The forward price is calculated considering the spot price of the equity, a prevailing risk-free interest rate, 
    any anticipated dividend yield, and the time left to the contract's maturity. 

    Initial Parameters
    ------------------
    - S0: Initial spot price of the equity.
    - T: Time to maturity in years. For instance: for a 8-day contract use 8/252.
    - r: Risk-free interest rate as a decimal in annual terms (e.g., 0.05 for 5%).
    - dividend_yield: Dividend yield of the equity as a decimal (annual continuous dividend yield). Default is 0.

    Methods
    -------
    - HistoricalSimulation(): Employs historical simulation to determine Value-at-Risk (VaR) and Expected Shortfall (ES) for forward contracts using past underlying asset returns.
    - mcModelGbm(): Simulates potential equity price paths using the Geometric Brownian Motion (GBM) model and computes the Value-at-Risk (VaR) and Expected Shortfall (ES)
    for a forward contract.
    - kupiec_test(): Applies the Kupiec backtest to assess the accuracy of the VaR model by comparing actual vs. expected exceptions.

    Attributes
    ----------
    - forward_price : Calculated forward price considering the spot price of the equity, prevailing risk-free interest rate, anticipated dividend yield, and time 
    left to the contract's maturity.
    - interval : Interval calculated by multiplying the time to maturity (T) by 252 (commonly used number of trading days in a year). 
    Useful for scaling the time to maturity to daily intervals.

    Notes
    -----
    The provided risk management package helps ensure the inputs' validity through various input checking functions. 

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************
    
    def __init__(self, S0: Union[int,float],
                       T: Union[int,float],
                       r: float,
                       dividend_yield: float = 0):
        
        from pyriskmgmt.inputsControlFunctions import (check_T, check_S0, check_r, check_dividend_yield)

        # check procedure
        self.S0 = check_S0(S0) ; self.T = check_T(T) ;  self.r = check_r(r) ; self.dividend_yield = check_dividend_yield(dividend_yield)

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.SupportFunctions import ForwardPrice # (spot_price, risk_free_rate, dividend_yield, time_to_maturity)

        forward_price = ForwardPrice(spot_price = self.S0, risk_free_rate = self.r, dividend_yield = self.dividend_yield, time_to_maturity = self.T)

        self.forward_price = round(forward_price,4) ; self.interval = int(self.T * 252)


    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, underlying_rets: np.ndarray, num_forward: int = 1, alpha: float = 0.05, warning: bool = True):

        """
        Main
        ----
        Estimates the Value-at-Risk (VaR) and Expected Shortfall (ES) for forward contracts using the historical simulation method based on past underlying asset returns.

        Parameters
        ----------
        - underlying_rets: Historical daily returns of the underlying asset on a daily return frequency. 
        Ensure that the frequency of the "underlying_rets" array is "daily".
        - num_forward: Number of forward contracts. A positive value indicates a long position, while a negative indicates a short position. Default is 1.
        - alpha: Significance level for computing VaR and ES, e.g., 0.05 for 95% confidence. Default is 0.05.
        - warning: Flag to activate a warning about the daily frequency of 'underlying_rets'. Default is True.

        Returns
        -------
        A dictionary with:
        - 'var': Computed Value-at-Risk.
        - 'es': Computed Expected Shortfall.
        - 'T': Time to maturity in days.
        - 'NumSims': Number of simulations drawn from the underlying_rets. This follows the 'interval' parameter.

        Notes
        -----
        A forward contract's long position commits to buy, while a short commits to sell. 
        This method caters to both, adjusting profit and loss calculations as needed. 
        - The frequency of the underlying_rets must be daily.
        """

        import warnings ; warnings.filterwarnings("ignore")
        
        from pyriskmgmt.inputsControlFunctions import (check_num_forward, check_alpha, validate_returns_single, check_warning)
        
        import numpy as np

        # check procedure
        num_forward = check_num_forward(num_forward) ; underlying_rets = validate_returns_single(underlying_rets) ; alpha = check_alpha(alpha)
        check_warning(warning) 

        if warning:
            print(f"WARNING:")
            print("The frequency of the underlying_rets must be daily.")
            print(f"This forward contract is due to expire in {self.interval} days,")
            print("and the method automatically aggregates the daily returns.")
            print(f"To suppress this warning, set 'warning=False' when calling the HistoricalSimulation() method.")
            
        # reshaping the underlying returns
        # calculating the number of periods based on the interval
        num_periods = len(underlying_rets) // self.interval * self.interval
        # trimming returns to be a multiple of the interval
        trimmed_returns = underlying_rets[len(underlying_rets) - num_periods:]
        # reshaping the returns according to the specified interval
        reshaped_returns = trimmed_returns.reshape((-1, self.interval))
        # aggregating the returns over the interval
        underlying_rets = np.product(1 + reshaped_returns, axis=1) - 1

        # starting the stock price simulation
        # initializing an array to store simulated stock prices
        SimuStockPrices = np.zeros(underlying_rets.shape[0]) 

        # simulating stock prices using the reshaped returns
        for t in range(underlying_rets.shape[0]):
            SimuStockPrices[t] = self.S0 * (1 + underlying_rets[t])

        # fetching the forward price
        forward_price = self.forward_price

        # initializing an array to store Profit & Loss (P&L) values
        PL = np.zeros(SimuStockPrices.shape[0])

        # Note about the nature of forward contracts
        # Long position: committing to buy the underlying asset in the future
        # Short position: committing to sell the underlying asset in the future

        # if holding a long forward position
        if num_forward > 0: 
            # calculating P&L for each simulated stock price in a long position
            for i in range(SimuStockPrices.shape[0]):
                PL[i] = (SimuStockPrices[i] - forward_price) * abs(num_forward)

        # if holding a short forward position
        elif num_forward < 0: 
            # calculating P&L for each simulated stock price in a short position
            for i in range(SimuStockPrices.shape[0]):
                PL[i] = (forward_price - SimuStockPrices[i]) * abs(num_forward)  # Using abs() since num_forward is negative

        # calculating the Value-at-Risk (VaR) for the specified alpha level
        VaR = np.quantile(PL, alpha) * -1
        # calculating the Expected Shortfall (ES) for the specified alpha level
        ES = np.mean(PL[PL < - VaR]) * -1 

        # packaging results into a dictionary
        HS = {"var" : round(VaR, 4),
              "es:" : round(ES, 4),
              "T" : self.interval,
              "NumSims" : len(PL)}

        return HS
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Monte Carlo Geometric Brownian Motion <<<<<<<<<<<<<<<

    def mcModelGbm(self, underlying_rets: np.ndarray, num_forward: int = 1, alpha: float = 0.05, vol: str = "simple", mu: str = "moving_average", num_sims: int = 10000,
                    p: int = 1, q: int = 1, lambda_ewma: float = 0.94, ma_window: int = 10,  warning: bool = True):
        
        """
        Main
        ----
        Estimates the Value-at-Risk (VaR) and Expected Shortfall (ES) for forward contracts using a Monte Carlo simulation based on the Geometric Brownian Motion (GBM)
        model and past underlying asset returns.

        Parameters
        ----------
        - underlying_rets: Historical daily returns of the underlying asset on a daily return frequency. 
        Ensure that the frequency of the "underlying_rets" array is "daily".
        - num_forward: Number of forward contracts. A positive value indicates a long position, while a negative indicates a short position. Default is 1.
        - alpha: Significance level for computing VaR and ES. Default is 0.05.
        - vol: Method to calculate the daily volatility of the underlying. It can be "simple", "garch" or "ewma". Default is "simple".
        - mu: Method to calculate the daily mean of the underlying. It can be "moving_average", "zero" or "constant". Default is "moving_average".
        - num_sims: Number of Monte Carlo simulations. Default is 10000.
        - p: Parameter for the p-order ARCH term. Default is 1.
        - q: Parameter for the q-order GARCH term. Default is 1.
        - lambda_ewma: Smoothing parameter for the EWMA method. Default is 0.94.
        - ma_window: Moving average window for calculating returns. Default is 10.
        - warning: Flag to activate a warning about the daily frequency of 'underlying_rets'. Set to False to suppress. Default is True.

        Returns
        -------
        A dictionary with:
        - 'var': Computed Value-at-Risk.
        - 'es': Computed Expected Shortfall.
        - 'T': Time to maturity in days.
        - 'NumSims': Number of Monte Carlo simulations drawn.
        - 'GbmDrift': Drift component of the GBM.
        - 'GbmSigma': Volatility component of the GBM.

        Notes
        -----
        The method considers both long and short positions in the forward contract and adjusts profit and loss calculations accordingly. 
        - The frequency of the underlying_rets must be daily.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_forward, check_alpha, validate_returns_single, check_warning)
        
        import numpy as np

        # check procedure
        num_forward = check_num_forward(num_forward) ; underlying_rets = validate_returns_single(underlying_rets) ; alpha = check_alpha(alpha)
        check_warning(warning) 

        if warning:
            print(f"WARNING:")
            print("The frequency of the underlying_rets must be daily.")
            print(f"This forward contract is due to expire in {self.interval} days,")
            print("and the method automatically aggregates the daily returns.")
            print(f"To suppress this warning, set 'warning=False' when calling the mcModelGbm() method.")

        warning = False

        # importing the necessary model from pyriskmgmt library
        from pyriskmgmt.equity_models import EquityPrmSingle

        # initializing the equity risk management model
        equity_rm = EquityPrmSingle(returns = underlying_rets, position = 1, interval = self.interval, alpha = alpha)

        # simulating stock price paths using the Geometric Brownian Motion (GBM) model
        GBM_PATH = equity_rm.mcModelGbm(S0 = self.S0, vol = vol, mu = mu, num_sims = num_sims, p = p, q = q, 
                                        lambda_ewma = lambda_ewma, ma_window = ma_window, return_gbm_path = True)

        # extracting drift and sigma (volatility) from the GBM simulation results
        drift = GBM_PATH[0]["GbmDrift"] ; sigma = GBM_PATH[0]["GbmSigma"]

        # extracting the simulated stock price paths
        GBM_PATH = GBM_PATH[1]

        # starting the simulation process using the GBM paths
        SimuStockPrices = GBM_PATH

        # fetching the forward price
        forward_price = self.forward_price

        # initializing an array to store Profit & Loss (P&L) values
        PL = np.zeros(SimuStockPrices.shape[0])

        # Note about the nature of forward contracts
        # Long position: committing to buy the underlying asset in the future
        # Short position: committing to sell the underlying asset in the future

        # if holding a long forward position
        if num_forward > 0:
            # calculating P&L for each simulated stock price in a long position
            for i in range(SimuStockPrices.shape[0]):
                PL[i] = (SimuStockPrices[i] - forward_price) * num_forward

        # if holding a short forward position
        elif num_forward < 0: 
            # calculating P&L for each simulated stock price in a short position
            for i in range(SimuStockPrices.shape[0]):
                PL[i] = (forward_price - SimuStockPrices[i]) * abs(num_forward)  # Using abs() since num_forward is negative
        
        losses = PL[PL < 0]

        # calculating the Value-at-Risk (VaR) for the specified alpha level
        VaR = np.quantile(losses, alpha) * -1
        # calculating the Expected Shortfall (ES) for the specified alpha level
        ES = np.mean(losses[losses < - VaR]) * -1

        # packaging results into a dictionary
        MC_GBM = {"var" : round(VaR, 4),
                  "es:" : round(ES, 4),
                  "T" : self.interval,
                  "NumSims" : len(PL),
                  "GbmDrift" : round(drift,4),
                  "GbmSigma" : round(sigma,4)}

        return MC_GBM
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_forward: int, S0_initial: float, test_returns: np.ndarray, var: float, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):

        """
        Main
        ----
        Tests the accuracy of a Value-at-Risk (VaR) model using the Kupiec test. The test compares the actual number of VaR breaches to the expected number, 
        given a certain confidence level.

        Parameters
        ----------
        - num_forward: Number of forward contracts. A positive value indicates a long position, while a negative indicates a short position. Default is 1.
        - S0_initial: Initial spot price of the equity at the beginning of the "test_returns" array.
        - test_returns: Historical daily returns used for the test. Should have daily frequency.
        Ensure that the frequency of the "test_returns" is "daily".
        - var: Predicted Value-at-Risk to be tested.
        - alpha: Significance level for computing VaR, e.g., 0.05 for 95% confidence. Default is 0.05.
        - alpha_test: Significance level for the Kupiec test statistic. Default is 0.05.
        - warning: Flag to activate a warning about the daily frequency of 'test_returns'. Set to False to suppress. Default is True.

        Returns
        -------
        A dictionary with:
        - 'test': Specifies the test name ("Kupiec").
        - 'var': Provided Value-at-Risk.
        - 'T': Time to maturity in days.
        - 'nObs': Number of observations.
        - 'nExc': Number of exceptions or breaches of the VaR.
        - 'alpha': Theoretical percentage of exceptions.
        - 'actualExc%': Actual percentage of exceptions in the dataset.
        - 'LR': Likelihood ratio statistic.
        - 'cVal': Critical value for the likelihood ratio at the specified significance level.
        - 'rejNull': Boolean indicating if the null hypothesis should be rejected.

        Notes
        -----
        The Kupiec test assesses the quality of VaR estimates by checking if the number of VaR breaches matches the theoretical expectation based on the given confidence level. 
        A failure of this test can indicate that the VaR model is not capturing the tail risks appropriately. 
        It's essential to ensure the validity of the provided returns. The likelihood ratio statistic is compared against a chi-squared distribution with one degree of freedom.
        - The frequency of the test_returns must be daily.
        - This method was written to aggregate returns by itself if the interval is > 1.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_forward, check_S0_initial, validate_returns_single,check_var,
                                                       check_alpha, check_alpha_test, check_warning)

        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_forward = check_num_forward(num_forward) ; S0_initial = check_S0_initial(S0_initial);  test_returns = validate_returns_single(test_returns) 
        var = check_var(var) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if warning:
            print("WARNING:")
            print("The frequency of the 'test_returns' array must be daily.")
            print("This method was written to aggregate returns by itself if the interval is > 1.")
            print("To mute this warning, add 'warning = False' as an argument of the kupiec_test() function.")

        # reshaping and aggregating returns; meant for individual returns (not matrix/vector of returns)
        num_periods = len(test_returns) // self.interval * self.interval  # determining the number of complete intervals
        trimmed_returns = test_returns[len(test_returns) - num_periods:]  # trimming returns to fit within the complete intervals
        reshaped_returns = trimmed_returns.reshape((-1, self.interval))  # reshaping returns into rows of 'self.interval' columns
        test_returns = np.product(1 + reshaped_returns, axis=1) - 1  # aggregating returns within each interval (row)

        # now we can proceed 

        # initializing Value-at-Risk
        VaR = var 

        # starting the simulation by setting initial stock prices
        SimuStockPrices = np.zeros(test_returns.shape[0]) 

        # simulating stock prices based on test returns
        for t in range(test_returns.shape[0]):
            SimuStockPrices[t] = self.S0 * (1 + test_returns[t])  # calculating stock price for each return

        # fetching the forward price
        forward_price = self.forward_price

        # initializing an array to store Profit & Loss (P&L) values
        PL = np.zeros(SimuStockPrices.shape[0])

        # Note about the nature of forward contracts
        # Long position: committing to buy the underlying asset in the future
        # Short position: committing to sell the underlying asset in the future

        # if holding a long forward position
        if num_forward > 0:
            # calculating P&L for each simulated stock price in a long position
            for i in range(SimuStockPrices.shape[0]):
                PL[i] = (SimuStockPrices[i] - forward_price) * abs(num_forward)

        # if holding a short forward position
        elif num_forward < 0: 
            # calculating P&L for each simulated stock price in a short position
            for i in range(SimuStockPrices.shape[0]):
                PL[i] = (forward_price - SimuStockPrices[i]) * abs(num_forward)  # Using abs() as the number of forwards is negative in this case

        # total number of observations and exceptions
        num_obs = len(PL)
        num_exceptions = np.sum(PL < - VaR)

        # % exceptions
        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "T" : self.interval,                    # Interval
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
##########################################################################################################################################################################
##########################################################################################################################################################################
## 8. EquityFutureSingle ################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EquityFutureSingle:

    """
    Main
    ----
    This class provides a comprehensive toolkit for traders and financial analysts to compute and assess the Value-at-Risk (VaR) and Expected Shortfall (ES) 
    specifically focused on the maturity of future contracts, a pivotal aspect in derivative trading and risk management.

    Incorporating two primary risk assessment techniques, this package allows for a dual approach:
    1. Historical Simulation: Harnessing the power of past data to provide insights into potential future risks.
    2. Monte Carlo Simulation using the Geometric Brownian Motion (GBM) Model: This method adopts stochastic processes to simulate potential 
    future stock price movements and provide a probabilistic estimate of future risks.

    For enhanced reliability in the risk management process, a Kupiec backtest has been integrated. This backtest is vital for not just 
    internal evaluations but is also aligned with regulatory requirements, ensuring that the VaR predictions stand up to rigorous testing.

    To ensure practical applicability, the future price calculation has been designed to factor in various real-world parameters like the spot price of the equity, 
    a prevailing risk-free interest rate, anticipated dividend yields, and the precise time left to a contract's maturity.

    Initial Parameters
    ------------------
    - S0: Initial spot price of the equity.
    - T: Time to maturity in years. For instance: for a 8-day contract use 8/252.
    - r: Risk-free interest rate as a decimal in annual terms (e.g., 0.05 for 5%).
    - dividend_yield: Dividend yield of the equity as a decimal (annual continuous dividend yield). Default is 0.
    - convenience_yield:: This is more common in the context of commodities. It's the non-monetary advantage or premium associated with holding an asset in its 
    physical form as opposed to a derivative form (annual continuous convenience yield). Default is 0.
    - storage_cost: This refers to the costs associated with storing a physical commodity. This is relevant for futures pricing and is part of 
    the cost of carry in financial models (annual continuous storage cost). Default is 0.

    Decision to Omit Margin Calls
    ------------------------------
    While the concept of margin calls is integral in many trading scenarios, this package has intentionally omitted it. The reasons are multifold:
    - Simplification: This package is designed as a foundational tool. Including margin calls might increase the complexity for users unfamiliar with the concept.
    - Customization: By providing a basic framework, users have the flexibility to enhance and build upon the package, integrating more sophisticated methods like 
    margin calls if they deem necessary.
    - Focus: The primary aim is to give a clear perspective on market risks, specifically from adverse price movements. Integrating margin calls might divert from this core focus.

    Methods
    -------
    - HistoricalSimulation(): Employs historical simulation to determine Value-at-Risk (VaR) and Expected Shortfall (ES) for forward contracts using past underlying asset returns.
    This method utilizes the mark-to-market approach and transports the P&L to maturity using the risk-free rate.
    - mcModelGbm(): Simulates potential equity price paths using the Geometric Brownian Motion (GBM) model and computes the Value-at-Risk (VaR) and Expected Shortfall (ES) 
    for forward contracts. Similar to the HistoricalSimulation method, this too incorporates the mark-to-market approach and adjusts the P&L to maturity using the risk-free rate.
    - kupiec_test(): Applies the Kupiec backtest to assess the accuracy of the VaR model by comparing actual vs. expected exceptions.

    Attributes
    ----------
    - future_price: Calculated future price considering the spot price of the equity, prevailing risk-free interest rate, anticipated dividend yield, convenience yield, storage costs, 
    and time left to the contract's maturity.
    - interval: Interval calculated by multiplying the time to maturity (T) by 252 (commonly used number of trading days in a year). Useful for scaling the time 
    to maturity to daily intervals.

    Notes
    -----
    The provided risk management package helps ensure the inputs' validity through various input checking functions. 

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************
    
    def __init__(self, S0: Union[int,float],
                       T: Union[int,float],
                       r: float,
                       dividend_yield: float = 0,
                       convenience_yield: float = 0,
                       storage_cost: float = 0):
        
        
        from pyriskmgmt.inputsControlFunctions import (check_T, check_S0, check_r, check_dividend_yield, check_convenience_yield, check_storage_cost)

        # check procedure
        self.S0 = check_S0(S0) ; self.T = check_T(T) ;  self.r = check_r(r)  ; self.dividend_yield = check_dividend_yield(dividend_yield)  
        self.convenience_yield = check_convenience_yield(convenience_yield) ; self.storage_cost = check_storage_cost(storage_cost)
        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.SupportFunctions import FuturesPrice # (spot_price, risk_free_rate, dividend_yield, convenience_yield, storage_cost, time_to_maturity)

        future_price = FuturesPrice(spot_price = self.S0, risk_free_rate = self.r, dividend_yield = self.dividend_yield,
                                      convenience_yield = self.convenience_yield, storage_cost = self.storage_cost, time_to_maturity = self.T)

        self.future_price = round(future_price,4) ; self.interval = int(self.T * 252)


    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, underlying_rets: np.ndarray, num_future: int = 1, alpha: float = 0.05, warning: bool = True):

        """
        Main
        ----
        Employs historical simulation to determine Value-at-Risk (VaR) and Expected Shortfall (ES) for future contracts using past underlying asset returns. 
        This method utilizes the mark-to-market approach, computing the daily mark-to-market losses and adjusting them to maturity using the risk-free rate.

        Parameters
        ----------
        - underlying_rets: Historical daily returns of the underlying asset on a daily return frequency. 
        Ensure that the frequency of the "underlying_rets" array is "daily".
        - num_future: Number of future contracts. A positive value indicates a long position, while a negative indicates a short position. Default is 1.
        - alpha: Significance level for computing VaR and ES, e.g., 0.05 for 95% confidence. Default is 0.05.
        - warning: Flag to activate a warning about the daily frequency of 'underlying_rets'. Default is True.

        Returns:
        --------
        - dict: A dictionary containing the Value at Risk (VaR), Expected Shortfall (ES), time to maturity (T), and the number of simulations performed (NumSims).

        Warnings:
        ---------
        If 'warning' is set to True, the method will print a warning to ensure the user understands the daily frequency requirement for the underlying returns.

        Notes:
        ------
        The method internally adjusts the P&L to maturity using the risk-free rate and provides a comprehensive historical analysis based on past asset returns.
        - The frequency of the underlying_rets must be daily.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_future, check_alpha, validate_returns_single, check_warning)
        
        import numpy as np 

        # check procedure
        num_future = check_num_future(num_future) ; underlying_rets = validate_returns_single(underlying_rets) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if warning:
            print("WARNING:")
            print("The frequency of the underlying_rets must be daily.")
            print(f"This future contract is due to expire in {self.interval} days,")
            print("and the method automatically calculates the daily mark-to-market losses.")
            print("To suppress this warning, set 'warning=False' when calling the HistoricalSimulation() method.")

        # extracting non-overlapping price paths for simulation
        n_periods = len(underlying_rets)
        n_extractions = n_periods // self.interval

        simPrices = np.zeros((self.interval, n_extractions))

        # populating the simulation matrix with price paths
        for i in range(0, n_periods, self.interval):
            end = i + self.interval
            if end > n_periods:
                break
            simPrices[:, i // self.interval] = self.S0 * np.cumprod(1 + underlying_rets[i: end])

        # setting the future price for calculations
        future_price = self.future_price 

        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Calculate discount factors for each time step
        discount_factors = np.exp(self.r * (np.arange(self.interval, 0, -1) / 252))

        # P&L calculation
        if num_future > 0:
            PL = (simPrices - future_price) * num_future
        else:
            PL = (future_price - simPrices) * abs(num_future)

        PL *= discount_factors[:, np.newaxis]

        # Flatten the PL matrix
        PL = PL.ravel(order='F') # flattening PL column-wise to maintain consecutive days in blocks
            
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PL = np.array(PL)

        losses = PL[PL<0]
        # calculating risk metrics: Value at Risk (VaR) and Expected Shortfall (ES)
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < -VaR]) * -1

        # storing the results in a dictionary
        HS = {"var": round(VaR,4),
              "es" : round(ES,4),
              "T" : self.interval,
              'NumSims' : simPrices.shape[1]}

        return HS

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Monte Carlo Geometric Brownian Motion <<<<<<<<<<<<<<<

    def mcModelGbm(self, underlying_rets: np.ndarray, num_future: int = 1, alpha: float = 0.05, vol: str = "simple",
                    mu: str = "moving_average", num_sims: int = 10000, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, ma_window: int = 10,
                    warning: bool = True):
        
        """
        Main
        ----
        Uses the Geometric Brownian Motion (GBM) model to simulate potential equity price paths and compute Value-at-Risk (VaR) and Expected Shortfall (ES) for future contracts. 
        This method employs the mark-to-market approach, computing the daily mark-to-market losses and adjusting them to maturity using the risk-free rate.

        Parameters
        ----------
        - underlying_rets: Historical daily returns of the underlying asset on a daily return frequency. 
        Ensure that the frequency of the "underlying_rets" array is "daily".
        - num_future: Number of future contracts. A positive value indicates a long position, while a negative indicates a short position. Default is 1.
        - alpha: Significance level for computing VaR and ES. Default is 0.05.
        - vol: Method to calculate the daily volatility of the underlying. It can be "simple", "garch" or "ewma". Default is "simple".
        - mu: Method to calculate the daily mean of the underlying. It can be "moving_average", "zero" or "constant". Default is "moving_average".
        - num_sims: Number of Monte Carlo simulations. Default is 10000.
        - p: Parameter for the p-order ARCH term. Default is 1.
        - q: Parameter for the q-order GARCH term. Default is 1.
        - lambda_ewma: Smoothing parameter for the EWMA method. Default is 0.94.
        - ma_window: Moving average window for calculating returns. Default is 10.
        - warning: Flag to activate a warning about the daily frequency of 'underlying_rets'. Set to False to suppress. Default is True.

        Returns:
        --------
        - dict: A dictionary containing Value at Risk (VaR), Expected Shortfall (ES), expiration date (ExpiDate), number of simulations (NumSims), 
        drift (GbmDrift), and volatility (GbmSigma).

        Warnings:
        ---------
        If 'warning' is set to True, the method will print a warning to ensure the user understands the daily frequency requirement for the underlying returns,
        and will highlight some possible interpretations of the calculated VaR.

        Notes:
        ------
        This method combines historical information with Monte Carlo simulation, providing a comprehensive risk assessment for future contracts based 
        on the Geometric Brownian Motion model. 
        It incorporates various methods to estimate drift and volatility, making it versatile for different scenarios.
        - The frequency of the underlying_rets must be daily.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_future, check_alpha, validate_returns_single, check_warning)
        
        import numpy as np

        # check procedure
        num_future = check_num_future(num_future) ; underlying_rets = validate_returns_single(underlying_rets) ; alpha = check_alpha(alpha)
        check_warning(warning) 

        if warning:
            print("WARNING:")
            print("The frequency of the underlying_rets must be daily.")
            print(f"This future contract is due to expire in {self.interval} days,")
            print("and the method automatically calculates the daily mark-to-market losses based on the GBM path.")
            print("To suppress this warning, set 'warning=False' when calling the mcModelGbm() method.")

        warning = False

        # using the EquityPrmSingle method to retrieve the drift and the sigma (daily) for the underlying equity

        from pyriskmgmt.equity_models import EquityPrmSingle

        equity_rm = EquityPrmSingle(returns = underlying_rets, position = 1, interval = 1 , alpha = alpha)

        GBM_PATH = equity_rm.mcModelGbm(S0 = self.S0, vol = vol, mu = mu, num_sims = 1, p = p, q = q, lambda_ewma = lambda_ewma, ma_window = ma_window,
                                                  return_gbm_path = False)
        
        drift = GBM_PATH["GbmDrift"] ; sigma = GBM_PATH["GbmSigma"]
        
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # GBM PROCESS

        # Initializing the array to store simulated stock prices
        SimuStockPrices = np.empty((self.interval, num_sims))

        # parameters and initial settings
        dt = 1 

        # Simulate the first step for all paths
        SimuStockPrices[0] = self.S0 * (1 + drift * dt + sigma * np.random.standard_normal(num_sims) * np.sqrt(dt))

        # Simulate subsequent steps
        for j in range(1, SimuStockPrices.shape[0]):
            SimuStockPrices[j] = SimuStockPrices[j-1] * (1 + drift * dt + sigma * np.random.standard_normal(num_sims) * np.sqrt(dt))

        # setting the future price for calculations
        future_price = self.future_price 

        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Calculate discount factors for each time step
        discount_factors = np.exp(self.r * (np.arange(self.interval, 0, -1) / 252))

        # P&L calculation
        if num_future > 0:
            PL = (SimuStockPrices - future_price) * num_future
        else:
            PL = (future_price - SimuStockPrices) * abs(num_future)

        PL *= discount_factors[:, np.newaxis]

        # Flatten the PL matrix
        PL = PL.ravel(order='F') # flattening PL column-wise to maintain consecutive days in blocks
            
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PL = np.array(PL)
        losses = PL[PL<0]
        # calculating risk metrics: Value at Risk (VaR) and Expected Shortfall (ES)
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < -VaR]) * -1

        # Storing the results
        MC_GBM = {"var" : round(VaR,4),
                  "es" : round(ES,4),
                  "ExpiDate" : self.interval,
                  "NumSims" : num_sims,
                  "GbmDrift" : drift,
                  "GbmSigma" : sigma}
        
        return MC_GBM
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_future: int, S0_initial: float, test_returns: np.ndarray, var: float, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):

        """
        Main
        ----
        Performs Kupiec's Proportional-of-Failures (POF) test on a given Value-at-Risk (VaR) model. The test examines if the observed frequency of exceptions 
        (losses exceeding VaR) matches the expected frequency under the VaR model.

        Parameters
        ----------
        - num_future: Number of future contracts. A positive value indicates a long position, while a negative indicates a short position. Default is 1.
        - S0_initial: Initial spot price of the equity at the beginning of the "test_returns" array.
        - test_returns: Historical daily returns used for the test. Should have daily frequency.
        Ensure that the frequency of the "test_returns" is "daily".
        - var: Predicted Value-at-Risk to be tested.
        - alpha: Significance level for computing VaR, e.g., 0.05 for 95% confidence. Default is 0.05.
        - alpha_test: Significance level for the Kupiec test statistic. Default is 0.05.
        - warning: Flag to activate a warning about the daily frequency of 'test_returns'. Set to False to suppress. Default is True.

        Returns:
        --------
        - dict: A dictionary containing the test name (test), Value at Risk (var), time interval (T), number of observations (nObs),
        number of exceptions (nExc), theoretical percentage of exceptions (alpha), actual percentage of exceptions (actualExc%), likelihood ratio (LR), critical value 
        (cVal), and a flag to indicate if the null hypothesis is rejected (rejNull).

        Warnings:
        ---------
        If 'warning' is set to True, the method will print a warning to ensure the user understands the daily frequency requirement for the test returns.

        Notes:
        ------
        The Kupiec test statistically verifies if the observed number of exceptions is consistent with what's expected from the chosen VaR model.
        A significant result indicates the VaR model may not be adequate.
        - The frequency of the test_returns must be daily.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_future, check_S0_initial, validate_returns_single,check_var,
                                                       check_alpha, check_alpha_test, check_warning)

        from scipy.stats import chi2 ; import numpy as np

        # check procedure
        num_future = check_num_future(num_future) ; S0_initial = check_S0_initial(S0_initial);  test_returns = validate_returns_single(test_returns) 
        var = check_var(var) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if warning:
            print("WARNING:")
            print("The frequency of the 'test_returns' must be daily.")
            print("To suppress this warning, set 'warning=False' when calling the kupiec_test() method.")

        # extracting non-overlapping price paths for simulation
        n_periods = len(test_returns)
        n_extractions = n_periods // self.interval

        simPrices = np.zeros((self.interval, n_extractions))

        # populating the simulation matrix with price paths
        for i in range(0, n_periods, self.interval):
            end = i + self.interval
            if end > n_periods:
                break
            simPrices[:, i // self.interval] = self.S0 * np.cumprod(1 + test_returns[i: end])

        # setting the future price for calculations
        future_price = self.future_price 

        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Calculate discount factors for each time step
        discount_factors = np.exp(self.r * (np.arange(self.interval, 0, -1) / 252))

        # P&L calculation
        if num_future > 0:
            PL = (simPrices - future_price) * num_future
        else:
            PL = (future_price - simPrices) * abs(num_future)

        PL *= discount_factors[:, np.newaxis]

        # Flatten the PL matrix
        PL = PL.ravel(order='F') # flattening PL column-wise to maintain consecutive days in blocks
            
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PL = np.array(PL) ; VaR = var
            
        # total number of observations and exceptions
        num_obs = len(PL) ; num_exceptions = np.sum(PL < - VaR)

        # % exceptions
        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "T" : self.interval,                    # Interval
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
##########################################################################################################################################################################
##########################################################################################################################################################################
## 9. EquityForwardPort ##################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EquityForwardPort:

    """
    Main
    ----
    This class offers tools to assess and compute the VaR (Value-at-Risk) and ES (Expected Shortfall) at the maturity of a portfolio of forward contracts, 
    which is fundamental in derivative trading and risk management. The risk assessment techniques for the portfolio 
    can include methods like historical simulation and Monte Carlo simulation using the Cholesky decomposition of the VCV.
    The class calculates the VaR and ES that will occur at the maturity of the last forward contracts in the portfolio. 
    It also moves money through time using the risk-free rate, allowing for a comprehensive understanding of potential future risks.

    Additionally, a Kupiec backtest can be integrated for evaluating the accuracy of the VaR predictions for the portfolio, 
    which is pivotal for regulatory and internal risk assessment purposes.

    The forward prices for the portfolio are calculated considering the spot prices of the equities, the prevailing risk-free interest rates, 
    any anticipated dividend yields, and the times left to the contracts' maturities.

    Initial Parameters
    ------------------
    - S0_s: Array of initial spot prices of the equities in the portfolio.
    - T_s: Array of times to maturity in years for the contracts in the portfolio.
      - For instance: expDays = np.array([5,8,12,15,18]); T_s = expDays/252
    - r_s: Array of risk-free interest rates as decimals for the contracts in the portfolio in annual terms (e.g., 0.05 for 5%).
    - dividend_yield_s: Array of dividend yields for each equities in the portfolio as decimals. (annual continuous dividend yields). Default is None.

    Methods
    -------
    - HistoricalSimulation(): Uses historical data to simulate the Value-at-Risk (VaR) and Expected Shortfall (ES) for the portfolio of forward contracts. 
    The method takes into account the expiration of the last forward contract, computes the P&L for each contract, and adjusts 
    them using the risk-free rate.
    - ff3(): Fetches data for the Fama-French three-factor (FF3) model, which includes Market return (MKT), Size (SMB: Small Minus Big), 
    and Value (HML: High Minus Low) from Ken French's Data Library.
    - mcModelRetsSimu(): This method simulates the future value at risk (VaR) and expected shortfall (ES) for forward contracts using Monte Carlo methods.
    - kupiec_test(): Applies the Kupiec backtest to assess the accuracy of the VaR model by comparing actual vs. expected exceptions.

    Attributes
    ----------
    - forward_prices: Array of calculated forward prices for each contract in the portfolio, 
    considering the spot prices of the equities, prevailing risk-free interest rates, anticipated dividend yields, and time left to the contracts' maturities.
    - summary: A DataFrame containing summarized information for each forward contract in the portfolio.
    This includes columns like 'ForwardPrice', 'S0', 'T', 'interval', 'r', and possibly 'dividend_yield', providing a comprehensive overview of the portfolio.

    Notes
    -----
    The provided risk management package helps ensure the validity of the inputs for the portfolio through various input checking functions.
    This class is suitable for managing portfolios of forward contracts, allowing for the simultaneous analysis of multiple contracts.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, S0_s: np.ndarray,
                 T_s: np.ndarray, 
                 r_s:  np.ndarray,
                 dividend_yield_s: np.ndarray = None):
        
        import numpy as np

        from pyriskmgmt.inputsControlFunctions import (check_S0_s, check_T_s, check_r_s, check_dividend_yield_s)

        # check procedure
        self.S0_s = check_S0_s(S0_s) ; self.T_s = check_T_s(T_s) ;  self.r_s = check_r_s(r_s) 

        if dividend_yield_s is None:
            self.dividend_yield_s = np.repeat(0,len(T_s))
        if dividend_yield_s is not None:
            self.dividend_yield_s = check_dividend_yield_s(dividend_yield_s)

        # ------------------------------------------------------------------------------------------------

        # checking the same length for all the imputs *args

        def check_same_length(*args):
            """
            Checks if all given numpy arrays have the same length.
            """
            it = iter(args)
            the_len = len(next(it))

            if not all(len(l) == the_len for l in it):
                raise ValueError("Not all the input arrays in the EquityForwardPort class have the same length!")
                       
        # Here we are ensuring that all arrays have the same length
        check_same_length(T_s, S0_s, r_s, self.dividend_yield_s)

        # ------------------------------------------------------------------------------------------------

        self.forward_prices = None 

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd 

        from pyriskmgmt.SupportFunctions import ForwardPrice # (spot_price, risk_free_rate, dividend_yield, time_to_maturity)

        summary = pd.DataFrame()

        # calculating forward prices and add to summary DataFrame
        for S0, T, dividend_yield, r in zip(self.S0_s, self.T_s, self.dividend_yield_s, self.r_s):
            price = ForwardPrice(S0, T, dividend_yield, r) ; interval = int(T * 252)

            if all(value == 0 for value in self.dividend_yield_s):
                summary = summary._append({'ForwardPrice' : price ,
                                           'S0' : S0,
                                            'T' : T,
                                            'interval': interval,
                                            'r' : r},
                                        ignore_index = True)
            else:
                summary = summary._append({'ForwardPrice': price,
                                           'S0': S0,
                                            'T' : T,
                                            'interval' : interval,
                                            'r' : r,
                                            "dividend_yield" : dividend_yield},
                                        ignore_index = True)

        # rounding all the values to 4 decimal places
        summary = summary.round(4) ; summary['interval'] = summary['interval'].astype(int)

        self.summary = summary ; self.forward_prices = summary["ForwardPrice"].values

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************    

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, num_forward_s: np.ndarray, underlying_rets: np.ndarray, alpha: float = 0.05, warning: bool = True):

        """
        Main
        ----
        This method assumes that the user provides synchronized inputs, i.e., the order of `num_forward_s`, `underlying_rets`, and the internal attributes 
        of the class should be consistent. If not, the results might not be accurate.
        The computations heavily depend on reshaping and realigning the input data structures to ensure synchronized calculations. 
        The method uses extensive matrix operations for efficiency.
        The method computes the VaR and ES at the expiration of the last forward contract. It also accounts for each contract's maturity and adjusts 
        potential gains or losses using the corresponding risk-free rate.

        Methodology
        -----------
        1. Data Processing:
        The method sorts and reorders the underlying return data based on the interval of each forward contract.
        It calculates simulated final prices for the underlying assets, considering different intervals.
        The function efficiently stacks or expands these simulated prices into a 4D data structure for synchronized calculations.
        2. Profit and Loss Calculation:
        It computes Profit and Loss (P&L) scenarios for each forward contract, adjusting them according to their maturity and risk-free rate. 
        The P&L is based on differences between simulated final prices and forward prices, accounting for long or short positions.
        4. Risk Metrics Calculation:
        The method calculates the VaR and ES using the aggregated P&L scenarios. VaR is computed as a quantile of the P&L distribution at the given significance level alpha. 
        ES is calculated as the average of the losses that exceed the calculated VaR.
        5. Output:
        The function returns a dictionary containing:
        Computed VaR and ES values.
        Maximum and minimum expiration intervals of the forward contracts.
        Total number of P&L scenarios that were simulated.


        Parameters
        -----------
        - num_forward_s: Array indicating the number of each forward contract in the portfolio. Positive values imply a long position, 
        while negative values imply a short position.
        - underlying_rets: A 2D array of historical returns for the underlying assets of the forward contracts. The frequency of these returns 
        should be daily and the same for every column in the "underlying_rets" array.
        - alpha: Significance level used to compute VaR and ES. Typically represents the probability of observing a loss greater 
        than VaR or the average of losses exceeding VaR. Default is 0.05.
        - warning: If set to True, it will print a warning regarding the consistency of the input data. Default is True.

        Notes
        -----
        - The frequency of the underlying_rets must be daily.
        - This method efficiently stacks or expands these simulated prices into a 4D data structure for synchronized calculations,
        but everything starts from daily returns.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_forward_s, check_alpha, validate_returns_port, check_warning)
        
        import numpy as np

        # check procedure
        num_forward_s = check_num_forward_s(num_forward_s) ; underlying_rets = validate_returns_port(underlying_rets) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if warning:
            print("WARNING:")
            print("The frequency of the underlying_rets must be daily.")
            print("Ensure also that every column in the 'underlying_rets' has the same daily frequency.")
            print("This method efficiently stacks or expands these simulated prices into a 4D data structure for synchronized calculations,")
            print("but everything starts from daily returns.")
            print("To suppress this warning, set 'warning=False' when calling the HistoricalSimulation() method.")
            
        # --------------------------------------------------------------------------------------------------------------------------------------------------

        # getting the order in which self.interval needs to be sorted
        sort_order = np.argsort(self.summary["interval"])

        # reordering the columns of underlying_rets based on the sorting order
        intervals_sorted = self.summary["interval"][sort_order].values ; underlying_rets_sorted = underlying_rets[:, sort_order] 
        num_forward_s_sorted = num_forward_s[sort_order] ; S0_s_sorted = self.S0_s[sort_order] 
        r_s_sorted = self.r_s[sort_order] ; forward_prices_sorted = self.forward_prices[sort_order] ; T_s_sorted = self.T_s[sort_order]
    
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # defining a function to calculate final prices after a given interval
        def generate_final_prices(returns, interval, S0):

            # trimming the returns to fit the interval size
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[:num_periods]

            # initializing the final_prices list
            final_prices = []

            # iterating over each possible start index in the trimmed returns
            for start_idx in range(0, num_periods - interval + 1):
                chunk = trimmed_returns[start_idx : start_idx + interval] ; final_price = S0 * np.prod(1 + chunk) ; final_prices.append(final_price)
            return np.array(final_prices).reshape(-1, 1)

        # initializing an empty list to store SimPrice2D results
        SimPrice2D = []

        # iterating over sorted intervals to compute SimPrice2D
        for index, interval in enumerate(intervals_sorted):
            final_prices = generate_final_prices(underlying_rets_sorted[:, index], interval, S0_s_sorted[index])
            SimPrice2D.append(final_prices)

        # defining a function to expand or stack arrays into a 4D structure
        def stack_or_expand(arrays):

            i = 0
            SimPrice4D = []

            # iterating through the provided arrays
            while i < len(arrays):
                current_array = arrays[i][:, :, np.newaxis]  # adding a third dimension
                # checking for arrays with matching shapes
                matching_arrays = [current_array]
                j = i + 1
                while j < len(arrays) and arrays[j].shape[0] == current_array.shape[0]:
                    matching_arrays.append(arrays[j][:, :, np.newaxis])  # adding a third dimension
                    j += 1
                # stacking matching arrays or using them as is
                if len(matching_arrays) > 1:
                    stacked_array = np.concatenate(matching_arrays, axis=2)
                    i = j
                else:
                    stacked_array = current_array
                    i += 1
                SimPrice4D.append(stacked_array)
            return SimPrice4D

        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # calling the stack_or_expand function
        SimPrice4D = stack_or_expand(SimPrice2D)

        # initializing Profit and Loss (P&L) list and getting the maximum T value
        PL = [] ; MaxT = max(T_s_sorted)

        forwardCount = 0

        # iterating through the SimPrice4D results
        for j in range(len(SimPrice4D)):

            SimPrice3D = SimPrice4D[j]

            # extracting relevant portions for the current 3D dataset
            num_forward_s_sorted_new = num_forward_s_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])]
            forward_prices_sorted_new = forward_prices_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])]
            r_s_sorted_new = r_s_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])] ; T_s_sorted_new = T_s_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])]

            Plmatrix3D = np.zeros_like(SimPrice3D)

            # iterating over the depth of the 3D dataset
            for index_inner, z in enumerate(range(SimPrice3D.shape[2])):

                dataset2D = SimPrice3D[:,:,z]
                PlMatrix2D = np.zeros_like(dataset2D)

                # iterating over columns of the 2D dataset
                for col in range(dataset2D.shape[1]):

                    # iterating over rows of the current column
                    for row in range(dataset2D.shape[0]):

                        # calculating P&L for positive num_forward_s_sorted_new values
                        if num_forward_s_sorted_new[index_inner] > 0:
                            PlMatrix2D[row,col] = (dataset2D[row,col] - forward_prices_sorted_new[index_inner]) * num_forward_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT -T_s_sorted_new[index_inner]))

                        # calculating P&L for negative num_forward_s_sorted_new values
                        elif num_forward_s_sorted_new[index_inner] < 0:
                            PlMatrix2D[row,col] = (forward_prices_sorted_new[index_inner] - dataset2D[row,col]) * num_forward_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT -T_s_sorted_new[index_inner]))

                    Plmatrix3D[:,:, index_inner] = PlMatrix2D

        # *******************************************************************************************************************************************

            pl = np.sum(Plmatrix3D, axis=2).ravel() 
            PL.extend(pl) 
            forwardCount += SimPrice3D.shape[2]

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PL = np.array(PL) ; VaR = np.quantile(PL, alpha) * -1 ; ES = np.mean(PL[PL < - VaR]) * - 1

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------

        # MaxExpiration, MinExpiration and MaxAggregation: finding the maximum and minimum expiration intervals

        MaxExpiration = max(self.summary["interval"]) ;  MinExpiration = min(self.summary["interval"])

        # returning the results in a dictionary format
        HS = {"var" : round(VaR, 4),
              "es" : round(ES, 4),
              "MaxExpi" : MaxExpiration,
              "MinExpi" : MinExpiration,
              "TotNumSimPL" : len(PL)}

        return HS
    
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
        
    # MONTECARLO MODEL [RETURNS - DISTIBUTION] <<<<<<<<<<<< 

    def mcModelRetsSimu(self, num_forward_s: np.ndarray, underlying_rets: np.ndarray, alpha: float = 0.05, sharpeDiag: bool = False , mappingRets: np.ndarray = None,
                         vol: str = "simple", num_sims: int = 250, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, warning: bool = True):
        
        """
        Main
        ----
        The `mcModelRetsSimu` function simulates Monte Carlo model returns based on specified configurations, accounting for 
        various risk metrics.
        The function assumes that all inputs, including `num_forward_s`, `underlying_rets`, and other parameters, are 
        synchronized and in order. The extensive matrix operations embedded within the function are utilized for efficiency 
        and precision. However, the onus of providing synchronized data rests on the user.

        Methodology
        -----------
        1. Data Processing:
        - It aligns and orders the underlying return data as per each forward contract's interval.
        - Simulated final prices of the underlying assets are deduced, based on various intervals.
        - The function crafts a 4D data structure by either stacking or expanding these simulated prices to facilitate synchronized computations.
        2. Profit and Loss Calculation:
        - P&L scenarios are calculated for every forward contract. Adjustments are made in line with the maturity and the risk-free rate of each contract.
        - P&L is derived from the discrepancy between the simulated final prices and the forward prices, taking into account whether the position is long or short.
        3. Risk Metrics Calculation:
        - VaR and ES are computed based on the amassed P&L scenarios. The VaR is determined as a P&L distribution quantile, considering the specified significance level alpha.
        - ES, on the other hand, represents the average of the losses surpassing the computed VaR.
        4. Output:
        - The final output is a dictionary encompassing the computed VaR and ES values, the maximum and minimum expiration intervals of the forward contracts, and the total number of simulated P&L scenarios.

        Parameters
        -----------
        - num_forward_s: Array that indicates the number of each forward contract in the portfolio. Positive values signify a long position, while negative values suggest a short position.
        - underlying_rets: A 2D array of historical returns for the underlying assets of the forward contracts. The frequency of these returns 
        should be daily and the same for every column in the "underlying_rets" array.
        - alpha: Significance level used to compute VaR and ES. Represents the probability of observing a loss greater 
        than VaR or the average of losses exceeding VaR. Default is 0.05.
        - sharpeDiag: If enabled, the function employs a Sharpe diagonal approach to map "underlying_rets" to 1 or more factors. Default is None.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on a daily frequency
        and align "mappingRets" with "underlying_rets" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Denotes the volatility model in use, with options being 'simple', 'ewma', or 'garch'. Default is "simple".
        - num_sims: Number of Monte Carlo simulations to be executed. Default is 250.
        - p: Order of the AR term for GARCH. Default is 1.
        - q: Order of the MA term for GARCH. Default is 1.
        - lambda_ewma: The smoothing factor designated for the EWMA volatility model. Default is 0.94.
        - warning: If enabled, the function will broadcast warnings pertinent to the input data's consistency. Default is True.

        Notes
        -----
        - The underlying_rets frequency must be set to daily.
        - Although the function efficiently stacks or expands the simulated prices into a 4D data structure to enable synchronized calculations, 
        the process is initiated from daily returns.
        """

        import warnings ; warnings.filterwarnings("ignore")
                        
        from pyriskmgmt.inputsControlFunctions import (check_num_forward_s, check_alpha, validate_returns_port, check_number_simulations, check_warning)
        
        import numpy as np ; import pandas as pd ; from joblib import Parallel, delayed
        
        # check procedure
        num_forward_s = check_num_forward_s(num_forward_s) ; underlying_rets = validate_returns_port(underlying_rets) ; alpha = check_alpha(alpha)
        num_sims = check_number_simulations(num_sims) ; check_warning(warning) 

        if warning:
            print("WARNING:")
            print("The frequency of the underlying_rets must be daily.")
            print("Ensure also that every column in the 'underlying_rets' has the same daily frequency.")
            print("This method efficiently stacks or expands these simulated prices into a 4D data structure for synchronized calculations,")
            print("but everything starts from daily returns.")
            print("To suppress this warning, set 'warning=False' when calling the mcModelRetsSimu() method.")

        if vol == "garch":
            
            if warning:
                from pyriskmgmt.SupportFunctions import garch_warnings
                garch_warnings("mcModelRetsSimu()")
                
            if warning and underlying_rets.shape[1] > 50 and not sharpeDiag:
                print("IMPORTANT:")
                print("The 'garch' option for the 'vol' parameter is currently in use, with an asset count exceeding 50.")
                print("This configuration can lead to slower processing and inefficient resource usage, due to the complexity ")
                print("of estimating the full variance-covariance matrix (VCV) in high-dimensional space.")
                print("It's strongly advised to employ a mapping approach instead. This approach maps the multiple assets to one or more factors,")
                print("effectively reducing the dimensionality of the problem.")
                print("This is not only more computationally efficient, but it also avoids the 'curse of dimensionality',")
                print("as it reduces the number of parameters needed to be estimated. ")
                print("The mapping approach aids in providing more stable and reliable estimations, especially in large, complex systems.")

        warning = False

        # all the other inputs will be handled here, in the EquityPrmPort class ---------------------------------------------------------------------------------

        from pyriskmgmt.equity_models import EquityPrmPort

        model = EquityPrmPort(returns = underlying_rets, positions = np.repeat(1 ,underlying_rets.shape[1]),
                              interval = 1, alpha = alpha) # here 'positions' is not pivotal... I need this class only to retrieve Monte Carlo simulated_rets
                                                           # also 'interval' must be one!
        
        simulated_rets = model.mcModelRetsSimu(sharpeDiag = sharpeDiag, mappingRets = mappingRets, vol = vol, num_sims = num_sims, p = p, q = q, 
                                                               lambda_ewma = lambda_ewma, warning = warning, return_simulated_rets = True)[1]

        # --------------------------------------------------------------------------------------------------------------------------------------------------

        # extracting datasets based on sorted intervals
        def extract_datasets(intervals_sorted, underlying_rets_sorted_aggregated):
            datasets = []  # initializing the list to hold datasets
            i = 0  # initializing the loop index
            # iterating through the sorted intervals
            while i < len(intervals_sorted):
                start = i  # setting start of possible range of duplicates
                # checking for consecutive identical intervals
                while i + 1 < len(intervals_sorted) and intervals_sorted[i] == intervals_sorted[i + 1]:
                    i += 1
                end = i  # setting end of possible range of duplicates
                # handling unique value case
                if start == end:
                    datasets.append(underlying_rets_sorted_aggregated[start])
                else:  # handling range of duplicates
                    # appending all the underlying returns for the range of duplicate intervals
                    datasets.extend(underlying_rets_sorted_aggregated[start:end+1])
                i += 1  # moving to the next index
            return datasets
        
        # reshape_and_aggregate_returns ---------------------------------------------------------------------------------------------------------------------

        def reshape_and_aggregate_returns(returns, interval):
            returns = returns.values
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[len(returns) - num_periods:]
            reshaped_returns = trimmed_returns.reshape((-1, interval)) ; aggregated_returns = np.product(1 + reshaped_returns, axis=1) - 1
            return aggregated_returns
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------
        
       # getting the sorting order of self.interval
        sort_order = np.argsort(self.summary["interval"])

        # reordering the columns of underlying_rets based on the sorting order

        intervals_sorted = self.summary["interval"][sort_order].values ; num_forward_s_sorted = num_forward_s[sort_order] ; S0_s_sorted = self.S0_s[sort_order] 
        r_s_sorted = self.r_s[sort_order] ; forward_prices_sorted = self.forward_prices[sort_order] ; T_s_sorted = self.T_s[sort_order]

        # Now everything is sorted by interval...
    
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # simulating historical VaR - ES values for different contracts - ready for the joblib process

        def single_HS_simulation(sim, simulated_rets, sort_order, S0_s_sorted, forward_prices_sorted, num_forward_s_sorted, T_s_sorted, r_s_sorted, intervals_sorted, alpha):

            # aggregating returns from sorted datasets
            underlying_rets_sorted_aggregated = [] ; underlying_rets = simulated_rets[:,:,sim] 
            underlying_rets_sorted = underlying_rets[:, sort_order] ; rets = pd.DataFrame(underlying_rets_sorted)

            # iterating through each interval to reshape and aggregate returns
            for interval in intervals_sorted:
                retsith = rets.apply(lambda x: reshape_and_aggregate_returns(x, interval)) ; underlying_rets_sorted_aggregated.append(retsith)

            # extracting datasets based on sorted returns and intervals
            datasets = extract_datasets(intervals_sorted, underlying_rets_sorted_aggregated)

            # initializing list to store Profit & Loss (P&L) values for each contract,  index and the maximum time horizon once for efficiency
            PLList = [] ; idx = 0 ; max_T = max(T_s_sorted)

            # traversing through the sorted intervals
            while idx < len(intervals_sorted):
                # initializing the length of subsequent identical intervals
                run_length = 1
                # counting identical subsequent intervals
                while idx + run_length < len(intervals_sorted) and intervals_sorted[idx] == intervals_sorted[idx + run_length]:
                    run_length += 1

                # extracting the dataset for the current interval index
                dataset = datasets[idx]

                # processing blocks of identical intervals <<<<<<<<<<<<<
                 
                if run_length > 1:
                    # converting dataset values into a numpy array
                    underlying_rets = dataset.iloc[:, idx:idx+run_length].values
                    # extracting values related to the current block of intervals
                    SO_S = np.array(S0_s_sorted[idx:idx+run_length])
                    num_forward_s = np.array(num_forward_s_sorted[idx:idx+run_length]) ; forward_prices = np.array(forward_prices_sorted[idx:idx+run_length])
                    r_s = np.array(r_s_sorted[idx:idx+run_length]) ; T_s = np.array(T_s_sorted[idx:idx+run_length])

                    # computing Profit & Loss (P&L) values for the current block of intervals
                    PLMATRIX = np.where(num_forward_s > 0, 
                                        (SO_S * (1 + underlying_rets) - forward_prices) * num_forward_s * np.exp(r_s * (max_T - T_s)), 
                                        (forward_prices - SO_S * (1 + underlying_rets)) * num_forward_s * np.exp(r_s * (max_T - T_s)))
                    
                    # aggregating the computed P&L values
                    PLList.append(np.sum(PLMATRIX, axis = 1))
                    # advancing the index to skip processed intervals
                    idx += run_length

                # processing individual intervals <<<<<<<<<<<<<<<

                else:
                    # extracting returns for the current interval index
                    underlying_rets = dataset[idx].values
                    # extracting values related to the current interval
                    SO_S = S0_s_sorted[idx] ; num_forward_s = num_forward_s_sorted[idx] ; forward_prices = forward_prices_sorted[idx]
                    r_s = r_s_sorted[idx] ; T_s = T_s_sorted[idx]

                    # simulating stock prices based on returns for the current interval
                    SimuStockPrices = SO_S * (1 + underlying_rets)
                    
                    # computing Profit & Loss (P&L) for long forward positions
                    if num_forward_s > 0:
                        PL = (SimuStockPrices - forward_prices) * num_forward_s * np.exp(r_s * (max_T - T_s))
                    # computing Profit & Loss (P&L) for short forward positions
                    else:
                        PL = (forward_prices - SimuStockPrices) * abs(num_forward_s) * np.exp(r_s * (max_T - T_s))

                    # aggregating the computed P&L values
                    PLList.append(PL)
                    # advancing to the next interval
                    idx += 1

            # flattening the aggregated P&L list into a single numpy array
            PL_results_flat = np.concatenate(PLList)
            # computing Value at Risk (VaR) and Expected Shortfall (ES) for the simulation
            VaR = np.quantile(PL_results_flat, alpha) * -1;  ES = np.mean(PL_results_flat[PL_results_flat < -VaR]) * -1 

            # returning VaR and ES for the simulation
            return VaR, ES
        
        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        def simulate_HS_VaR_Es_parallel(simulated_rets, sort_order, S0_s_sorted, forward_prices_sorted, num_forward_s_sorted,
                                         T_s_sorted, r_s_sorted, intervals_sorted, alpha):
            
            # def single_HS_simulation(sim, simulated_rets, sort_order, S0_s_sorted, forward_prices_sorted, num_forward_s_sorted, T_s_sorted, r_s_sorted, intervals_sorted)
            
            results = Parallel(n_jobs=-1)( delayed(single_HS_simulation)(sim, simulated_rets, sort_order, S0_s_sorted, forward_prices_sorted, num_forward_s_sorted,
                                                                       T_s_sorted, r_s_sorted, intervals_sorted, alpha) for sim in range(num_sims))
            
            VaRs = [res[0] for res in results] ; ESs = [res[1] for res in results] ; VaR = np.mean(VaRs) ; ES = np.mean(ESs)

            return VaR, ES
        
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # just some aesthetics
        from pyriskmgmt.SupportFunctions import DotPrinter

        # starting the animation
        my_dot_printer = DotPrinter(f"\r {num_sims} Scenarios for {len(self.S0_s)} Forwards - joblib - VaR & ES Parallel Computation") ; my_dot_printer.start()

        VaR, ES = simulate_HS_VaR_Es_parallel(simulated_rets, sort_order, S0_s_sorted, forward_prices_sorted, num_forward_s_sorted,
                                               T_s_sorted, r_s_sorted, intervals_sorted, alpha)
        
        # stopping the animation when minimization is done
        my_dot_printer.stop()
        print("\rParallel Computation --->  Done", end=" " * 95)

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # MaxExpiration, MinExpiration and MaxAggregation: finding the maximum and minimum expiration intervals

        MaxExpiration = max(self.summary["interval"]) ;  MinExpiration = min(self.summary["interval"])

        # ---------------------------------------------------------------------------------------------------------------------------------------------

        LenSinglePl = [] ; underlying_rets_sorted_aggregated = [] ; retsith = pd.DataFrame(simulated_rets[:,:,0]) # just for the dimension

        for interval in intervals_sorted:
            retsith.apply(lambda x: reshape_and_aggregate_returns(x, interval)) ; underlying_rets_sorted_aggregated.append(retsith) 
            LenSinglePl.append(retsith.shape[0])

        TotLenPL = np.sum(np.unique(LenSinglePl)); TotMCLenPL = num_sims * TotLenPL

        # -----------------------------------------------------------------------------------------------------------------------------------------

        # returning the results in a dictionary format
        MC_DIST = {"var" : round(VaR, 4),
                   "es" : round(ES,4),
                   "MaxExpi" : MaxExpiration,
                   "MinExpi" : MinExpiration,
                   "TotNumSimPL" : TotMCLenPL}
                
        return MC_DIST

    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_forward_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray, var: float, alpha: float = 0.05,
                     alpha_test: float = 0.05, warning = True):

        """
        Main
        ----
        The `kupiec_test` method performs the Kupiec test for the backtesting of Value at Risk (VaR). This test focuses on verifying whether the number of 
        exceptions (times the losses exceed VaR) matches the expected number given a confidence level.
        The method processes input data, checks and ensures that the forward contracts and returns are synchronized, and computes the P&L scenarios. It 
        then calculates the number of exceptions and compares the observed exceptions against the expected ones using a likelihood ratio test.

        Parameters
        ----------
        - num_forward_s: Array indicating the number of each forward contract in the portfolio.
        - S0_initial_s: Array of the initial prices for each forward contract's underlying asset at the beginning of the "test_returns" array.
        - test_returns: A 2D array representing historical return data for backtesting.
        - var: Value at Risk (VaR) level to be tested against.
        - alpha: The significance level used for the VaR calculation. Default is 0.05.
        - alpha_test: Significance level used for the likelihood ratio test. Default is 0.05.
        - warning: If set to True, the function will display warnings related to the input data's consistency. Default is True.

        Returns
        -------
        - dict: A dictionary containing the results of the Kupiec test, including:
            - Test name
            - VaR level
            - Total number of observations
            - Number of exceptions observed
            - Theoretical percentage of exceptions based on alpha
            - Actual percentage of exceptions
            - Likelihood ratio
            - Critical value from chi-squared distribution
            - Boolean indicating whether to reject the null hypothesis

        Notes
        -----
        - The test_returns frequency must be set to daily.
        - The likelihood ratio is used to test the accuracy of the VaR model. A significant result suggests the VaR model may not be accurate at the 
        specified confidence level.
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        import numpy as np ; from scipy.stats import chi2 


        from pyriskmgmt.inputsControlFunctions import (check_num_forward_s, check_S0_initial_s, validate_returns_port, check_var,
                                                       check_alpha, check_alpha_test, check_warning)
        
        # check procedure
        num_forward_s = check_num_forward_s(num_forward_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s) ; test_returns = validate_returns_port(test_returns) 
        var = check_var(var) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if warning:
            print("WARNING:")
            print("Ensure that the test_returns frequency is set to daily.")
            print("To mute this warning, add 'warning = False' as an argument of the kupiec_test() function.")

        # --------------------------------------------------------------------------------------------------------------------------------------------------

        # getting the order in which self.interval needs to be sorted
        sort_order = np.argsort(self.summary["interval"])

        # reordering the columns of test_returns based on the sorting order
        intervals_sorted = self.summary["interval"][sort_order].values ; test_returns_sorted = test_returns[:, sort_order] 
        num_forward_s_sorted = num_forward_s[sort_order] ; S0_s_sorted = self.S0_s[sort_order] 
        r_s_sorted = self.r_s[sort_order] ; forward_prices_sorted = self.forward_prices[sort_order] ; T_s_sorted = self.T_s[sort_order]
    
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # defining a function to calculate final prices after a given interval
        def generate_final_prices(returns, interval, S0):

            # trimming the returns to fit the interval size
            num_periods = len(returns) // interval * interval ; trimmed_returns = returns[:num_periods]

            # initializing the final_prices list
            final_prices = []

            # iterating over each possible start index in the trimmed returns
            for start_idx in range(0, num_periods - interval + 1):
                chunk = trimmed_returns[start_idx : start_idx + interval] ; final_price = S0 * np.prod(1 + chunk) ; final_prices.append(final_price)
            return np.array(final_prices).reshape(-1, 1)

        # initializing an empty list to store SimPrice2D results
        SimPrice2D = []

        # iterating over sorted intervals to compute SimPrice2D
        for index, interval in enumerate(intervals_sorted):
            final_prices = generate_final_prices(test_returns_sorted[:, index], interval, S0_s_sorted[index])
            SimPrice2D.append(final_prices)

        # defining a function to expand or stack arrays into a 4D structure
        def stack_or_expand(arrays):

            i = 0
            SimPrice4D = []

            # iterating through the provided arrays
            while i < len(arrays):
                current_array = arrays[i][:, :, np.newaxis]  # adding a third dimension
                # checking for arrays with matching shapes
                matching_arrays = [current_array]
                j = i + 1
                while j < len(arrays) and arrays[j].shape[0] == current_array.shape[0]:
                    matching_arrays.append(arrays[j][:, :, np.newaxis])  # adding a third dimension
                    j += 1
                # stacking matching arrays or using them as is
                if len(matching_arrays) > 1:
                    stacked_array = np.concatenate(matching_arrays, axis=2)
                    i = j
                else:
                    stacked_array = current_array
                    i += 1
                SimPrice4D.append(stacked_array)
            return SimPrice4D

        # -------------------------------------------------------------------------------------------------------------------------------------------------

        # calling the stack_or_expand function
        SimPrice4D = stack_or_expand(SimPrice2D)

        # initializing Profit and Loss (P&L) list and getting the maximum T value
        PL = [] ; MaxT = max(T_s_sorted)

        forwardCount = 0

        # iterating through the SimPrice4D results
        for j in range(len(SimPrice4D)):

            SimPrice3D = SimPrice4D[j]

            # extracting relevant portions for the current 3D dataset
            num_forward_s_sorted_new = num_forward_s_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])]
            forward_prices_sorted_new = forward_prices_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])]
            r_s_sorted_new = r_s_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])] 
            T_s_sorted_new = T_s_sorted[forwardCount : (forwardCount + SimPrice3D.shape[2])]

            Plmatrix3D = np.zeros_like(SimPrice3D)

            # iterating over the depth of the 3D dataset
            for index_inner, z in enumerate(range(SimPrice3D.shape[2])):

                dataset2D = SimPrice3D[:,:,z]
                PlMatrix2D = np.zeros_like(dataset2D)

                # iterating over columns of the 2D dataset
                for col in range(dataset2D.shape[1]):

                    # iterating over rows of the current column
                    for row in range(dataset2D.shape[0]):

                        # calculating P&L for positive num_forward_s_sorted_new values
                        if num_forward_s_sorted_new[index_inner] > 0:
                            PlMatrix2D[row,col] = (dataset2D[row,col] - forward_prices_sorted_new[index_inner]) * num_forward_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT -T_s_sorted_new[index_inner]))

                        # calculating P&L for negative num_forward_s_sorted_new values
                        elif num_forward_s_sorted_new[index_inner] < 0:
                            PlMatrix2D[row,col] = (forward_prices_sorted_new[index_inner] - dataset2D[row,col]) * num_forward_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT -T_s_sorted_new[index_inner]))

                    Plmatrix3D[:,:, index_inner] = PlMatrix2D

        # *******************************************************************************************************************************************

            pl = np.sum(Plmatrix3D, axis=2).ravel() ; PL.extend(pl) ; forwardCount += SimPrice3D.shape[2]

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        VaR = var ;  PL = np.array(PL) ; num_exceptions = np.sum(PL < - VaR)
        
        # total number of observations -----------------------------------------------------------------------------------

        num_obs = len(PL)

        # % exceptions -------------------------------------------------------------------------------------------------------

        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test
            "var" : VaR,                            # Value at Risk
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
##########################################################################################################################################################################
##########################################################################################################################################################################
## 10. EquityFuturePort ##################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class EquityFuturePort: 

    """
    EquityFuturePort 
    ----------------
    This class offers tools to assess and compute the VaR (Value-at-Risk) and ES (Expected Shortfall) at the maturity of a portfolio of futures contracts 
    integral to derivatives trading and risk management. The risk assessment techniques for the portfolio incorporate methods 
    such as historical simulation and Monte Carlo simulation using the Cholesky decomposition of the VCV. 
    The class calculates the VaR and ES set to occur at the maturity of the last future contract in the portfolio. It emphasizes the market-to-market approach, 
    reflecting the continuous revaluation of contracts to account for daily price changes. The class also integrates the notion of moving money through time 
    using the risk-free rate, enhancing the understanding of potential future risks.
    
    A Kupiec backtest can be integrated for evaluating the accuracy of the VaR predictions for the portfolio, critical for both regulatory and internal risk assessment.
    The future prices for the portfolio consider the equities' spot prices, prevailing risk-free interest rates, any expected dividend yields, convenience yields, storage costs, 
    and the times left to the contracts' maturities.

    Initial Parameters
    ------------------
    - S0_s: Array of initial spot prices of the equities in the portfolio.
    - T_s: Array of times to maturity in years for the contracts in the portfolio.
      - For instance: expDays = np.array([5,8,12,15,18]); T_s = expDays/252
    - r_s: Array of risk-free interest rates as decimals for the contracts in the portfolio in annual terms (e.g., 0.05 for 5%).
    - dividend_yield_s: Array of dividend yields for each equities in the portfolio as decimals (annual continuous dividend yields). Default is None.
    - convenience_yield_s: Array of the contracts' convenience yields in the portfolio as decimals (annual convenience dividend yields). Default is None.
    - storage_cost_s: Array of the contracts' storage costs in the portfolio as decimals (annual storage costs). Default is None.

    Decision to Omit Margin Calls
    ------------------------------
    - Notably, this class does not factor in margin calls. In futures trading, margin calls occur when the value of an investor's margin account falls below the broker's required amount.
    When the account value drops below the margin requirement, brokers demand that the investor deposit additional money to cover the potential loss. 
    This ensures that the account is always sufficiently funded to cover the contract's obligations. 
    However, in the context of this class, the emphasis is placed squarely on evaluating the inherent market risk of a portfolio of futures.
    By focusing on market risk, the class prioritizes understanding the potential shifts in portfolio value due to market factors, without the added complexity of
    handling margin requirements and associated cash flows. As such, while margin calls are pivotal in real-world futures trading, ensuring appropriate levels of
    funds and managing daily changes, their exclusion here allows users to concentrate on the primary goal: assessing market risk in a streamlined manner.

    - Furthermore, it's essential to note that the `EquityFuturePort` class serves as a foundational package, designed with extensibility in mind. 
    While the current version prioritizes the evaluation of market risk without the intricacies of margin calls, this design choice does not limit the potential of the package.
    Developers and risk management professionals can leverage this foundational structure to build upon it, introducing more advanced features or specific modifications
    tailored to unique requirements. By starting with a streamlined and focused version, the package offers a clean slate, 
    reducing the initial learning curve and providing a clear path for enhancements. This modularity ensures that the class can evolve to cater to a broader spectrum
    of risk management needs, with the possibility of integrating margin calls or other advanced features in subsequent iterations, 
    depending on the evolving requirements of the derivatives trading community.

    Methods
    -------
    - HistoricalSimulation(): Uses historical data to simulate the VaR and ES for the futures contract portfolio. Takes into account the expiration of the last contract, 
    computes the P&L for each contract, and adjusts using the risk-free rate.
    - ff3(): Fetches data for the Fama-French three-factor (FF3) model, including Market return (MKT), Size (SMB), and Value (HML).
    - mcModelRetsSimu(): Simulates the VaR and ES for future contracts using Monte Carlo methods.
    - kupiec_test(): Uses the Kupiec backtest to evaluate the VaR model's accuracy through actual vs. expected exceptions comparison.

    Attributes
    ----------
    - future_prices: An array derived from the summary DataFrame that holds the computed futures prices for the contracts in the portfolio. 
    It essentially captures the expected market prices of the equities at the time of the futures contracts' maturity.
    - summary: A pandas DataFrame that provides an overview of the futures portfolio. It includes details like futures prices, spot prices, risk-free rates, 
    time to maturity, and other relevant details.

    Notes
    -----
    The provided risk management package includes various input checking functions to ensure portfolio input validity. 
    This class is apt for forward contracts portfolio management, supporting the simultaneous analysis of multiple contracts.

    Limitations and Extensibility
    ------------------------------
    The `pyriskmgmt` package is intended to serve as either a straightforward tool for basic pricing processes and risk assessment or as a foundational 
    starting point for more advanced, customized development. Users seeking a more nuanced or precise tool are encouraged to extend and adapt the existing
    functionalities to better align with their specific requirements and investment strategies.
    """

    import warnings ; warnings.filterwarnings("ignore")

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, S0_s: np.ndarray,
                 T_s: np.ndarray, 
                 r_s:  np.ndarray,
                 dividend_yield_s: np.ndarray = None, convenience_yield_s : np.ndarray = None, storage_cost_s : np.ndarray = None):
        
        import numpy as np

        from pyriskmgmt.inputsControlFunctions import (check_S0_s, check_T_s, check_r_s, check_dividend_yield_s,
                                                       check_convenience_yield_s, check_storage_cost_s)

        # check procedure
        self.S0_s = check_S0_s(S0_s) ; self.T_s = check_T_s(T_s) ;  self.r_s = check_r_s(r_s) 

        if dividend_yield_s is None: self.dividend_yield_s = np.repeat(0,len(T_s)) 
        if convenience_yield_s is None: self.convenience_yield_s = np.repeat(0,len(T_s))
        if storage_cost_s is None: self.storage_cost_s = np.repeat(0,len(T_s))

        if dividend_yield_s is not None: self.dividend_yield_s = check_dividend_yield_s(dividend_yield_s) 
        if convenience_yield_s is not None: self.convenience_yield_s = check_convenience_yield_s(convenience_yield_s)
        if storage_cost_s is not None: self.storage_cost_s = check_storage_cost_s(storage_cost_s)

        # ------------------------------------------------------------------------------------------------

        # checking the same length for all the imputs *args

        def check_same_length(*args):
            """
            Checks if all given numpy arrays have the same length.
            """
            it = iter(args)
            the_len = len(next(it))

            if not all(len(l) == the_len for l in it):
                raise ValueError("Not all the input arrays in the EquityFuturePort class have the same length!")
                       
        # Here we are ensuring that all arrays have the same length
        check_same_length(T_s, S0_s, r_s, self.dividend_yield_s, self.convenience_yield_s, self.storage_cost_s)

        # ------------------------------------------------------------------------------------------------

        self.future_prices = None 

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import pandas as pd 

        from pyriskmgmt.SupportFunctions import FuturesPrice # (spot_price, risk_free_rate, dividend_yield, convenience_yield, storage_cost, time_to_maturity)

        summary = pd.DataFrame()

        # calculating forward prices and add to summary DataFrame
        for S0, r, dividend_yield, convenience_yield, storage_cost, T in zip(self.S0_s, self.r_s, self.dividend_yield_s, self.convenience_yield_s, 
                                                                             self.storage_cost_s, self.T_s):
            
            price = FuturesPrice(S0, r, dividend_yield, convenience_yield, storage_cost, T) ; interval = int(T * 252)

            # checking if all values in the lists are zeros
            if all(value == 0 for value in self.dividend_yield_s) and all(value == 0 for value in self.convenience_yield_s) and all(value == 0 for value in self.storage_cost_s):
                summary = summary._append({'FuturePrice': price, 'S0': S0, 'T': T, 'interval': interval, 'r': r}, ignore_index=True)
            else:
                append_dict = {'FuturePrice': price, 'S0': S0, 'T': T, 'interval': interval, 'r': r}

                # Check individual lists and add the non-zero attributes
                if not all(value == 0 for value in self.dividend_yield_s):
                    append_dict["dividend_yield"] = dividend_yield

                if not all(value == 0 for value in self.convenience_yield_s):
                    append_dict["convenience_yield"] = convenience_yield 

                if not all(value == 0 for value in self.storage_cost_s):
                    append_dict["storage_cost"] = storage_cost 

                summary = summary._append(append_dict, ignore_index=True)

        # rounding all the values to 4 decimal places
        summary = summary.round(4) ; summary['interval'] = summary['interval'].astype(int)

        self.summary = summary ; self.future_prices = summary["FuturePrice"].values

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************    

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, num_future_s: np.ndarray, underlying_rets: np.ndarray, alpha: float = 0.05, warning: bool = True):

        """
        Main
        ----
        This method employs a historical simulation approach to compute the Value at Risk (VaR) and Expected Shortfall (ES) 
        for a portfolio of future contracts at the expiration of the last future contract. The method leverages daily mark-to-market 
        (MTM) P&L computations, adjusting them through time using the risk-free rate. Synchronization and consistency of the provided 
        inputs are of utmost importance. The computations heavily involve reshaping and realigning the input data structures and 
        extensively utilize matrix operations for efficient processing.

        Methodology
        -----------
        1. Data Processing:
        The method sorts and reorders the return data based on the expiration interval of each future contract. It then 
        simulates final prices for the underlying assets based on the corresponding intervals. This results in a synchronized 4D data 
        structure of simulated prices.
        2. Profit and Loss Calculation:
        The method computes the daily mark-to-market (MTM) P&L scenarios for each future contract, adjusting them across time 
        using the associated risk-free rate. P&L is derived from the differences between simulated final prices and futures 
        prices, considering the position (long or short) of each contract.
        3. Risk Metrics Calculation:
        The aggregated P&L scenarios are used to compute VaR and ES. VaR is the quantile of the P&L distribution at the given 
        significance level alpha. ES represents the average of the losses exceeding the computed VaR.
        4. Output:
        Returns a dictionary encompassing the calculated VaR and ES, maximum and minimum expiration intervals of the future 
        contracts, and the total number of simulated P&L scenarios.

        Parameters
        ----------
        - num_future_s: Array indicating the number of each future contract in the portfolio. Positive values denote a long position, 
        and negative values denote a short position.
        - underlying_rets: A 2D array of historical returns for the underlying assets of the future contracts. The frequency of these returns
        must be daily in order to compute the daily mark-to-market (MTM) P&L for each contract.
        - alpha: Significance level used to compute VaR and ES. Represents the probability of observing a loss greater 
        than VaR or the average of losses exceeding VaR. Default is 0.05.
        - warning: If set to True, it will print a warning regarding the consistency of the input data. Default is True.

        Notes
        -----
        - It's crucial that the 'underlying_rets' are provided with a daily frequency.
        - The computations primarily rely on the generation of a synchronized 4D data structure from daily returns.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from pyriskmgmt.inputsControlFunctions import (check_num_future_s, check_alpha, validate_returns_port, check_warning)
        
        import numpy as np 

        # check procedure
        num_future_s = check_num_future_s(num_future_s) ; underlying_rets = validate_returns_port(underlying_rets) ; alpha = check_alpha(alpha) ; check_warning(warning)

        if warning:
            print("WARNING:")
            print("It's crucial that the 'underlying_rets' are provided with a daily frequency.")
            print("This method adjusts as necessary and calculates the VaR at the expiration of the last future contract.")
            print("It computes the daily mark-to-market (MTM) P&L for each contract, adjusting them through time using the risk-free rate.")
            print("Also ensure that the 'underlying_rets' matches the 'num_future_s' in an element-wise way.")
            print("To suppress this warning, set 'warning=False' in this 'HistoricalSimulation()' method.")

        # -----------------------------------------------------------------------------------------------------------------------------------------

        # getting the sorting order of self.interval
        sort_order = np.argsort(self.summary["interval"])
        # reordering the columns of underlying_rets based on the sorting order
        intervals_sorted = self.summary["interval"][sort_order].values ; underlying_rets_sorted = underlying_rets[:, sort_order] 
        num_futures_s_sorted = num_future_s[sort_order] ; S0_s_sorted = self.S0_s[sort_order]  ; r_s_sorted = self.r_s[sort_order] 
        futures_prices_sorted = self.future_prices[sort_order] ; T_s_sorted = self.T_s[sort_order]

        # Now everything is sorted by interval...

        # -----------------------------------------------------------------------------------------------------------------------------------------

        # defining the function to simulate future stock prices based on provided underlying returns, interval, and initial stock price (S0)

        def FuturesMatrixSimPrices3DCreator(underlying_rets_ith, interval, S0):

            # determining the total number of periods from the underlying returns
            n_periods = len(underlying_rets_ith) ; n_extractions = n_periods // interval
            # initializing a matrix with zeros to store simulated prices
            simPrices = np.zeros((interval, n_extractions))
            # iterating over the underlying returns in steps of the provided interval
            for i in range(0, n_periods, interval):
                # determining the end of the current interval
                end = i + interval
                # breaking the loop if we exceed the number of available periods
                if end > n_periods:
                    break
                # calculating and storing simulated prices for the current interval
                simPrices[:, i // interval] = S0 * np.cumprod(1 + underlying_rets_ith[i: end])
            # returning the matrix with simulated prices
            return simPrices
        
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # accumulating counts for each interval.
        # for each interval, it checks how many intervals are greater than or equal to the current interval
        # and appends the total count to the list.
        counts = []
        for interval in intervals_sorted:
            count = np.sum(intervals_sorted >= interval) ; counts.append(count)
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # creating a 4D structure to store the simulated prices for each asset and each interval.

        SimPrice4D = []
        # iterating over all intervals.
        for idx in range(len(intervals_sorted)):
            # getting the number of assets we will consider for the current interval.
            NumAssetsToSelect = counts[idx]
            # generating a list of indices corresponding to all assets.
            full_indices = np.arange(underlying_rets.shape[1])
            # selecting the last 'NumAssetsToSelect' indices from the generated list.
            indices_to_select = full_indices[ - NumAssetsToSelect:] 
            n_extractions = underlying_rets_sorted.shape[0] // (max(intervals_sorted))
            # initializing a 3D list for the current interval to store simulated prices for each asset.
            SimPrice3D = np.zeros((max(intervals_sorted), n_extractions, NumAssetsToSelect))
            # iterating over the selected asset indices.
            for index, value in enumerate(indices_to_select):
                # fetching the underlying returns and initial stock price (S0) for the current asset.
                underlying_rets_ith = underlying_rets_sorted[:,value] ; S0 = S0_s_sorted[value]
                # using the FuturesMatrixSimPrices3DCreator function to simulate future stock prices for the current asset and the current interval.
                simPrices = FuturesMatrixSimPrices3DCreator(underlying_rets_ith, max(intervals_sorted), S0)
                # appending the simulated prices for the current asset to the 3D list.
                SimPrice3D[:,:,index] = simPrices
            # once we have simulated prices for all selected assets for the current interval, we append the 3D list to the 4D list.
            SimPrice4D.append(SimPrice3D)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        unique_arrays = [] ; seen_shapes = set()

        for array3D in SimPrice4D:
            # checking if the shape of this 3D array is not in the seen_shapes.
            if array3D.shape not in seen_shapes:
                seen_shapes.add(array3D.shape)  # adding this shape to the set.
                unique_arrays.append(array3D)   # appending this 3D array to the unique list.

        SimPrice4D = unique_arrays

        # *******************************************************************************************************

        unique_interval_sorted = np.unique(intervals_sorted)

        # Initializing an empty list to store Profit & Loss values.
        PL = []
        # Getting the maximum expiration time.
        MaxT = max(T_s_sorted)

        prevSimDays = 0
        # Iterating over each entry in SimPrice4D.
        for j, dataset3D in enumerate(SimPrice4D): 
            days_to_simulate = unique_interval_sorted[j] - prevSimDays ; dataset3DShape2 = dataset3D.shape[2] ; relevant_slice = - dataset3DShape2
            # Slicing the relevant portions based on the depth of the current dataset.
            num_futures_s_sorted_new = num_futures_s_sorted[relevant_slice:] ; futures_prices_sorted_new = futures_prices_sorted[relevant_slice:]
            r_s_sorted_new = r_s_sorted[relevant_slice:]

            # ********************************************************************************************************

            # initializing a 3D matrix of zeros with shape (days_to_simulate, dataset3D.shape[1], dataset3DShape2)
            PlMatrix3D = np.zeros((days_to_simulate, dataset3D.shape[1], dataset3DShape2))

            # looping through each slice of the third dimension of dataset3D
            for index_inner in range(dataset3DShape2):
                # extracting a 2D slice from dataset3D for the given index_inner 
                dataset2D = dataset3D[prevSimDays: prevSimDays + days_to_simulate, :, index_inner]
                # initializing a 2D matrix of zeros with the same shape as dataset2D
                PlMatrix2D = np.zeros_like(dataset2D)

                # iterating through each column of dataset2D
                for col in range(dataset2D.shape[1]):
                    # iterating through each row of the current column in dataset2D
                    for row in range(dataset2D.shape[0]):

                        # checking if the value at index_inner in num_futures_s_sorted_new is positive
                        if num_futures_s_sorted_new[index_inner] > 0:
                            # calculating the profit or loss (P&L) for the given cell in dataset2D using a specific formula
                            PlMatrix2D[row,col] = (dataset2D[row,col] - futures_prices_sorted_new[index_inner]) * num_futures_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT - ((prevSimDays + row + 1) / 252)))

                        # checking if the value at index_inner in num_futures_s_sorted_new is negative
                        elif num_futures_s_sorted_new[index_inner] < 0:

                            # calculating the profit or loss (P&L) for the given cell in dataset2D using a different formula
                            PlMatrix2D[row,col] = (futures_prices_sorted_new[index_inner] - dataset2D[row,col] ) * num_futures_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT - ((prevSimDays + row + 1) / 252)))

                # updating the 3D matrix PlMatrix3D with the values from PlMatrix2D for the current slice
                PlMatrix3D[:, :, index_inner] = PlMatrix2D

            # *******************************************************************************************************************************************

            pl = np.sum(PlMatrix3D, axis=2).ravel() ; PL.extend(pl) ; prevSimDays += days_to_simulate

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PL = np.array(PL) ; VaR = np.quantile(PL, alpha) * -1 ; ES = np.mean(PL[PL < - VaR]) * - 1

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------

        # MaxExpiration, MinExpiration and MaxAggregation: finding the maximum and minimum expiration intervals

        MaxExpiration = max(self.summary["interval"]) ;  MinExpiration = min(self.summary["interval"])

        # returning the results in a dictionary format
        HS = {"var" : round(VaR, 4),
              "es" : round(ES, 4),
              "MaxExpi" : MaxExpiration,
              "MinExpi" : MinExpiration,
              "TotNumSimPL" : len(PL)}

        return HS
    
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
        
    # MONTECARLO MODEL [RETURNS - DISTIBUTION] <<<<<<<<<<<< 

    def mcModelRetsSimu(self, num_future_s: np.ndarray, underlying_rets: np.ndarray, alpha: float = 0.05, sharpeDiag: bool = False, mappingRets: np.ndarray = None,
                         vol: str = "simple" ,num_sims: int = 250, p: int = 1, q: int = 1, lambda_ewma: float = 0.94, warning: bool = True):
        
        """
        Main
        ----
        The `mcModelRetsSimu` function simulates Monte Carlo model returns based on specified configurations, accounting for various risk metrics.
        The function assumes that all inputs, including `num_future_s`, `underlying_rets`, and other parameters, are synchronized and in order.
        The extensive matrix operations embedded within the function are utilized for efficiency 
        and precision. However, the onus of providing synchronized data rests on the user.

        Methodology
        -----------
        1. Data Processing:
        The method sorts and reorders the return data based on the expiration interval of each future contract. It then 
        simulates final prices for the underlying assets based on the corresponding intervals. This results in a synchronized 4D data 
        structure of simulated prices.
        2. Profit and Loss Calculation:
        The method computes the daily mark-to-market (MTM) P&L scenarios for each future contract, adjusting them across time 
        using the associated risk-free rate. P&L is derived from the differences between simulated final prices and futures 
        prices, considering the position (long or short) of each contract.
        3. Risk Metrics Calculation:
        The aggregated P&L scenarios are used to compute VaR and ES. VaR is the quantile of the P&L distribution at the given 
        significance level alpha. ES represents the average of the losses exceeding the computed VaR.
        4. Output:
        Returns a dictionary encompassing the calculated VaR and ES, maximum and minimum expiration intervals of the future 
        contracts, and the total number of simulated P&L scenarios.

        Parameters
        -----------
        - num_future_s: Array that indicates the number of each future contract in the portfolio. Positive values signify a long position, while negative values suggest 
        a short position.
        - underlying_rets: A 2D array presenting historical returns of the future contracts' underlying assets. 
        The underlying_rets frequency must be set to "daily".
        - alpha: Significance level used to compute VaR and ES. Represents the probability of observing a loss greater 
        than VaR or the average of losses exceeding VaR. Default is 0.05.
        - sharpeDiag: If enabled, the function employs a Sharpe diagonal approach to map "underlying_rets" to 1 or more factors. Default is None.
        - mappingRets: Indicates the returns of one or more factors used in the mapping process. Make sure the data is on a daily frequency
        and align "mappingRets" with "underlying_rets" in a time-wise manner. Required if sharpeDiag = True. Default is None.
        - vol: Denotes the volatility model in use, with options being 'simple', 'ewma', or 'garch'. Default is "simple".
        - num_sims: Number of Monte Carlo simulations to be executed. Default is 250.
        - p: Order of the AR term for GARCH. Default is 1.
        - q: Order of the MA term for GARCH. Default is 1.
        - lambda_ewma: The smoothing factor designated for the EWMA volatility model. Default is 0.94.
        - warning: If enabled, the function will broadcast warnings pertinent to the input data's consistency. Default is True.

        Notes
        -----
        - The underlying_rets frequency must be set to daily.
        - Although the function efficiently stacks or expands the simulated prices into a 4D data structure to enable synchronized calculations, 
        the process is initiated from daily returns.
        """

        import warnings ; warnings.filterwarnings("ignore")
                        
        from pyriskmgmt.inputsControlFunctions import (check_num_future_s, check_alpha, validate_returns_port, check_number_simulations, check_warning)
        
        import numpy as np ; from joblib import Parallel, delayed
        
        # check procedure
        num_future_s = check_num_future_s(num_future_s) ; underlying_rets = validate_returns_port(underlying_rets) ; alpha = check_alpha(alpha)
        num_sims = check_number_simulations(num_sims) ; check_warning(warning) 

        if warning:
            print("WARNING:")
            print("It's crucial that the 'underlying_rets' are provided with a daily frequency.")
            print("This method adjusts as necessary and calculates the VaR at the expiration of the last future contract.")
            print("It computes the daily mark-to-market (MTM) P&L for each contract, adjusting them through time using the risk-free rate.")
            print("Also ensure that the 'underlying_rets' matches the 'num_future_s' in an element-wise way.")
            print("To suppress this warning, set 'warning=False' in this 'mcModelRetsSimu()' method.")

        if vol == "garch":
            
            if warning:
                from pyriskmgmt.SupportFunctions import garch_warnings
                garch_warnings("mcModelRetsSimu()")
                
            if warning and underlying_rets.shape[1] > 50 and not sharpeDiag:
                print("IMPORTANT:")
                print("The 'garch' option for the 'vol' parameter is currently in use, with an asset count exceeding 50.")
                print("This configuration can lead to slower processing and inefficient resource usage, due to the complexity ")
                print("of estimating the full variance-covariance matrix (VCV) in high-dimensional space.")
                print("It's strongly advised to employ a mapping approach instead. This approach maps the multiple assets to one or more factors,")
                print("effectively reducing the dimensionality of the problem.")
                print("This is not only more computationally efficient, but it also avoids the 'curse of dimensionality',")
                print("as it reduces the number of parameters needed to be estimated. ")
                print("The mapping approach aids in providing more stable and reliable estimations, especially in large, complex systems.")

        warning = False

        # all the other inputs will be handled here, in the EquityPrmPort class ---------------------------------------------------------------------------------

        from pyriskmgmt.equity_models import EquityPrmPort

        model = EquityPrmPort(returns = underlying_rets, positions = np.repeat(1 ,underlying_rets.shape[1]),
                              interval = 1, alpha = alpha) # here 'positions' is not pivotal... I need this class only to retrieve Monte Carlo simulated_rets
        
        simulated_rets = model.mcModelRetsSimu(sharpeDiag = sharpeDiag, mappingRets = mappingRets, vol = vol, num_sims = num_sims, p = p, q = q, 
                                                               lambda_ewma = lambda_ewma, warning = warning, return_simulated_rets = True)[1]        
            
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # getting the sorting order of self.interval
        sort_order = np.argsort(self.summary["interval"])
        # reordering the columns of underlying_rets based on the sorting order
        intervals_sorted = self.summary["interval"][sort_order].values ; num_futures_s_sorted = num_future_s[sort_order] ; S0_s_sorted = self.S0_s[sort_order] 
        r_s_sorted = self.r_s[sort_order] ; futures_prices_sorted = self.future_prices[sort_order] ; T_s_sorted = self.T_s[sort_order]

        # Now everything is sorted by interval...

        # -----------------------------------------------------------------------------------------------------------------------------------------

        # defining the function to simulate future stock prices based on provided underlying returns, interval, and initial stock price (S0)

        def FuturesMatrixSimPrices3DCreator(underlying_rets_ith, interval, S0):

            # determining the total number of periods from the underlying returns
            n_periods = len(underlying_rets_ith) ; n_extractions = n_periods // interval
            # initializing a matrix with zeros to store simulated prices
            simPrices = np.zeros((interval, n_extractions))
            # iterating over the underlying returns in steps of the provided interval
            for i in range(0, n_periods, interval):
                # determining the end of the current interval
                end = i + interval
                # breaking the loop if we exceed the number of available periods
                if end > n_periods:
                    break
                # calculating and storing simulated prices for the current interval
                simPrices[:, i // interval] = S0 * np.cumprod(1 + underlying_rets_ith[i: end])
            # returning the matrix with simulated prices
            return simPrices
        
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # accumulating counts for each interval.
        # for each interval, it checks how many intervals are greater than or equal to the current interval
        # and appends the total count to the list.
        counts = []
        for interval in intervals_sorted:
            count = np.sum(intervals_sorted >= interval) ; counts.append(count)

        # -----------------------------------------------------------------------------------------------------------------------------------------

        VaRS = np.zeros(num_sims) ; ESs = np.zeros(num_sims)

        ######################################################################################################################

        def simulation_work(sim):
            underlying_rets = simulated_rets[:,:,sim] ;  underlying_rets_sorted = underlying_rets[:, sort_order]

            # creating a 4D structure to store the simulated prices for each asset and each interval.

            SimPrice4D = []
            # iterating over all intervals.
            for idx in range(len(intervals_sorted)):
                # getting the number of assets we will consider for the current interval and generating a list of indices corresponding to all asset
                NumAssetsToSelect = counts[idx] ; full_indices = np.arange(underlying_rets.shape[1])
                # selecting the last 'NumAssetsToSelect' indices from the generated list.
                indices_to_select = full_indices[ - NumAssetsToSelect:] ; n_extractions = underlying_rets_sorted.shape[0] // (max(intervals_sorted))
                # initializing a 3D list for the current interval to store simulated prices for each asset.
                SimPrice3D = np.zeros((max(intervals_sorted), n_extractions, NumAssetsToSelect))
                # iterating over the selected asset indices.
                for index, value in enumerate(indices_to_select):
                    # fetching the underlying returns and initial stock price (S0) for the current asset.
                    underlying_rets_ith = underlying_rets_sorted[:,value]   
                    S0 = S0_s_sorted[value]
                    # using the FuturesMatrixSimPrices3DCreator function to simulate future stock prices for the current asset and the current interval.
                    simPrices = FuturesMatrixSimPrices3DCreator(underlying_rets_ith, max(intervals_sorted), S0)
                    # appending the simulated prices for the current asset to the 3D list.
                    SimPrice3D[:,:,index] = simPrices
                # once we have simulated prices for all selected assets for the current interval, we append the 3D list to the 4D list.
                SimPrice4D.append(SimPrice3D)

            # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            unique_arrays = [] ; seen_shapes = set()

            for array3D in SimPrice4D:
                # checking if the shape of this 3D array is not in the seen_shapes.
                if array3D.shape not in seen_shapes:
                    seen_shapes.add(array3D.shape)  # adding this shape to the set.
                    unique_arrays.append(array3D)   # appending this 3D array to the unique list.

            SimPrice4D = unique_arrays

            # ***************************************************************************************

            unique_interval_sorted = np.unique(intervals_sorted)

            # initializing an empty list to store Profit & Loss values.
            PL = []
            # getting the maximum expiration time.
            MaxT = max(T_s_sorted)

            prevSimDays = 0
            # iterating over each entry in SimPrice4D.
            for j, dataset3D in enumerate(SimPrice4D): 

                days_to_simulate = unique_interval_sorted[j] - prevSimDays ; dataset3DShape2 = dataset3D.shape[2] ; relevant_slice = - dataset3DShape2

                # slicing the relevant portions based on the depth of the current dataset.
                num_futures_s_sorted_new = num_futures_s_sorted[relevant_slice:] ; futures_prices_sorted_new = futures_prices_sorted[relevant_slice:] 
                r_s_sorted_new = r_s_sorted[relevant_slice:]

                # ***********************************************************************************************************************************

                # initializing a 3D matrix of zeros with shape (days_to_simulate, dataset3D.shape[1], dataset3DShape2)
                PlMatrix3D = np.zeros((days_to_simulate, dataset3D.shape[1], dataset3DShape2))
                # looping through each slice of the third dimension of dataset3D
                for index_inner in range(dataset3DShape2):
                    # extracting a 2D slice from dataset3D for the given index_inner 
                    dataset2D = dataset3D[prevSimDays: prevSimDays + days_to_simulate, :, index_inner]
                    # initializing a 2D matrix of zeros with the same shape as dataset2D
                    PlMatrix2D = np.zeros_like(dataset2D)

                    # iterating through each column of dataset2D
                    for col in range(dataset2D.shape[1]):
                        # iterating through each row of the current column in dataset2D
                        for row in range(dataset2D.shape[0]):

                            # checking if the value at index_inner in num_futures_s_sorted_new is positive
                            if num_futures_s_sorted_new[index_inner] > 0:
                                # calculating the profit or loss (P&L) for the given cell in dataset2D using a specific formula
                                PlMatrix2D[row,col] = (dataset2D[row,col] - futures_prices_sorted_new[index_inner]) * num_futures_s_sorted_new[index_inner]
                                PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT - ((prevSimDays + row + 1) / 252)))

                            # checking if the value at index_inner in num_futures_s_sorted_new is negative
                            elif num_futures_s_sorted_new[index_inner] < 0:

                                # calculating the profit or loss (P&L) for the given cell in dataset2D using a different formula
                                PlMatrix2D[row,col] = (futures_prices_sorted_new[index_inner] - dataset2D[row,col] ) * num_futures_s_sorted_new[index_inner]
                                PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT - ((prevSimDays + row + 1) / 252)))

                    # updating the 3D matrix PlMatrix3D with the values from PlMatrix2D for the current slice
                    PlMatrix3D[:, :, index_inner] = PlMatrix2D

                # *******************************************************************************************************************************************
            
                pl = np.sum(PlMatrix3D, axis=2).ravel() ; PL.extend(pl) ; prevSimDays += days_to_simulate

            # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            PL = np.array(PL) ; VaR = np.quantile(PL, alpha) * -1 ; ES = np.mean(PL[PL < - VaR]) * - 1

            return VaR, ES, PL
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

        # just some aesthetics
        from pyriskmgmt.SupportFunctions import DotPrinter

        # starting the animation
        my_dot_printer = DotPrinter(f"\r {num_sims} Scenarios for {len(self.S0_s)} Futures - joblib - VaR & ES Parallel Computation") ; my_dot_printer.start()

        results = Parallel(n_jobs=-1)(delayed(simulation_work)(sim) for sim in range(num_sims))

        # stopping the animation when minimization is done
        my_dot_printer.stop()
        print("\rParallel Computation --->  Done", end=" " * 95)

        VaRS, ESs, PL = zip(*results) ; VaR = np.mean(VaRS) ; ES = np.mean(ESs)

        # ------------------------------------------------------------------------------------------------------------------------------------------------

        # MaxExpiration, MinExpiration and MaxAggregation: finding the maximum and minimum expiration intervals

        MaxExpiration = max(self.summary["interval"]) ;  MinExpiration = min(self.summary["interval"])

        # returning the results in a dictionary format
        MC_DIST = {"var" : round(VaR, 4),
                   "es" : round(ES,4),
                   "MaxExpi" : MaxExpiration,
                   "MinExpiration" : MinExpiration,
                   "TotNumSimPL" : len(PL[0]) * num_sims}

        return MC_DIST
    
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 

    # KUPICEC TEST <<<<<<<<<<<<

    def kupiec_test(self, num_future_s: np.ndarray, S0_initial_s: np.ndarray, test_returns: np.ndarray, var: float, alpha: float = 0.05, alpha_test: float = 0.05, warning = True):

        """
        Main
        ----
        The `kupiec_test` method performs the Kupiec test for the backtesting of Value at Risk (VaR). This test focuses on verifying whether the number of 
        exceptions (times the losses exceed VaR) matches the expected number given a confidence level.
        The method processes input data, checks and ensures that the future contracts and returns are synchronized, and computes the P&L scenarios. It 
        then calculates the number of exceptions and compares the observed exceptions against the expected ones using a likelihood ratio test.

        Parameters
        ----------
        - num_future_s: Array indicating the number of each forward contract in the portfolio.
        - S0_initial_s: Array of the initial prices for each forward contract's underlying asset at the beginning of the "test_returns" array.
        - test_returns: A 2D array representing historical return data for backtesting.
        Ensure that the frequency of the "test_returns" array is "daily".
        - var: Value at Risk (VaR) to be tested (in monetary value).
        - alpha: The significance level used for the VaR calculation. Default is 0.05.
        - alpha_test: Significance level used for the likelihood ratio test. Default is 0.05.
        - warning: If set to True, the function will display warnings related to the input data's consistency. Default is True.

        Returns
        -------
        - dict: A dictionary containing the results of the Kupiec test, including:
            - Test name
            - VaR level
            - Total number of observations
            - Number of exceptions observed
            - Theoretical percentage of exceptions based on alpha
            - Actual percentage of exceptions
            - Likelihood ratio
            - Critical value from chi-squared distribution
            - Boolean indicating whether to reject the null hypothesis

        Notes
        -----
        - The test_returns frequency must be set to daily.
        - The likelihood ratio is used to test the accuracy of the VaR model. A significant result suggests the VaR model may not be accurate at the 
        specified confidence level.
        """

        import warnings ; warnings.filterwarnings("ignore")
    
        import numpy as np ; from scipy.stats import chi2 

        from pyriskmgmt.inputsControlFunctions import (check_num_future_s, check_S0_initial_s, validate_returns_port, check_var,
                                                       check_alpha, check_alpha_test, check_warning)
        
        # check procedure
        num_future_s = check_num_future_s(num_future_s) ; S0_initial_s = check_S0_initial_s(S0_initial_s) ; test_returns = validate_returns_port(test_returns) 
        var = check_var(var) ; alpha = check_alpha(alpha) ; alpha_test = check_alpha_test(alpha_test) ; check_warning(warning) 

        if warning:
            print("WARNING:")
            print("It's crucial that the 'test_returns' are provided with a daily frequency.")
            print("This method adjusts as necessary and calculates the VaR at the expiration of the last future contract.")
            print("It computes the daily mark-to-market (MTM) P&L for each contract, adjusting them through time using the risk-free rate")
            print("and compare it with the provided Value At Risk.")
            print(" Also ensure that the 'test_returns' matches the 'num_future_s' in an element-wise way. ")
            print("To suppress this warning, set 'warning=False' in this 'kupiec_test()' method.")

        # -----------------------------------------------------------------------------------------------------------------------------------------

        # getting the sorting order of self.interval
        sort_order = np.argsort(self.summary["interval"])
        # reordering the columns of test_returns based on the sorting order
        intervals_sorted = self.summary["interval"][sort_order].values ; test_returns_sorted = test_returns[:, sort_order] 
        num_futures_s_sorted = num_future_s[sort_order] ; S0_s_sorted = self.S0_s[sort_order]  ; r_s_sorted = self.r_s[sort_order] 
        futures_prices_sorted = self.future_prices[sort_order] ; T_s_sorted = self.T_s[sort_order]

        # Now everything is sorted by interval...

        # -----------------------------------------------------------------------------------------------------------------------------------------

        # defining the function to simulate future stock prices based on provided underlying returns, interval, and initial stock price (S0)

        def FuturesMatrixSimPrices3DCreator(test_returns_ith, interval, S0):

            # determining the total number of periods from the underlying returns
            n_periods = len(test_returns_ith) ; n_extractions = n_periods // interval
            # initializing a matrix with zeros to store simulated prices
            simPrices = np.zeros((interval, n_extractions))
            # iterating over the underlying returns in steps of the provided interval
            for i in range(0, n_periods, interval):
                # determining the end of the current interval
                end = i + interval
                # breaking the loop if we exceed the number of available periods
                if end > n_periods:
                    break
                # calculating and storing simulated prices for the current interval
                simPrices[:, i // interval] = S0 * np.cumprod(1 + test_returns_ith[i: end])
            # returning the matrix with simulated prices
            return simPrices
        
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # accumulating counts for each interval.
        # for each interval, it checks how many intervals are greater than or equal to the current interval
        # and appends the total count to the list.
        counts = []
        for interval in intervals_sorted:
            count = np.sum(intervals_sorted >= interval) ; counts.append(count)
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # creating a 4D structure to store the simulated prices for each asset and each interval.

        SimPrice4D = []
        # iterating over all intervals.
        for idx in range(len(intervals_sorted)):
            # getting the number of assets we will consider for the current interval.
            NumAssetsToSelect = counts[idx]
            # generating a list of indices corresponding to all assets.
            full_indices = np.arange(test_returns.shape[1])
            # selecting the last 'NumAssetsToSelect' indices from the generated list.
            indices_to_select = full_indices[ - NumAssetsToSelect:] 
            n_extractions = test_returns_sorted.shape[0] // (max(intervals_sorted))
            # initializing a 3D list for the current interval to store simulated prices for each asset.
            SimPrice3D = np.zeros((max(intervals_sorted), n_extractions, NumAssetsToSelect))
            # iterating over the selected asset indices.
            for index, value in enumerate(indices_to_select):
                # fetching the underlying returns and initial stock price (S0) for the current asset.
                test_returns_ith = test_returns_sorted[:,value] ; S0 = S0_s_sorted[value]
                # using the FuturesMatrixSimPrices3DCreator function to simulate future stock prices for the current asset and the current interval.
                simPrices = FuturesMatrixSimPrices3DCreator(test_returns_ith, max(intervals_sorted), S0)
                # appending the simulated prices for the current asset to the 3D list.
                SimPrice3D[:,:,index] = simPrices
            # once we have simulated prices for all selected assets for the current interval, we append the 3D list to the 4D list.
            SimPrice4D.append(SimPrice3D)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        unique_arrays = [] ; seen_shapes = set()

        for array3D in SimPrice4D:
            # checking if the shape of this 3D array is not in the seen_shapes.
            if array3D.shape not in seen_shapes:
                seen_shapes.add(array3D.shape)  # adding this shape to the set.
                unique_arrays.append(array3D)   # appending this 3D array to the unique list.

        SimPrice4D = unique_arrays

        # *******************************************************************************************************

        unique_interval_sorted = np.unique(intervals_sorted)

        # Initializing an empty list to store Profit & Loss values.
        PL = []
        # Getting the maximum expiration time.
        MaxT = max(T_s_sorted)

        prevSimDays = 0
        # Iterating over each entry in SimPrice4D.
        for j, dataset3D in enumerate(SimPrice4D): 
            days_to_simulate = unique_interval_sorted[j] - prevSimDays ; dataset3DShape2 = dataset3D.shape[2] ; relevant_slice = - dataset3DShape2
            # Slicing the relevant portions based on the depth of the current dataset.
            num_futures_s_sorted_new = num_futures_s_sorted[relevant_slice:] ; futures_prices_sorted_new = futures_prices_sorted[relevant_slice:]
            r_s_sorted_new = r_s_sorted[relevant_slice:]

            # ********************************************************************************************************

            # initializing a 3D matrix of zeros with shape (days_to_simulate, dataset3D.shape[1], dataset3DShape2)
            PlMatrix3D = np.zeros((days_to_simulate, dataset3D.shape[1], dataset3DShape2))

            # looping through each slice of the third dimension of dataset3D
            for index_inner in range(dataset3DShape2):
                # extracting a 2D slice from dataset3D for the given index_inner 
                dataset2D = dataset3D[prevSimDays: prevSimDays + days_to_simulate, :, index_inner]
                # initializing a 2D matrix of zeros with the same shape as dataset2D
                PlMatrix2D = np.zeros_like(dataset2D)

                # iterating through each column of dataset2D
                for col in range(dataset2D.shape[1]):
                    # iterating through each row of the current column in dataset2D
                    for row in range(dataset2D.shape[0]):

                        # checking if the value at index_inner in num_futures_s_sorted_new is positive
                        if num_futures_s_sorted_new[index_inner] > 0:
                            # calculating the profit or loss (P&L) for the given cell in dataset2D using a specific formula
                            PlMatrix2D[row,col] = (dataset2D[row,col] - futures_prices_sorted_new[index_inner]) * num_futures_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT - ((prevSimDays + row + 1) / 252)))

                        # checking if the value at index_inner in num_futures_s_sorted_new is negative
                        elif num_futures_s_sorted_new[index_inner] < 0:

                            # calculating the profit or loss (P&L) for the given cell in dataset2D using a different formula
                            PlMatrix2D[row,col] = (futures_prices_sorted_new[index_inner] - dataset2D[row,col] ) * num_futures_s_sorted_new[index_inner]
                            PlMatrix2D[row,col] = PlMatrix2D[row,col] * np.exp(r_s_sorted_new[index_inner] * (MaxT - ((prevSimDays + row + 1) / 252)))

                # updating the 3D matrix PlMatrix3D with the values from PlMatrix2D for the current slice
                PlMatrix3D[:, :, index_inner] = PlMatrix2D

            # *******************************************************************************************************************************************

            pl = np.sum(PlMatrix3D, axis=2).ravel() ; PL.extend(pl) ; prevSimDays += days_to_simulate

       # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        PL = np.array(PL)  ; VaR = var ; num_exceptions = np.sum(PL < - VaR)
        
        # total number of observations ----------------------------------------------------------------------------------------

        num_obs = len(PL)

        # % exceptions -------------------------------------------------------------------------------------------------------

        per_exceptions = round(num_exceptions/num_obs,4)

        p = alpha

        likelihood_ratio = -2 * (np.log(((1 - p) ** (num_obs - num_exceptions)) * (p ** num_exceptions)) - np.log( ((1 - (num_exceptions / num_obs)) ** (num_obs - num_exceptions)) * ((num_exceptions / num_obs) ** num_exceptions)))
        likelihood_ratio = round(likelihood_ratio,5)

        critical_value = chi2.ppf(1 - alpha_test, df=1)

        reject_null = likelihood_ratio > critical_value

        return {
            "test" : "Kupiec",                      # Test 
            "var" : VaR,                            # Value at Risk
            "nObs" : num_obs,                       # Number of observations
            "nExc" : num_exceptions,                # Number of exceptions
            "alpha" : alpha,                        # Theoretical % of exceptions
            "actualExc%" : per_exceptions,          # Real % of exceptions
            "LR" : likelihood_ratio,                # Likelihood ratio
            "cVal" : round(critical_value, 5),      # Critical value
            "rejNull" : reject_null                 # Reject null hypothesis
        }
    
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 
    # ******************************************************************************************************************************************************************
    # ****************************************************************************************************************************************************************** 
    




            


            






















    






        







        

        


            













        

        





        





 

        