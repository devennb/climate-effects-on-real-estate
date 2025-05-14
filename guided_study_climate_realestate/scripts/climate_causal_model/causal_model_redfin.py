import warnings
import pandas as pd 
import numpy as np 
import geopandas as gpd 
from scipy.stats import ranksums
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression, Lasso

import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

from data_loader import DataLoaderRedfin #adjust this later...
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('REDFIN MODEL')

class DBLRedfin: 
    def __init__(self, state, window_size=1):

        logger.info('Instantiating Data: Redfin-Model')
        dl = DataLoaderRedfin()
        self.state = state
        self.dataset = dl.retrieve_state_data_snapshot(state=state)
        self.dataset = self.dataset.replace('NA',0.0).astype(np.float32)

        logger.info('Assign Treatment Groups')
        self.dataset['risk_regime'] = self.dataset['totalLossesZip'] > 10000
        self.dataset = self.dataset.loc[self.dataset['INVENTORY']>20]

        logger.info(f'Setting recovery window: {window_size} post-disaster')
        self.risk_window = window_size
        
        if self.risk_window > 1:
            mask = pd.concat([self.dataset.reset_index().groupby(['zip'])['risk_regime'].shift(i).bfill() for i in range(self.risk_window)], axis=1)
            mask.index = self.dataset.index
            self.dataset['risk_regime'] = mask.any(axis=1)          
  
        irs_tax_return_feats = list(self.dataset.columns[:10])
        housing_market_feats = [
            'HOMES_SOLD',
            'PENDING_SALES',
            'NEW_LISTINGS',
            'INVENTORY',
            'SOLD_ABOVE_LIST',
            'OFF_MARKET_IN_TWO_WEEKS'
        ]

        self.ttl_feats = irs_tax_return_feats + housing_market_feats
        logger.info(f"Model Features: {','.join([str(i) for i in self.ttl_feats])}")

        self.causal_intervention_variable = 'risk_regime'
        self.outcome_variable = 'MEDIAN_SALE_PRICE_MOM'
        logger.info(f'Outcome Variable: {self.outcome_variable}')

    def generate_propensity_score(self, class_weight=0.5):
        logger.info('Fitting the treatment model via a logistic regression')
        X = self.dataset[self.ttl_feats].values
        T = self.dataset[self.causal_intervention_variable].astype(bool)
        lr = LogisticRegression(
            class_weight={True:class_weight,False:1-class_weight}, 
            max_iter=1000
        ) 
        lr.fit(X,T)

        self.dataset['propensity_score'] = lr.predict_proba(X)[:,1]

    def generate_enhanced_propensity_score(self, scale_factor): 
        logger.info('Fitting the treatment model via the enhanced propensity scoring methodology')
        X = self.dataset[self.ttl_feats].values
        T = self.dataset[self.causal_intervention_variable].astype(bool)
        lr = LinearRegression()
        lr.fit(X,T)

        p = lr.predict(X)
        ps = 1 - (np.exp(scale_factor + p))/(1 + np.exp(p))

        self.dataset['propensity_score'] = ps

    def isolate_causal_effect(self, plot=False):

        #run the outcome model... 

        logger.info('Fitting the Outcome Models')
        self.generate_enhanced_propensity_score(scale_factor=0.01)

        X = self.dataset[self.ttl_feats].values

        ttl_dataset_0 = self.dataset.loc[self.dataset[self.causal_intervention_variable]==False].drop(columns=[self.causal_intervention_variable])
        X_0, y_0 = ttl_dataset_0[self.ttl_feats].values, ttl_dataset_0[self.outcome_variable]
        y_0m = LinearRegression() 
        y_0m.fit(X_0, y_0)
        y_0h = y_0m.predict(X)


        ttl_dataset_1 = self.dataset.loc[self.dataset[self.causal_intervention_variable]==True].drop(columns=[self.causal_intervention_variable])
        X_1, y_1 = ttl_dataset_1[self.ttl_feats].values, ttl_dataset_1[self.outcome_variable]
        y_1m =  LinearRegression() 
        y_1m.fit(X_1, y_1)
        y_1h = y_1m.predict(X)

        logger.info('Apply the debiasing correction. Calculating the treatment effects...')
        self.dataset['y_0'] = y_0h 
        self.dataset['y_0_corrected'] = self.dataset['y_0'] + ((self.dataset[self.outcome_variable]-y_0h)/(1-self.dataset['propensity_score']))*(self.dataset[self.causal_intervention_variable].astype(np.int32)==0)
        self.dataset['y_1'] = y_1h 
        self.dataset['y_1_corrected'] = self.dataset['y_1'] + ((self.dataset[self.outcome_variable]-y_1h)/(self.dataset['propensity_score']))*(self.dataset[self.causal_intervention_variable].astype(np.int32)==1)
        self.dataset['treatment_effect'] = self.dataset['y_1_corrected']-self.dataset['y_0_corrected']
        self.dataset['treatment_effect_no_corr'] = self.dataset['y_1']-self.dataset['y_0']

        logger.info('Generate ATE sampling distribution')
        diffs_dbl = self.dataset['treatment_effect']
        diffs_reg_only = self.dataset['treatment_effect_no_corr']
        bootstrapped_sample_dist_dbl = pd.Series([diffs_dbl.sample(n=len(diffs_dbl), replace=True).mean() for _ in np.arange(10000)])
       
        obs = diffs_dbl.mean()
        obs_reg_only = diffs_reg_only.mean()
        print(f"Average Treatment Effect, Flooding effects on house prices {self.state}: {obs}")
        CI = [
            bootstrapped_sample_dist_dbl.quantile(0.025), 
            bootstrapped_sample_dist_dbl.quantile(0.975)
        ]

        logger.info('Building output distribution report')
        self.report = {
            'State': self.state, 
            'Recovery Window': self.risk_window,
            'ATE Estimate': obs, 
            'ATE Confidence Bounds (95 pct)': CI, 
            'P-value': np.count_nonzero(bootstrapped_sample_dist_dbl > 0)/len(bootstrapped_sample_dist_dbl),
            'Significant?': (np.count_nonzero(bootstrapped_sample_dist_dbl > 0)/len(bootstrapped_sample_dist_dbl)) < 0.06
        }

        if plot:

            plt.hist(bootstrapped_sample_dist_dbl, bins=50, label='Doubly Robust Estimation')
      
            plt.title(f'ATE Estimation (Window Size={self.risk_window})')

            plt.axvline(obs, label=f"Observed ATE: {obs}",color='red')

            #plt.axvline(obs_reg_only, label=f"Observed ATE [No DRL Estimation]: {obs_reg_only}",color='orange')
            plt.axvline(CI[0], label=f"95 pct CI Lower: {CI[0]}",color='red',linestyle='--')
            plt.axvline(CI[1], label=f"95 pct CI Upper: {CI[1]}",color='red',linestyle='--')
            plt.legend()
            plt.show()

        return self.dataset, self.report
    
if __name__ == '__main__':
     dbl = DBLRedfin(state='LA', window_size=6)
     _, re = dbl.isolate_causal_effect()
     print(re)

        