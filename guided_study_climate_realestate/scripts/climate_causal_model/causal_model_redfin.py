import warnings
import pandas as pd 
import numpy as np 
from scipy.stats import ranksums
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

from climate_causal_model.data_loader import DataLoaderRedfin #adjust this later...
from climate_causal_model.census_data_loader import DataLoaderCensus
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('REDFIN MODEL')

class DBLRedfin: 

    '''
    Doubly Robust Estimation Framework via Redfin Dataset
    Eval causal effect of hurricanes/flooding disasters (via NFIP claims) on property values (Redfin), controlled for zip-level socioeconomic attributes 
    '''

    def __init__(self, dataset, causal_intervention_variable, outcome_variable, ttl_feats):
        '''
        Runs the dataloader to retrieve state-specific dataset for the model (indexed by zip-code,month,year)
        Assigns control/treatment groups via thresholding on claims/losses 
        Adds recovery window into treatment groups
        '''
        self.dataset = dataset 
        self.causal_intervention_variable = causal_intervention_variable
        self.outcome_variable = outcome_variable
        self.ttl_feats = ttl_feats 
        

    def generate_propensity_score(self, class_weight=0.5):
        '''
        Propensity Score Model (P(T|X))
        '''

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
        '''
        Propensity score correction for imbalanced samples per (@deven add paper citation here). 
        Important for high-granularity, large datasets with relatively infrequent treatments
        '''

        logger.info('Fitting the treatment model via the enhanced propensity scoring methodology')

        X = self.dataset[self.ttl_feats].values
        T = self.dataset[self.causal_intervention_variable].astype(bool)
        lr = LinearRegression()
        lr.fit(X,T)

        p = lr.predict(X)
        ps = 1 - (np.exp(scale_factor + p))/(1 + np.exp(p))

        self.dataset['propensity_score'] = ps

    def generate_report(self):
        '''
        Spits out report, you must run 'isolate_causal_effect' before invoking this method 
        '''

        assert hasattr(self, 'report')
        return self.report 
    
    def retrieve_dataset(self):
        '''
        Spits out dataset for full reference
        '''

        return self.dataset 

    def isolate_causal_effect(self, plot=False):
        '''
        Calculates the causal effect (ATE) via doubly robust estimation
        (1) calculates the propensity score model (P(T|X))
        (2) calculates outcome model for both treatment scenarios P(Y|X,T=1),P(Y|X,T=0)
        (3) combines (1) and (2) to add debiasing factor for P(Y|X,T=1) and P(Y|X,T=0), then calculates the difference btw the two
        (4) estimates ATE using a non-parametric bootstrapped sampling simulation (E[Y|X,T=1 - Y|X,T=0]) over 10000 runs
        '''
        #run the outcome model... 

        logger.info('Running Propensity Score Model')
        self.generate_enhanced_propensity_score(scale_factor=0.01)

        X = self.dataset[self.ttl_feats].values

        #outcome modeling via simple linear regression
        logger.info('Running Outcome Model')
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

        logger.info('Calculating Treatment Effects')
        self.dataset['y_0'] = y_0h 
        self.dataset['y_0_corrected'] = self.dataset['y_0'] + ((self.dataset[self.outcome_variable]-y_0h)/(1-self.dataset['propensity_score']))*(self.dataset[self.causal_intervention_variable].astype(np.int32)==0)
        self.dataset['y_1'] = y_1h 
        self.dataset['y_1_corrected'] = self.dataset['y_1'] + ((self.dataset[self.outcome_variable]-y_1h)/(self.dataset['propensity_score']))*(self.dataset[self.causal_intervention_variable].astype(np.int32)==1)
        self.dataset['treatment_effect'] = self.dataset['y_1_corrected']-self.dataset['y_0_corrected']
        self.dataset['treatment_effect_no_corr'] = self.dataset['y_1']-self.dataset['y_0']


        diffs_dbl = self.dataset['treatment_effect']
        diffs_reg_only = self.dataset['treatment_effect_no_corr']

        #create sample dist to estimate uncertainty of the ATE val, confidence intervals etc...
        logger.info('Estimating ATE Sampling Distribution and Building Output Report')
        bootstrapped_sample_dist_dbl = pd.Series([diffs_dbl.sample(n=len(diffs_dbl), replace=True).mean() for _ in np.arange(10000)])
       
        obs = diffs_dbl.mean()
        obs_reg_only = diffs_reg_only.mean()
        print(f"Average Treatment Effect, Flooding effects on house prices {self.state}: {obs}")
        CI = [
            bootstrapped_sample_dist_dbl.quantile(0.025), 
            bootstrapped_sample_dist_dbl.quantile(0.975)
        ]

        self.report = {
            'Region': self.state, 
            'ATE Estimate': obs, 
            'ATE Confidence Bounds (95 pct)': CI, 
            'P-value': np.count_nonzero(bootstrapped_sample_dist_dbl > 0)/len(bootstrapped_sample_dist_dbl),
            'Significant?': (np.count_nonzero(bootstrapped_sample_dist_dbl > 0)/len(bootstrapped_sample_dist_dbl)) < 0.06
        }

        if plot:

            plt.hist(bootstrapped_sample_dist_dbl, bins=20, label='Doubly Robust Estimation')
      
            plt.axvline(obs, label=f"Observed ATE: {obs}",color='red')

            #plt.axvline(obs_reg_only, label=f"Observed ATE [No DRL Estimation]: {obs_reg_only}",color='orange')
            plt.axvline(CI[0], label=f"95 pct CI Lower: {CI[0]}",color='red',linestyle='--')
            plt.axvline(CI[1], label=f"95 pct CI Upper: {CI[1]}",color='red',linestyle='--')
            plt.legend()
            plt.show()
    
class DBLRedfinCensus(DBLRedfin):
   
    def __init__(self, query, parent_dir, query_type='state'):
        '''
        Redfin model using Census attributes as independent variables + NFIP claims as treatment
        '''

        logger.info('Instantiating Data: Redfin-Model')

        assert query_type in ['msa','state']

        dl = DataLoaderCensus(parent_dir=parent_dir)
        self.state = query
        self.dataset = dl.retrieve_state_data_snapshot(query, query_type, table_name=f'{self.state.split(',')[0].replace(' ','')}census').drop_duplicates()
      
        self.census = list(dl.census_features.keys())

       # dataset = dataset.copy()
        feats = self.census + ['homes_sold','home_price']
        for f in feats: 
            self.dataset[f] = self.dataset[f].replace(0,0.001)
            self.dataset[f] = self.dataset.groupby(['zip'])[f].diff() \
                            / self.dataset.groupby(['zip'])[f].shift(1)
            self.dataset[f] = self.dataset[f].clip(
                upper=-3,
                lower=3
            )

        self.ttl_feats = self.census + ['homes_sold']

        #define intervention + outcome vars. change me as needed 
        self.dataset = self.dataset.dropna()
        
        self.causal_intervention_variable = 'risk_regime'
        self.outcome_variable = 'home_price'


    def pairwise_analysis(self, plot):
        '''
        Runs a pairwise analysis of regional treatment effects across growth/decay neighborhoods using a basic socioeconomic index threshold 
        Must run the causal model before invoking this method
        '''

        assert 'treatment_effect' in self.dataset.columns

        self.dataset = self.dataset[[
            'zip','PER CAPITA INCOME','TOTAL POPULATION ESTIMATE','homes_sold','treatment_effect','treatment_effect_no_corr', 
            self.outcome_variable, self.causal_intervention_variable
        ]]

        logger.info('Calculating socioeconomic partitioning index')
        self.dataset['income_decay'] = self.dataset['PER CAPITA INCOME'] < 0
        self.dataset['population_decay'] = self.dataset['TOTAL POPULATION ESTIMATE'] < 0
        self.dataset['marketdemand_decay'] = self.dataset['homes_sold'] < 0

        self.dataset['marginalized'] = self.dataset[['income_decay','population_decay','marketdemand_decay']].sum(axis=1) >= 2

        
        self.dataset.to_csv(f'zcta_results_{self.state}.csv')

        logger.info('Running pairwise simulation and generating report')
        obs_sample_marg = self.dataset.loc[self.dataset['marginalized']==True]['treatment_effect']
        ATE_marg = obs_sample_marg.mean()
        obs_sample_non_marg = self.dataset.loc[self.dataset['marginalized']==False]['treatment_effect']
        ATE_non_marg = obs_sample_non_marg.mean()

        bootstrapped_sample_marg = \
        pd.Series([obs_sample_marg.sample(n=len(obs_sample_marg), replace=True).mean() for _ in np.arange(10000)])
        bootstrapped_sample_nonmarg = \
        pd.Series([obs_sample_non_marg.sample(n=len(obs_sample_marg), replace=True).mean() for _ in np.arange(10000)])

        t_statistic,p_value = stats.ttest_ind(bootstrapped_sample_marg,bootstrapped_sample_nonmarg,equal_var=True)
        self.pairwise_report = {
            'Region': self.state,
            'ATE_disadvantaged': ATE_marg, 
            'ATE_nondisadvantaged': ATE_non_marg, 
            't_statistic': t_statistic, 
            'p_value': p_value
        }

        if plot: 
            plt.rcParams['figure.figsize']=(15,10)
            bootstrapped_sample_marg.hist(alpha=0.75,bins=25,label='Disadvantaged Communities')
            bootstrapped_sample_nonmarg.hist(alpha=0.75,bins=25,label='Growth Communities')
            plt.title(f'ATE Pairwise Sampling Distributions: {self.state}')
            plt.xlabel('ATE Sampling Estimates: Flood Disasters x Home Price')
            plt.axvline(ATE_marg,color='blue',label=f'[Disadvantaged] Estimated ATE: {ATE_marg}')
            plt.axvline(ATE_non_marg,color='red',label=f'[Growth] Estimated ATE: {ATE_non_marg}')
            plt.legend()
            plt.savefig(f'outputs/plots/{self.state}_plot.png')
            plt.show()


        return self.pairwise_report
    
#if __name__ == '__main__':

 #   states = [
  #      'Miami, FL', 'New York, NY', 
   # ]

#    outputs = []
 #   outputsALL = []
  #  pairwise = []
   # for msa in states: 
    #    dbl = DBLRedfinCensus(query=msa, query_type='msa')
     #   dd, re = dbl.isolate_causal_effect()
      #  outputs.append(re)
       # outputsALL.append(dd)

#        re_pair = dbl.pairwise_analysis(plot=True)
 #       pairwise.append(re_pair)


  

        