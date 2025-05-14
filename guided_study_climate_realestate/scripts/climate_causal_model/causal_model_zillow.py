import random 
import os
import pandas as pd 
import numpy as np 
import glob
import duckdb
import geopandas as gpd 
from scipy.stats import ranksums
from sklearn.linear_model import LogisticRegression,LinearRegression,Lasso
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
import warnings 
import logging

from data_loader import DataLoaderZillow #adjust this later...

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ZILLOW MODEL')

class DBLZillow: 
    
    def __init__(self, state, time_window): 
        
        logger.info('Instantiating Data: Zillow-Model')
        self.dataloader = DataLoaderZillow()
        self.state = state 
        self.n = time_window

        #find a better way to do this...prevents db locking from occurring
        if not hasattr(self.dataloader, 'con'):
            self.dataloader.initialize_ddb()
        else: 
            self.dataloader.con = duckdb.connect(self.dataloader.DUCKDB_location)
    
        datasets = self.dataloader.retrieve_regional_data_snapshot(f', {state}', time_window_size=self.n)
       # self.dataloader.con.close()
        for n, df in datasets.items():
            logger.info(f'Instantiating dataset {n} into memory')
            setattr(self, n, df)

        self.macroeconomic_indicators = self.dataloader.retrieve_macroeconomic_data() 

    def generate_real_estate_shiftindex(self):

        logger.info('Building real-estate shift index (outcome variable)')
        data_hp = self.ZILLOW.set_index('zip').drop(columns=['Metro','CountyName']).T.bfill()

        data_hp['year'] = [int(i.split('-')[0]) for i in data_hp.index]
        data_grp_yr = data_hp.groupby(['year']).mean()

        logger.info('Apply CPI correction')
        cpi = self.macroeconomic_indicators\
                 .loc[self.macroeconomic_indicators['year'].between(data_hp['year'].min(),data_hp['year'].max())][['consumer_price_idx']]\
                 .dropna()\
                 .iloc[1:-1,:]

        cpi['year'] = [i.year for i in cpi.index]

        cpi_grp = cpi.groupby(['year']).mean()
        cpi_idx = cpi_grp.iloc[0]/cpi_grp
        cpi_corrected = data_grp_yr.mul(cpi_idx['consumer_price_idx'].to_numpy(),axis=0)

        return ((cpi_corrected.iloc[-1]/cpi_corrected.iloc[0])-1)*100
    
    def build_features(self):

        logger.info('Building Features')
        socioeconomic_tax_confounders = self.IRS_SOCIO.copy()
        num_rets_feats = [
            'numberTaxReturns',
            'elderlyReturns', 
            'returnsTotalwSalariesWages',
            'returnsDependentCareCredit', 
            'returnsEducationCredit'
        ] 
        piv=socioeconomic_tax_confounders.pivot(
            index='zip', columns='adjGrossIncomeTaxBracket', values='numberTaxReturns'
        )
        piv_norm = piv.div(piv.sum(axis=1),axis=0)
        grp = socioeconomic_tax_confounders.groupby('zip')[num_rets_feats].sum()
        grp = grp.div(grp['numberTaxReturns'],axis=0).drop(columns=['numberTaxReturns'])

        feature_matrix = pd.concat((piv_norm,grp),axis=1)

        risk_variables = self.NFIP_CLAIMS.copy()

        risk_variables['waterAreaProp'] = risk_variables['waterAreaTotal'] / (risk_variables['landAreaTotal']+risk_variables['waterAreaTotal'])
        zip_ = \
            risk_variables.groupby(['zip']).agg(
                numberEvents=pd.NamedAgg(column='floodEvent',aggfunc='count'), 
                totalClaims=pd.NamedAgg(column='totalClaimZip',aggfunc='sum'), 
                totalClaimCounts=pd.NamedAgg(column='claimCounts',aggfunc='sum'),
                totalLosses=pd.NamedAgg(column='totalLossesZip',aggfunc='sum'), 
                waterAreaProp=pd.NamedAgg(column='waterAreaProp',aggfunc='max')
        )

        zip_['log(totalClaims)'] = np.log(zip_['totalClaims']+0.000001) 
        zip_['log(totalLosses)'] = np.log(zip_['totalLosses']+0.000001) 
        
        zip_ = zip_.drop(columns=['totalClaims','totalLosses']) 

        rfs = [
            'numberEvents', 
            'totalClaimCounts',
            'log(totalLosses)', 
        ]

        rf_matrix = zip_[rfs].values
        
        logger.info(f"Calculating Standardized Risk Index with Variables: {','.join(rfs)}")
        rf_matrix_st = StandardScaler().fit_transform(rf_matrix).mean(axis=1)
        zip_['risk_score'] = rf_matrix_st
        zip_['risk_group'] = rf_matrix_st > rf_matrix_st.mean()
        self.RISK_VARIABLES = zip_
        self.FEATURES = pd.concat((feature_matrix, zip_[['risk_group']]),axis=1).dropna()
        
        return self.FEATURES
    
    def calculate_propensity_scores(self, plot=False): 

        logger.info('Fitting the treatment model via a logistic regression')

        assert hasattr(self, 'FEATURES')
        prop_score_lr = LogisticRegression() 

        X = self.FEATURES.iloc[:,:-1].values
        y = self.FEATURES['risk_group'].astype(bool)

        prop_score_lr.fit(X,y) 
        self.FEATURES['propensity_score'] = prop_score_lr.predict_proba(X)[:,1]

        if plot: 
            self.FEATURES.loc[self.FEATURES['risk_group']]['propensity_score'].hist(alpha=0.5,label='risky',bins=25)
            self.FEATURES.loc[self.FEATURES['risk_group']==False]['propensity_score'].hist(alpha=0.5,label='non-risky',bins=25)
            plt.title('Propensity Score Distributions')
            plt.legend()
            plt.show()

        return prop_score_lr, self.FEATURES
    
    def run_DRL(self):
        
        self.discount_factor = None
        self.calculate_propensity_scores() 
        
        home_pr_indices = self.generate_real_estate_shiftindex()
        home_pr_indices.name = 'price_shift'
        
        feature_matrix = self.FEATURES.copy() 

        logger.info('Fitting the Outcome Models')
        ttl_dataset = pd.concat((feature_matrix, home_pr_indices),axis=1).dropna()
        ttl_dataset_0 = ttl_dataset.loc[ttl_dataset['risk_group']==False].drop(columns=['risk_group'])
        X_0, y_0 = ttl_dataset_0.iloc[:, :-1].values, ttl_dataset_0['price_shift']
        y_0m = Lasso(alpha=0.01)
        y_0m.fit(X_0, y_0)
        y_0h = y_0m.predict(ttl_dataset.iloc[:,:-2].values)


        ttl_dataset_1 = ttl_dataset.loc[ttl_dataset['risk_group']==True].drop(columns=['risk_group'])
        X_1, y_1 = ttl_dataset_1.iloc[:, :-1].values, ttl_dataset_1['price_shift']
        y_1m = Lasso(alpha=0.01)
        y_1m.fit(X_1, y_1)
        y_1h = y_1m.predict(ttl_dataset.iloc[:,:-2].values)
  
        logger.info('Apply the debiasing correction. Calculating the treatment effects...')
        ttl_dataset['y_0'] = y_0h 
        ttl_dataset['y_0_corrected'] = ttl_dataset['y_0'] + ((ttl_dataset['price_shift']-y_0h)/(1-ttl_dataset['propensity_score']))*(ttl_dataset['risk_group'].astype(np.int32)==0)
        ttl_dataset['y_1'] = y_1h 
        ttl_dataset['y_1_corrected'] = ttl_dataset['y_1'] + ((ttl_dataset['price_shift']-y_1h)/ttl_dataset['propensity_score'])*(ttl_dataset['risk_group'].astype(np.int32)==1)
        ttl_dataset['treatment_effect'] = ttl_dataset['y_1_corrected']-ttl_dataset['y_0_corrected']
        ttl_dataset['treatment_effect_no_corr'] = ttl_dataset['y_1']-ttl_dataset['y_0']

        self.DRL_result = ttl_dataset
        
        return ttl_dataset

    def build_treatment_effect_report(self, plot=False): 
        
        logger.info('Generate ATE sampling distribution')

        assert hasattr(self, 'DRL_result')
        
        diffs = self.DRL_result['treatment_effect']
        
        bootstrapped_sample_dist = pd.Series([diffs.sample(n=len(diffs), replace=True).mean() for _ in np.arange(10000)])
        obs = diffs.mean()
        print(f"Average Treatment Effect, Flooding effects on house prices {self.state}: {obs}")
        CI = [
            bootstrapped_sample_dist.quantile(0.025), 
            bootstrapped_sample_dist.quantile(0.975)
        ]

        logger.info('Building output distribution report')
        self.report = {
            'State': self.state, 
            'Discount Factor': self.discount_factor,
            'ATE Estimate': obs, 
            'ATE Confidence Bounds (95 pct)': CI, 
            'P-value': np.count_nonzero(bootstrapped_sample_dist > 0)/len(bootstrapped_sample_dist),
            'Significant?': (np.count_nonzero(bootstrapped_sample_dist > 0)/len(bootstrapped_sample_dist)) < 0.06
        }

        if plot: 
            plt.hist(bootstrapped_sample_dist, bins=50,)
            plt.title('ATE Estimation')

            plt.axvline(diffs.mean(), label=f"Observed ATE: {diffs.mean()}",color='red')
            plt.axvline(CI[0], label=f"95 pct CI Lower: {CI[0]}",color='red',linestyle='--')
            plt.axvline(CI[1], label=f"95 pct CI Upper: {CI[1]}",color='red',linestyle='--')
            plt.legend()
            plt.show()
        
        return self.DRL_result, self.report
    