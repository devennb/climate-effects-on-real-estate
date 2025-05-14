
import logging 
import os
import pandas as pd 
import numpy as np 
import glob
import duckdb
import geopandas as gpd 
import requests
from configs import read_yaml_as_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DATALOADER')

class DataLoaderRedfin: 
    
    def __init__(
            self, 
            data_config = read_yaml_as_dict()
        ): 

        logger.info('Reading dataset file locations')
        datasets = data_config['datasets']
        for var_name, path in datasets.items(): 
            setattr(self, var_name, path)
        
        logger.info('Reading FRED API Endpoint + Key')
        fred_api_params = data_config['apis']
        for var_name, path in fred_api_params.items(): 
            setattr(self, var_name, path)

        logger.info('Initializing DUCKDB local database')
        try:
            self.con = self.initialize_ddb(data_config)

        except Exception as e: 
            logger.warning(e) 
            logger.warning('Could not create database connection.')
            pass 

    def retrieve_state_data_snapshot(self, state):

        logger.info(f'Retrieving Combined Dataset for State: {state}')
        assert hasattr(self, 'con')

        logger.info('Gathering NFIP Claims/Losses data + Joining on Redfin Real Estate dataset')
        subqueryclaims_re = f'''
        with realestate_data as (
        select * from redfin_dataset where STATE_CODE = '{state}'
        )
        select *
        from realestate_data 
        left join nfip_claims_zip using (zip,month,year)
        order by zip,year,month
        ;
        '''

        logger.info('Gathering Socioeconomic Context Variables')
        subquery_irs = f'''
        select * from irs_zip where STATE = '{state}'
        ;
        '''

        claims_realestate = self.con.sql(subqueryclaims_re).df()
        socioeconomic_tax_confounders = self.con.sql(subquery_irs).df() 

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

        all_data = feature_matrix.merge(
            claims_realestate.set_index('zip'), 
            how='inner', 
            on=['zip']
        )

        logger.info('Building Combined Dataset with Preliminary Treatment Assignment')
        all_data = all_data\
            .reset_index()\
            .sort_values(by=['zip','year','month'])\
            .set_index(['zip','year','month'])\
            .drop(columns=['STATE_CODE','state'])\
            .fillna(0)\
            .replace('NA',0.0)\
            .astype(np.float32)
        
        all_data['risk_regime'] = all_data['totalLossesZip'] > 0

        return all_data

        
    def retrieve_macroeconomic_data(self):

        logger.info('Retrieving macroeconomic time-series datasets')
        if hasattr(self, 'macro_data'):
            return self.macro_data 
            
        econ_dataset_codes = {
            'fed_funding_rate':'DFF', 
            '30yr_mortgage_avg':'MORTGAGE30US', 
            '15yr_mortgage_avg':'MORTGAGE15US',  
            'consumer_price_idx':'CPIAUCSL', 
            'real_disposable_income':'DSPIC96', 
            'real_disp_income_per_capita':'A229RX0'
        }

        data=[]
        for col, code in econ_dataset_codes.items(): 
            params = {
                'series_id': code, 'api_key': self.FRED_api_key, 'file_type': 'json'
            }
            try: 
                resp = requests.get(self.FRED_api_endpoint, params=params).json()
                resp_df = pd.DataFrame(resp['observations'])\
                    .drop(columns=['realtime_start','realtime_end'])\
                    .rename(columns={'value':col})
                resp_df[col] = resp_df[col].astype(float)
                resp_df['date'] = pd.to_datetime(resp_df['date'])
                resp_df['year'] = resp_df['date'].dt.year
                resp_df['month'] = resp_df['date'].dt.month 
                resp_df_monthly = resp_df\
                    .drop(columns=['date'])\
                    .groupby(['year','month'])[[col]].mean()
            
                data.append(resp_df_monthly)
            except Exception as e: 
                print(e)
                continue
                        
        cct = pd.concat(data,axis=1)\
            .sort_values(by=['year','month'])\
            .reset_index()
        cct['date'] = cct['year'].astype(str) + '-' + cct['month'].astype(str) 
        cct['date'] = pd.to_datetime(cct['date']) 
        cct = cct.set_index('date')
        self.macro_data = cct
        return cct
    

    def initialize_ddb(self, config):
        self.DUCKDB_location = config['databases']['duckdb_location_redfin']
        logger.info(f'Building local database at {self.DUCKDB_location}')
        if os.path.exists(self.DUCKDB_location): 
            return duckdb.connect(self.DUCKDB_location)
            
        base_query = '''
        drop table if exists nfip_claims
        ;

        create table nfip_claims as 
        select  
            id,
            asOfDate, 
            dateOfLoss,
            policyCount as insuredUnits, 
            baseFloodElevation, 
            ratedFloodZone, 
            occupancyType, 
            originalConstructionDate, 
            originalNBDate, 
            coalesce(amountPaidOnBuildingClaim,0) +                  
            coalesce(amountPaidOnContentsClaim,0) +                  
            coalesce(amountPaidOnIncreasedCostOfComplianceClaim,0) as totalClaim, 
            coalesce(totalBuildingInsuranceCoverage,0) + 
            coalesce(totalContentsInsuranceCoverage,0) as totalCovered, 
            coalesce(buildingDamageAmount,0) + coalesce(contentsDamageAmount,0) as totalDamageLoss, 
            buildingDeductibleCode, 
            contentsDeductibleCode,
            causeOfDamage, 
            buildingPropertyValue, 
            floodproofedIndicator, 
            floodEvent, 
            state, 
            reportedZipCode, 
            censusTract, 
            latitude, 
            longitude
        from read_csv('{claims_data_path}', strict_mode=False)
        ;

        drop table if exists nfip_claims_zip 
        ; 

        create table nfip_claims_zip as 
        select 
            reportedZipCode as zip, 
            state,
            extract('year' from dateOfLoss) as year,
            extract('month' from dateOfLoss) as month,
            count(id) as claimCounts,
            count(floodEvent) as numEvents,
            sum(insuredUnits) as policyCounts, 
            sum(totalClaim) as totalClaimZip, 
            sum(totalDamageLoss) as totalLossesZip
        from nfip_claims
        where dateOfLoss >= '2000-01-01'
        group by 1,2,3,4
        order by 5 desc
        ;

        drop table if exists irs_zip
        ;

        create table irs_zip as 
        select 
            STATE, 
            ZIPCODE as zip, 
            AGI_STUB as adjGrossIncomeTaxBracket,
            N1 as numberTaxReturns, 
            MARS1 as singleStatusTotalReturns, 
            MARS2 as marriedStatusTotalReturns, 
            MARS4 as HoHTotalReturns, 
            N2 as totalIndividuals,
            VITA as volunteerAssistedReturns, 
            ELDERLY as elderlyReturns, 
            A00100 as adjustedGrossIncome, 
            A02650 as totalIncome, 
            N00200 as returnsTotalwSalariesWages, 
            N00300 as returnsTotalTaxableInterest, 
            A00300 as taxableInterestAmt,
            SCHF   as returnsTotalFarm, 
            A18450 as stateLocalSalesTaxTotal, 
            N18500 as realEstateTaxTotal, 
            N18800 as returnsTotalPersonalPropertyTax, 
            A18800 as propertyTaxAmtTotal, 
            N19300 as returnsTotalMortgageInterestPaid, 
            A19300 as mortgageInterestPaidTotal,
            N07225 as returnsDependentCareCredit, 
            A07225 as dependentCareCreditTotal, 
            N07230 as returnsEducationCredit, 
            A07230 as educationCreditTotal, 
            N85770 as returnsPremiumsCredit, ---aids in offsetting health insurance premiums
            A85770 as premiumsCreditTotal, 
        from read_csv('{irs_data_path}', strict_mode=False)
        ;
    
        drop table if exists redfin_dataset
        ;

        create table redfin_dataset as 
        select 
            extract('year' from PERIOD_BEGIN) as year, 
            extract('month' from PERIOD_BEGIN) as month,
            substring(REGION,11,6) as zip, 
            STATE_CODE,
            MEDIAN_SALE_PRICE, 
            MEDIAN_SALE_PRICE_MOM, 
            MEDIAN_PPSF, 
            HOMES_SOLD, 
            HOMES_SOLD_MOM, 
            PENDING_SALES, 
            PENDING_SALES_MOM, 
            NEW_LISTINGS, 
            NEW_LISTINGS_MOM, 
            INVENTORY, 
            INVENTORY_MOM, 
            SOLD_ABOVE_LIST, 
            SOLD_ABOVE_LIST_MOM, 
            OFF_MARKET_IN_TWO_WEEKS, 
            OFF_MARKET_IN_TWO_WEEKS_MOM 
        from read_csv('{redfin_data_path}', header=true, sep = '\t')
        where PROPERTY_TYPE = 'Single Family Residential' 
            and PERIOD_BEGIN >= '2015-01-01'
        '''

        con = duckdb.connect(self.DUCKDB_location)
        con.sql(
            base_query.format(
                claims_data_path=self.claims_data_path,
                redfin_data_path=self.redfin_data_path, 
                irs_data_path=self.irs_data_path, 
                zip_geos_pd=self.zip_geos_pd
            )
        )
        return con
    
class DataLoaderZillow(DataLoaderRedfin):

    def retrieve_regional_data_snapshot(self, region_str, time_window_size=2):
        
        logger.info(f'Retrieving Combined Dataset for Specified Region: {region_str}')
        assert hasattr(self, 'con') 

        logger.info('Retrieving Zillow data')
        yr_upper_bound = 2020 #look at pre-covid
        yr_lower_bound = yr_upper_bound - time_window_size

        df = self.con.sql(f"select * from zillow_home_prices where Metro ilike '%{region_str}%'").df()
        cols = ['RegionName', 'Metro','CountyName'] + \
            [
                i for i in df.columns 
                if len(i.split('-')) == 3 
                    and int(i.split('-')[0]) < yr_upper_bound
                    and int(i.split('-')[0]) >= yr_lower_bound
        ]
        zillow_df = df[cols]
        zillow_df = zillow_df.rename(
            columns={
                'RegionName':'zip'
            }
        )
        
        zip_codes = zillow_df['zip'].unique()

        logger.info('Retrieving socioeconomic context variables')
        zips_str = ','.join(zip_codes)
        irs_df = self.con.sql(f'select * from irs_zip where zip in ({zips_str})').df()

        logger.info('Retrieving NFIP claims/losses dataset')
        claims_df = self.con.sql(
            f'''
             select * 
             from (select * from nfip_claims_zip where zip in ({zips_str})) nfip
             inner join zip_geos using (zip)
             where dateOfLoss between '{yr_lower_bound}-01-01' and '{yr_upper_bound}-01-01'
             order by dateOfLoss
             ;
             '''
        ).df()

        return {
            'ZILLOW' : zillow_df, 
            'IRS_SOCIO': irs_df, 
            'NFIP_CLAIMS': claims_df
        }
    
    def initialize_ddb(self, config):
        self.DUCKDB_location = config['databases']['duckdb_location_zillow']
        logger.info(f'Building local database at {self.DUCKDB_location}')
        if os.path.exists(self.DUCKDB_location): 
            return duckdb.connect(self.DUCKDB_location)
            
        base_query = '''
        drop table if exists nfip_claims
        ;

        create table nfip_claims as 
        select  
            id,
            asOfDate, 
            dateOfLoss,
            policyCount as insuredUnits, 
            baseFloodElevation, 
            ratedFloodZone, 
            occupancyType, 
            originalConstructionDate, 
            originalNBDate, 
            coalesce(amountPaidOnBuildingClaim,0) +                  
            coalesce(amountPaidOnContentsClaim,0) +                  
            coalesce(amountPaidOnIncreasedCostOfComplianceClaim,0) as totalClaim, 
            coalesce(totalBuildingInsuranceCoverage,0) + 
            coalesce(totalContentsInsuranceCoverage,0) as totalCovered, 
            coalesce(buildingDamageAmount,0) + coalesce(contentsDamageAmount,0) as totalDamageLoss, 
            buildingDeductibleCode, 
            contentsDeductibleCode,
            causeOfDamage, 
            buildingPropertyValue, 
            floodproofedIndicator, 
            floodEvent, 
            state, 
            reportedZipCode, 
            censusTract, 
            latitude, 
            longitude
        from read_csv('{claims_data_path}', strict_mode=False)
        ;

        drop table if exists nfip_claims_zip 
        ; 

        create table nfip_claims_zip as 
        select 
            reportedZipCode as zip, 
            state,
            floodEvent,
            dateOfLoss,
            count(id) as claimCounts,
            sum(insuredUnits) as policyCounts, 
            sum(totalClaim) as totalClaimZip, 
            sum(totalDamageLoss) as totalLossesZip
        from nfip_claims
        group by 1,2,3,4
        order by 5 desc
        ;

        drop table if exists irs_zip
        ;
    
        create table irs_zip as 
        select 
            STATE, 
            ZIPCODE as zip, 
            AGI_STUB as adjGrossIncomeTaxBracket,
            N1 as numberTaxReturns, 
            MARS1 as singleStatusTotalReturns, 
            MARS2 as marriedStatusTotalReturns, 
            MARS4 as HoHTotalReturns, 
            N2 as totalIndividuals,
            VITA as volunteerAssistedReturns, 
            ELDERLY as elderlyReturns, 
            A00100 as adjustedGrossIncome, 
            A02650 as totalIncome, 
            N00200 as returnsTotalwSalariesWages, 
            N00300 as returnsTotalTaxableInterest, 
            A00300 as taxableInterestAmt,
            SCHF   as returnsTotalFarm, 
            A18450 as stateLocalSalesTaxTotal, 
            N18500 as realEstateTaxTotal, 
            N18800 as returnsTotalPersonalPropertyTax, 
            A18800 as propertyTaxAmtTotal, 
            N19300 as returnsTotalMortgageInterestPaid, 
            A19300 as mortgageInterestPaidTotal,
            N07225 as returnsDependentCareCredit, 
            A07225 as dependentCareCreditTotal, 
            N07230 as returnsEducationCredit, 
            A07230 as educationCreditTotal, 
            N85770 as returnsPremiumsCredit, ---aids in offsetting health insurance premiums
            A85770 as premiumsCreditTotal, 
        from read_csv('{irs_data_path}', strict_mode=False)
        ;
    
        drop table if exists zillow_home_prices
        ;

        create table zillow_home_prices as 
        select * 
        from read_csv('{zillow_data_path}', strict_mode=False)
        ;

        drop table if exists zip_geos
        ;

        create table zip_geos as 
        select * 
        from {zip_geos_pd}
        ;
        '''

        self.logger.info('Reading in Census Geospatial Parameters')
        read_jsons = [
            gpd.read_file(gj)[['ZCTA5CE10','geometry','ALAND10','AWATER10']] 
            for gj in glob.glob(self.ZCTA_geo_datapath)
        ]
        json_to_dfs = pd.concat(read_jsons)
        json_to_dfs = json_to_dfs.rename(
                columns={
                    'ZCTA5CE10':'zip', 
                    'ALAND10':'landAreaTotal', 
                    'AWATER10':'waterAreaTotal'
                }
            )
        con = duckdb.connect(self.DUCKDB_location)
        con.sql(
            base_query.format(
                claims_data_path=self.claims_data_path,
                zillow_data_path=self.zillow_data_path, 
                irs_data_path=self.irs_data_path, 
                zip_geos_pd=self.zip_geos_pd
            )
        )
        return con
    



    

