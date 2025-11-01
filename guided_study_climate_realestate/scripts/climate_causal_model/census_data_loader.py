
from data_loader import DataLoaderRedfin
from configs import read_yaml_as_dict
from tqdm import tqdm 
import pandas as pd
import requests 
import logging 
import numpy as np 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CENSUSDATALOADER')

class DataLoaderCensus(DataLoaderRedfin):

    FEATURE_MAP = {
        'TOTAL POPULATION ESTIMATE': 'B01001_001E', 
        'POPULATION W': 'B01001A_001E',
        'POPULATION B': 'B01001B_001E',
        'POPULATION AAPI': 'B01001D_001E',
        'POPULATION H': 'B01001I_001E',
        'MEDIAN_AGE': 'B01002_001E',
        'PER CAPITA INCOME': 'B19301_001E',
        'PER CAPITA INCOME W': 'B19301A_001E',
        'PER CAPITA INCOME B': 'B19301B_001E',
        'PER CAPITA INCOME AAPI': 'B19301D_001E',
        'PER CAPITA INCOME H': 'B19301I_001E',
        'PER CAPITA INCOME A15-24': 'B19049_002E',
        'PER CAPITA INCOME A25-44': 'B19049_003E',
        'PER CAPITA INCOME A45-64': 'B19049_004E',
        'PER CAPITA INCOME A65+': 'B19049_005E',
        'TOTAL HH POP': 'B07013_001E',
        'SAME H 1 YR':'B07013_002E',
        'SAME STATE':'B07013_004E',
        'DIFFERENT STATE':'B07013_013E',
        'ABROAD': 'B07013_015E'
    }
    
    def __init__(
            self, 
            census_features=[],
            data_config = read_yaml_as_dict()
        ):

        #first do the standard workflow, initiate db connection
        super().__init__(self, data_config)

        if len(census_features) == 0: #do all features 
            self.census_features = self.FEATURE_MAP
        else:
            self.census_features = {
                i:j for i,j in self.FEATURE_MAP.items() if j in census_features
            }

        
    def retrieve_state_data_snapshot(self, state):

        logger.info(f'Retrieving Combined Dataset for State: {state}')
        assert hasattr(self, 'con')
        try: 
            census_data = self.retrieve_census_data_by_state(
                state, 
                tbl_name='census'
            )
        except Exception as e:
            print(e)
            print('Failed to generate census data due to above reason. Terminating...')
            return  
        
        logger.info('Gathering NFIP Claims/Losses data + Joining on Redfin Real Estate dataset')
        subqueryclaims_re = f'''
         with realestate_data as (
         select * from redfin_dataset where STATE_CODE = 'FL'
         )
         select 
            year, 
            zip::INT as zip, 
            STATE_CODE, 
            avg(MEDIAN_SALE_PRICE::DOUBLE) as home_price, 
            sum(claimCounts) as claimCounts, 
            sum(numEvents) as numEvents, 
            sum(totalClaimZip) as totalClaimZip, 
            sum(totalLossesZip) as totalLossesZip
         from realestate_data 
         left join nfip_claims_zip using (zip,month,year)
         where 
            MEDIAN_SALE_PRICE <> 'NA'
         group by 1,2,3
         order by zip,year
         ;
        '''

        claims_realestate = self.con.sql(subqueryclaims_re).df()

        ttl = claims_realestate.merge(
            census_data, how='inner', on=['year', 'zip']
        )

        ttl = ttl.fillna(0)

        ttl['risk_regime'] = ttl['totalLossesZip'] > 0
        
        return ttl
    
    def retrieve_census_data_by_state(self, state, tbl_name):

        zips = self.con.sql(f'''select distinct zip from redfin_dataset where STATE_CODE = '{state}' ;''').df()['zip']
        var_codes = ','.join(list(self.census_features.values()))
        headers = requests.utils.default_headers()
        headers.update(
            {
                'User-Agent': 'PostmanRuntime/7.43.4',
            }
        )

        years = np.arange(2015,2025)
        blocks_it = np.arange(0,len(zips),len(zips)//10)
        all_ = []

        for year in tqdm(years):
            for idx, _ in list(enumerate(blocks_it))[:-1]:

                i_start = blocks_it[idx]
                i_end = blocks_it[idx+1]

                zips_block = zips.loc[i_start:i_end]
                usgis_id_block = ','.join('860Z200US' + zips_block)
                url_block = f'https://api.census.gov/data/{year}/acs/acs5?get=NAME,{var_codes}&ucgid={usgis_id_block}' 

                print(url_block)

                response = requests.get(url_block, headers=headers)

                assert response.status_code == 200
    
                data = response.json() 

                values = data[1:]
                columns = ['ZCTA'] + list(self.features.keys()) + ['ucgid']
                tbl = pd.DataFrame(data=values,columns=columns)
                tbl['year'] = int(year)
                tbl['zip']=tbl['ZCTA'].str.strip('ZCTA5 ')
                tbl = tbl.drop(columns=['ucgid','ZCTA'])
                tbl = tbl.astype(np.float64)
                all_.append(tbl)

        data = pd.concat(all_)

        #option to save data as csv? 
        self.con.sql(
        f'''
        create table {tbl_name} as 
        select * from data
        ;
        '''
        )

        return data


