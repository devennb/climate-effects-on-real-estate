from setuptools import setup, find_packages

def parse_requirements(): 
    pass 

setup(
    name='climate_causal_model',
    version='0.1.9',
    packages=find_packages(),
    package_data={
        'climate_causal_model':['*','*/*.yaml']
    },
    install_requires=[
          "numpy>1.0",
          "pandas",
          "duckdb", 
          "scikit-learn",
          "geopandas", 
          "matplotlib"
    ],
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

