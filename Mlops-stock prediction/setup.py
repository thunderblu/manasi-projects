from setuptools import setup, find_packages

setup(
    name="stock_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'yfinance',
        'pandas',
        'numpy',
        'scikit-learn',
        'pytest',
        'pytest-cov'
    ]
) 