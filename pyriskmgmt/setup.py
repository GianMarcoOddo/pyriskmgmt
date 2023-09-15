# ------------------------------------------------------------------------------------------
# Author: GIAN MARCO ODDO
# Module: setup.py
#     - Part of the pyriskmgmt package.
# Contact Information:
#   - Email: gian.marco.oddo@usi.ch
#   - LinkedIn: https://www.linkedin.com/in/gian-marco-oddo-8a6b4b207/
#   - GitHub: https://github.com/GianMarcoOddo
# Feel free to reach out for any questions or further clarification on this code.
# ------------------------------------------------------------------------------------------

from setuptools import setup, find_packages
import pathlib

# Reading the content of the README.md file
here = pathlib.Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="pyriskmgmt",
    version="1.1.3",
    packages=find_packages(),
    install_requires=[
        'yfinance',
        'numpy',
        'pandas',
        'scipy',
        'rpy2',
        'requests',
        'QuantLib',
        'arch',
        'joblib',
        'bs4',
        'seaborn',
        'matplotlib'],
    author="Gian Marco Oddo",
    author_email="gian.marco.oddo@usi.ch",
    description="The pyriskmgmt package is designed to offer a straightforward but comprehensive platform for risk assessment, targeting the calculation of Value at Risk (VaR) and Expected Shortfall (ES) across various financial instruments. While providing a solid foundation, the package also allows for more specialized development to meet users' specific investment strategies and risk requirements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GianMarcoOddo/pyriskmgmt",
    license="MIT",
)
