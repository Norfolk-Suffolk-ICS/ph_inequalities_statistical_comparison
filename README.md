# Repository introduction 

This repository contains a python package to automate statistical comparison of health indicators across various domains of inequalities within a snowflake environment.

Currently, the package enables statistical comparison of:
- proportions (crude and/or standardized)
- rates (crute and/or standardised)

For any other indicator types (e.g the mean of a numeric variable) further module development will be required.

Helper functions are prefixed with an `_`. Public functions (that the user should call) have no prefix appended.

## Using this package

This package can be installed from the command line by running
`pip install git+https://github.com/Norfolk-Suffolk-ICS/ph_inequalities_statistical_comparison.git`

## Other files in this repository

- User guide: [ph-inequalities-statistical-comparison-guide.md](./ph-inequalities-statistical-comparison-guide.md)
- python module: [ph_inequalities_statistical_comparison.py](./ph_inequalities_statistical_comparison.py)
- test suite for the module: [test_ph_inequalities_statistical_comparison.py](./test_ph_inequalities_statistical_comparison.py)
- details of statistical approach: [statistical-documentation.py](./statistical-documentation.py)
- snowflake integration issues: [further-issues.md](./further-issues.md)