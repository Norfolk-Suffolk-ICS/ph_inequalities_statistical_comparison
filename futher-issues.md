# Integration with snowflake as a stored procedure

The dependencies for this module are `polars`, `scipy`, `numpy`, and `pandas`. They are confirmed as available on the Snowflake Snowpark for Python library list: https://repo.anaconda.com/pkgs/snowflake/

Use of these packages requires a named stored procedure rather than an anonymous one.

# Why polars is the main dataframe engine

It could be expected that one of pandas, polars or snowpark could have been used to develop this module.

However, the wrangling and manipulation required in this module does not lend itself to pure snowpark.

Pandas or polars are sufficiently flexible to conduct the analysis. However, polars is a faster, multi-threaded and more memory efficient data wrangling solution. It is also more readable and so easier to debug.

# Possible additional wrappers to make a stored procedure.

Ingress of data will likely need a chained `snowpark.to_arrow` to `polars.from_arrow` wrapper function.

Unfortunately, egress will need to go via pandas (the only reason it is a dependency) with something like a chained `polars.to_pandas` and `session.create_dataframe`