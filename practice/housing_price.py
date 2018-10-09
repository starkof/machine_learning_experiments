

import pandas as pd
import numpy as np

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

housing_data = pd.read_csv('data/california_housing_data.csv')
print(housing_data.head())
