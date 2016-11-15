import wishbone

# Plotting and miscellaneous imports
import os
# Load sample data
scdata = wishbone.wb.SCData.from_fcs(os.path.expanduser('~/.wishbone/data/sample_masscyt.fcs'), cofactor=None)
# subsample
data = scdata.data  # .loc[::12, :]
print('Data shape', data.shape)

data.to_csv('masscytof.csv')
print('data processed - head\n', data.head())

# run wishbone notebook to get data