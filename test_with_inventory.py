import numpy as np
import pandas as pd
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats import zscore

def find_sub_series(pattern, series, window):
    len_p = len(pattern)
    len_s = len(series)
    print(len_p, len_s, window)
    if len_p > len_s and window > len_s:
        print("Length of series data is smaller than pattern's length")
        return -1, -1, -1
    min_distance = sys.maxsize
    min_path = []
    min_step = 0
    win_size = min(len(pattern), window)
    for step in range(len_s - win_size):
        sub_series = series[step:step+win_size]

        distance, path = fastdtw(pattern, sub_series, dist=euclidean)
        if distance < min_distance:
            # print(step)
            min_distance = distance
            min_path = path
            min_step = step
    return min_step, min_distance, min_path

def resampling_dataframe(df, date_column, date_format, value_column):
    # parsing datetime format
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    # reset index as Date
    df = df.set_index([date_column])
    # z score normalization
    # df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df[value_column] = (df[value_column] - df[value_column].mean())/df[value_column].std(ddof=0)
    # df[value_column] = zscore(df[value_column])
    # add missed date using resampling
    df = df.resample('D').mean().reset_index().rename(columns={pattern_df.index.name:date_column})
    # interpolation
    df = df.interpolate(method='linear')
    return df


# read dataframes
inventory_df = pd.read_csv('inventory1.csv', header=0).dropna()
pattern_df = pd.read_csv('pattern.csv', header=0).dropna()
# pattern_df = resampling_dataframe(pattern_df, 'Date', "%m/%d/%Y", "CL")
# inventory_df = resampling_dataframe(inventory_df, 'SeriesDate', "%m/%d/%Y", "Inventory")

print(len(inventory_df.index))
print(len(pattern_df.index))

print(inventory_df.head(30))
print(pattern_df.head(30))

pattern_list = []
for i, x in enumerate(pattern_df.to_records(index=False)):
    pattern_list.append((i, x[1]))

inventory_list = []
series_date_list = []
for i, x in enumerate(inventory_df.to_records(index=False)):
    series_date_list.append(x[0])
    inventory_list.append((i, x[1]))
p_x, p_y = zip(*pattern_list)
i_x, i_y = zip(*inventory_list)

# plt.plot(p_x, p_y)
plt.plot(i_x, i_y)
# plt.show()
# pattern_list = [tuple(i, x[1]) for i, x in enumerate(pattern_df.to_records(index=False))]

window_size = len(pattern_list)
real_window_size = min(len(pattern_list), window_size)
step, distance, path = find_sub_series(pattern_list, inventory_list, window_size)
print('matched date is :', series_date_list[step])
sub_sin2=plt.plot(i_x[step:step+real_window_size],i_y[step:step+real_window_size])
plt.setp(sub_sin2,color='g',linewidth=2.0)

sub_sin1=plt.plot(i_x[step:step+real_window_size],p_y[0:real_window_size])
plt.setp(sub_sin1,color='r',linewidth=2.0)
plt.xlabel('from ' + series_date_list[step] + ' to ' + series_date_list[step + window_size-1], fontsize=18)
plt.show()

