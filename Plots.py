import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import os

# Specify the folder path
folder_path = '/Users/christianrobertson/Desktop/Senior project CSv'
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]

for file in all_files:
    # Load the data
    stock_data = pd.read_csv(os.path.join(folder_path, file), index_col='Date', parse_dates=True)

    # Set up the plot
    plt.figure(figsize=(15,10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
    x_dates = stock_data.index.date

    # Plot the data
    plt.plot(x_dates, stock_data['High'], label='High')
    plt.plot(x_dates, stock_data['Low'], label='Low')
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.title(f"Stock Data for {os.path.splitext(file)[0]}")  # Set the title to the stock's name
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()