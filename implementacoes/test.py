import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import datetime  # Import the datetime module

# Create a date range
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 12, 31)
date_range = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date - start_date).days)]

# Create fictional data
data = {
    'Date': date_range,
    'Open': np.random.uniform(1000, 2000, len(date_range)),
    'Close': np.random.uniform(900, 1900, len(date_range))
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Convert Date column to numeric value
df['Date'] = mdates.date2num(df['Date'])

# Add regression line to plot
coefficients_open = np.polyfit(df['Date'], df['Open'], 1)
p_open = np.poly1d(coefficients_open)

coefficients_close = np.polyfit(df['Date'], df['Close'], 1)
p_close = np.poly1d(coefficients_close)

fig, ax = plt.subplots()
ax.plot(df['Date'], df['Open'], '.', label='Open Price')
ax.plot(df['Date'], p_open(df['Date']), '-', label='Open Regression Line')
ax.plot(df['Date'], df['Close'], '.', label='Close Price')
ax.plot(df['Date'], p_close(df['Date']), '-', label='Close Regression Line')
ax.set_title('DIJA Stock Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()

# Format x-axis labels as dates
date_form = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.gcf().autofmt_xdate()

plt.show()