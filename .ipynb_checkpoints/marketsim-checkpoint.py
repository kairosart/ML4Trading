"""Market simulator that processes a dataframe instead of a csv file"""

import pandas as pd
import numpy as np
import datetime as dt
#import matplotlib.pyplot as plt
from analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data
from util import normalize_data, fetchOnlineData


# Add plotly for interactive charts
from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

def compute_portvals_single_symbol(df_orders, symbol, start_val=1000000, 
    commission=9.95, impact=0.005):
    """Compute portfolio values for a single symbol.

    Parameters:
    df_orders: A dataframe with orders for buying or selling stocks. There is
    no value for cash (i.e. 0).
    symbol: The stock symbol whose portfolio values need to be computed
    start_val: The starting value of the portfolio (initial cash available)
    commission: The fixed amount in dollars charged for each transaction
    impact: The amount the price moves against the trader compared to the 
    historical data at each transaction
    
    Returns:
    portvals: A dataframe with one column containing the value of the portfolio
    for each trading day
    """

    # Sort the orders dataframe by date
    df_orders.sort_index(ascending=True, inplace=True)
    
    # Convert Shares column to float
    df_orders['Shares'] = df_orders['Shares'].astype(float)

    # Get the start and end dates
    start_date = df_orders.index.min()
    end_date = df_orders.index.max()

    # Create a dataframe with adjusted close prices for the symbol and for cash
    df_prices = fetchOnlineData(start_date, end_date, symbol)
    
    # Convert Adj Close column to float
    df_prices['Adj Close'] = df_prices['Adj Close'].astype(float)
    df_prices["cash"] = 1.0
    
    # Fill NAN values if any
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    df_prices.fillna(1.0, inplace=True)
    
    # Create a dataframe that represents changes in the number of shares by day
    df_trades = df_prices.copy()
    # Delete Adj Close column
    del df_trades['Adj Close']
    # Rename column symbol
    df_trades.rename(columns={'Symbol':symbol}, inplace=True)
    # Set to 0 all values of symbol column
    df_trades[symbol] = 0.0
    
    # Create a dataframe that represents on each particular day how much of
    # each asset in the portfolio
    df_holdings = df_trades.copy()
    

    
    for index, row in df_orders.iterrows():
        # Total value of shares purchased or sold
        price = df_prices.loc[index, "Adj Close"]
        shares =  row["Shares"]
        traded_share_value = price * shares
        
        # Transaction cost 
        transaction_cost = float(commission) + float(impact) * traded_share_value

        # Update the number of shares and cash based on the type of transaction
        # Note: The same asset may be traded more than once on a particular day
        # If the shares were bought
        if row["Shares"] > 0:
            df_trades.loc[index, "Adj Close"] = price + shares
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
        # If the shares were sold
        elif row["Shares"] < 0:
            df_trades.loc[index, "Adj Close"] = price + shares
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
    
    
    

    for row_count in range(len(df_holdings)):
        # In the first row, the number shares are the same as in df_trades, 
        # but start_val must be added to cash
        if row_count == 0:
            df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1].copy()
            df_holdings.iloc[0, -1] = df_trades.iloc[0, -1] + float(start_val)
        # The rest of the rows show cumulative values
        else:
            df_holdings.iloc[row_count] = df_holdings.iloc[row_count-1] \
                                            + df_trades.iloc[row_count]
        row_count += 1

    # Create a dataframe that represents the monetary value of each asset 
    df_value = df_prices * df_holdings
    
    # Create portvals dataframe
    portvals = pd.DataFrame(df_value.sum(axis=1), df_value.index, ["port_val"])
    return portvals

def market_simulator(df_orders, df_orders_benchmark, symbol, start_val=1000000, commission=9.95, 
    impact=0.005, daily_rf=0.0, samples_per_year=252.0, vertical_lines=False, title="Title", xtitle="X title", ytitle="Y title"):
    """
    This function takes in and executes trades from orders dataframes
    Parameters:
    df_orders: A dataframe that contains portfolio orders
    df_orders_benchmark: A dataframe that contains benchmark orders
    start_val: The starting cash in dollars
    commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    impact: The amount the price moves against the trader compared to the historical data at each transaction
    daily_rf: Daily risk-free rate, assuming it does not change
    samples_per_year: Sampling frequency per year
    vertical_lines: Showing vertical lines for buy and sell orders
    title: Chart title
    xtitle: Chart X axe title
    ytitle: Chart Y axe title
    
    Returns:
    Print out final portfolio value of the portfolio, as well as Sharpe ratio, 
    cumulative return, average daily return and standard deviation of the portfolio and Benchmark.
    Plot a chart of the portfolio and benchmark performances
    """
    
    # Process portfolio orders
    portvals = compute_portvals_single_symbol(df_orders=df_orders, symbol=symbol,
        start_val=start_val, commission=commission, impact=impact)

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals,
     daily_rf=daily_rf, samples_per_year=samples_per_year)
    
    # Process benchmark orders
    portvals_bm = compute_portvals_single_symbol(df_orders=df_orders_benchmark, 
        symbol=symbol, start_val=start_val, commission=commission, impact=impact)
    
    # Get benchmark stats
    cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm = get_portfolio_stats(portvals_bm,
     daily_rf=daily_rf, samples_per_year=samples_per_year)

    # Get Final values
    final_value = portvals.iloc[-1, -1]
    final_value_bm = portvals_bm.iloc[-1, -1]
    
    # Get orders number
    orders_count = len(portvals.index)
    
    # Compare portfolio against Benchmark
    print ("Sharpe Ratio of Portfolio: {}".format(sharpe_ratio))
    print ("Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_bm))
    print ()
    print ("Cumulative Return of Portfolio: {}".format(cum_ret))
    print ("Cumulative Return of Benchmark : {}".format(cum_ret_bm))
    print ()
    print ("Standard Deviation of Portfolio: {}".format(std_daily_ret))
    print ("Standard Deviation of Benchmark : {}".format(std_daily_ret_bm))
    print ()
    print ("Average Daily Return of Portfolio: {}".format(avg_daily_ret))
    print ("Average Daily Return of Benchmark : {}".format(avg_daily_ret_bm))
    print ()
    print ("Final Portfolio Value: {}".format(portvals.iloc[-1, -1]))
    print ("Final Benchmark Value: {}".format(portvals_bm.iloc[-1, -1]))
    print ()
    print ("Portfolio Orders count: {}".format(len(portvals.index)))
    

    # Rename columns and normalize data to the first date of the date range
    portvals.rename(columns={"port_val": "Portfolio"}, inplace=True)
    portvals_bm.rename(columns={"port_val": "Benchmark"}, inplace=True)
    
    plot_norm_data_vertical_lines(df_orders, portvals, portvals_bm, vertical_lines, title, xtitle, ytitle)
    
    return orders_count, sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, final_value

def plot_norm_data_vertical_lines(df_orders, portvals, portvals_bm, vert_lines=False, title="Title", xtitle="X title", ytitle="Y title"):
    """Plots portvals and portvals_bm, showing vertical lines for buy and sell orders
    
    Parameters:
    df_orders: A dataframe that contains portfolio orders
    portvals: A dataframe with one column containing daily portfolio value
    portvals_bm: A dataframe with one column containing daily benchmark value
    vert_lines: Show buy and sell signals in chart
    title: Chart title
    xtitle: Chart X axe title
    ytitle: Chart Y axe title
    Returns: Plot a chart of the portfolio and benchmark performances
    """
    # Normalize data
    #portvals = normalize_data(portvals)
    #portvals_bm = normalize_data(portvals_bm)
    df = portvals_bm.join(portvals)
  
    # Min range
    if (df.loc[:, "Benchmark"].min() < df.loc[:, "Portfolio"].min()):
        min_range = df.loc[:, "Benchmark"].min()
    else:
        min_range = df.loc[:, "Portfolio"].min()
        
    # Max range    
    if (df.loc[:, "Benchmark"].max() > df.loc[:, "Portfolio"].max()):
        max_range = df.loc[:, "Benchmark"].max()
    else:
        max_range = df.loc[:, "Portfolio"].max()  
        
    # Plot the normalized benchmark and portfolio        
    trace_bench = go.Scatter(
                x=df.index,
                y=df.loc[:, "Benchmark"],
                name = "Benchmark",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

    trace_porfolio = go.Scatter(
                x=df.index,
                y=df.loc[:, "Portfolio"],
                name = "Portfolio",
                line = dict(color = '#000000'),
                opacity = 0.8)

    data = [trace_bench, trace_porfolio]
    
    

    
    # Plot the vertical lines for buy and sell signals
    shapes = list()
    if vert_lines:
        buy_line = []
        sell_line = []
        for date in df_orders.index:
            if df_orders.loc[date, "Order"] == "BUY":
                buy_line.append(date)
            else:
                sell_line.append(date)
        # Vertical lines
        line_size = max_range + (max_range * 10 / 100)

        # Buy line
        for i in buy_line:
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': 0,
                           'x1': i,
                           'y1': line_size,
                           'line': {
                                'color': 'rgb(0, 102, 34)',
                                'width': 1,
                                'dash': 'dash',
                            },
                          })
        # Sell line
        for i in sell_line:
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': 0,
                           'x1': i,
                           'y1': line_size,
                           'line': {
                                'color': 'rgb(255, 0, 0)',
                                'width': 1,
                                'dash': 'dash',
                            },
                          })
    
    layout = dict(
        autosize=True,
        shapes=shapes,
        margin=go.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        title = title,
        xaxis = dict(
                title=xtitle,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label='1m',
                            step='month',
                            stepmode='backward'),
                        dict(count=6,
                            label='6m',
                            step='month',
                            stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                range = [portvals.index[0], portvals.index[-1]]),
            
        yaxis = dict(
                title=ytitle,
                range = [min_range - (min_range * 10 / 100) ,max_range + (max_range * 10 / 100)]),
                    
        )
        
    
        
        

    fig = dict(data=data, layout=layout)
    iplot(fig)
    