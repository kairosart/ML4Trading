"""Implement technical indicators"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import datetime as dt
from util import get_exchange_days, normalize_data


# Add plotly for interactive charts
from plotly.offline import init_notebook_mode, iplot, plot
#init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

def get_momentum(price, window=10):
    """Calculate momentum indicator: 
    momentum[t] = (price[t]/price[t-window]) - 1
    Parameters:
    price: Price, typically adjusted close price, series of a symbol
    window: Number of days to look back
    
    Returns: Momentum, series of the same size as input data
    """    
    momentum = pd.Series(np.nan, index=price.index)
    momentum.iloc[window:] = price.iloc[window:] / price.values[:-window] - 1
    return momentum

def get_sma_indicator(price, rolling_mean):
    """Calculate simple moving average indicator, i.e. price / rolling_mean.
    Parameters:
    price: Price, typically adjusted close price, series of a symbol
    rolling_mean: Rolling mean of a series
    Returns: The simple moving average indicator
    """
    return price / rolling_mean - 1

def get_sma(values, window):
    """Return Simple moving average of given values, using specified window size."""
    sma = pd.Series(values.rolling(window,center=False).mean()) 
    q = (sma / values) - 1 
    return sma, q

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    #values.rolling(window).mean
    return values.rolling(window).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # todo: Compute and return rolling standard deviation
    #return pd.rolling_std(values, window=window)
    return values.rolling(window).std()

def get_bollinger_bands(rolling_mean, rolling_std, num_std=2):
    """Calculate upper and lower Bollinger Bands.

    Parameters:
    rolling_mean: Rolling mean of a series
    rolling_meanstd: Rolling std of a series
    num_std: Number of standard deviations for the bands

    Returns: Bollinger upper band and lower band
    """
    upper_band = rolling_mean + rolling_std * num_std
    lower_band = rolling_mean - rolling_std * num_std
    return upper_band, lower_band

def compute_bollinger_value(price, rolling_mean, rolling_std):
    """Output a value indicating how many standard deviations 
    a price is from the mean.

    Parameters:
    price: Price, typically adjusted close price, series of a symbol
    rolling_mean: Rolling mean of a series
    rolling_std: Rolling std of a series

    Returns:
    bollinger_val: the number of standard deviations a price is from the mean
    """

    bollinger_val = (price - rolling_mean) / rolling_std
    return bollinger_val


def get_RSI(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    
    
    return rsi


def plot_stock_prices(df_index, sym_price, symbol, title="Stock prices", xlabel="Date", ylabel="Price", fig_size=(12, 6)):
    """Plot Stock Prices.

    Parameters:
    df_index: Date index
    sym_price: Price, typically adjusted close price, series of symbol
    title: Chart title
    xlabel: X axis title
    ylable: Y axis title
    fig_size: Width and height of the chart in inches
    
    Returns:
    Plot cumulative return
    """
    trace_prices = go.Scatter(
                x=df_index,
                y=sym_price,
                name = symbol,
                line = dict(color = '#17BECF'),
                fill='tonexty',
                opacity = 0.8)

    
        

    data = [trace_prices]

    layout = dict(
        title = title,
        showlegend=True,
        legend=dict(
                orientation="h"),
        margin=go.layout.Margin(
            l=50,
            r=10,
            b=100,
            t=100,
            pad=4
        ),        
        xaxis = dict(
                title=xlabel,
                linecolor='#000', linewidth=1,
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
                range = [df_index.values[0], df_index.values[1]]),
            
        yaxis = dict(
                title=ylabel,
                linecolor='#000', linewidth=1
                ),
    )
        

    fig = dict(data=data, layout=layout)
    chart = plot(fig, show_link=False, output_type='div')
    return chart
    
def plot_cum_return(epoch, cum_return, title="Cumulative Return",
                  fig_size=(12, 6)):
    """Plot cumulative return.

    Parameters:
    epoch: one forward pass and one backward pass of all the training examples
    cum_retirm: cumulative return
    fig_size: Width and height of the chart in inches
    
    Returns:
    Plot cumulative return
    """
    trace_cum_r = go.Scatter(
                x=epoch,
                y=cum_return,
                name = "CR",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

    
        

    data = [trace_cum_r]

    layout = dict(
        title = title,
        
        xaxis = dict(
                title='Epoch',
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
                range = [epoch[0], epoch[-1]]),
            
        yaxis = dict(
                title='Cumulative return (%)'
                ),
    )
        
        
        

    fig = dict(data=data, layout=layout)
    chart = plot(fig, show_link=False, output_type='div')
    return chart


def plot_momentum(df_index, sym_price, sym_mom, title="Momentum Indicator",
                  fig_size=(12, 6)):
    """Plot momentum and prices for a symbol.

    Parameters:
    df_index: Date index
    sym_price: Price, typically adjusted close price, series of symbol
    sym_mom: Momentum of symbol
    fig_size: Width and height of the chart in inches
    
    Returns:
    Plot momentum and prices on the sample plot with two scales
    """
    trace_symbol = go.Scatter(
                x=df_index,
                y=sym_price,
                name = "AMZN",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

    trace_momentum = go.Scatter(
                x=df_index,
                y=sym_mom,
                name = "Momentum",
                yaxis='y2',
                line = dict(color = '#FF8000'),
                fill='tonexty',
                opacity = 0.8)
        

    data = [trace_symbol, trace_momentum]

    layout = dict(
        title = title,
        showlegend=True,

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        legend=dict(
                orientation="h"),
        xaxis = dict(
                title='Dates',
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
                range = [df_index.values[0], df_index.values[1]]),
            
        yaxis = dict(
                title='Adjusted Closed Price'
                ),
                    
        yaxis2=dict(
                title='M. Quantitative',
                overlaying='y',
                side='right'
                )
    )
        
        
        

    fig = dict(data=data, layout=layout)
    chart = plot(fig, show_link=False, output_type='div')
    return chart
    
def plot_sma_indicator(dates, df_index, sym_price, symbol, sma_indicator, sma_quality, 
                       title="SMA Indicator", fig_size=(12, 6)):
    """Plot SMA indicator, price and SMA quality for a symbol.

    Parameters:
    dates: Range of dates
    df_index: Date index
    sym_price: Price, typically adjusted close price, series of symbol
    symbol: Stock symbol
    sma_indicator: The simple moving average indicator
    sma_quality: SMA quality
    title: The chart title
    fig_size: Width and height of the chart in inches

    Returns:
    Plot all the three series on the same plot
    """
    trace_symbol = go.Scatter(
                x=df_index,
                y=sym_price,
                name = symbol,
                line = dict(color = '#17BECF'),
                opacity = 0.8)

    trace_sma = go.Scatter(
                x=df_index,
                y=sma_indicator,
                name = "SMA",
                line = dict(color = '#FF8000'),
                opacity = 0.8)
        
    trace_q = go.Scatter(
                x=df_index,
                y=sma_quality,
                name = "SMA Quantity",
                line = dict(color = '#04B404'),
                opacity = 0.8)
        
    data = [trace_symbol, trace_sma, trace_q]

    layout = dict(
        title = title,
        showlegend=True,

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        legend=dict(
                orientation="h"),
        xaxis = dict(
                title='Dates',
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
                range = [dates.values[0], dates.values[1]]),
            
        yaxis = dict(
                title='Price')
                    
        )
        
        

    fig = dict(data=data, layout=layout)
    chart = plot(fig, show_link=False, output_type='div')
    return chart

def plot_momentum_sma_indicator(dates, df_index, sym_price, sma_indicator, momentum, 
                       title="MOMENTUM/ SMA Indicator", fig_size=(12, 6)):
    """Plot Momentum/SMA cross indicator for a symbol.

    Parameters:
    dates: Range of dates
    df_index: Date index
    sym_price: Price, typically adjusted close price, series of symbol
    sma_indicator: The simple moving average indicator
    Momentum: Momentum
    title: The chart title
    fig_size: Width and height of the chart in inches

    Returns:
    Plot Momentum/SMA cross points
    """
    

    trace_sma = go.Scatter(
                x=df_index,
                y=sma_indicator,
                name = "SMA",
                line = dict(color = '#FF8000'),
                opacity = 0.8)
        
    trace_momentum = go.Scatter(
                x=df_index,
                y=momentum,
                name = "Momentum",
                line = dict(color = '#04B404'),
                opacity = 0.8)
        
    data = [trace_sma, trace_momentum]

    layout = dict(
        title = title,
        xaxis = dict(
                title='Dates',
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
                range = [dates.values[0], dates.values[1]]),
            
        yaxis = dict(
                title='Price')
                    
        )
        
        

    fig = dict(data=data, layout=layout)
    chart = plot(fig, show_link=False, output_type='div')
    return chart  
    
def plot_bollinger(dates, df_index, sym_price, symbol, upper_band, lower_band, bollinger_val, 
                   num_std=1, title="Bollinger Indicator", fig_size=(12, 6)):
    """Plot Bollinger bands and value for a symbol.

    Parameters:
    dates: Range of dates
    df_index: Date index
    sym_price: Price, typically adjusted close price, series of symbol
    symbol: Stock symbol    
    upper_band: Bollinger upper band
    lower_band: Bollinger lower band
    bollinger_val: The number of standard deviations a price is from the mean
    num_std: Number of standard deviations for the bands
    fig_size: Width and height of the chart in inches

    Returns:
    Plot Bollinger bands and Bollinger value
    """
    trace_symbol = go.Scatter(
                x=df_index,
                y=sym_price,
                name = symbol,
                line = dict(color = '#17BECF'),
                opacity = 0.8)

    trace_upper = go.Scatter(
                x=df_index,
                y=upper_band,
                name = "Upper band",
                line = dict(color = '#04B404'),
                opacity = 0.8)
        
    trace_lower = go.Scatter(
                x=df_index,
                y=lower_band,
                name = "Lower band",
                line = dict(color = '#FF0000'),
                opacity = 0.8)
        
    trace_Rolling = go.Scatter(
                x=df_index,
                y=bollinger_val,
                name = "Rolling Mean",
                line = dict(color = '#FF8000'),
                opacity = 0.8)
        
    data = [trace_symbol, trace_upper, trace_lower, trace_Rolling]

    layout = dict(
        title = title,
        showlegend=True,

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        legend=dict(
                orientation="h"),
        xaxis = dict(
                    title='Dates',
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
                    range = [dates.values[0], dates.values[1]]),
            
        yaxis = dict(
                    title='Price')
                    
        )
        
        

    fig = dict(data=data, layout=layout)
    chart = plot(fig, show_link=False, output_type='div')
    return chart

def plot_rsi_indicator(dates, df_index, sym_price, symbol, rsi_indicator, window=14, 
                       title="RSI Indicator", fig_size=(12, 6)):
    """Plot Relative Strength Index (RSI) of given values, using specified window size."""  
    '''
    Parameters:
    dates: Range of dates
    df_index: Date index
    sym_price: Price series of symbol
    symbol: Stock symbol
    rsi_indicator: RSI indicator
    window: Window size
    title: The chart title
    fig_size: Width and height of the chart in inches

    Returns:
    Plot price, RSI, Overbought line and Oversold line
    '''
    
    # Price line
    trace_symbol = go.Scatter(
                x=df_index,
                y=sym_price,
                name = symbol,
                line = dict(color = '#17BECF'),
                opacity = 0.8)

    # RSI line
    trace_rsi = go.Scatter(
                x=df_index,
                y=rsi_indicator,
                name = "RSI",
                line = dict(color = '#FF8000'),
                opacity = 0.8)

    # Overbought line
    trace_ob = go.Scatter(
                x=df_index,
                y=np.repeat(70, len(df_index)),
                name = "Overbought",
                line = dict(color = '#04B404',
                           dash = 'dash')
                )
    # Oversold line
    trace_os = go.Scatter(
                x=df_index,
                y=np.repeat(30, len(df_index)),
                name = "Oversold",
                line = dict(color = '#FF0000',
                           dash = 'dash')
                )

    # Signal line
    trace_signal = go.Scatter(
                x=df_index,
                y=np.repeat(50, len(df_index)),
                name = "Signal line",
                line = dict(color = '#000000',
                           dash = 'dot')
                )

    # Subplots
    fig = tools.make_subplots(rows=2, cols=1, print_grid=False,
                              subplot_titles=('Price', 'Relative Strength Index (RSI)'))
    fig.append_trace(trace_symbol, 1, 1)
    fig.append_trace(trace_ob, 2, 1)
    fig.append_trace(trace_os, 2, 1)
    fig.append_trace(trace_rsi, 2, 1)
    fig.append_trace(trace_signal, 2, 1)
    layout = dict(
        title = title,
        
        xaxis = dict(
                    title='Dates',
                    range = [dates.values[0], dates.values[1]]),

        yaxis = dict(
                    title='Price')

        )



    fig['layout'].update(height=600, title='Overbought-Oversold',
                        showlegend=True,

                        margin=go.layout.Margin(
                            l=50,
                            r=50,
                            b=100,
                            t=100,
                            pad=4
                            ),
                        legend=dict(orientation="h"))
    chart = plot(fig, show_link=False, output_type='div')
    return chart

    
def plot_performance(perform_df, title="In-sample vs Out of sample performance",
                  fig_size=(12, 6)):
    """Plot In-sample and Out of sample performances.

    Parameters:
    perform_df: Performance dataframe
    title: Chart title
    fig_size: Width and height of the chart in inches
    
    Returns:
    Plot In-sample and Out of sample performances.
    """
    trace1 = go.Bar(
        x=['Sharpe Ratio'],
        y=perform_df.loc['Sharpe Ratio', ['In-sample']],
        name='In-sample'
    )
   
    trace2 = go.Bar(
        x=['Sharpe Ratio'],
        y=perform_df.loc['Sharpe Ratio', ['Out of sample']],
        name='Out of sample'
    )
    
    trace3 = go.Bar(
        x=['Cum. Return'],
        y=perform_df.loc['Cumulative Return', ['In-sample']],
        name='In-sample'
    )
   
    trace4 = go.Bar(
        x=['Cum. Return'],
        y=perform_df.loc['Cumulative Return', ['Out of sample']],
        name='Out of sample'
    )
    
    trace5 = go.Bar(
        x=['Standard Deviation'],
        y=perform_df.loc['Standard Deviation', ['In-sample']],
        name='In-sample'
    )
   
    trace6 = go.Bar(
        x=['Standard Deviation'],
        y=perform_df.loc['Standard Deviation', ['Out of sample']],
        name='Out of sample'
    )
    
    trace7 = go.Bar(
        x=['Average Daily Return'],
        y=perform_df.loc['Average Daily Return', ['In-sample']],
        name='In-sample'
    )
   
    trace8 = go.Bar(
        x=['Average Daily Return'],
        y=perform_df.loc['Average Daily Return', ['Out of sample']],
        name='Out of sample'
    )
    
    trace9 = go.Bar(
        x=['Final Portfolio Value'],
        y=perform_df.loc['Final Portfolio Value', ['In-sample']],
        name='In-sample'
    )
   
    trace10 = go.Bar(
        x=['Final Portfolio Value'],
        y=perform_df.loc['Final Portfolio Value', ['Out of sample']],
        name='Out of sample'
    )
    
    # Subplots
    fig = tools.make_subplots(rows=3, cols=2, print_grid=False, 
                        specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]])
    # Sharpe ratio
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    
    # Cumulative return
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 1, 2)
    
    # Standard Deviation
    fig.append_trace(trace5, 2, 1)
    fig.append_trace(trace6, 2, 1)
    
    # Average Daily Return
    fig.append_trace(trace7, 2, 2)
    fig.append_trace(trace8, 2, 2)    

    # Final Portfolio Value
    fig.append_trace(trace9, 3, 1)
    fig.append_trace(trace10, 3, 1) 
    
    layout = go.Layout(
        barmode='group'
    )
        
    fig['layout'].update(height=600, width=600, title=title)    

    iplot(fig)    
    
