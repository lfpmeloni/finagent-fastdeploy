import requests
import json
import pandas as pd
import numpy as np
import os
import yfinance as yf
import scipy
from scipy.stats import norm
from datetime import datetime, timedelta, date
import sys

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

# Function to get the next Friday if needed
def get_next_friday(date):
    if 0 <= date.weekday() <= 3:
        # Monday to Thursday -> Get this Friday
        return date + timedelta(days=(4 - date.weekday()))
    elif date.weekday() == 4:
        # If it's already Friday, return the next Friday
        return date + timedelta(days=7)
    else:
        # Saturday or Sunday -> Get next Friday
        return date + timedelta(days=((4 - date.weekday()) % 7))
    
# Get options data
def get_cboe_option_data(index):
    print("Getting CBOE Option Data for " + index)
    response = requests.get(url="https://cdn.cboe.com/api/global/delayed_quotes/options/" + index + ".json")
    options = response.json()
    
    # Get SPX Spot
    spotPrice = options["data"]["close"]
    #print(spotPrice)

    # Get SPX Options Data
    data_df = pd.DataFrame(options["data"]["options"])
    
    quote = options['data']
    quote.pop('options')
    spot_price = quote.get('current_price', None)
    print(f"Underlying index price: {spot_price}")

    #data_df[['symbol', 'expiration_date', 'put_call', 'strike_price']] = data_df.option.str.extract(r'([A-Z]+)(\d{6})([CP])(\d+)')
    #data_df['expiration_date'] = pd.to_datetime(data_df['expiration_date'], yearfirst=True)

    #for c in ['strike_price', 'open_interest', 'iv', 'gamma', 'last_trade_price', 'bid', 'ask', 'volume', 'delta']:
    #        data_df[c] = pd.to_numeric(data_df[c], errors='coerce')
    #data_df['strike_price'] = data_df['strike_price'] / 1000
    ##snapshot_time = pd.to_datetime(data['timestamp'])
    #data_df['days_to_expiration'] = np.busday_count(
    #    pd.Series(snapshot_time).dt.date.values.astype('datetime64[D]'), 
    #    data_df['expiration_date'].dt.date.values.astype('datetime64[D]')
    #) / 262

    # Add SPXspot column with the spot_price value
    data_df['SPXspot'] = spot_price

    data_df['CallPut'] = data_df['option'].str.slice(start=-9,stop=-8)
    data_df['ExpirationDate'] = data_df['option'].str.slice(start=-15,stop=-9)
    data_df['ExpirationDate'] = pd.to_datetime(data_df['ExpirationDate'], format='%y%m%d')
    data_df['Strike'] = data_df['option'].str.slice(start=-8,stop=-3)
    data_df['Strike'] = data_df['Strike'].str.lstrip('0')

    # drop the data if the ExpirationDate is less than today
    data_df = data_df[data_df['ExpirationDate'] >= datetime.now()]
    
    data_df_calls = data_df.loc[data_df['CallPut'] == "C"]
    data_df_puts = data_df.loc[data_df['CallPut'] == "P"]
    data_df_calls = data_df_calls.reset_index(drop=True)
    data_df_puts = data_df_puts.reset_index(drop=True)

    df = data_df_calls[['ExpirationDate','option','last_trade_price','change','bid','ask','volume','iv','delta','gamma', 'vega', 'theta', 'rho', 'theo', 'open_interest','Strike']]
    df_puts = data_df_puts[['ExpirationDate','option','last_trade_price','change','bid','ask','volume','iv','delta','gamma', 'vega', 'theta', 'rho', 'theo', 'open_interest','Strike']]
    df_puts.columns = ['put_exp','put_option','put_last_trade_price','put_change','put_bid','put_ask','put_volume','put_iv','put_delta','put_gamma','put_vega', 'put_theta', 'put_rho', 'put_theo', 'put_open_interest','put_strike']

    df = pd.concat([df, df_puts], axis=1)

    df['check'] = np.where((df['ExpirationDate'] == df['put_exp']) & (df['Strike'] == df['put_strike']), 0, 1)

    if df['check'].sum() != 0:
        print("PUT CALL MERGE FAILED - OPTIONS ARE MISMATCHED.")
        exit()

    df.drop(['put_exp', 'put_strike', 'check'], axis=1, inplace=True)

    #print(df)

    df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                'CallIV','CallDelta','CallGamma', 'CallVega', 'CallTheta', 'CallRho', 'CallTheo', 'CallOpenInt','StrikePrice','Puts','PutLastSale',
                'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutVega', 'PutTheta', 'PutRho', 'PutTheo', 'PutOpenInt']

    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
    df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
    df['StrikePrice'] = df['StrikePrice'].astype(float)
    df['CallIV'] = df['CallIV'].astype(float)
    df['PutIV'] = df['PutIV'].astype(float)
    df['CallGamma'] = df['CallGamma'].astype(float)
    df['CallVega'] = df['CallVega'].astype(float)
    df['CallTheta'] = df['CallTheta'].astype(float)
    df['CallRho'] = df['CallRho'].astype(float)
    df['CallTheo'] = df['CallTheo'].astype(float)
    df['PutGamma'] = df['PutGamma'].astype(float)
    df['PutVega'] = df['PutVega'].astype(float)
    df['PutTheta'] = df['PutTheta'].astype(float)
    df['PutRho'] = df['PutRho'].astype(float)
    df['PutTheo'] = df['PutTheo'].astype(float)
    df['CallOpenInt'] = df['CallOpenInt'].astype(float)
    df['PutOpenInt'] = df['PutOpenInt'].astype(float)

    # ---=== CALCULATE SPOT GAMMA ===---
    # Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price 
    # To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

    df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9

    # GEX Exposure Code
    df['SPXprice'] = spot_price

    # Calculate NetGexCall and NetGexPut and add them as new columns
    df['NetGexCall'] = np.where(df['TotalGamma'] > 0, df['TotalGamma'] * df['StrikePrice'], 0)
    df['NetGexPut'] = np.where(df['TotalGamma'] < 0, df['TotalGamma'] * df['StrikePrice'] * -1, 0)

     # Calculate NetGexCall1 and NetGexPut1 and add them as new columns without multiplying by strike_price
    df['NetGexCall1'] = np.where(df['TotalGamma'] > 0, df['TotalGamma'], 0)
    df['NetGexPut1'] = np.where(df['TotalGamma'] < 0, df['TotalGamma'] * -1, 0)

    # Calculate CallGammaOI and PutGammaOI
    df['CallGEXOI'] = df['CallGamma'] * df['CallOpenInt']
    df['PutGEXOI'] = df['PutGamma'] * df['PutOpenInt']

    # Calculate GEX Volume and GEX Open Interest for calls and puts
    df['CallGEXVolume'] = df['CallGamma'] * df['CallVol']
    df['PutGEXVolume'] = df['PutGamma'] * df['PutVol']

    # Calculate NetGammaOI
    df['NetGEXOI'] = df['CallGEXOI'] - df['PutGEXOI']
    df['TotalGEXOI'] = df['CallGEXOI'] + df['PutGEXOI']
    df['NetGEXVolume'] = df['CallGEXVolume'] - df['PutGEXVolume']
    df['TotalGEXVolume'] = df['CallGEXVolume'] + df['PutGEXVolume']
    df['NetVolume'] = df['CallVol'] - df['PutVol']
    df['NetOpenInterest'] = df['CallOpenInt'] - df['PutOpenInt']

    df['TotalVolume'] = df['CallVol'] + df['PutVol']
    df['TotalOpenInterest'] = df['CallOpenInt'] + df['PutOpenInt']

    df['CallGEXOI'] = df['CallGEXOI'].astype(float)
    df['PutGEXOI'] = df['PutGEXOI'].astype(float)
    df['CallGEXVolume'] = df['CallGEXVolume'].astype(float)
    df['PutGEXVolume'] = df['PutGEXVolume'].astype(float)
    df['NetGEXOI'] = df['NetGEXOI'].astype(float)
    df['NetGEXVolume'] = df['NetGEXVolume'].astype(float)
    df['NetVolume'] = df['NetVolume'].astype(float)
    df['NetOpenInterest'] = df['NetOpenInterest'].astype(float)
    df['TotalVolume'] = df['TotalVolume'].astype(float)
    df['TotalOpenInterest'] = df['TotalOpenInterest'].astype(float)

    df['CallVolOI'] = df['CallVol'] * df['CallOpenInt']
    df['PutVolOI'] = df['PutVol'] * df['PutOpenInt']
    df['NetVolOI'] = df['CallVolOI'] - df['PutVolOI']
    
    # Create a new column CallStrikeVol - CallVol*(CallDelta + StrikePrice)
    df['CallDeltaOI'] = df['CallOpenInt'] * df['CallDelta']
    df['PutDeltaOI'] = df['PutOpenInt'] * df['PutDelta']
    df['CallStrikeVol'] = df['CallVol'] * (df['CallDelta'] + df['StrikePrice'])
    df['PutStrikeVol'] = df['PutVol'] * (df['PutDelta'] + df['StrikePrice'])
    df['TotalStrikeVol'] = df['CallStrikeVol'] - df['PutStrikeVol']
    df['CallStrikeOI'] = df['CallOpenInt'] * (df['CallDelta'] + df['StrikePrice'])
    df['PutStrikeOI'] = df['PutOpenInt'] * (df['PutDelta'] + df['StrikePrice'])
    df['TotalStrikeOI'] = df['CallStrikeOI'] - df['PutStrikeOI']
    df['CallBidOI'] = df['CallOpenInt'] * df['CallBid']
    df['PutBidOI'] = df['PutOpenInt'] * df['PutBid']
    df['CallBidVol'] = df['CallVol'] * df['CallBid']
    df['PutBidVol'] = df['PutVol'] * df['PutBid']
    df['CallWall'] = df['CallGamma'] + df['CallVol'] + df['CallOpenInt']
    df['PutWall'] = df['PutGamma'] + df['PutVol'] + df['PutOpenInt']

    df['CallStrikeVol'] = df['CallStrikeVol'].astype(float)
    df['CallVol'] = df['CallVol'].astype(float)
    df['PutStrikeVol'] = df['PutStrikeVol'].astype(float)
    df['PutVol'] = df['PutVol'].astype(float)
    df['CallStrikeOI'] = df['CallStrikeOI'].astype(float)
    df['PutStrikeOI'] = df['PutStrikeOI'].astype(float)
    df['CallDeltaOI'] = df['CallDeltaOI'].astype(float)
    df['PutDeltaOI'] = df['PutDeltaOI'].astype(float)
    df['CallBidOI'] = df['CallBidOI'].astype(float)
    df['PutBidOI'] = df['PutBidOI'].astype(float)
    df['CallBidVol'] = df['CallBidVol'].astype(float)
    df['PutBidVol'] = df['PutBidVol'].astype(float)
    df['CallWall'] = df['CallWall'].astype(float)
    df['PutWall'] = df['PutWall'].astype(float)

    # Initialize an empty column to store the result
    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['CallBidOI'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['CallBidOI'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    # Add the results back to the DataFrame
    df['CumCallBidOI'] = results

    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['PutBidOI'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['PutBidOI'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    # Add the results back to the DataFrame
    df['CumPutBidOI'] = results

    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['CallBidVol'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['CallBidVol'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    # Add the results back to the DataFrame
    df['CumCallBidVol'] = results

    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['PutBidVol'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['PutBidVol'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    # Add the results back to the DataFrame
    df['CumPutBidVol'] = results

    df['CumTotalBidOI'] = df['CumCallBidOI'] + df['CumPutBidOI']
    df['CumTotalBidVol'] = df['CumCallBidVol'] + df['CumPutBidVol']

    df['CumCallBidOI'] = df['CumCallBidOI'].astype(float)
    df['CumPutBidOI'] = df['CumPutBidOI'].astype(float)
    df['CumCallBidVol'] = df['CumCallBidVol'].astype(float)
    df['CumPutBidVol'] = df['CumPutBidVol'].astype(float)
    df['CumTotalBidOI'] = df['CumTotalBidOI'].astype(float)
    df['CumTotalBidVol'] = df['CumTotalBidVol'].astype(float)
    

    # For 0DTE options, I'm setting DTE = 1 day, otherwise they get excluded
    df['daysTillExp'] = [1/262 if (np.busday_count(date.today(), x.date())) == 0 \
                            else np.busday_count(date.today(), x.date())/262 for x in df.ExpirationDate]

    df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]

    dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
    return df, dfAgg, spotPrice

def calc_vanna_vectorized(S, K, vol, T, delta):
    """Vectorized calculation of Vanna for all rows."""
    valid = (T > 0) & (vol > 0)  # Ensure T and vol are positive
    sqrt_T = np.sqrt(T[valid])
    vol_sqrt_T = vol[valid] * sqrt_T
    log_term = np.log(S / K[valid])
    dp = (log_term + (0.5 * vol[valid]**2) * T[valid]) / vol_sqrt_T

    vanna = np.zeros_like(K, dtype=np.float64)
    vanna[valid] = (
        -np.exp(-delta[valid] * T[valid]) * log_term / (vol_sqrt_T**2)
    )  # Simplified Vanna formula
    return vanna

def calc_vanna_exposure_vectorized(spot_price, df):
    df = df.copy()
    """Calculate total vanna exposure for all spot levels."""
    call_vanna = calc_vanna_vectorized(spot_price, df['StrikePrice'].values, df['CallIV'].values, 
                                       df['daysTillExp'].values, df['CallDelta'].values)
    put_vanna = calc_vanna_vectorized(spot_price, df['StrikePrice'].values, df['PutIV'].values, 
                                      df['daysTillExp'].values, df['PutDelta'].values)

    df['callVannaEx'] = call_vanna * spot_price * df['CallOpenInt'].values
    df['putVannaEx'] = put_vanna * spot_price * df['PutOpenInt'].values

    # Total vanna exposures
    total_vanna_exposure = df['callVannaEx'].sum() - df['putVannaEx'].sum()

    return total_vanna_exposure

# Optimized Black-Scholes Gamma calculation
def calc_gamma_vectorized(S, K, vol, T, q):
    """Vectorized calculation of Gamma for all rows."""
    valid = (T > 0) & (vol > 0)
    sqrt_T = np.sqrt(T[valid])
    vol_sqrt_T = vol[valid] * sqrt_T
    log_term = np.log(S / K[valid])
    dp = (log_term + (0.5 * vol[valid]**2) * T[valid]) / vol_sqrt_T

    gamma = np.zeros_like(K, dtype=np.float64)
    gamma[valid] = np.exp(-q * T[valid]) * norm.pdf(dp) / (S * vol_sqrt_T)
    return gamma

def calc_gamma_exposure_vectorized(spot_price, df):
    df = df.copy()
    """Calculate total gamma exposure for all spot levels."""
    # Vectorized calculation of Gamma
    call_gamma = calc_gamma_vectorized(spot_price, df['StrikePrice'].values, df['CallIV'].values, 
                                       df['daysTillExp'].values, 0)
    put_gamma = calc_gamma_vectorized(spot_price, df['StrikePrice'].values, df['PutIV'].values, 
                                      df['daysTillExp'].values, 0)

    df['callGammaEx'] = call_gamma * spot_price * df['CallOpenInt'].values
    df['putGammaEx'] = put_gamma * spot_price * df['PutOpenInt'].values

    # Total gamma exposures
    total_gamma_exposure = df['callGammaEx'].sum() - df['putGammaEx'].sum()

    return total_gamma_exposure

def find_zero_vanna(values, levels):
    """
    Find the level where the values cross zero. 
    If no crossing is found, fall back to the weighted average strike.
    """
    # Find indices where zero crossing happens
    zero_cross_idx = np.where(np.diff(np.sign(values)))[0]
    
    if len(zero_cross_idx) > 0:
        # Calculate the zero crossing using linear interpolation
        neg_value = values[zero_cross_idx]
        pos_value = values[zero_cross_idx + 1]
        neg_level = levels[zero_cross_idx]
        pos_level = levels[zero_cross_idx + 1]

        zero_crossing = pos_level - ((pos_level - neg_level) * pos_value / (pos_value - neg_value))
        return zero_crossing[0]  # Return the first zero crossing
    else:
        # Fallback to weighted average strike
        weighted_avg_strike = np.average(levels, weights=np.abs(values))
        return weighted_avg_strike

def find_zero_gamma(totalGamma, levels):
    """
    Find the zero gamma crossing point.

    Parameters:
        totalGamma (np.array): Gamma values for all spot levels.
        levels (np.array): Spot levels corresponding to totalGamma.

    Returns:
        float or None: Zero gamma crossing point, or None if no crossing is found.
    """
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
    if len(zeroCrossIdx) == 0:
        return None  # No zero-crossing found

    # Extract the first crossing
    idx = zeroCrossIdx[0]
    negGamma = totalGamma[idx]
    posGamma = totalGamma[idx + 1]
    negStrike = levels[idx]
    posStrike = levels[idx + 1]

    # Calculate zero gamma as a scalar
    zeroGamma = posStrike - ((posStrike - negStrike) * posGamma / (posGamma - negGamma))
    return zeroGamma

def find_zero_gamma_levels(df, fromStrike, toStrike):
    # Generate spot levels
    levels = np.linspace(fromStrike, toStrike, 240)

    # Calculate gamma exposures for all spot levels
    gamma_results = [calc_gamma_exposure_vectorized(spot, df) 
                    for spot in levels]

    # Extract results for each gamma exposure type
    #totalGamma = zip(*gamma_results)

    # Convert to arrays and normalize
    totalGamma = np.array(gamma_results) / 10**9

    zeroGamma = find_zero_gamma(totalGamma, levels)
    return zeroGamma

def find_zero_vanna_levels(df, fromStrike, toStrike):
    # Generate spot levels
    levels = np.linspace(fromStrike, toStrike, 240)

    # Calculate Vanna exposures for all spot levels
    vanna_results = [calc_vanna_exposure_vectorized(spot, df) 
                     for spot in levels]

    # Extract results for each Vanna exposure type
    #totalVanna = zip(*vanna_results)

    # Convert to arrays and normalize
    totalVanna = np.array(vanna_results) / 10**9

    zeroVanna = find_zero_vanna(totalVanna, levels)  # Reuse the same interpolation function

    return zeroVanna

def calculate_gex_ladder(df):
    df_sorted = df.sort_values(by=['ExpirationDate', 'StrikePrice'])
     # Find the first and second expiration dates
    first_expiration_date = df_sorted['ExpirationDate'].min()
    second_expiration_date = df_sorted['ExpirationDate'].unique()[1]

    # Filter rows for the first expiration date
    first_expiration_data = df_sorted[df_sorted['ExpirationDate'] == first_expiration_date]

    # Filter rows for the second expiration date
    second_expiration_data = df_sorted[df_sorted['ExpirationDate'] == second_expiration_date]

    # Calculate sum of NetGexCall and NetGexCall1 in the first expiration
    sum_NetGexCall = first_expiration_data['NetGexCall'].sum()
    sum_NetGexCall1 = first_expiration_data['NetGexCall1'].sum()
    call_wall_0 = sum_NetGexCall / sum_NetGexCall1 if sum_NetGexCall1 != 0 else 0

    # Calculate sum of NetGexPut and NetGexPut1 in the first expiration
    sum_NetGexPut = first_expiration_data['NetGexPut'].sum()
    sum_NetGexPut1 = first_expiration_data['NetGexPut1'].sum()
    put_wall_0 = sum_NetGexPut / sum_NetGexPut1 if sum_NetGexPut1 != 0 else 0

    # Calculate sum of NetGexCall and NetGexCall1 in the second expiration
    sum_NetGexCall_1 = second_expiration_data['NetGexCall'].sum()
    sum_NetGexCall1_1 = second_expiration_data['NetGexCall1'].sum()
    call_wall_1 = sum_NetGexCall_1 / sum_NetGexCall1_1 if sum_NetGexCall1_1 != 0 else 0

    # Calculate sum of NetGexPut and NetGexPut1 in the second expiration
    sum_NetGexPut_1 = second_expiration_data['NetGexPut'].sum()
    sum_NetGexPut1_1 = second_expiration_data['NetGexPut1'].sum()
    put_wall_1 = sum_NetGexPut_1 / sum_NetGexPut1_1 if sum_NetGexPut1_1 != 0 else 0

    # Calculate sum of NetGexCall and NetGexCall1 for all expirations
    total_NetGexCall = df_sorted['NetGexCall'].sum()
    total_NetGexCall1 = df_sorted['NetGexCall1'].sum()
    call_wall = total_NetGexCall / total_NetGexCall1 if total_NetGexCall1 != 0 else 0

    # Calculate sum of NetGexPut and NetGexPut1 for all expirations
    total_NetGexPut = df_sorted['NetGexPut'].sum()
    total_NetGexPut1 = df_sorted['NetGexPut1'].sum()
    put_wall = total_NetGexPut / total_NetGexPut1 if total_NetGexPut1 != 0 else 0

    # Calculate avg_wall_0 and avg_wall
    avg_wall_0 = (call_wall_0 + put_wall_0) / 2
    avg_wall_1 = (call_wall_1 + put_wall_1) / 2
    avg_wall = (call_wall + put_wall) / 2

    return first_expiration_data, second_expiration_data, call_wall_0, put_wall_0, call_wall_1, put_wall_1, call_wall, put_wall, avg_wall_0, avg_wall_1, avg_wall

def get_additional_gex_values(df, spotPrice):
    """
    Calculate additional GEX values for the given DataFrame and spot price.
    Returns results in the same format as the original implementation.
    """
    import numpy as np
    
    # Sort and filter the dataframe for the relevant strike range
    #df_sorted = df.sort_values(by=['ExpirationDate', 'StrikePrice'])
    #strike_range = (0.9 * spotPrice, 1.1 * spotPrice)
    #df_filtered = df_sorted[(df_sorted['StrikePrice'] >= strike_range[0]) & (df_sorted['StrikePrice'] <= strike_range[1])]

    # Pre-compute masks for calls above and puts below the spot price
    calls_above_spot = df['StrikePrice'] > spotPrice
    puts_below_spot = df['StrikePrice'] < spotPrice

    # Define a helper function for repeated operations
    def get_resistance_support(column):
        """Find resistance and support levels based on the given column."""
        call_resistance = df.loc[calls_above_spot].nlargest(1, column)['StrikePrice'].iloc[0] \
            if not df.loc[calls_above_spot].empty else None
        put_support = df.loc[puts_below_spot].nlargest(1, column)['StrikePrice'].iloc[0] \
            if not df.loc[puts_below_spot].empty else None
        return call_resistance, put_support

    # Calculate resistance and support levels
    metrics_columns = ['CallOpenInt', 'PutOpenInt', 'CallVol', 'PutVol', 'CallGEXOI', 'PutGEXOI',
                       'CallGEXVolume', 'PutGEXVolume', 'CallWall', 'PutWall', 'NetGEXOI',
                       'NetGEXVolume', 'CallVolOI', 'PutVolOI', 'NetVolOI']
    resistance_support = {col: get_resistance_support(col) for col in metrics_columns}

    # Aggregate data for volume and open interest calculations
    callbid_vol = np.divide(df['CallBidVol'].sum(), df['CallVol'].sum(), where=df['CallVol'].sum() != 0)
    putbid_vol = np.divide(df['PutBidVol'].sum(), df['PutVol'].sum(), where=df['PutVol'].sum() != 0)
    volume = callbid_vol - putbid_vol

    calloi_vol = np.divide(df['CallOpenInt'].sum(), df['CallVol'].sum(), where=df['CallVol'].sum() != 0)
    putoi_vol = np.divide(df['PutOpenInt'].sum(), df['PutVol'].sum(), where=df['PutVol'].sum() != 0)
    open_interest = calloi_vol - putoi_vol

    return (
        resistance_support['CallOpenInt'][0], resistance_support['PutOpenInt'][1],
        resistance_support['CallVol'][0], resistance_support['PutVol'][1],
        resistance_support['CallGEXOI'][0], resistance_support['PutGEXOI'][1],
        resistance_support['CallGEXVolume'][0], resistance_support['PutGEXVolume'][1],
        resistance_support['CallWall'][0], resistance_support['PutWall'][1],
        resistance_support['NetGEXOI'][0], resistance_support['NetGEXOI'][1],
        resistance_support['NetGEXVolume'][0], resistance_support['NetGEXVolume'][1],
        resistance_support['CallVolOI'][0], resistance_support['PutVolOI'][1],
        resistance_support['NetVolOI'][0], resistance_support['NetVolOI'][1],
        callbid_vol, putbid_vol, volume, calloi_vol, putoi_vol, open_interest
    )

def get_flip_pain_points(df1):
    df = df1.copy()
    #dfNext["CumGamma"] = dfNext["TotalGamma"].cumsum()
    # Initialize an empty column to store the result
    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['TotalGamma'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['TotalGamma'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    # Add the results back to the DataFrame
    df['CumGamma'] = results
    zero_gamma_idx = df["CumGamma"].abs().idxmin()
    zero_gamma = df.loc[zero_gamma_idx, "StrikePrice"]   

    # Total Positive and Negative Gamma-based metrics
    tot_pos_vol = df[df["TotalGamma"] > 0]["CallVol"].sum() + df[df["TotalGamma"] > 0]["PutVol"].sum()
    tot_pos_oi = df[df["TotalGamma"] > 0]["CallOpenInt"].sum() + df[df["TotalGamma"] > 0]["PutOpenInt"].sum()
    tot_neg_vol = df[df["TotalGamma"] < 0]["CallVol"].sum() + df[df["TotalGamma"] < 0]["PutVol"].sum()
    tot_neg_oi = df[df["TotalGamma"] < 0]["CallOpenInt"].sum() + df[df["TotalGamma"] < 0]["PutOpenInt"].sum()
    tot_vol = tot_pos_vol/tot_neg_vol
    tot_oi = tot_pos_oi/tot_neg_oi

    # Gamma Flips
    gamma_flip1 = df[df["TotalGamma"] < 0]["StrikePrice"].max()
    gamma_flip2 = df[df["TotalGamma"] > 0]["StrikePrice"].min()

    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['TotalVolume'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['TotalVolume'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    df['CumVolume'] = results

    # Find the pain point strike based on volume
    pain_volume_idx = df['CumVolume'].abs().idxmin()
    pain_volume_strike = df.loc[pain_volume_idx, 'StrikePrice']

    #dfNext['CumOpenInterest'] = dfNext['TotalOpenInterest'].cumsum()
    results = []
    # Iterate through each row to apply the logic dynamically
    for i in range(len(df)):
        # Calculate cumulative sum from row 10 up to the current row
        cumulative_sum = df['TotalOpenInterest'][:i + 1].sum()

        # Calculate sum of all rows below the current row
        remaining_sum = df['TotalOpenInterest'][i + 1:].sum()

        # Compute the result for the current row
        result = cumulative_sum - remaining_sum
        results.append(result)

    df['CumOpenInterest'] = results

    # Find the pain point strike based on open interest
    pain_oi_idx = df['CumOpenInterest'].abs().idxmin()
    pain_oi_strike = df.loc[pain_oi_idx, 'StrikePrice']

    zero_pos_vol = df.loc[zero_gamma_idx:, 'TotalVolume'].sum()

    # Find the strike for zero positive volume
    zero_pos_strike_idx = df.loc[zero_gamma_idx:].query("CumVolume >= @zero_pos_vol").index.min()
    zero_pos_strike = df.loc[zero_pos_strike_idx, 'StrikePrice'] if zero_pos_strike_idx is not None else None

    # Zero Negative Volume: Total volume below zero_gamma_idx
    zero_neg_vol = df.loc[:zero_gamma_idx - 1, 'TotalVolume'].sum()

    # Find the strike for zero negative volume
    zero_neg_strike_idx = df.loc[:zero_gamma_idx - 1].query("CumVolume <= @zero_neg_vol").index.max()
    zero_neg_strike = df.loc[zero_neg_strike_idx, 'StrikePrice'] if zero_neg_strike_idx is not None else None

    return zero_gamma, tot_vol, tot_oi, gamma_flip1, gamma_flip2, pain_volume_strike, pain_oi_strike, zero_pos_strike, zero_neg_strike

def calculate_flow_indicator(df):
    multiplier = 1000000
    # if symbol == 'MES':
    #     multiplier = 1000000
    # elif symbol == 'MNQ':
    #     multiplier = 150000
    # elif symbol == 'M2K':
    #     multiplier = 25000

    def safe_value(value):
        return value if value != 0 and value != 0.0 else 0.01

    red_oi = safe_value(df['PutBidOI'].sum() / multiplier)
    green_oi = safe_value(df['CallBidOI'].sum() / multiplier)
    delta_oi = safe_value(green_oi - red_oi)
    red_vol = safe_value(df['PutBidVol'].sum() / multiplier)
    green_vol = safe_value(df['CallBidVol'].sum() / multiplier)
    delta_vol = safe_value(green_vol - red_vol)
    eth_blue = safe_value(green_vol / red_vol)
    eth_purple = safe_value((red_vol / green_vol) * -1)

    return red_oi, green_oi, delta_oi, red_vol, green_vol, delta_vol, eth_blue, eth_purple

def calculate_flow_levels_for_expiration(df, spotPrice):
    vCallPrice = df['CallStrikeVol'].sum()/df['CallVol'].sum()
    vPutPrice = df['PutStrikeVol'].sum()/df['PutVol'].sum()
    vCallOiPrice = df['CallStrikeOI'].sum()/df['CallOpenInt'].sum()
    vPutOiPrice = df['PutStrikeOI'].sum()/df['PutOpenInt'].sum()
    vAvgPrice = (vCallPrice + vPutPrice) / 2
    vAvgOiPrice = (vCallOiPrice + vPutOiPrice) / 2
    vRes1Price = (vCallPrice + vAvgPrice) / 2
    vSup1Price = (vPutPrice + vAvgPrice) / 2
    vRes2Price = (vCallPrice + vRes1Price)/2
    vSup2Price = (vPutPrice + vSup1Price)/2
    vRes3Price = (vCallPrice - (vRes2Price - vCallPrice))
    vSup3Price = (vPutPrice - (vSup2Price - vPutPrice))
    vRes4Price = (vRes3Price - (vCallPrice - vRes3Price))
    vSup4Price = (vSup3Price - (vPutPrice - vSup3Price))
    extremeValue = pd.concat([df.nsmallest(10, 'TotalGamma'), df.nlargest(10, 'TotalGamma')])
    resistances = [x for x in extremeValue['StrikePrice'] if x >= spotPrice]
    supports = [x for x in extremeValue['StrikePrice'] if x < spotPrice]
    redOi, greenOi, deltaOi, redVol, greenVol, deltaVol, ethBlue, ethPurple = calculate_flow_indicator(df)

    return vCallPrice, vPutPrice, vCallOiPrice, vPutOiPrice, vAvgPrice, vAvgOiPrice, vRes1Price, vSup1Price, vRes2Price, vSup2Price, vRes3Price, vSup3Price,  \
        vRes4Price, vSup4Price, resistances, supports, \
        redOi, greenOi, deltaOi, redVol, greenVol, deltaVol, ethBlue, ethPurple

def get_gex_and_flow_levels(symbol, today):
    print("Getting Gex and Flow Levels")
    df, dfAgg, spotPrice = get_cboe_option_data(symbol)
    strikes = dfAgg.index.values
    fromStrike = 0.9 * spotPrice
    toStrike = 1.1 * spotPrice

    # if symbol == "_SPX" or symbol == "SPY":
    #     convertedSymbol = "MES"
    #     if symbol == "_SPX":
    #         priceRatio = round(get_current_price("ES=F"),2) / round(get_current_price("^GSPC"),2)
    #     else:
    #         priceRatio = round(get_current_price("ES=F"),2) / round(get_current_price("SPY"),2)
    # elif symbol == "_NDX" or symbol == "QQQ":
    #     convertedSymbol = "MNQ"
    #     if symbol == "_NDX":
    #         priceRatio = round(get_current_price("NQ=F"),2) / round(get_current_price("^NDX"),2)
    #     else:
    #         priceRatio = round(get_current_price("NQ=F"),2) / round(get_current_price("QQQ"),2)
    # elif symbol == "_RUT" or symbol == "IWM":
    #     convertedSymbol = "M2K"
    #     if symbol == "_RUT":
    #         priceRatio = round(get_current_price("M2K=F"),2) / round(get_current_price("^RUT"),2)
    #     else:
    #         priceRatio = round(get_current_price("M2K=F"),2) / round(get_current_price("IWM"),2)

    # Gex Ladder
    print("Calculating GEX Ladder")
    first_expiration_data, second_expiration_data, call_wall_0, put_wall_0, call_wall_1, put_wall_1, call_wall, put_wall, avg_wall_0, avg_wall_1, avg_wall = calculate_gex_ladder(df)
    max_call_strike = df.loc[df['NetGexCall1'].idxmax()]['StrikePrice']
    max_put_strike = df.loc[df['NetGexPut1'].idxmax()]['StrikePrice']
    max_call_strike_0 = first_expiration_data.loc[first_expiration_data['NetGexCall1'].idxmax()]['StrikePrice']
    max_put_strike_0 = first_expiration_data.loc[first_expiration_data['NetGexPut1'].idxmax()]['StrikePrice']
    max_call_strike_1 = second_expiration_data.loc[second_expiration_data['NetGexCall1'].idxmax()]['StrikePrice']
    max_put_strike_1 = second_expiration_data.loc[second_expiration_data['NetGexPut1'].idxmax()]['StrikePrice']

    df_sorted = df.sort_values(by=['ExpirationDate', 'StrikePrice'])

    # Filter the dataframe to get only the strikes between fromStrike and toStrike
    df_filtered = df_sorted[(df_sorted['StrikePrice'] >= fromStrike) & (df_sorted['StrikePrice'] <= toStrike)]

    # Create a dataframe for next expiration, next Friday and next weekly and next monthly
    nextExpiry = df_filtered.loc[df_filtered['ExpirationDate'] > datetime.now(), 'ExpirationDate'].min()
    secondExpiry = df_filtered.loc[df_filtered['ExpirationDate'] > nextExpiry, 'ExpirationDate'].min()
    nextMonthlyExp = df_filtered.loc[df_filtered['IsThirdFriday'], 'ExpirationDate'].min()
    nextWeeklyExp = get_next_friday(nextExpiry)

    # Create a dataframe for next expiration, next Friday and next weekly and next monthly
    dfNext = df_filtered.loc[df_filtered['ExpirationDate'] == nextExpiry]
    dfSecond = df_filtered.loc[df_filtered['ExpirationDate'] == secondExpiry]
    dfMonthly = df_filtered.loc[df_filtered['ExpirationDate'] == nextMonthlyExp]
    dfWeekly = df_filtered.loc[df_filtered['ExpirationDate'] == nextWeeklyExp]


    print("Calculating Flow Levels")
    vCallPrice, vPutPrice, vCallOiPrice, vPutOiPrice, vAvgPrice, vAvgOiPrice, vRes1Price, vSup1Price, vRes2Price, vSup2Price, vRes3Price, vSup3Price,  \
        vRes4Price, vSup4Price, resistances, supports, \
        redOi, greenOi, deltaOi, redVol, greenVol, deltaVol, ethBlue, ethPurple = calculate_flow_levels_for_expiration(df_filtered, spotPrice)

    vCallPrice_1, vPutPrice_1, vCallOiPrice_1, vPutOiPrice_1, vAvgPrice_1, vAvgOiPrice_1, vRes1Price_1, vSup1Price_1, vRes2Price_1, vSup2Price_1, vRes3Price_1, vSup3Price_1,  \
        vRes4Price_1, vSup4Price_1, resistances_1, supports_1, \
        redOi_1, greenOi_1, deltaOi_1, redVol_1, greenVol_1, deltaVol_1, ethBlue_1, ethPurple_1 = calculate_flow_levels_for_expiration(dfNext, spotPrice)   
    
    vCallPrice_2, vPutPrice_2, vCallOiPrice_2, vPutOiPrice_2, vAvgPrice_2, vAvgOiPrice_2, vRes1Price_2, vSup1Price_2, vRes2Price_2, vSup2Price_2, vRes3Price_2, vSup3Price_2, \
    vRes4Price_2, vSup4Price_2, resistances_2, supports_2, \
        redOi_2, greenOi_2, deltaOi_2, redVol_2, greenVol_2, deltaVol_2, ethBlue_2, ethPurple_2 = calculate_flow_levels_for_expiration(dfSecond, spotPrice)  
    
    vCallPrice_w, vPutPrice_w, vCallOiPrice_w, vPutOiPrice_w, vAvgPrice_w, vAvgOiPrice_w, vRes1Price_w, vSup1Price_w, vRes2Price_w, vSup2Price_w, vRes3Price_w, vSup3Price_w, \
    vRes4Price_w, vSup4Price_w, resistances_w, supports_w, \
        redOi_w, greenOi_w, deltaOi_w, redVol_w, greenVol_w, deltaVol_w, ethBlue_w, ethPurple_w = calculate_flow_levels_for_expiration(dfWeekly, spotPrice)
    
    vCallPrice_m, vPutPrice_m, vCallOiPrice_m, vPutOiPrice_m, vAvgPrice_m, vAvgOiPrice_m, vRes1Price_m, vSup1Price_m, vRes2Price_m, vSup2Price_m, vRes3Price_m, vSup3Price_m,  \
    vRes4Price_m, vSup4Price_m, resistances_m, supports_m, \
    redOi_m, greenOi_m, deltaOi_m, redVol_m, greenVol_m, deltaVol_m, ethBlue_m, ethPurple_m = calculate_flow_levels_for_expiration(dfMonthly, spotPrice)

    print("Calculating Gamma Levels")
    zeroGamma = find_zero_gamma_levels(df, fromStrike, toStrike)
    zeroGamma_1 = find_zero_gamma_levels(dfNext, fromStrike, toStrike)
    zeroGamma_2 = find_zero_gamma_levels(dfSecond, fromStrike, toStrike)
    zeroGamma_w = find_zero_gamma_levels(dfWeekly, fromStrike, toStrike)
    zeroGamma_m = find_zero_gamma_levels(dfMonthly, fromStrike, toStrike)

    print("Calculating Vanna Levels")
    zeroVanna = find_zero_vanna_levels(df, fromStrike, toStrike)
    zeroVanna_1 = find_zero_vanna_levels(dfNext, fromStrike, toStrike)
    zeroVanna_2 = find_zero_vanna_levels(dfSecond, fromStrike, toStrike)
    zeroVanna_w = find_zero_vanna_levels(dfWeekly, fromStrike, toStrike)
    zeroVanna_m = find_zero_vanna_levels(dfMonthly, fromStrike, toStrike)

    print("Calculating Additional Gex Levels")
    call_resistance_oi, put_support_oi, call_resistance_vol, put_support_vol, call_resistance_gex_oi, put_support_gex_oi, call_resistance_gex_vol, \
        put_support_gex_vol, call_resistance_wall, put_support_wall, call_resistance_net_gex_oi, put_support_net_gex_oi,  \
        call_resistance_net_gex_vol, put_support_net_gex_vol, call_resistance_vol_oi, put_support_vol_oi, call_resistance_net_vol_oi, put_support_net_vol_oi, \
        callbid_vol, putbid_vol, volume, calloi_vol, putoi_vol, open_interest = get_additional_gex_values(df_filtered, spotPrice)

    call_resistance_oi_1, put_support_oi_1, call_resistance_vol_1, put_support_vol_1, call_resistance_gex_oi_1, put_support_gex_oi_1, call_resistance_gex_vol_1, \
        put_support_gex_vol_1, call_resistance_wall_1, put_support_wall_1, call_resistance_net_gex_oi_1, put_support_net_gex_oi_1,  \
        call_resistance_net_gex_vol_1, put_support_net_gex_vol_1, call_resistance_vol_oi_1, put_support_vol_oi_1, call_resistance_net_vol_oi_1, put_support_net_vol_oi_1, \
        callbid_vol_1, putbid_vol_1, volume_1, calloi_vol_1, putoi_vol_1, open_interest_1 = get_additional_gex_values(dfNext, spotPrice)

    call_resistance_oi_2, put_support_oi_2, call_resistance_vol_2, put_support_vol_2, call_resistance_gex_oi_2, put_support_gex_oi_2, call_resistance_gex_vol_2, \
        put_support_gex_vol_2, call_resistance_wall_2, put_support_wall_2, call_resistance_net_gex_oi_2, put_support_net_gex_oi_2,  \
        call_resistance_net_gex_vol_2, put_support_net_gex_vol_2, call_resistance_vol_oi_2, put_support_vol_oi_2, call_resistance_net_vol_oi_2, put_support_net_vol_oi_2, \
        callbid_vol_2, putbid_vol_2, volume_2, calloi_vol_2, putoi_vol_2, open_interest_2 = get_additional_gex_values(dfSecond, spotPrice)
    
    call_resistance_oi_w, put_support_oi_w, call_resistance_vol_w, put_support_vol_w, call_resistance_gex_oi_w, put_support_gex_oi_w, call_resistance_gex_vol_w, \
        put_support_gex_vol_w, call_resistance_wall_w, put_support_wall_w, call_resistance_net_gex_oi_w, put_support_net_gex_oi_w,  \
        call_resistance_net_gex_vol_w, put_support_net_gex_vol_w, call_resistance_vol_oi_w, put_support_vol_oi_w, call_resistance_net_vol_oi_w, put_support_net_vol_oi_w, \
        callbid_vol_w, putbid_vol_w, volume_w, calloi_vol_w, putoi_vol_w, open_interest_w = get_additional_gex_values(dfWeekly, spotPrice)
    
    call_resistance_oi_m, put_support_oi_m, call_resistance_vol_m, put_support_vol_m, call_resistance_gex_oi_m, put_support_gex_oi_m, call_resistance_gex_vol_m, \
        put_support_gex_vol_m, call_resistance_wall_m, put_support_wall_m, call_resistance_net_gex_oi_m, put_support_net_gex_oi_m,  \
        call_resistance_net_gex_vol_m, put_support_net_gex_vol_m, call_resistance_vol_oi_m, put_support_vol_oi_m, call_resistance_net_vol_oi_m, put_support_net_vol_oi_m, \
        callbid_vol_m, putbid_vol_m, volume_m, calloi_vol_m, putoi_vol_m, open_interest_m = get_additional_gex_values(dfMonthly, spotPrice)
    
    zero_gamma, tot_vol, tot_oi, gamma_flip1, gamma_flip2, pain_volume_strike, pain_oi_strike, zero_pos_strike, zero_neg_strike = get_flip_pain_points(df_filtered)

    zero_gamma_1, tot_vol_1, tot_oi_1, gamma_flip1_1, gamma_flip2_1, pain_volume_strike_1, pain_oi_strike_1, zero_pos_strike_1, zero_neg_strike_1 = get_flip_pain_points(dfNext)

    zero_gamma_2, tot_vol_2, tot_oi_2, gamma_flip1_2, gamma_flip2_2, pain_volume_strike_2, pain_oi_strike_2, zero_pos_strike_2, zero_neg_strike_2 = get_flip_pain_points(dfSecond)

    zero_gamma_w, tot_vol_w, tot_oi_w, gamma_flip1_w, gamma_flip2_w, pain_volume_strike_w, pain_oi_strike_w, zero_pos_strike_w, zero_neg_strike_w = get_flip_pain_points(dfWeekly)

    zero_gamma_m, tot_vol_m, tot_oi_m, gamma_flip1_m, gamma_flip2_m, pain_volume_strike_m, pain_oi_strike_m, zero_pos_strike_m, zero_neg_strike_m = get_flip_pain_points(dfMonthly)
        
    print("Calculating All Expiration Levels")
    gexLadder = {
        'symbol': symbol,
        'processTime': today.strftime('%Y-%m-%d %H:%M:00'),
        'max_call_strike': max_call_strike,
        'max_put_strike': max_put_strike,
        'max_call_strike_0': max_call_strike_0,
        'max_put_strike_0': max_put_strike_0,
        'call_wall_0': call_wall_0,
        'put_wall_0': put_wall_0,
        'max_call_strike_1': max_call_strike_1,
        'max_put_strike_1': max_put_strike_1,
        'call_wall_1': call_wall_1,
        'put_wall_1': put_wall_1,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'avg_wall_0': avg_wall_0,
        'avg_wall_1': avg_wall_1,
        'avg_wall': avg_wall,
        'spotPrice': spotPrice,
    }
    
    allExpiration = {
        'symbol': symbol,
        'expiration': 'All',
        'processTime': today.strftime('%Y-%m-%d %H:%M:00'),
        'vol_call': vCallPrice,
        'vol_put': vPutPrice,
        'oi_call': vCallOiPrice,
        'oi_put': vPutOiPrice,
        'vol_avg': vAvgPrice,
        'oi_avg': vAvgOiPrice,
        'vol_resistance1': vRes1Price,
        'vol_support1': vSup1Price,
        'vol_resistance2': vRes2Price,
        'vol_support2': vSup2Price,
        'vol_resistance3': vRes3Price,
        'vol_support3': vSup3Price,
        'vol_resistance4': vRes4Price,
        'vol_support4': vSup4Price,
        'resistances': [x for x in resistances],
        'supports': [x for x in supports],
        'zero_gamma': zeroGamma,
        'zero_vanna': zeroVanna,
        'callFlow': greenOi,
        'deltaFlow': deltaOi,
        'putFlow': redOi,
        'callOrderFlow': greenVol,
        'putOrderFlow': redVol,
        'deltaOrderFlow': deltaVol,
        'ethBlue': ethBlue,
        'ethPurple': ethPurple,
        'call_resistance_oi': call_resistance_oi,
        'put_support_oi': put_support_oi,
        'call_resistance_vol': call_resistance_vol,
        'put_support_vol': put_support_vol,
        'call_resistance_gex_oi': call_resistance_gex_oi,
        'put_support_gex_oi': put_support_gex_oi,
        'call_resistance_gex_vol': call_resistance_gex_vol,
        'put_support_gex_vol': put_support_gex_vol,
        'call_resistance_wall': call_resistance_wall,
        'put_support_wall': put_support_wall,
        'call_resistance_net_gex_oi': call_resistance_net_gex_oi,
        'put_support_net_gex_oi': put_support_net_gex_oi,
        'call_resistance_net_gex_vol': call_resistance_net_gex_vol,
        'put_support_net_gex_vol': put_support_net_gex_vol,
        'call_resistance_vol_oi': call_resistance_vol_oi,
        'put_support_vol_oi': put_support_vol_oi,
        'call_resistance_net_vol_oi': call_resistance_net_vol_oi,
        'put_support_net_vol_oi': put_support_net_vol_oi,
        'callbid_vol': callbid_vol,
        'putbid_vol': putbid_vol,
        'tot_vol': volume,
        'calloi_vol': calloi_vol,
        'putoi_vol': putoi_vol,
        'tot_oi': open_interest,
        'zero_gamma1': zero_gamma,
        'gamma_flip1': gamma_flip1,
        'gamma_flip2': gamma_flip2,
        'pain_volume_strike': pain_volume_strike,
        'pain_oi_strike': pain_oi_strike,
        'zero_pos_strike': zero_pos_strike,
        'zero_neg_strike': zero_neg_strike,
        'tot_vol_ratio': tot_vol,
        'tot_oi_ratio': tot_oi,
        'spotPrice': spotPrice,
    }

    print("Calculating First Expiration Levels")
    firstExpiration = {
        'symbol': symbol,
        'expiration': 'First',
        'processTime': today.strftime('%Y-%m-%d %H:%M:00'),
        'vol_call_1': vCallPrice_1,
        'vol_put_1': vPutPrice_1,
        'oi_call_1': vCallOiPrice_1,
        'oi_put_1': vPutOiPrice_1,
        'vol_avg_1': vAvgPrice_1,
        'oi_avg_1': vAvgOiPrice_1,
        'vol_resistance1_1': vRes1Price_1,
        'vol_support1_1': vSup1Price_1,
        'vol_resistance2_1': vRes2Price_1,
        'vol_support2_1': vSup2Price_1,
        'vol_resistance3_1': vRes3Price_1,
        'vol_support3_1': vSup3Price_1,
        'vol_resistance4_1': vRes4Price_1,
        'vol_support4_1': vSup4Price_1,
        'resistances_1': [x for x in resistances_1],
        'supports_1': [x for x in supports_1],
        'zero_gamma_1': zeroGamma_1,
        'zero_vanna_1': zeroVanna_1,
        'callFlow_1': greenOi_1,
        'deltaFlow_1': deltaOi_1,
        'putFlow_1': redOi_1,
        'callOrderFlow_1': greenVol_1,
        'putOrderFlow_1': redVol_1,
        'deltaOrderFlow_1': deltaVol_1,
        'ethBlue_1': ethBlue_1,
        'ethPurple_1': ethPurple_1,
        'call_resistance_oi_1': call_resistance_oi_1,
        'put_support_oi_1': put_support_oi_1,
        'call_resistance_vol_1': call_resistance_vol_1,
        'put_support_vol_1': put_support_vol_1,
        'call_resistance_gex_oi_1': call_resistance_gex_oi_1,
        'put_support_gex_oi_1': put_support_gex_oi_1,
        'call_resistance_gex_vol_1': call_resistance_gex_vol_1,
        'put_support_gex_vol_1': put_support_gex_vol_1,
        'call_resistance_wall_1': call_resistance_wall_1,
        'put_support_wall_1': put_support_wall_1,
        'call_resistance_net_gex_oi_1': call_resistance_net_gex_oi_1,
        'put_support_net_gex_oi_1': put_support_net_gex_oi_1,
        'call_resistance_net_gex_vol_1': call_resistance_net_gex_vol_1,
        'put_support_net_gex_vol_1': put_support_net_gex_vol_1,
        'call_resistance_vol_oi_1': call_resistance_vol_oi_1,
        'put_support_vol_oi_1': put_support_vol_oi_1,
        'call_resistance_net_vol_oi_1': call_resistance_net_vol_oi_1,
        'put_support_net_vol_oi_1': put_support_net_vol_oi_1,
        'callbid_vol_1': callbid_vol_1,
        'putbid_vol_1': putbid_vol_1,
        'tot_vol_1': volume_1,
        'calloi_vol_1': calloi_vol_1,
        'putoi_vol_1': putoi_vol_1,
        'tot_oi_1': open_interest_1,
        'zero_gamma1_1': zero_gamma_1,
        'gamma_flip1_1': gamma_flip1_1,
        'gamma_flip2_1': gamma_flip2_1,
        'pain_volume_strike_1': pain_volume_strike_1,
        'pain_oi_strike_1': pain_oi_strike_1,
        'zero_pos_strike_1': zero_pos_strike_1,
        'zero_neg_strike_1': zero_neg_strike_1,
        'tot_vol_ratio_1': tot_vol_1,
        'tot_oi_ratio_1': tot_oi_1,
        'spotPrice': spotPrice,
    }

    print("Calculating Second Expiration Levels")
    secondExpiration = {
        'symbol': symbol,
        'expiration': 'Second',
        'processTime': today.strftime('%Y-%m-%d %H:%M:00'),
        'vol_call_2': vCallPrice_2,
        'vol_put_2': vPutPrice_2,
        'oi_call_2': vCallOiPrice_2,
        'oi_put_2': vPutOiPrice_2,
        'vol_avg_2': vAvgPrice_2,
        'oi_avg_2': vAvgOiPrice_2,
        'vol_resistance1_2': vRes1Price_2,
        'vol_support1_2': vSup1Price_2,
        'vol_resistance2_2': vRes2Price_2,
        'vol_support2_2': vSup2Price_2,
        'vol_resistance3_2': vRes3Price_2,
        'vol_support3_2': vSup3Price_2,
        'vol_resistance4_2': vRes4Price_2,
        'vol_support4_2': vSup4Price_2,
        'resistances_2': [x for x in resistances_2],
        'supports_2': [x for x in supports_2],
        'zero_gamma_2': zeroGamma_2,
        'zero_vanna_2': zeroVanna_2,
        'callFlow_2': greenOi_2,
        'deltaFlow_2': deltaOi_2,
        'putFlow_2': redOi_2,
        'callOrderFlow_2': greenVol_2,
        'putOrderFlow_2': redVol_2,
        'deltaOrderFlow_2': deltaVol_2,
        'ethBlue_2': ethBlue_2,
        'ethPurple_2': ethPurple_2,
        'call_resistance_oi_2': call_resistance_oi_2,
        'put_support_oi_2': put_support_oi_2,
        'call_resistance_vol_2': call_resistance_vol_2,
        'put_support_vol_2': put_support_vol_2,
        'call_resistance_gex_oi_2': call_resistance_gex_oi_2,
        'put_support_gex_oi_2': put_support_gex_oi_2,
        'call_resistance_gex_vol_2': call_resistance_gex_vol_2,
        'put_support_gex_vol_2': put_support_gex_vol_2,
        'call_resistance_wall_2': call_resistance_wall_2,
        'put_support_wall_2': put_support_wall_2,
        'call_resistance_net_gex_oi_2': call_resistance_net_gex_oi_2,
        'put_support_net_gex_oi_2': put_support_net_gex_oi_2,
        'call_resistance_net_gex_vol_2': call_resistance_net_gex_vol_2,
        'put_support_net_gex_vol_2': put_support_net_gex_vol_2,
        'call_resistance_vol_oi_2': call_resistance_vol_oi_2,
        'put_support_vol_oi_2': put_support_vol_oi_2,
        'call_resistance_net_vol_oi_2': call_resistance_net_vol_oi_2,
        'put_support_net_vol_oi_2': put_support_net_vol_oi_2,
        'callbid_vol_2': callbid_vol_2,
        'putbid_vol_2': putbid_vol_2,
        'tot_vol_2': volume_2,
        'calloi_vol_2': calloi_vol_2,
        'putoi_vol_2': putoi_vol_2,
        'tot_oi_2': open_interest_2,
        'zero_gamma1_2': zero_gamma_2,
        'gamma_flip1_2': gamma_flip1_2,
        'gamma_flip2_2': gamma_flip2_2,
        'pain_volume_strike_2': pain_volume_strike_2,
        'pain_oi_strike_2': pain_oi_strike_2,
        'zero_pos_strike_2': zero_pos_strike_2,
        'zero_neg_strike_2': zero_neg_strike_2,
        'tot_vol_ratio_2': tot_vol_2,
        'tot_oi_ratio_2': tot_oi_2,
        'spotPrice': spotPrice,
    }

    print("Calculating Weekly Expiration Levels")
    weeklyExpiration = {
        'symbol': symbol,
        'expiration': 'Weekly',
        'processTime': today.strftime('%Y-%m-%d %H:%M:00'),
        'vol_call_w': vCallPrice_w,
        'vol_put_w': vPutPrice_w,
        'oi_call_w': vCallOiPrice_w,
        'oi_put_w': vPutOiPrice_w,
        'vol_avg_w': vAvgPrice_w,
        'oi_avg_w': vAvgOiPrice_w,
        'vol_resistance1_w': vRes1Price_w,
        'vol_support1_w': vSup1Price_w,
        'vol_resistance2_w': vRes2Price_w,
        'vol_support2_w': vSup2Price_w,
        'vol_resistance3_w': vRes3Price_w,
        'vol_support3_w': vSup3Price_w,
        'vol_resistance4_w': vRes4Price_w,
        'vol_support4_w': vSup4Price_w,
        'resistances_w': [x for x in resistances_w],
        'supports_w': [x for x in supports_w],
        'zero_gamma_w': zeroGamma_w,
        'zero_vanna_w': zeroVanna_w,
        'callFlow_w': greenOi_w,
        'deltaFlow_w': deltaOi_w,
        'putFlow_w': redOi_w,
        'callOrderFlow_w': greenVol_w,
        'putOrderFlow_w': redVol_w,
        'deltaOrderFlow_w': deltaVol_w,
        'ethBlue_w': ethBlue_w,
        'ethPurple_w': ethPurple_w,
        'call_resistance_oi_w': call_resistance_oi_w,
        'put_support_oi_w': put_support_oi_w,
        'call_resistance_vol_w': call_resistance_vol_w,
        'put_support_vol_w': put_support_vol_w,
        'call_resistance_gex_oi_w': call_resistance_gex_oi_w,
        'put_support_gex_oi_w': put_support_gex_oi_w,
        'call_resistance_gex_vol_w': call_resistance_gex_vol_w,
        'put_support_gex_vol_w': put_support_gex_vol_w,
        'call_resistance_wall_w': call_resistance_wall_w,
        'put_support_wall_w': put_support_wall_w,
        'call_resistance_net_gex_oi_w': call_resistance_net_gex_oi_w,
        'put_support_net_gex_oi_w': put_support_net_gex_oi_w,
        'call_resistance_net_gex_vol_w': call_resistance_net_gex_vol_w,
        'put_support_net_gex_vol_w': put_support_net_gex_vol_w,
        'call_resistance_vol_oi_w': call_resistance_vol_oi_w,
        'put_support_vol_oi_w': put_support_vol_oi_w,
        'call_resistance_net_vol_oi_w': call_resistance_net_vol_oi_w,
        'put_support_net_vol_oi_w': put_support_net_vol_oi_w,
        'callbid_vol_w': callbid_vol_w,
        'putbid_vol_w': putbid_vol_w,
        'tot_vol_w': volume_w,
        'calloi_vol_w': calloi_vol_w,
        'putoi_vol_w': putoi_vol_w,
        'tot_oi_w': open_interest_w,
        'zero_gamma1_w': zero_gamma_w,
        'gamma_flip1_w': gamma_flip1_w,
        'gamma_flip2_w': gamma_flip2_w,
        'pain_volume_strike_w': pain_volume_strike_w,
        'pain_oi_strike_w': pain_oi_strike_w,
        'zero_pos_strike_w': zero_pos_strike_w,
        'zero_neg_strike_w': zero_neg_strike_w,
        'tot_vol_ratio_w': tot_vol_w,
        'tot_oi_ratio_w': tot_oi_w,
        'spotPrice': spotPrice,
    }

    print ("Calculating Monthly Expiration Levels")
    monthlyExpiration = {
        'symbol': symbol,
        'expiration': 'Monthly',
        'processTime': today.strftime('%Y-%m-%d %H:%M:00'),
        'vol_call_m': vCallPrice_m,
        'vol_put_m': vPutPrice_m,
        'oi_call_m': vCallOiPrice_m,
        'oi_put_m': vPutOiPrice_m,
        'vol_avg_m': vAvgPrice_m,
        'oi_avg_m': vAvgOiPrice_m,
        'vol_resistance1_m': vRes1Price_m,
        'vol_support1_m': vSup1Price_m,
        'vol_resistance2_m': vRes2Price_m,
        'vol_support2_m': vSup2Price_m,
        'vol_resistance3_m': vRes3Price_m,
        'vol_support3_m': vSup3Price_m,
        'vol_resistance4_m': vRes4Price_m,
        'vol_support4_m': vSup4Price_m,
        'resistances_m': [x for x in resistances_m],
        'supports_m': [x for x in supports_m],
        'zero_gamma_m': zeroGamma_m,
        'zero_vanna_m': zeroVanna_m,
        'callFlow_m': greenOi_m,
        'deltaFlow_m': deltaOi_m,
        'putFlow_m': redOi_m,
        'callOrderFlow_m': greenVol_m,
        'putOrderFlow_m': redVol_m,
        'deltaOrderFlow_m': deltaVol_m,
        'ethBlue_m': ethBlue_m,
        'ethPurple_m': ethPurple_m,
        'call_resistance_oi_m': call_resistance_oi_m,
        'put_support_oi_m': put_support_oi_m,
        'call_resistance_vol_m': call_resistance_vol_m,
        'put_support_vol_m': put_support_vol_m,
        'call_resistance_gex_oi_m': call_resistance_gex_oi_m,
        'put_support_gex_oi_m': put_support_gex_oi_m,
        'call_resistance_gex_vol_m': call_resistance_gex_vol_m,
        'put_support_gex_vol_m': put_support_gex_vol_m,
        'call_resistance_wall_m': call_resistance_wall_m,
        'put_support_wall_m': put_support_wall_m,
        'call_resistance_net_gex_oi_m': call_resistance_net_gex_oi_m,
        'put_support_net_gex_oi_m': put_support_net_gex_oi_m,
        'call_resistance_net_gex_vol_m': call_resistance_net_gex_vol_m,
        'put_support_net_gex_vol_m': put_support_net_gex_vol_m,
        'call_resistance_vol_oi_m': call_resistance_vol_oi_m,
        'put_support_vol_oi_m': put_support_vol_oi_m,
        'call_resistance_net_vol_oi_m': call_resistance_net_vol_oi_m,
        'put_support_net_vol_oi_m': put_support_net_vol_oi_m,
        'callbid_vol_m': callbid_vol_m,
        'putbid_vol_m': putbid_vol_m,
        'tot_vol_m': volume_m,
        'calloi_vol_m': calloi_vol_m,
        'putoi_vol_m': putoi_vol_m,
        'tot_oi_m': open_interest_m,
        'zero_gamma1_m': zero_gamma_m,
        'gamma_flip1_m': gamma_flip1_m,
        'gamma_flip2_m': gamma_flip2_m,
        'pain_volume_strike_m': pain_volume_strike_m,
        'pain_oi_strike_m': pain_oi_strike_m,
        'zero_pos_strike_m': zero_pos_strike_m,
        'zero_neg_strike_m': zero_neg_strike_m,
        'tot_vol_ratio_m': tot_vol_m,
        'tot_oi_ratio_m': tot_oi_m,
        'spotPrice': spotPrice,
    }
    
    #gex_flow_levels = pd.DataFrame()
    #gex_flow_levels = pd.concat([gex_flow_levels, pd.DataFrame([esData])])
    # Round values to two decimal places
    #columns_to_round = [col for col in gex_flow_levels.columns if col != 'priceRatio']
    #gex_flow_levels[columns_to_round] = gex_flow_levels[columns_to_round].round(2)

    #eturn gex_flow_levels, df
    return gexLadder, allExpiration, firstExpiration, secondExpiration, weeklyExpiration, monthlyExpiration, df

def round_dict_values(data, instrument, exclude_keys=None):
    """
    Round all numerical values in a nested dictionary to the nearest tick based on the instrument.
    Round excluded keys to 2 decimal places.

    :param data: Dictionary with numerical values.
    :param instrument: The instrument (e.g., 'MES', 'MNQ', 'M2K').
    :param exclude_keys: List of keys to round to 2 decimal places instead of nearest tick.
    :return: Rounded dictionary.
    """

    # Function to get tick size
    def get_tick_size(symbol):
        tick_sizes = {'MES': 0.25, 'MNQ': 0.25, 'M2K': 0.1, 'SPY' : 0.01, 'QQQ' : 0.01, 'IWM' : 0.01}
        if symbol == '_SPX':
            return 0.01  # Two increments of 0.01 for _SPX
        elif symbol.isalpha():  # Assume all equities are alphabetic symbols
            return 0.01
        else:
            return tick_sizes.get(symbol, None)  # Return from predefined values if available
    
    def round_to_ticks(value, instrument):
        # Define tick sizes for instruments
        tick_size = get_tick_size(instrument)
        if tick_size is None:
            raise ValueError(f"Unsupported instrument: {instrument}")
        return round(value / tick_size) * tick_size

    def round_nested(obj):
        if isinstance(obj, dict):
            return {
                key: (
                    round(value, 4) if key in exclude_keys else round_nested(value)
                ) if isinstance(value, (int, float)) else round_nested(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [round_nested(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return round_to_ticks(obj, instrument)
        else:
            return obj

    # Default exclude_keys to an empty list if None
    exclude_keys = exclude_keys or []
    return round_nested(data)

def get_levels(symbol):
    print("Getting GEX and Flow Levels")
    today = pd.to_datetime(datetime.now())
    gexLadder, allExpiration, firstExpiration, secondExpiration, weeklyExpiration, monthlyExpiration, df = get_gex_and_flow_levels(symbol, today)

    # Round the new data based on tick size
    exclude_keys = ["callFlow_1", "deltaFlow_1", "putFlow_1", "callOrderFlow_1", "putOrderFlow_1",
                    "deltaOrderFlow_1", "ethBlue_1", "ethPurple_1", "callbid_vol_1", "putbid_vol_1",
                    "tot_vol_1", "calloi_vol_1", "putoi_vol_1", "tot_oi_1", "tot_vol_ratio_1", "tot_oi_ratio_1",
                    "callFlow_2", "deltaFlow_2", "putFlow_2", "callOrderFlow_2", "putOrderFlow_2",
                    "deltaOrderFlow_2", "ethBlue_2", "ethPurple_2", "callbid_vol_2", "putbid_vol_2",
                    "tot_vol_2", "calloi_vol_2", "putoi_vol_2", "tot_oi_2", "tot_vol_ratio_2", "tot_oi_ratio_2",
                    "callFlow_w", "deltaFlow_w", "putFlow_w", "callOrderFlow_w", "putOrderFlow_w",
                    "deltaOrderFlow_w", "ethBlue_w", "ethPurple_w", "callbid_vol_w", "putbid_vol_w",
                    "tot_vol_w", "calloi_vol_w", "putoi_vol_w", "tot_oi_w", "tot_vol_ratio_w", "tot_oi_ratio_w",
                    "callFlow_m", "deltaFlow_m", "putFlow_m", "callOrderFlow_m", "putOrderFlow_m",
                    "deltaOrderFlow_m", "ethBlue_m", "ethPurple_m", "callbid_vol_m", "putbid_vol_m",
                    "tot_vol_m", "calloi_vol_m", "putoi_vol_m", "tot_oi_m", "tot_vol_ratio_m", "tot_oi_ratio_m",
                    "callFlow", "deltaFlow", "putFlow", "callOrderFlow", "putOrderFlow",
                    "deltaOrderFlow", "ethBlue", "ethPurple", "callbid_vol", "putbid_vol",
                    "tot_vol", "calloi_vol", "putoi_vol", "tot_oi", "tot_vol_ratio", "tot_oi_ratio",
                    "priceRatio"]  # Exclude "excluded" from rounding
    gexLadder_new = round_dict_values(gexLadder, symbol, exclude_keys)

    # Append the data into dataframe
    gex_ladder = pd.DataFrame()
    gex_ladder = pd.concat([gex_ladder, pd.DataFrame([gexLadder_new])])

    gex_flow_and_levels = pd.DataFrame()
    firstExpiration_new = round_dict_values(firstExpiration, symbol, exclude_keys)
    secondExpiration_new = round_dict_values(secondExpiration, symbol, exclude_keys)
    weeklyExpiration_new = round_dict_values(weeklyExpiration, symbol, exclude_keys)
    monthlyExpiration_new = round_dict_values(monthlyExpiration, symbol, exclude_keys)
    allExpiration_new = round_dict_values(allExpiration, symbol, exclude_keys)
    gex_flow_and_levels = pd.concat([gex_flow_and_levels, pd.DataFrame([allExpiration_new, firstExpiration_new, secondExpiration_new, weeklyExpiration_new, monthlyExpiration_new])])

    return gex_ladder, gex_flow_and_levels, df

def write_or_append_gex_data(instrument, new_data, fileName, dataPath, latestFileName):
    """
    Write or append gexLadder data to a JSON file. Create directories if needed.
    Rounds numerical values in new_data based on the instrument's tick size.
    
    :param new_data: The dictionary or list object to write/append.
    :param fileName: The file to write or append to.
    :param dataPath: The directory path for the file.
    :param latestFileName: The file to store the latest gex data.
    :param instrument: The instrument (e.g., 'MES', 'MNQ', 'M2K') for rounding.
    """
    # Round the new data based on tick size
    exclude_keys = ["callFlow_1", "deltaFlow_1", "putFlow_1", "callOrderFlow_1", "putOrderFlow_1",
                    "deltaOrderFlow_1", "ethBlue_1", "ethPurple_1", "callbid_vol_1", "putbid_vol_1",
                    "tot_vol_1", "calloi_vol_1", "putoi_vol_1", "tot_oi_1", "tot_vol_ratio_1", "tot_oi_ratio_1",
                    "callFlow_2", "deltaFlow_2", "putFlow_2", "callOrderFlow_2", "putOrderFlow_2",
                    "deltaOrderFlow_2", "ethBlue_2", "ethPurple_2", "callbid_vol_2", "putbid_vol_2",
                    "tot_vol_2", "calloi_vol_2", "putoi_vol_2", "tot_oi_2", "tot_vol_ratio_2", "tot_oi_ratio_2",
                    "callFlow_w", "deltaFlow_w", "putFlow_w", "callOrderFlow_w", "putOrderFlow_w",
                    "deltaOrderFlow_w", "ethBlue_w", "ethPurple_w", "callbid_vol_w", "putbid_vol_w",
                    "tot_vol_w", "calloi_vol_w", "putoi_vol_w", "tot_oi_w", "tot_vol_ratio_w", "tot_oi_ratio_w",
                    "callFlow_m", "deltaFlow_m", "putFlow_m", "callOrderFlow_m", "putOrderFlow_m",
                    "deltaOrderFlow_m", "ethBlue_m", "ethPurple_m", "callbid_vol_m", "putbid_vol_m",
                    "tot_vol_m", "calloi_vol_m", "putoi_vol_m", "tot_oi_m", "tot_vol_ratio_m", "tot_oi_ratio_m",
                    "callFlow", "deltaFlow", "putFlow", "callOrderFlow", "putOrderFlow",
                    "deltaOrderFlow", "ethBlue", "ethPurple", "callbid_vol", "putbid_vol",
                    "tot_vol", "calloi_vol", "putoi_vol", "tot_oi", "tot_vol_ratio", "tot_oi_ratio",
                    "priceRatio"]  # Exclude "excluded" from rounding
    #new_data = round_dict_values(new_data, instrument)
    new_data = round_dict_values(new_data, instrument, exclude_keys)


    # Convert new_data to an array if it's a single dictionary
    if isinstance(new_data, dict):
        new_data = [new_data]

    # Check if the file exists
    if os.path.exists(fileName):
        # Read existing data
        with open(fileName, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []  # Initialize as empty if the file is corrupted or empty

        # Append or merge based on the structure of existing data
        if isinstance(existing_data, list):
            if isinstance(new_data, list):
                existing_data.extend(new_data)  # Extend if both are lists
            else:
                existing_data.append(new_data)  # Append if new_data is a single dictionary
        elif isinstance(existing_data, dict):
            if isinstance(new_data, dict):
                existing_data.update(new_data)  # Merge dictionaries
            else:
                raise ValueError(f"Cannot append list data to a dictionary in {fileName}.")
        else:
            raise ValueError(f"Unsupported data type in {fileName}: {type(existing_data)}.")

        # Write back the updated data
        with open(fileName, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        # Create directory if needed
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        # Write new data to file
        with open(fileName, "w") as outfile:
            json.dump(new_data, outfile)

    # Overwrite the latest gex data to the latest file
    with open(latestFileName, "w") as outfile:
        json.dump(new_data, outfile)

