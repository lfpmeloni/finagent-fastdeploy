import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from helpers.dutils import decorate_all_methods
from helpers.summarizeutils import get_next_weekday


# from finrobot.utils import decorate_all_methods, get_next_weekday
from functools import wraps
from typing import Annotated, List

def init_fmp_api(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global fmp_api_key
        if os.environ.get("FMP_API_KEY") is None:
            print("Please set the environment variable FMP_API_KEY to use the FMP API.")
            return None
        else:
            fmp_api_key = os.environ["FMP_API_KEY"]
            print("FMP api key found successfully.")
            return func(*args, **kwargs)

    return wrapper


@decorate_all_methods(init_fmp_api)
class fmpUtils:

    def get_target_price(
        ticker_symbol: Annotated[str, "ticker symbol"],
        date: Annotated[str, "date of the target price, should be 'yyyy-mm-dd'"],
    ) -> str:
        """Get the target price for a given stock on a given date"""
        # API URL
        url = f"https://financialmodelingprep.com/api/v4/price-target?symbol={ticker_symbol}&apikey={fmp_api_key}"

        price_target = "Not Given"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            est = []

            date = datetime.strptime(date, "%Y-%m-%d")
            for tprice in data:
                tdate = tprice["publishedDate"].split("T")[0]
                tdate = datetime.strptime(tdate, "%Y-%m-%d")
                if abs((tdate - date).days) <= 999:
                    est.append(tprice["priceTarget"])

            if est:
                price_target = f"{np.min(est)} - {np.max(est)} (md. {np.median(est)})"
            else:
                price_target = "N/A"
        else:
            return f"Failed to retrieve data: {response.status_code}"

        return price_target

    def get_company_profile(
        ticker_symbol: Annotated[str, "ticker symbol"],
    ) -> str:
        """Get the url and filing date of the 10-K report for a given stock and year"""

        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}?apikey={fmp_api_key}"

        news = None
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            companyName = data[0]["companyName"]
            sector = data[0]["sector"]
            ipoDate = data[0]["ipoDate"]
            mktCap = data[0]["mktCap"]
            currency = data[0]["currency"]
            country = data[0]["country"]
            symbol = data[0]["symbol"]
            exchange = data[0]["exchange"]
            industry = data[0]["industry"]
            description = data[0]["description"]

            # print(data)
            if len(data) == 0:
                print(f"No profile found for symbol {ticker_symbol} from fmp!")
            formatted_str = (
            f"[Company Introduction]:\n\n{companyName} is a leading entity in the {sector} sector. "
            f"Incorporated and publicly traded since {ipoDate}, the company has established its reputation as "
            f"one of the key players in the market. As of today, {companyName} has a market capitalization "
            f"of {mktCap:.2f} in {currency}."
            f"\n\n{companyName} operates primarily in the {country}, trading under the ticker {symbol} on the {exchange}. "
            f"As a dominant force in the {industry} space, the company continues to innovate and drive "
            f"progress within the industry.  The detail description on the company's business and products are: {description}"
            )

            return formatted_str
        else:
            return f"Failed to retrieve data: {response.status_code}"
        
    def get_company_news(
        ticker_symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[
            str,
            "start date of the search period for the company's basic financials, yyyy-mm-dd",
        ],
        end_date: Annotated[
            str,
            "end date of the search period for the company's basic financials, yyyy-mm-dd",
        ],
        max_news_num: Annotated[
            int, "maximum number of news to return, default to 10"
        ] = 25,
    ) -> pd.DataFrame:
        """Get the url and filing date of the 10-K report for a given stock and year"""

        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker_symbol}&apikey={fmp_api_key}"

        news = None
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            # print(data)
            if len(data) == 0:
                print(f"No company news found for symbol {ticker_symbol} from fmp!")
            news = [
                {
                    #"date": datetime.fromtimestamp(n["publishedDate"]).strftime("%Y-%m-%d %H%M%S"),
                    "date": n["publishedDate"],
                    "headline": n["title"],
                    "summary": n["text"],
                }
                for n in data
            ]

            if len(news) > max_news_num:
                news = random.choices(news, k=max_news_num)
            news.sort(key=lambda x: x["date"])
            output = pd.DataFrame(news)
            return output
        else:
            return f"Failed to retrieve data: {response.status_code}"
        
    def get_sec_report(
        ticker_symbol: Annotated[str, "ticker symbol"],
        fyear: Annotated[
            str,
            "year of the 10-K report, should be 'yyyy' or 'latest'. Default to 'latest'",
        ] = "latest",
    ) -> str:
        """Get the url and filing date of the 10-K report for a given stock and year"""

        url = f"https://financialmodelingprep.com/api/v3/sec_filings/{ticker_symbol}?type=10-k&page=0&apikey={fmp_api_key}"

        filing_url = None
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            # print(data)
            if fyear == "latest":
                filing_url = data[0]["finalLink"]
                filing_date = data[0]["fillingDate"]
            else:
                for filing in data:
                    if filing["fillingDate"].split("-")[0] == fyear:
                        filing_url = filing["finalLink"]
                        filing_date = filing["fillingDate"]
                        break

            return f"Link: {filing_url}\nFiling Date: {filing_date}"
        else:
            return f"Failed to retrieve data: {response.status_code}"

    def get_earning_calls(
        ticker_symbol: Annotated[str, "ticker symbol"],
        year: Annotated[
            str,
            "year of the earning calls, should be 'yyyy' or 'latest'. Default to 'latest'",
        ] = "latest",
    ) -> str:
        """Get the url and filing date of the 10-K report for a given stock and year"""

        if year is None or year == "latest":
            year = datetime.now().year
            if datetime.now().month < 3:
                year = int(year) - 1

        url = f"https://financialmodelingprep.com/api/v4/batch_earning_call_transcript/{ticker_symbol}?year={year}&apikey={fmp_api_key}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            # transcripts = [
            #     {
            #         "quarter": n["quarter"],
            #         "year": n["year"],
            #         "date": n["date"],
            #         "content": n["content"],
            #     }
            #     for n in data
            # ]
            transcripts = data[0]['content']
            #output = pd.DataFrame(transcripts)
            return transcripts
        else:
            return f"Failed to retrieve data: {response.status_code}"
        
    def get_historical_market_cap(
        ticker_symbol: Annotated[str, "ticker symbol"],
        date: Annotated[str, "date of the market cap, should be 'yyyy-mm-dd'"],
    ) -> str:
        """Get the historical market capitalization for a given stock on a given date"""
        date = get_next_weekday(date).strftime("%Y-%m-%d")
        url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker_symbol}?limit=100&from={date}&to={date}&apikey={fmp_api_key}"

        mkt_cap = None
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            mkt_cap = data[0]["marketCap"]
            return mkt_cap
        else:
            return f"Failed to retrieve data: {response.status_code}"

    def get_historical_bvps(
        ticker_symbol: Annotated[str, "ticker symbol"],
        target_date: Annotated[str, "date of the BVPS, should be 'yyyy-mm-dd'"],
    ) -> str:
        """Get the historical book value per share for a given stock on a given date"""
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker_symbol}?limit=40&apikey={fmp_api_key}"
        response = requests.get(url)
        data = response.json()

        if not data:
            return "No data available"

        closest_data = None
        min_date_diff = float("inf")
        target_date = datetime.strptime(target_date, "%Y-%m-%d")
        for entry in data:
            date_of_data = datetime.strptime(entry["date"], "%Y-%m-%d")
            date_diff = abs(target_date - date_of_data).days
            if date_diff < min_date_diff:
                min_date_diff = date_diff
                closest_data = entry

        if closest_data:
            return closest_data.get("bookValuePerShare", "No BVPS data available")
        else:
            return "No close date data found"
        
    def get_financial_metrics(
        ticker_symbol: Annotated[str, "ticker symbol"],
        years: Annotated[int, "number of the years to search from, default to 4"] = 4
    ) -> pd.DataFrame:
        """Get the financial metrics for a given stock for the last 'years' years"""
        # Base URL setup for FMP API
        base_url = "https://financialmodelingprep.com/api/v3"
        # Create DataFrame
        df = pd.DataFrame()

        # Iterate over the last 'years' years of data
        for year_offset in range(years):
            # Construct URL for income statement and ratios for each year
            income_statement_url = f"{base_url}/income-statement/{ticker_symbol}?limit={years}&apikey={fmp_api_key}"
            ratios_url = (
                f"{base_url}/ratios/{ticker_symbol}?limit={years}&apikey={fmp_api_key}"
            )
            key_metrics_url = f"{base_url}/key-metrics/{ticker_symbol}?limit={years}&apikey={fmp_api_key}"

            # Requesting data from the API
            income_data = requests.get(income_statement_url).json()
            key_metrics_data = requests.get(key_metrics_url).json()
            ratios_data = requests.get(ratios_url).json()

            # Extracting needed metrics for each year
            if income_data and key_metrics_data and ratios_data:
                metrics = {
                    "Revenue": round(income_data[year_offset]["revenue"] / 1e6),
                    "Revenue Growth": "{}%".format(round(((income_data[year_offset]["revenue"] - income_data[year_offset - 1]["revenue"]) / income_data[year_offset - 1]["revenue"])*100,1)),
                    "Gross Revenue": round(income_data[year_offset]["grossProfit"] / 1e6),
                    "Gross Margin": round((income_data[year_offset]["grossProfit"] / income_data[year_offset]["revenue"]),2),
                    "EBITDA": round(income_data[year_offset]["ebitda"] / 1e6),
                    "EBITDA Margin": round((income_data[year_offset]["ebitdaratio"]),2),
                    "FCF": round(key_metrics_data[year_offset]["enterpriseValue"] / key_metrics_data[year_offset]["evToOperatingCashFlow"] / 1e6),
                    "FCF Conversion": round(((key_metrics_data[year_offset]["enterpriseValue"] / key_metrics_data[year_offset]["evToOperatingCashFlow"]) / income_data[year_offset]["netIncome"]),2),
                    "ROIC":"{}%".format(round((key_metrics_data[year_offset]["roic"])*100,1)),
                    "EV/EBITDA": round((key_metrics_data[year_offset][
                        "enterpriseValueOverEBITDA"
                    ]),2),
                    "PE Ratio": round(ratios_data[year_offset]["priceEarningsRatio"],2),
                    "PB Ratio": round(key_metrics_data[year_offset]["pbRatio"],2),
                }
                # Append the year and metrics to the DataFrame
                # Extracting the year from the date
                year = income_data[year_offset]["date"][:4]
                df[year] = pd.Series(metrics)

        df = df.sort_index(axis=1)

        return df

    def get_competitor_financial_metrics(
        ticker_symbol: Annotated[str, "ticker symbol"], 
        competitors: Annotated[List[str], "list of competitor ticker symbols"],  
        years: Annotated[int, "number of the years to search from, default to 4"] = 4
    ) -> dict:
        """Get financial metrics for the company and its competitors."""
        base_url = "https://financialmodelingprep.com/api/v3"
        all_data = {}

        symbols = [ticker_symbol] + competitors  # Combine company and competitors into one list
    
        for symbol in symbols:
            income_statement_url = f"{base_url}/income-statement/{symbol}?limit={years}&apikey={fmp_api_key}"
            ratios_url = f"{base_url}/ratios/{symbol}?limit={years}&apikey={fmp_api_key}"
            key_metrics_url = f"{base_url}/key-metrics/{symbol}?limit={years}&apikey={fmp_api_key}"

            income_data = requests.get(income_statement_url).json()
            ratios_data = requests.get(ratios_url).json()
            key_metrics_data = requests.get(key_metrics_url).json()

            metrics = {}

            if income_data and ratios_data and key_metrics_data:
                for year_offset in range(years):
                    metrics[year_offset] = {
                        "Revenue": round(income_data[year_offset]["revenue"] / 1e6),
                        "Revenue Growth": (
                            "{}%".format((round(income_data[year_offset]["revenue"] - income_data[year_offset - 1]["revenue"] / income_data[year_offset - 1]["revenue"])*100,1))
                            if year_offset > 0 else None
                        ),
                        "Gross Margin": round((income_data[year_offset]["grossProfit"] / income_data[year_offset]["revenue"]),2),
                        "EBITDA Margin": round((income_data[year_offset]["ebitdaratio"]),2),
                        "FCF Conversion": round((
                            key_metrics_data[year_offset]["enterpriseValue"] 
                            / key_metrics_data[year_offset]["evToOperatingCashFlow"] 
                            / income_data[year_offset]["netIncome"]
                            if key_metrics_data[year_offset]["evToOperatingCashFlow"] != 0 else None
                        ),2),
                        "ROIC":"{}%".format(round((key_metrics_data[year_offset]["roic"])*100,1)),
                        "EV/EBITDA": round((key_metrics_data[year_offset]["enterpriseValueOverEBITDA"]),2),
                    }

            df = pd.DataFrame.from_dict(metrics, orient='index')
            df = df.sort_index(axis=1)
            all_data[symbol] = df

        return all_data

    def get_ratings(
        ticker_symbol: Annotated[str, "ticker symbol"],
    ) -> dict:
        """Get the stable ratings for a given stock"""
        # Base URL setup for FMP API
        base_url = "https://financialmodelingprep.com/stable"
        ratingsUrl = f"{base_url}/ratings-historical?symbol={ticker_symbol}&apikey={fmp_api_key}"
        # Create DataFrame
        ratings_data = requests.get(ratingsUrl).json()
        return ratings_data
    
    def get_financial_scores(
        ticker_symbol: Annotated[str, "ticker symbol"],
    ) -> dict:
        """Get the stable ratings for a given stock"""
        # Base URL setup for FMP API
        base_url = "https://financialmodelingprep.com/stable"
        scoreUrl = f"{base_url}/financial-scores?symbol={ticker_symbol}&apikey={fmp_api_key}"
        # Create DataFrame
        score_data = requests.get(scoreUrl).json()
        return score_data