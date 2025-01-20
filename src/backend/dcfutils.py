import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from dutils import decorate_all_methods
from helpers import get_next_weekday
import re
from tenacity import RetryError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain.schema import Document
import json
from typing import List
import ast 

# from finrobot.utils import decorate_all_methods, get_next_weekday
from functools import wraps
from typing import Annotated, List

def init_dcf_api(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global dcf_api_key
        if os.environ.get("DCF_API_KEY") is None:
            print("Please set the environment variable DCF_API_KEY to use the DCF API.")
            return None
        else:
            dcf_api_key = os.environ["DCF_API_KEY"]
            print("DCF api key found successfully.")
            return func(*args, **kwargs)

    return wrapper

@decorate_all_methods(init_dcf_api)
class DcfUtils:

    def correct_date(yr, dt):
        """Some transcripts have incorrect date, correcting it

        Args:
            yr (int): actual
            dt (datetime): given date

        Returns:
            datetime: corrected date
        """
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        if dt.year != yr:
            dt = dt.replace(year=yr)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def extract_speakers(cont: str) -> List[str]:
        """Extract the list of speakers

        Args:
            cont (str): transcript content

        Returns:
            List[str]: list of speakers
        """
        pattern = re.compile(r"\n(.*?):")
        matches = pattern.findall(cont)

        return list(set(matches))
    
    def clean_speakers(speaker):
        speaker = re.sub("\n", "", speaker)
        speaker = re.sub(":", "", speaker)
        return speaker
    
    def get_earnings_transcript(quarter: str, ticker: str, year: int):
        """Get the earnings transcripts

        Args:
            quarter (str)
            ticker (str)
            year (int)
        """
        response = requests.get(
            f"https://discountingcashflows.com/api/transcript/?ticker={ticker}&quarter={quarter}&year={year}&key={dcf_api_key}"
        )

        resp_text = json.loads(response.text)
        # speakers_list = extract_speakers(resp_text[0]["content"])
        corrected_date = DcfUtils.correct_date(resp_text[0]["year"], resp_text[0]["date"])
        resp_text[0]["date"] = corrected_date
        return resp_text[0]
    
    def get_earnings_all_quarters_data(quarter: str, ticker: str, year: int):
        docs = []
        resp_dict = DcfUtils.get_earnings_transcript(quarter, ticker, year)

        content = resp_dict["content"]
        pattern = re.compile(r"\n(.*?):")
        matches = pattern.finditer(content)

        speakers_list = []
        ranges = []
        for match_ in matches:
            # print(match.span())
            span_range = match_.span()
            # first_idx = span_range[0]
            # last_idx = span_range[1]
            ranges.append(span_range)
            speakers_list.append(match_.group())
        speakers_list = [DcfUtils.clean_speakers(sl) for sl in speakers_list]

        for idx, speaker in enumerate(speakers_list[:-1]):
            start_range = ranges[idx][1]
            end_range = ranges[idx + 1][0]
            speaker_text = content[start_range + 1 : end_range]

            docs.append(
                Document(
                    page_content=speaker_text,
                    metadata={"speaker": speaker, "quarter": quarter},
                )
            )

        docs.append(
            Document(
                page_content=content[ranges[-1][1] :],
                metadata={"speaker": speakers_list[-1], "quarter": quarter},
            )
        )
        return docs, speakers_list

    def get_earning_calls(ticker: str) -> str:
        
        url = f"https://discountingcashflows.com/api/transcript/list/?ticker={ticker}&key={dcf_api_key}"

        response = requests.get(url)

        if response.status_code == 200:
            data = ast.literal_eval(response.text)
            quarter, year = data[0][0], data[0][1]

            resp_dict = DcfUtils.get_earnings_transcript("Q" + str(quarter), ticker, year)

            transcripts = resp_dict["content"]
            return transcripts
        else:
            return f"Failed to retrieve data: {response.status_code}"
        
    def get_earnings_all_docs(ticker: str, year: int):
        earnings_docs = []
        earnings_call_quarter_vals = []
        print("Earnings Call Q1")
        try:
            docs, speakers_list_1 = DcfUtils.get_earnings_all_quarters_data("Q1", ticker, year)
            earnings_call_quarter_vals.append("Q1")
            earnings_docs.extend(docs)
        except RetryError:
            print(f"Don't have the data for Q1")
            speakers_list_1 = []

        print("Earnings Call Q2")
        try:
            docs, speakers_list_2 = DcfUtils.get_earnings_all_quarters_data("Q2", ticker, year)
            earnings_call_quarter_vals.append("Q2")
            earnings_docs.extend(docs)
        except RetryError:
            print(f"Don't have the data for Q2")
            speakers_list_2 = []
        print("Earnings Call Q3")
        try:
            docs, speakers_list_3 = DcfUtils.get_earnings_all_quarters_data("Q3", ticker, year)
            earnings_call_quarter_vals.append("Q3")
            earnings_docs.extend(docs)
        except RetryError:
            print(f"Don't have the data for Q3")
            speakers_list_3 = []
        print("Earnings Call Q4")
        try:
            docs, speakers_list_4 = DcfUtils.get_earnings_all_quarters_data("Q4", ticker, year)
            earnings_call_quarter_vals.append("Q4")
            earnings_docs.extend(docs)
        except RetryError:
            print(f"Don't have the data for Q4")
            speakers_list_4 = []
        return (
            earnings_docs,
            earnings_call_quarter_vals,
            speakers_list_1,
            speakers_list_2,
            speakers_list_3,
            speakers_list_4,
        )
    