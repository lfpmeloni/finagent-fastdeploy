from typing import List, Dict, Any
import pandas as pd
import yfinance as yf

from autogen_core import AgentId
from autogen_core import default_subscription
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool

from agents.base_agent import BaseAgent
from context.cosmos_memory import CosmosBufferedChatCompletionContext
from helpers.fmputils import *
from helpers.yfutils import *
from helpers.analyzer import *
from datetime import date, timedelta, datetime

from typing import List, Dict, Any
import os
import requests
from autogen_core.tools import FunctionTool, Tool

async def fetch_and_analyze_fundamentals(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetch up to 5 years of fundamental data (Income Statement, Balance Sheet, Cash Flow)
    from Financial Modeling Prep, then compute ratios (ROE, ROA, placeholders for 
    Altman Z-score, Piotroski F-score, etc.) for the given ticker.

    Returns a JSON-serializable dict with:
      - 5-year income statements
      - 5-year balance sheets
      - 5-year cash flows
      - A 'ratios_scores' section (ROE, ROA, AltmanZ, PiotroskiF)
      - Any notes or error messages
    """

    result = {
        "ticker_symbol": ticker_symbol,
        "financial_metrics": [],
        "ratings": {},
        "financial_scores": []
    }

    try:
        financialMetrics = fmpUtils.get_financial_metrics(ticker_symbol)
        ratings = fmpUtils.get_ratings(ticker_symbol)
        finacialScores = fmpUtils.get_financial_scores(ticker_symbol)

        result["financial_metrics"] = financialMetrics
        result["ratings"] = ratings
        result["financial_scores"] = finacialScores

    except Exception as e:
        result["notes"].append(f"Exception during fetch: {e}")

    return result


def get_fundamental_analysis_tools() -> List[Tool]:
    """
    Return a list of Tools for the Fundamental Analysis Agent 
    that fetch data from Financial Modeling Prep (FMP).
    """
    return [
        FunctionTool(
            fetch_and_analyze_fundamentals,
            description=(
                "Fetch fundamental data (Income, Balance, Cash Flow)"
                "and compute ratios (ROE, ROA, Altman Z, Piotroski, etc.) for a given ticker."
            ),
        )
    ]

@default_subscription
class FundamentalAnalysisAgent(BaseAgent):
    """
    A dedicated agent to perform fundamental analysis over the last ~5 years
    by pulling data from Financial Modeling Prep (FMP).
    Computes key ratios or scores (ROE, ROA, Altman Z, Piotroski, etc.).
    """
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        session_id: str,
        user_id: str,
        memory: CosmosBufferedChatCompletionContext,
        fundamental_analysis_tools: List[Tool],
        fundamental_analysis_tool_agent_id: AgentId,
    ):
        super().__init__(
            "FundamentalAnalysisAgent",
            model_client,
            session_id,
            user_id,
            memory,
            fundamental_analysis_tools,
            fundamental_analysis_tool_agent_id,
            system_message=dedent(
                """
                You are a Fundamental Analysis Agent. 
                Your role is to retrieve and analyze up to 5 years of fundamental data 
                (cash flow, income statements, balance sheets) for a given ticker 
                using the Financial Modeling Prep API. 
                You also compute basic ratios like ROE, ROA, and placeholders for 
                Altman Z-score and Piotroski F-score. 
                Return the data and computations in structured JSON.
                """
            )
        )