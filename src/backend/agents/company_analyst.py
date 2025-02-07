from typing import List

from autogen_core import AgentId
from autogen_core import default_subscription
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool
from typing_extensions import Annotated

from agents.base_agent import BaseAgent
from context.cosmos_memory import CosmosBufferedChatCompletionContext
from helpers.fmputils import *
from helpers.yfutils import *
from datetime import date, timedelta, datetime

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."

# Define Company Analyst tools (functions)
async def get_company_info(ticker_symbol: str) -> str:
    return (
        f"##### Get Company Information\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Company Information:** {fmpUtils.get_company_profile(ticker_symbol)}\n"
        f"{formatting_instructions}"
    )

async def get_analyst_recommendations(ticker_symbol: str) -> str:
    return (
        f"##### Get Company Recommendations\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Recommendations:** {yfUtils.get_analyst_recommendations(ticker_symbol)}\n"
        f"{formatting_instructions}"
    )

async def get_stock_data(ticker_symbol: str) -> str:
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    return (
        f"##### Stock Data from Yahoo Finance\n"
        f"**Company Name:** {ticker_symbol}\n\n"
        f"**Start Date:** {start_date}\n"
        f"**End Date:** {end_date}\n\n"
        f"**Stock Data:** {yfUtils.get_stock_data(ticker_symbol, start_date, end_date)}\n"
        f"{formatting_instructions}"
    )

async def get_financial_metrics(ticker_symbol: str) -> str:
    years = 4
    return (
        f"##### Get Financial Information\n"
        f"**Company Name:** {ticker_symbol}\n\n"
        f"**Years:** {years}\n\n"
        f"**Financial Information:** {fmpUtils.get_financial_metrics(ticker_symbol, years)}\n"
        f"{formatting_instructions}"
    )

async def get_company_news(ticker_symbol: str) -> str:
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    return (
        f"##### Get Company News\n"
        f"**Company Name:** {ticker_symbol}\n\n"
        #f"**Company News:** {fmpUtils.get_company_news(ticker_symbol, start_date, end_date)}\n"
        f"**Company News:** {yfUtils.get_company_news(ticker_symbol, start_date, end_date)}\n"
        f"{formatting_instructions}"
    )

async def get_sentiment_analysis(ticker_symbol: str) -> str:
    return (
        f"##### Get Company Information\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"{formatting_instructions}"
    )

# async def analyze_predict_company(ticker_symbol: str) -> str:
#     return (
#         f"##### Analyze and Prediction\n"
#         f"**Company Name:** {ticker_symbol}\n\n"
#         f"{formatting_instructions}"
#     )

# Create the Company Analyst Tools list
def get_company_analyst_tools() -> List[Tool]:
    return [
        FunctionTool(
            get_company_info, 
            description="get a company's profile information",
        ),
        FunctionTool(
           get_stock_data, 
           description="retrieve stock price data for designated ticker symbol",
        ),
        FunctionTool(
            get_financial_metrics, 
            description="get latest financial basics for a designated company",
        ),
        FunctionTool(
            get_company_news, 
            description="retrieve market news related to designated company",
        ),
        FunctionTool(
            get_analyst_recommendations, 
            description="get analyst recommendation for a designated company",
        ),
        FunctionTool(
            get_sentiment_analysis, 
            description="Analyze the data that you have access to like news and analyst recommendations and provide a sentiment analysis, positive or negative outlook",
        ),
        # FunctionTool(
        #     analyze_predict_company, 
        #     description="Analyze and predict the future of a designated company",
        # ),
    ]


@default_subscription
class CompanyAnalystAgent(BaseAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        session_id: str,
        user_id: str,
        memory: CosmosBufferedChatCompletionContext,
        ca_tools: List[Tool],
        ca_tool_agent_id: AgentId,
    ):
        super().__init__(
            "CompanyAnalystAgent",
            model_client,
            session_id,
            user_id,
            memory,
            ca_tools,
            ca_tool_agent_id,
            system_message="You are an AI Agent. You have knowledge about stock market, company information, company news, analyst recommendation and company's financial data and metrics."
    #         system_message="As a Company Analyst, one must possess strong analytical and problem-solving abilities, collect necessary financial information and aggregate them based on client's requirement."
    # "For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
        )
