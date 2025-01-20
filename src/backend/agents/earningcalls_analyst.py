from typing import List

from autogen_core import AgentId
from autogen_core import default_subscription
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool
from typing_extensions import Annotated

from agents.base_agent import BaseAgent
from context.cosmos_memory import CosmosBufferedChatCompletionContext
from fmputils import *
from yfutils import *
from datetime import date, timedelta, datetime
from helpers import summarize, summarizeTopic
from dcfutils import DcfUtils

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown."
latestEarnings = None

# Define HR tools (functions)
async def get_earning_calls_transcript(ticker_symbol: str, year:str) -> str:
    global latestEarnings
    print("Calling get_earning_calls_transcript")
    if year is None or year == "latest":
        year = datetime.now().year
        if datetime.now().month < 3:
            year = int(year) - 1

    if latestEarnings is None or len(latestEarnings) == 0:
        #latestEarnings = fmpUtils.get_earning_calls(ticker_symbol, year)
        latestEarnings = DcfUtils.get_earning_calls(ticker_symbol)
    return (
        f"##### Get Earning Calls\n"
        f"{formatting_instructions}"
    )

async def summarize_transcripts(ticker_symbol:str, year:str) -> str:
    global latestEarnings
    if latestEarnings is None or len(latestEarnings) == 0:
        #latestEarnings = fmpUtils.get_earning_calls(ticker_symbol, year)
        latestEarnings = DcfUtils.get_earning_calls(ticker_symbol)
    print("*"*35)
    print("Calling summarize_transcripts")
    summarized = summarize(latestEarnings)
    print("*"*35)
    return (
        f"##### Summarized transcripts\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Summary:** {summarized}\n"
        f"{formatting_instructions}"
    )

async def management_positive_outlook(ticker_symbol: str, year:str) -> str:
    global latestEarnings
    if latestEarnings is None or len(latestEarnings) == 0:
        #latestEarnings = fmpUtils.get_earning_calls(ticker_symbol, year)
        latestEarnings = DcfUtils.get_earning_calls(ticker_symbol)
    print("*"*35)
    print("Calling management_positive_outlook")
    positiveOutlook = summarizeTopic(latestEarnings, 'Management Positive Outlook')
    print("*"*35)
    return (
        f"##### Management Positive Outlook\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Topic Summary:** {positiveOutlook}\n"
        f"{formatting_instructions}"
    )

async def management_negative_outlook(ticker_symbol: str, year:str) -> str:
    global latestEarnings
    if latestEarnings is None or len(latestEarnings) == 0:
        #latestEarnings = fmpUtils.get_earning_calls(ticker_symbol, year)
        latestEarnings = DcfUtils.get_earning_calls(ticker_symbol)
    print("*"*35)
    print("Calling management_negative_outlook")
    negativeOutlook = summarizeTopic(latestEarnings, 'Management Negative Outlook')
    print("*"*35)
    years = 4
    return (
        f"##### Management Negative Outlook\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Topic Summary:** {negativeOutlook}\n"
        f"{formatting_instructions}"
    )

async def future_growth_opportunity(ticker_symbol: str, year:str) -> str:
    global latestEarnings
    if latestEarnings is None or len(latestEarnings) == 0:
        #latestEarnings = fmpUtils.get_earning_calls(ticker_symbol, year)
        latestEarnings = DcfUtils.get_earning_calls(ticker_symbol)
    print("*"*35)
    print("Calling management_negative_outlook")
    futureGrowth = summarizeTopic(latestEarnings, 'Future Growth Opportunities')
    print("*"*35)
    return (
        f"##### Future Growth and Opportunities\n"
        f"**Company Name:** {ticker_symbol}\n\n"
        f"**Topic Summary:** {futureGrowth}\n"
        f"{formatting_instructions}"
    )

# async def analyze_predict_transcript(ticker_symbol: str) -> str:
#     return (
#         f"##### Transcription Analyze and Prediction\n"
#         f"**Company Name:** {ticker_symbol}\n\n"
#         f"{formatting_instructions}"
#     )

# Create the Company Analyst Tools list
def get_earning_calls_analyst_tools() -> List[Tool]:
    return [
        FunctionTool(
            get_earning_calls_transcript, 
            description="get a earning call's transcript for a company",
        ),
        FunctionTool(
           summarize_transcripts, 
           description="summarize the earning call's transcript for a company",
        ),
        FunctionTool(
            management_positive_outlook, 
            description="From the extracted earning call's transcript, identify the management's positive outlook for a company",
        ),
        FunctionTool(
            management_negative_outlook, 
            description="From the extracted earning call's transcript, identify the management's negative outlook for a company",
        ),
        FunctionTool(
            future_growth_opportunity, 
            description="From the extracted earning call's transcript, identify the future growth and opportunities for a company",
        ),
        # FunctionTool(
        #     analyze_predict_transcript, 
        #     description="Analyze and predict the future of a designated company based on the information from the earning call's transcript",
        # ),
    ]


@default_subscription
class EarningCallsAnalystAgent(BaseAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        session_id: str,
        user_id: str,
        memory: CosmosBufferedChatCompletionContext,
        earning_calls_analyst_tools: List[Tool],
        earning_calls_analyst_tool_agent_id: AgentId,
    ):
        super().__init__(
            "EarningCallsAnalystAgent",
            model_client,
            session_id,
            user_id,
            memory,
            earning_calls_analyst_tools,
            earning_calls_analyst_tool_agent_id,
            system_message="You are an AI Agent. You have knowledge about the management positive and negative outlook, future growths and opportunities based on the earning call transcripts."
        )
