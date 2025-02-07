from typing import List

from autogen_core import AgentId
from autogen_core import default_subscription
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool

from agents.base_agent import BaseAgent
from context.cosmos_memory import CosmosBufferedChatCompletionContext
from helpers.fmputils import *
from helpers.yfutils import *
from helpers.analyzer import *

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown."

async def analyze_and_predict(ticker_symbol: str) -> str:
    return (
        f"##### Analyze and Prediction\n"
        f"**Company Name:** {ticker_symbol}\n\n"
        #f"{formatting_instructions}"
    )

# Create the Company Analyst Tools list
def get_forecaster_tools() -> List[Tool]:
    return [
        FunctionTool(
            analyze_and_predict, 
            description="analyze and make predictions for the company",
        ),
    ]


@default_subscription
class ForecasterAgent(BaseAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        session_id: str,
        user_id: str,
        memory: CosmosBufferedChatCompletionContext,
        forecaster_tools: List[Tool],
        forecaster_tool_agent_id: AgentId,
    ):
        super().__init__(
            "ForecasterAgent",
            model_client,
            session_id,
            user_id,
            memory,
            forecaster_tools,
            forecaster_tool_agent_id,
            #system_message="You are an AI Agent. You have knowledge about the SEC annual (10-K) and quarterly (10-Q) reports.  SEC reports includes the information about income statement, balance sheet, cash flow, risk assessment, competitor analysis, business highlights and business information."
            system_message=dedent(
            f"""
            Analyze all the data that you have capture for the company. 
            with 2-4 most important factors respectively and keep them concise. Most factors should be inferred only from the data you have access to
            The data consists of one or more the following:
            - Business Overview
            - Risk Assessment
            - Market Position
            - Income Statement
            - Segment Statement
            - Income Summarization
            - Competitor Analysis
            - Business Highlights
            - Business Information
            - Earnings Call Transcripts
            - SEC Reports
            - Analyst Reports
            - News
            - Stock Price Data

            Then make a rough prediction (e.g. up/down by X-Y% or a price range between X-Y) of the company stock price movement for next week. Provide a summary analysis to support your prediction.
            """
            )
        )
