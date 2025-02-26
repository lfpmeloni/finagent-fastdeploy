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
from typing import List, Dict, Any
import json

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown."

async def analyze_and_predict(analysis_result: Dict[str, Any]) -> str:
    """
    Takes the JSON output from ExtendedCombinedAnalysisAgent (technical indicators,
    candlestick patterns, fundamentals, news sentiment, final decision),
    and uses an LLM to produce a structured forecast with:
      1) A multi-section format (Introduction, Technical, Fundamental, etc.)
      2) An explanation of probability/score as confidence (e.g., 70% => "moderately strong")
      3) A final recommendation
      4) Legal disclaimers

    Returns a markdown or text response with these structured sections.
    """
    # Convert analysis_result into a JSON string
    analysis_json_str = json.dumps(analysis_result, indent=2)

    # Extract the final probability from the JSON for prompt usage
    final_decision = analysis_result.get("final_decision", {})
    probability_value = final_decision.get("probability", None)
    rating_value = final_decision.get("rating", "hold")

    # We can provide instructions to interpret the confidence level:
    # e.g., 0.0-0.33 => "low confidence", 0.33-0.66 => "moderate confidence", 0.66-1.0 => "high confidence"
    # We'll do a bit of logic to embed in the prompt. Alternatively, let the LLM do it entirely.
    confidence_descriptor = "moderate"
    if probability_value is not None:
        if probability_value <= 0.33:
            confidence_descriptor = "low"
        elif probability_value >= 0.66:
            confidence_descriptor = "high"
        else:
            confidence_descriptor = "moderate"

    # Construct a detailed prompt with strict output structure
    prompt = f"""
    You are a specialized financial analysis LLM. You have received a JSON structure that
    represents an extended analysis of a stock, including:
      - Technical signals (RSI, MACD, Bollinger, EMA crossover, Stochastics, ADX)
      - Candlestick pattern detections (TA-Lib)
      - Basic fundamentals (P/E ratios, etc.)
      - News sentiment
      - A final numeric probability (score) and rating (Buy/Sell/Hold).

    The JSON data is:

    ```
    {analysis_json_str}
    ```

    **Please return your answer in the following sections:**

    1) **Introduction**
       - Briefly introduce the analysis.

    2) **Technical Overview**
       - Summarize the key technical indicators and any candlestick patterns.
       - Explain whether they are bullish, bearish, or neutral.

    3) **Fundamental Overview**
       - Mention any notable fundamental data (like forwardPE, trailingPE, etc.) 
         and how it influences the outlook.

    4) **News & Sentiment**
       - Highlight the sentiment score (range: -1.0 to +1.0). 
         Explain if it's a tailwind (positive) or headwind (negative).

    5) **Probability & Confidence**
       - The system’s final probability is **{probability_value}** (range: 0.0 to 1.0).
       - Interpret it as **{confidence_descriptor}** confidence 
         (e.g., <=0.33 => "low", 0.33-0.66 => "moderate", >=0.66 => "high").
       - Elaborate how confident or uncertain this rating might be based on
         conflicting signals, volatility, etc.

    6) **Final Recommendation**
       - Based on the system’s final rating: **{rating_value}**.
       - Explain briefly why you agree or disagree, or how you interpret it.

    7) **Disclaimers**
       - Include disclaimers such as "Past performance is not indicative of future results."
       - Remind the user that this is not guaranteed investment advice.
       - Encourage further research before making any decisions.

    Please format your response in **Markdown**, with headings for each section
    and bullet points where appropriate. 
    """

    return prompt
    # Now we call the LLM with this prompt. We'll mock the response for this example.
    # In real usage, you'd do something like:
    # response = await model_client.get_chat_completion(
    #     system_message="You are a financial analysis LLM.",
    #     user_message=prompt,
    #     temperature=0.7,
    #     max_tokens=1200,
    # )
    #

# Create the Company Analyst Tools list
def get_forecaster_tools() -> List[Tool]:
    return [
        FunctionTool(
            analyze_and_predict, 
            description=(
                "Interprets the JSON output from ExtendedCombinedAnalysisAgent. "
                "Generates a final Buy/Sell/Hold recommendation with a structured rationale, "
                "risk factors, disclaimers, and an explanation of the probability or confidence."
            ),
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
            You are a Forecaster and Analysis Agent. 
            Your role is to interpret the output of an extended technical & fundamental analysis pipeline 
            and additional data from the list of one or more the following:
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
            Produce a final recommendation (Buy, Sell, or Hold) with 
            a structured format and thorough, bullet-pointed explanation. 
            You must mention the final probability, interpret it as confidence level, 
            and provide disclaimers like "Past performance is not indicative of future results.
            """
            )
        )
