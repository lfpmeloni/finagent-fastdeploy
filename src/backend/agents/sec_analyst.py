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
from analyzer import *
from reports import ReportLabUtils
from charting import ReportChartUtils

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."
businessOverview = None
riskAssessment = None
marketPosition = None
incomeStatement = None
incomeSummarization = None
segmentStatement = None

async def analyze_company_description(ticker_symbol:str, year:str) -> str:
    global marketPosition
    companyDesc = ReportAnalysisUtils.analyze_company_description(ticker_symbol, year)
    marketPosition = summarize(companyDesc)
    return (
        f"##### Company Description\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Company Analysis:** {marketPosition}\n"
        f"{formatting_instructions}"
    )

async def analyze_business_highlights(ticker_symbol:str, year:str) -> str:
    global businessOverview
    businessHighlights = ReportAnalysisUtils.analyze_business_highlights(ticker_symbol, year)
    businessOverview = summarize(businessHighlights)
    return (
        f"##### Business Highlights\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Business Highlights:** {businessOverview}\n"
        f"{formatting_instructions}"
    )

async def get_competitors_analysis(ticker_symbol:str, year:str) -> str:
    compAnalysis = ReportAnalysisUtils.get_competitors_analysis(ticker_symbol, year)
    summarized = summarize(compAnalysis)
    return (
        f"##### Competitor Analysis\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Competitor Analysis:** {summarized}\n"
        f"{formatting_instructions}"
    )

async def get_risk_assessment(ticker_symbol:str, year:str) -> str:
    global riskAssessment
    riskAssess = ReportAnalysisUtils.get_risk_assessment(ticker_symbol, year)
    riskAssessment = summarize(riskAssess)
    return (
        f"##### Risk Assessment\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Risk Assessment Analysis:** {riskAssessment}\n"
        f"{formatting_instructions}"
    )

async def analyze_segment_stmt(ticker_symbol:str, year:str) -> str:
    global segmentStatement
    segmentStmt = ReportAnalysisUtils.analyze_segment_stmt(ticker_symbol, year)
    segmentStatement = summarize(segmentStmt)
    return (
        f"##### Segment Statement\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Segment Statement Analysis:** {segmentStatement}\n"
        f"{formatting_instructions}"
    )

async def analyze_cash_flow(ticker_symbol:str, year:str) -> str:
    cashFlow = ReportAnalysisUtils.analyze_cash_flow(ticker_symbol, year)
    summarized = summarize(cashFlow)
    return (
        f"##### Cash Flow\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Cash Flow Analysis:** {summarized}\n"
        f"{formatting_instructions}"
    )

async def analyze_balance_sheet(ticker_symbol:str, year:str) -> str:
    balanceSheet = ReportAnalysisUtils.analyze_balance_sheet(ticker_symbol, year)
    summarized = summarize(balanceSheet)
    return (
        f"##### Balance Sheet\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Balance Sheet Analysis:** {summarized}\n"
        f"{formatting_instructions}"
    )

async def analyze_income_stmt(ticker_symbol:str, year:str) -> str:
    global incomeStatement
    incomeStmt = ReportAnalysisUtils.analyze_income_stmt(ticker_symbol, year)
    incomeStatement = summarize(incomeStmt)
    return (
        f"#####Income Statement\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Income Statement Analysis:** {incomeStatement}\n"
        f"{formatting_instructions}"
    )

async def income_summarization(ticker_symbol:str, year:str) -> str:
    global incomeSummarization
    global incomeStatement
    global segmentStatement
    if incomeStatement is None or len(incomeStatement) == 0:
        incomeStmt = ReportAnalysisUtils.analyze_income_stmt(ticker_symbol, year)
        incomeStatement = summarize(incomeStmt)
    if segmentStatement is None or len(segmentStatement) == 0:
        segmentStmt = ReportAnalysisUtils.analyze_segment_stmt(ticker_symbol, year)
        segmentStatement = summarize(segmentStmt)
    incomeSummary = ReportAnalysisUtils.income_summarization(ticker_symbol, year, incomeStatement, segmentStatement)
    incomeSummarization = summarize(incomeSummary)
    return (
        f"#####Income Statement\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Income Statement Analysis:** {incomeSummarization}\n"
        f"{formatting_instructions}"
    )

async def build_annual_report(ticker_symbol:str, year:str) -> str:
    global businessOverview
    global riskAssessment
    global marketPosition
    global incomeSummarization
    if businessOverview is None or len(businessOverview) == 0:
        businessHighlights = ReportAnalysisUtils.analyze_business_highlights(ticker_symbol, year)
        businessOverview = summarize(businessHighlights)
    
    if riskAssessment is None or len(riskAssessment) == 0:
        riskAssess = ReportAnalysisUtils.get_risk_assessment(ticker_symbol, year)
        riskAssessment = summarize(riskAssess)

    if marketPosition is None or len(marketPosition) == 0:
        companyDesc = ReportAnalysisUtils.analyze_company_description(ticker_symbol, year)
        marketPosition = summarize(companyDesc)

    if incomeSummarization is None or len(incomeSummarization) == 0:
        incomeSummary = await income_summarization(ticker_symbol, year)
        incomeSummarization = summarize(incomeSummary)
    
    secReport = fmpUtils.get_sec_report(ticker_symbol, year)
    if secReport.find("Date: ") > 0:
        index = secReport.find("Date: ")
        filingDate = secReport[index:].split()[1]
    else:
        filingDate = datetime.now()

    #Convert filing date to datetime and then convert to a formatted string
    if isinstance(filingDate, datetime):
        filingDate = filingDate.strftime("%Y-%m-%d")
    else:
        filingDate = datetime.strptime(filingDate, "%Y-%m-%d").strftime("%Y-%m-%d")


    ReportChartUtils.get_share_performance(ticker_symbol, filingDate, "reports\\")
    ReportChartUtils.get_pe_eps_performance(ticker_symbol, filingDate, 4, "reports\\")
    reportOut = ReportLabUtils.build_annual_report(ticker_symbol, "reports\\", incomeSummarization,
                            marketPosition, businessOverview, riskAssessment, None, "reports\\stock_performance.png", "reports\\pe_performance.png", filingDate)
    return (
        f"#####Build Annual Report\n"
        f"**Company Name:** {ticker_symbol}\n"
        f"**Report Saved at :** reports\\{ticker_symbol}_Equity_Research_report.pdf\n"
        f"{formatting_instructions}"
    )

# Create the Company Analyst Tools list
def get_sec_analyst_tools() -> List[Tool]:
    return [
        FunctionTool(
            analyze_company_description, 
            description="analyze the company description for a company from the SEC report",
        ),
        FunctionTool(
           analyze_business_highlights, 
           description="analyze the business highlights for a company from the SEC report",
        ),
        FunctionTool(
            get_competitors_analysis, 
            description="analyze the competitors analysis for a company from the SEC report",
        ),
        FunctionTool(
            get_risk_assessment, 
            description="analyze the risk assessment for a company from the SEC report",
        ),
        FunctionTool(
            analyze_segment_stmt, 
            description="analyze the segment statement for a company from the SEC report",
        ),
        FunctionTool(
            analyze_cash_flow, 
            description="analyze the cash flow for a company from the SEC report",
        ),
        FunctionTool(
            analyze_balance_sheet, 
            description="analyze the balance sheet for a company from the SEC report",
        ),
        FunctionTool(
            analyze_income_stmt, 
            description="analyze the income statement for a company from the SEC report",
        ),
        FunctionTool(
            build_annual_report, 
            description="build the annual report for a company from the SEC report",
        ),
    ]


@default_subscription
class SecAnalystAgent(BaseAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        session_id: str,
        user_id: str,
        memory: CosmosBufferedChatCompletionContext,
        sec_analyst_tools: List[Tool],
        sec_analyst_tool_agent_id: AgentId,
    ):
        super().__init__(
            "SecAnalystAgent",
            model_client,
            session_id,
            user_id,
            memory,
            sec_analyst_tools,
            sec_analyst_tool_agent_id,
            system_message="You are an AI Agent. You have knowledge about the SEC annual (10-K) and quarterly (10-Q) reports.  SEC reports includes the information about income statement, balance sheet, cash flow, risk assessment, competitor analysis, business highlights and business information."
        )
