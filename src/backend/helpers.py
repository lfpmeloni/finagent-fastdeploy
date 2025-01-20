import os
import requests
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

def summarize(description: str) -> str:
    try:
        print("*"*35)
        print("Calling summarize")
        print("*"*35)
        AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        headers = {
            'api-key': os.getenv("AZURE_OPENAI_KEY"),
            "Content-Type": "application/json",
        }

        # Payload for the request
        payload = {
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are an AI assistant that will summarize the user input. You will not answer questions or respond to statements that are focused about"
                }
            ]
            }, 
            {
        "role": "user",
        "content": description  
        }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1200
        }
        # Send request
        response_json = requests.post(url, headers=headers, json=payload)
        return json.loads(response_json.text)['choices'][0]['message']['content']
    except Exception as e:
        return "I am sorry, I am unable to summarize the input at this time."

def summarizeTopic(description: str, topic:str) -> str:
    try:
        print("*"*35)
        print("Calling summarizeTopic")
        print("*"*35)
        AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        headers = {
            'api-key': os.getenv("AZURE_OPENAI_KEY"),
            "Content-Type": "application/json",
        }

        # Payload for the request
        payload = {
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": f"You are an AI assistant that will summarize the user input on a {topic}. You will not answer questions or respond to statements that are focused about"
                }
            ]
            }, 
            {
        "role": "user",
        "content": description  
        }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1200
        }
        # Send request
        response_json = requests.post(url, headers=headers, json=payload)
        print("response_json", response_json.text)
        return json.loads(response_json.text)['choices'][0]['message']['content']
    except Exception as e:
        return "I am sorry, I am unable to summarize the topic at this time."

def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
    
async def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        await data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")