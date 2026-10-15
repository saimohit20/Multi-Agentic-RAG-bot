from typing import Callable
import os
import openai
from azure.identity import get_bearer_token_provider, DefaultAzureCredential

AZURE_ENDPOINT = "https://gum1u-m7konvso-swedencentral.cognitiveservices.azure.com/" #"https://aim205-openai.openai.azure.com/"
API_VERSION = "2023-09-01-preview" #os.environ.get("API_VERSION", "2024-02-15-preview") #"2024-02-15-preview"

def _get_azure_token_provider() -> Callable:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(exclude_managed_identity_credential=True),
        "https://cognitiveservices.azure.com/.default",
    )
    return token_provider

def get_openai_client(
    azure_endpoint: str = AZURE_ENDPOINT, api_version: str = API_VERSION
) -> openai.OpenAI:
    client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_ad_token_provider=_get_azure_token_provider(),
    )

    return client