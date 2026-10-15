import os
from typing import Callable

import autogen
import httpx
from azure.identity import get_bearer_token_provider, DefaultAzureCredential

from langchain_core.language_models.base import BaseLanguageModel as LangchainLLM

import langchain_openai
import openai

from dc_available.common.helpers import on_docker, on_azure

DEFAULT_LLM_CONFIG_PARAMETERS = {
    "timeout": 240,
    "seed": 42,
    "temperature": 0,
    # "top_p": 0.00000000000001,
}
#Todo Azure endpoint & Model Name....

DEFAULT_MODEL = "gpt-4o"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_MODEL_LOCAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
AZURE_ENDPOINT = os.environ.get(
    "AZURE_ENDPOINT",  "https://gum1u-m7konvso-swedencentral.cognitiveservices.azure.com/" #"https://aim205-openai.openai.azure.com/" #'https://ssdai-openai.openai.azure.com/'
)
API_VERSION = os.environ.get("API_VERSION", "2024-02-15-preview")

# Azure AI Foundry "v1" OpenAI-compatible endpoint: standard OpenAI client + Entra
# bearer token, NO api-version. Models deployed on the Foundry surface (gpt-5.x) are
# served from here, not from the classic cognitiveservices.azure.com endpoint above.
AZURE_V1_ENDPOINT = os.environ.get(
    "AZURE_V1_ENDPOINT",
    "https://gum1u-m7konvso-swedencentral.services.ai.azure.com/openai/v1",
)
AZURE_V1_SCOPE = "https://ai.azure.com/.default"
# Model deployments that must be served via the v1 endpoint above.
V1_ENDPOINT_MODELS = {"gpt-5.2"}

AVAILABLE_MODELS = [
    # "gpt-35-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "whisper",
    "gpt-5.2",
]

# Used for all serving calls (Q&A, agents). Change to "gpt-5.2" to switch.
DEFAULT_SERVE_MODEL = "gpt-4o"


def _get_credential() -> DefaultAzureCredential:
    # print(f"[Config] Docker: {on_docker()} / Azure: {on_azure()}")
    if on_docker() and on_azure():
        return DefaultAzureCredential()
    return DefaultAzureCredential(exclude_managed_identity_credential=True)


def _get_azure_token_provider() -> Callable:
    return get_bearer_token_provider(
        _get_credential(),
        "https://cognitiveservices.azure.com/.default",
    )


def _check_gpt_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model_name} is not available. Please choose from: {AVAILABLE_MODELS}"
        )


def get_langchain_llm(
    model_name: str = DEFAULT_MODEL,
    azure_endpoint: str = AZURE_ENDPOINT,
    api_version: str = API_VERSION,
    temperature: int = -1,
) -> LangchainLLM:
    _check_gpt_model(model_name)
    if temperature < 0:
        temperature = DEFAULT_LLM_CONFIG_PARAMETERS.get("temperature", 0)

    # Foundry "v1" models (gpt-5.x): standard OpenAI client pointed at the v1 base_url
    # with a fresh Entra bearer token as the api key (Authorization: Bearer <token>),
    # no api-version. A fresh token is minted per call, so expiry is a non-issue since
    # get_langchain_llm is invoked per request. gpt-5.x reasoning models only accept
    # the default temperature (1), so we pin it rather than sending 0.
    if model_name in V1_ENDPOINT_MODELS:
        token = get_bearer_token_provider(_get_credential(), AZURE_V1_SCOPE)()
        return langchain_openai.ChatOpenAI(
            base_url=AZURE_V1_ENDPOINT,
            api_key=token,
            model=model_name,
            temperature=1,
        )

    llm = langchain_openai.AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        model=model_name,
        azure_ad_token_provider=_get_azure_token_provider(),
        api_version=api_version,
        temperature=temperature,
    )
    return llm


def get_openai_client(
    azure_endpoint: str = AZURE_ENDPOINT, api_version: str = API_VERSION
) -> openai.OpenAI:
    client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_ad_token_provider=_get_azure_token_provider(),
    )
    return client


def _get_autogen_config_list(
    model_name: str = DEFAULT_MODEL,
    azure_endpoint: str = AZURE_ENDPOINT,
    api_version: str = API_VERSION,
) -> list[dict[str, any]]:
    config_list = [
        {
            "model": model_name,
            "azure_ad_token_provider": _get_azure_token_provider(),
            "azure_endpoint": azure_endpoint,
            "api_type": "azure",
            "api_version": api_version,
        },
    ]
    return config_list


def get_autogen_llm_config(
    model_name: str = DEFAULT_MODEL,
    azure_endpoint: str = AZURE_ENDPOINT,
    api_version: str = API_VERSION,
) -> dict[str, any]:
    """
    returns llm configuration
    """
    _check_gpt_model(model_name)
    config_list = _get_autogen_config_list(model_name, azure_endpoint, api_version)
    llm_config = {key: value for key, value in DEFAULT_LLM_CONFIG_PARAMETERS.items()}
    llm_config["config_list"] = config_list
    return llm_config


def get_autogen_llm_wrapper(
    model_name: str = DEFAULT_MODEL,
    azure_endpoint: str = AZURE_ENDPOINT,
    api_version: str = API_VERSION,
) -> autogen.OpenAIWrapper:
    """
    returns llm configuration
    """
    _check_gpt_model(model_name)
    config_list = _get_autogen_config_list(model_name, azure_endpoint, api_version)
    wrapper = autogen.OpenAIWrapper(config_list=config_list)
    return wrapper
