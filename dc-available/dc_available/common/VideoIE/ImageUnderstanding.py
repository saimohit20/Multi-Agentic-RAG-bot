from pathlib import Path
import base64
from openai import OpenAI

from .AzureClient import get_openai_client

PROMPT_Default = "Please describe the provided image."
PROMPT_GeneralSlide = "Given this PPT slide, describe its content in English."
PROMPT_Slide_woBottonPart = "Given this PPT slide, describe its content in English. Ignore the bottom part that includes the slide number, footnotes, company logo, etc., if they present."
PROMPT_SlidePresent_woBottonPart = "In this given image, a man is presenting a slide.  Please describe the content of this slide in English.  When describing the slide, please ignore the slide's bottom part that includes the slide number, footnotes, company logo, etc."

def create_message_for_image_understanding(
    base64_image: str,
    prompt: str,
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information.",
                }
            ],        
        },
        {
            "role": "user",
            "content": [                
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {   "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    return messages

def img_to_bytes(img_or_path: str | Path, encoding: str = "utf-8") -> str:
    img_bytes = Path(img_or_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode(encoding)
    return encoded

def generate_image_description(
    image_path: str | Path,
    prompt: str = PROMPT_Default,
    client: OpenAI = None,
    model: str = "gpt-4o",
) -> str:
    if client is None:
        client = get_openai_client()
    messages = create_message_for_image_understanding(img_to_bytes(image_path), prompt)
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        max_tokens=800,
        temperature=0.7,
        top_p=0.95
    )
    output = response.choices[0].message.content
    return output

def generate_slide_understanding_result(
    image_path: str | Path,
    slideType: str, #allowed values: "PresentedSlide", "GeneralSlide", "SlideWoBottomInfo", "GeneralImage"
    client: OpenAI = None,
    model: str = "gpt-4o",
) -> str:
    prompt = PROMPT_Default
    if slideType == "PresentedSlide":
        prompt = PROMPT_SlidePresent_woBottonPart
    if slideType == "GeneralSlide":
        prompt = PROMPT_GeneralSlide
    elif slideType == "SlideWoBottomInfo":
        prompt = PROMPT_Slide_woBottonPart
    
    if client is None:
        client = get_openai_client()
    messages = create_message_for_image_understanding(img_to_bytes(image_path), prompt)
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        max_tokens=800,
        temperature=0.7,
        top_p=0.95
    )
    output = response.choices[0].message.content
    return output

if __name__ == "__main__":
    print('Generated description for input image:\n')
    #active_prompt = PROMPT_GeneralSlide
    #print(generate_image_description('image\\PPT_page_example.png', active_prompt))
    
    #active_prompt = PROMPT_Slide_woBottonPart
    #print(generate_image_description('image\\Slide_7.png', active_prompt))
    
    #active_prompt = PROMPT_SlidePresent_woBottonPart
    #print(generate_image_description('image\\SceenShot_7.png', active_prompt))

    print(generate_slide_understanding_result('image\\SceenShot_7.png', "PresentedSlide"))
