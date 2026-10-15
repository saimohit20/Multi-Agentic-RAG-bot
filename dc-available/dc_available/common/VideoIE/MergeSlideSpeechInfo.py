from pathlib import Path
from openai import OpenAI

from .ImageUnderstanding import generate_slide_understanding_result
from .Speech2Text import generate_speech_transcription
from .AzureClient import get_openai_client

PROMPT_Merge_OverallHead = "Given a video of someone presenting a slide, the text description of the slide (Part_1) and the speech narrative of the presentation (Part_2) are provided below.  Please consolidate the two parts into a nice overall description of the video content. \n\n"
PROMPT_Merge_SlideHead = "----------------------------------------------\nPart_1. The text desciption of slide:\n\n"
PROMPT_Merge_SpeechHead = "\n\n----------------------------------------------\nPart_2. The speech narrative of the presentation:\n\n"

def create_message(
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
                {   "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    return messages

def generate_consolidation_response(
    prompt: str,
    model: str = "gpt-4o",
    client: OpenAI = None,
) -> str:
    if client is None:
        client = get_openai_client()
    messages = create_message(prompt)
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        max_tokens=800,
        temperature=0.7,
        top_p=0.95
    )
    output = response.choices[0].message.content
    return output

def generate_consolidation_prompt(slide_understanding_result, speech_transcript):
    prompt = PROMPT_Merge_OverallHead + PROMPT_Merge_SlideHead + slide_understanding_result + PROMPT_Merge_SpeechHead + speech_transcript
    return prompt

if __name__ == "__main__":
    
    print('Start to generate slide understanding result:\n')    
    slide_understanding_result = generate_slide_understanding_result('image\\SceenShot_7.png', "PresentedSlide")
    print("the slide understanding result is: \n\n", slide_understanding_result)

    print('\n\nStart to generate speech transcription:\n')
    audio_file= open("audio\\audio7.mp3", "rb")
    transcription = generate_speech_transcription(audio_file)
    speech_transcript = transcription.text
    print("transcription is: \n\n", speech_transcript)

    print('\n\nStart to consolidate slide and audio info:\n')
    prompt = PROMPT_Merge_OverallHead + PROMPT_Merge_SlideHead + slide_understanding_result + PROMPT_Merge_SpeechHead + speech_transcript
    #prompt = generate_consolidation_prompt(slide_understanding_result, speech_transcript)
    
    print('The consolidation prompt is:\n\n', prompt)
    consolidated_result_slide_plus_audio = generate_consolidation_response(prompt, "gpt-4o") #using "gpt-4" is also fine.
    print('\n\n***************\nThe consolidated understanding is:\n\n', consolidated_result_slide_plus_audio)

    print('\n\nstart to write the output file')
    with open("Result_overall_7_mergedByGPT4.txt", 'w', encoding="utf-8") as out:
        out.write("slide understanding result:------\n\n" + slide_understanding_result 
                  + "\n\nspeech transcription:------\n\n" + speech_transcript
                  + "\n\nconsolidated overall understanding:------\n\n" + consolidated_result_slide_plus_audio)
