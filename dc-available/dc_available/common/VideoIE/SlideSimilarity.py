from skimage.metrics import structural_similarity as ssim  
import cv2
from pathlib import Path
from openai import OpenAI

from .ImageUnderstanding import img_to_bytes
from .AzureClient import get_openai_client

THRESHOLD_SLIDE_SIMILARITY = 0.8  #This threshold is needed only if identical_slide_detection_with_cv() is used to identify same slide in video screen shot. Need to be tuned for differnt video recording settings.
PROMPT_SLIDE_COMPARISON = """
For the two images provided, each should show a man presenting a slide.  Are the slide shown in the first image the same as the one shown in the second image? Please start your answer with "Yes. ", "No. ", or "Mostly similar but with minor difference(s).", and then give the explanation. Note that if the titles on the two slides are different, the two slides should be deemed as different and the answer should be "No.".

Also note if one of the two images is not about a man presenting a slide, the answer should be "No.".
If both images are not about a man presenting a slide, determine whether the two images are similer, and answer "Yes. ", "No. ", or "Mostly similar but with minor difference(s).", and then give the explanation.
"""




#The thre_correct, thre_wrong are two thresholds to tune for three caterories (similar, difference, ambiguous)
def image_comparison_with_cv(imageA_gray, imageB_gray, thre_correct = 0.868, thre_wrong = 0.80):  
    # Load the images  
    score, _ = ssim(imageA_gray, imageB_gray, full=True)
    print("similarity score is: ", score)  
    identical_slide = False
    if score > thre_correct:
        return 1 #means definitely similar or same
    elif score < thre_wrong:
        return -1 #means definitely not similar
    else:
        return 0 #means somehow ambiguous

def identical_slide_detection_with_cv(imageA_path, imageB_path):  
    # Load the images  
    imageA = cv2.imread(imageA_path)  
    imageB = cv2.imread(imageB_path)  
    
    # Convert the images to grayscale  
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)  
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)  
      
    # Compute SSIM between two images  
    score, _ = ssim(imageA_gray, imageB_gray, full=True)
    print("similarity score is: ", score)  
    identical_slide = False
    if score > THRESHOLD_SLIDE_SIMILARITY:
        identical_slide = True
    # The score represents the structural similarity index between the two images  
    # If the score is above a certain threshold, we consider the images to be largely the same  
    return identical_slide

def create_message_for_slide_comparison(
    base64_image1: str,
    base64_image2: str,
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
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"},
                },
                {   "type": "text", 
                    "text": PROMPT_SLIDE_COMPARISON
                },
            ],
        }
    ]
    return messages

def identical_slide_detection_with_llm(
    image_path1: str | Path,
    image_path2: str | Path,
    client: OpenAI = None,
    model: str = "gpt-4o",
) -> str:  
    if client is None:
        client = get_openai_client()

    try:
        messages = create_message_for_slide_comparison(img_to_bytes(image_path1),img_to_bytes(image_path2))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95
        )
        answer = response.choices[0].message.content
        print("the llm's judgement is: ", answer)
        result = 0
        if answer[0:3] == "Yes":
            result = 1
        elif answer[0:2] == "No":
            result = -1
        elif answer.find("Sorry") != -1 or answer.find("sorry") != -1:
            result = -1
        return result

    except Exception as e:
        # Handle Azure OpenAI content filter errors
        if hasattr(e, 'response') and hasattr(e, 'status_code'):
            if e.status_code == 400:
                error_msg = str(e)
                if any(keyword in error_msg for keyword in [
                    "content_filter",
                    "ResponsibleAIPolicyViolation",
                    "content management policy",
                    "jailbreak"
                ]):
                    print(
                        f"Warning: Content filter triggered for slide comparison between {image_path1} and {image_path2}")
                    print(f"Content filter reason: {error_msg}")
                    print("Falling back to default similarity assumption...")
                    # Return a neutral result when content filter is triggered
                    return 0  # or -1 if you want to assume they're different by default

        # Handle other OpenAI API errors
        if "openai" in str(type(e)).lower():
            print(f"OpenAI API error during slide comparison: {e}")
            print(f"Images: {image_path1} vs {image_path2}")
            print("Falling back to default similarity assumption...")
            return 0

        # Handle general exceptions (file access, network issues, etc.)
        print(f"Unexpected error during slide comparison: {e}")
        print(f"Images: {image_path1} vs {image_path2}")
        print("Falling back to default similarity assumption...")
        return 0


if __name__ == "__main__":
    
    image1_path = 'video\\TrainingExample_Digit\\Kapitel8_slides\\slide_0001.jpg'
    image2_path = 'video\\TrainingExample_Digit\\Kapitel8_slides\\slide_0002.jpg'
   
    identical_slide = identical_slide_detection_with_cv(image1_path, image2_path) #return value is True or False
    #identical_slide = identical_slide_detection_with_llm(image1_path, image2_path) #return 0 (mostly similar but with difference), 1 (yes, identical), -1 (no, different slide)
    print(identical_slide)  
    
    
