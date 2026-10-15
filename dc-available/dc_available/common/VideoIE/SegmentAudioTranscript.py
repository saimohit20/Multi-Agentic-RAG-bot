import os 
import ast
from skimage.metrics import structural_similarity as ssim  
import cv2

from .Speech2Text import read_in_transcript_file
from .Speech2Text import extract_sentences
from .ExtractScreenShotsFromVideo import extract_frame_at_given_time
  
def read_slides_time_info_file(file_path):  
    slides_info = []  
      
    with open(file_path, 'r') as file:  
        for line in file:  
            parts = line.strip().split(':')  
            if len(parts) == 2:  
                filename = parts[0].strip()  
                # Use ast.literal_eval to safely evaluate the times string into a list  
                times = ast.literal_eval(parts[1].strip())  
                min_time = float(times[0])
                max_time = float(times[-1])
                slides_info.append({'filename': filename, 'min_time': min_time, 'max_time': max_time, 'sentences': ''})

    #print(slides_info)
    return slides_info

def get_audio_transcript_per_slide(distinct_slides_path, whole_audio_transcript, video_path, ignore_video_head=True):
    if (not os.path.exists(distinct_slides_path)) or (not os.path.isfile(whole_audio_transcript)) or (not os.path.isfile(video_path)):
        print("directory and/or file not found")        
        return False
    
    #read in distinct slides information
    slides_time_info_file = os.path.join(distinct_slides_path, "slides_time_info.txt")
    if not os.path.isfile(slides_time_info_file):
        print("missing slides_time_info.txt file")
        return False

    distinct_slides = read_slides_time_info_file(slides_time_info_file)
    #print(distinct_slides)

    transcript_text, language, time_segments = read_in_transcript_file(whole_audio_transcript)
    segments = ast.literal_eval(time_segments)
    sentences = extract_sentences(segments)  
    active_slide = 0
    print("active_slide: ", active_slide)
    for sen in sentences: 
        sen_text = sen['text']
        #print(sen_text)
        start_time = round(sen['start_time'] * 100) / 100
        end_time = round(sen['end_time'] * 100) / 100
        
        #if start1 < end2 and start2 < end1: 
        if start_time < distinct_slides[active_slide]['max_time'] and distinct_slides[active_slide]['min_time'] < end_time: #if there is overlap
            distinct_slides[active_slide]['sentences'] += sen_text.strip() + " "
            distinct_slides[active_slide]['min_time'] = min(start_time, distinct_slides[active_slide]['min_time'])
            distinct_slides[active_slide]['max_time'] = max(end_time, distinct_slides[active_slide]['max_time'])
        
        elif start_time >= distinct_slides[active_slide]['max_time']:
            if active_slide == (len(distinct_slides) - 1):  #to do: further improve
                distinct_slides[active_slide]['sentences'] += sen_text.strip() + " "
                distinct_slides[active_slide]['max_time'] = end_time
            
            elif end_time > distinct_slides[active_slide+1]['min_time']:
                active_slide = active_slide + 1
                print("active_slide: ", active_slide)
                distinct_slides[active_slide]['sentences'] += sen_text.strip() + " "
                distinct_slides[active_slide]['min_time'] = start_time
                distinct_slides[active_slide]['max_time'] = max(end_time, distinct_slides[active_slide]['max_time'])
            
            else: #if end_time <= distinct_slides[active_slide+1]['min_time']
                middle_time = (start_time + end_time) / 2
                extract_frame_at_given_time(video_path, middle_time, 'tmp.jpg')
                slide_cur = os.path.join(distinct_slides_path, distinct_slides[active_slide]['filename'])
                slide_next = os.path.join(distinct_slides_path, distinct_slides[active_slide+1]['filename'])
                image_to_compare = cv2.imread('tmp.jpg')
                image_slide_cur = cv2.imread(slide_cur)
                image_slide_next = cv2.imread(slide_next)
                image_to_compare_gray = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2GRAY)
                image_slide_cur_gray = cv2.cvtColor(image_slide_cur, cv2.COLOR_BGR2GRAY)
                image_slide_next_gray = cv2.cvtColor(image_slide_next, cv2.COLOR_BGR2GRAY)
                score_compare_with_cur_slide, _ = ssim(image_to_compare_gray, image_slide_cur_gray, full=True)
                print("score_compare_with_cur_slide: ", score_compare_with_cur_slide)
                score_compare_with_next_slide, _ = ssim(image_to_compare_gray, image_slide_next_gray, full=True)
                print("score_compare_with_next_slide: ", score_compare_with_next_slide)
                #to do: to further improve algorithm
                if score_compare_with_cur_slide >= score_compare_with_next_slide:
                    distinct_slides[active_slide]['sentences'] += sen_text.strip() + " "
                    distinct_slides[active_slide]['max_time'] = end_time
                else:
                    active_slide = active_slide + 1
                    print("active_slide: ", active_slide)
                    distinct_slides[active_slide]['sentences'] += sen_text.strip() + " "
                    distinct_slides[active_slide]['min_time'] = start_time

    if ignore_video_head == True and len(distinct_slides) > 1 and len(distinct_slides[0]['sentences']) > 0:
        distinct_slides[1]['sentences'] = distinct_slides[0]['sentences'] + distinct_slides[1]['sentences']
        distinct_slides[1]['min_time'] = 0.0
        distinct_slides[0]['max_time'] = 0.0
        distinct_slides[0]['sentences'] = ""

    speech_transcription_per_slide_file = os.path.join(distinct_slides_path, "speech_transcription_per_slide.txt")
    with open(speech_transcription_per_slide_file, 'w', encoding="utf-8") as out: 
        for slide in distinct_slides:
            out.write('------------\n')
            out.write(slide['filename'] + '\n')
            out.write("min_time: " + str(slide['min_time']) + '\n')
            out.write("max_time: " + str(slide['max_time']) + '\n')
            out.write("sentences: " + slide['sentences'] + '\n\n')            

    if os.path.exists('tmp.jpg'):
        os.remove('tmp.jpg')
    
    return True


def read_in_transcript_per_slide_file(file_path):  
    # List to hold all slides' content as dictionaries  
    slides = []  
      
    # Open and read the file content  
    with open(file_path, 'r') as file:  
        content = file.read()  
      
    # Split the content by the delimiter "------------"  
    entries = content.split("------------")  
      
    # Process each entry  
    for entry in entries:  
        # Skip empty entries  
        if not entry.strip():  
            continue  
          
        # Split the entry by "min_time:", "max_time:", and "sentences:"  
        parts = entry.split('min_time: ')  
        filename = parts[0].strip()  
        min_time, rest = parts[1].split('max_time: ')  
        max_time, sentences = rest.split('sentences: ')  
          
        # Create a dictionary for the current slide  
        slide_info = {  
            'filename': filename,  
            'start_time': float(min_time.strip()),  
            'end_time': float(max_time.strip()),  
            'sentences': sentences.strip()  
        }  
          
        # Add the dictionary to the list of slides  
        slides.append(slide_info)  
      
    return slides  
    

if __name__ == "__main__":

    distinct_slides_path = "video\\Kapitel5_slides"
    whole_audio_transcript = "audio\\transcript_audio5.txt"    
    video_path = 'video\\Kapitel5.mp4'
    success = get_audio_transcript_per_slide(distinct_slides_path, whole_audio_transcript, video_path)

    '''
    speech_transcription_per_slide_file = "video\\Kapitel5_slides\\speech_transcription_per_slide.txt"
    slides_transcription_info = read_in_transcript_per_slide_file(speech_transcription_per_slide_file)
    for s in slides_transcription_info:
        print(s)
    '''
