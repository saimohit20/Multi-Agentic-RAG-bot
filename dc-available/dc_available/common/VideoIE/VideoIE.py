import os

from .Speech2Text import generate_speech_transcription
from .ExtractScreenShotsFromVideo import sparse_extract_distince_slides_from_video
from .SegmentAudioTranscript import get_audio_transcript_per_slide, read_in_transcript_per_slide_file
from .ImageUnderstanding import generate_slide_understanding_result
from .MergeSlideSpeechInfo import generate_consolidation_prompt, generate_consolidation_response


#For this function, the video must be mp4 file; the audio must be mp3 file. 
#ignoreVideoHead should be set as True if there is an entry part to ignore per video.  True for DC training videos
#slide_sampling_rate can be set as 30, which is a safer but shlower choice.  For many cases, 60 may already work well.
def training_video_information_extraction(training_video, extracted_audio, ignore_video_head=True, slide_sampling_rate=60, write_to_file=True):
    transcription_file = extracted_audio.replace(".mp3", "__mp3_transcript.txt")
    slides_dir = training_video.replace(".mp4", "__mp4_slides") # e.g.,"video\\Kapitel5_slides"
    speech_transcription_per_slide_file = os.path.join(slides_dir, "speech_transcription_per_slide.txt")
    understanding_dir = training_video.replace(".mp4", "__mp4_understanding")
    if not os.path.exists(understanding_dir):
        os.makedirs(understanding_dir)
    
    print('start speech-to-text')
    audio_file= open(extracted_audio, "rb")
    transcription = generate_speech_transcription(audio_file)
    with open(transcription_file, 'w', encoding="utf-8") as file:  
        file.write(str(transcription))
    print('speech-to-text done')

    print('start to extract distinct slides from video')    
    sparse_extract_distince_slides_from_video(training_video, slides_dir, slide_sampling_rate, "LLM") #60 means every 60 seconds, extract a frame to try
    print('slides extraction done')
    
    print('start to segment video into slide sections and generate a speech-to-text transcription per slide section')
    #transcription per slide and slide segmentation times are stored in speech_transcription_per_slide.txt under slides_dir
    get_audio_transcript_per_slide(slides_dir, transcription_file, training_video, ignore_video_head)
    print('transcription and video segmentation done')
    
    slides_info = read_in_transcript_per_slide_file(speech_transcription_per_slide_file)
    outputs = []
    for sID, slide in enumerate(slides_info):
        slide_image_file_name = slide['filename']
        segment_start = slide['start_time'] #float
        segment_end = slide['end_time'] #float
        slide_transcript = slide['sentences'] 
        #print(slide_image_file, segment_start, segment_end, slide_transcript)
        if segment_end == segment_start or len(slide_transcript) == 0:
            continue

        print("processing ", slide_image_file_name)
        print("--start to generate slide-image understanding result")
        slide_image = os.path.join(slides_dir, slide_image_file_name)        
        slide_image_understanding = generate_slide_understanding_result(slide_image, "PresentedSlide")
        #print(slide_image_understanding)
        
        print("--start to generate consolidated understanding result")
        if slide_image_understanding.find("Sorry") != -1 or slide_image_understanding.find("sorry") != -1:
            slide_image_understanding = ''
            consolidated_understanding = slide_transcript
        else:     
            prompt = generate_consolidation_prompt(slide_image_understanding, slide_transcript)
            consolidated_understanding = generate_consolidation_response(prompt, "gpt-4o") #using "gpt-4" is also fine.
        #print(consolidated_understanding)

        print("--generate file")
        if write_to_file:
            output_file_name = slide_image_file_name.replace('.jpg', '.txt')
            output_file = os.path.join(understanding_dir, output_file_name)
            with open(output_file, 'w', encoding="utf-8") as out:
                out.write("slide name: " + slide_image_file_name + '\n')
                out.write("start time (seconds): " + f"{segment_start:.2f}" + '\n')
                out.write("end time (seconds): " + f"{segment_end:.2f}" + '\n\n')
                out.write('----------------------------------------------\n')
                out.write('Part_1. The text description of slide:\n\n')
                out.write(slide_image_understanding + '\n\n')
                out.write('----------------------------------------------\n')
                out.write('Part_2. The speech narrative of the presentation:\n\n')
                out.write(slide_transcript + '\n\n')
                out.write('----------------------------------------------\n')
                out.write('Info-Consolidation Output:\n\n')
                out.write(consolidated_understanding + '\n')
        slide["slide_image_understanding"] = slide_image_understanding
        slide["consolidated_understanding"] = consolidated_understanding
        outputs.append(slide)
    return outputs


if __name__ == "__main__":
    
    training_video = 'video\\Kapitel5.mp4'
    extracted_audio = 'video\\Kapitel5.mp3' 
    #See vlc_commands.txt for how to extract audio from video. 
    #Attention: Do NOT use moviepy to extract audio, as the audio/video duration will be different.
    #Suggest to use mp3 instead of .wav format for audio.

    training_video_information_extraction(training_video, extracted_audio)






