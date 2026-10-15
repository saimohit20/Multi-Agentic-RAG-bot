from .VideoIE import training_video_information_extraction

training_video = 'video\\TrainingExample_SmartSystems\\Kapitel16.mp4'
extracted_audio = 'video\\TrainingExample_SmartSystems\\Kapitel16.mp3' 
#See Readme file for how to extract audio from video. 
#Attention: Do NOT use moviepy to extract audio, as the audio/video duration will be different.
#Suggest to use mp3 instead of .wav format for audio.

#This function generates two subfolders: *_slides (storing images of distinct slides)and  *_understanding (storing understanding results per slide plus time duration)
slide_sampling_rate = 60 #set slide_sampling_rate=60 may already work well. though set it as 30 will be a safer (but slower) choice for video frequently changing slides.
ignore_video_head = True #ignoreVideoHead set as True if there is a same entry part to ignore per video.  True for DC training videos
training_video_information_extraction(training_video, extracted_audio, ignore_video_head, slide_sampling_rate) 
