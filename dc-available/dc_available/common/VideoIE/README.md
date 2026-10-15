This respository is for training video (slide presentation style) information extraction.  

To run this repository, the following packages need to be installed:
********************
opencv-python (tested with opencv-python-4.9.0.80)

scikit-image (tested with scikit-image-0.22.0)

openai==1.12.0

azure-identity==1.15.0

python3 (tested with Python 3.11.8)
********************

Given an input video (x.mp4) to process:
====================
Step1: Extract the audio from the video. Make use to use the mp3 audio format to save the audio.

The audio can be extracted using VLC, either with graphic interface or with following command:

"C:\Program Files\VideoLAN\VLC\vlc.exe" "video_dir\x.mp4" --no-sout-video --sout-audio --sout="#transcode{acodec=mp3,ab=128,channels=2,samplerate=44100}:std{access=file,mux=raw,dst='audio_dir\x.mp3'}"

Attention: When extracting audio, do NOT use moviePy python package.  The duration of audio extracted by moviepy is longer than the video file. 
           The audio must have the same duration as video file, because we need to align the audio with video for slide segmentation. 

------------
Step2: Run main.py, using the video and audio files as two inputs.

In the directory stores the input video x.mp4, two folders will be generated: "x_understanding" and "x_slides".
    
In "x_understanding", an description file is created for each distinct slide in the video.  The desciption file shows the following information: 
    
       (1) the slide name (you can find the corresponding slide image in "x_slides")
       
       (2) the time duration for this slide
       
       (3) The text desciption of slide
       
       (4) The speech narrative for that presentation duration
       
       (5) The consolidated understanding result of (3) and (4)
       
       
Example output files provided in the "video" folder
========================
The original video files can be downloaded from the following link:
 https://bosch-my.sharepoint.com/personal/mdm1lo_bosch_com/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fmdm1lo%5Fbosch%5Fcom%2FDocuments%2FGenAI%5FCR%2DDC%5FCase%2FDigitalTransformation 

Suggest to put video(mp4) and audio(mp3) in the same folder. Then all the created files/folders will be in the same folder.  Otherwise, the speech transcription will be in the same folder as the audio file, while all the other generated files/folders in the same folder as the video.

     
To do for the demo:
===============
The repository serves the functionality to extract understanding results from provided video(s).

For the demo, the next steps are: 

(1) create indexing for the videos based on the understanding results,

(2) RAG function with the indexing database, 

(3) an interface that can:

           a) list the link relevant video
           
           b) show the image of the most relevant slide, together with the slide's time duration in the video
           
           c) illustrate the consolidated understanding info of that slide.
