import cv2  
import os
from skimage.metrics import structural_similarity as ssim

from .SlideSimilarity import identical_slide_detection_with_llm
from .SlideSimilarity import image_comparison_with_cv

THRESHOLD = 0.8  #This threshold is needed in extract_distince_slides_from_video_using_cv(). Need to be tuned for differnt video recording settings. 

def extract_frame_at_given_time(video_path, time_in_seconds, output_path):  
    # Capture the video from the given path  
    video = cv2.VideoCapture(video_path)  
      
    # Check if video opened successfully  
    if not video.isOpened():  
        print("Error: Could not open video.")  
        return  
      
    # Set the video position to the specific time in milliseconds  
    video.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds*1000)  
      
    # Read the frame at the specified time  
    success, image = video.read()  
      
    # If the frame was read successfully  
    if success:  
        # Write the frame to the output path as an image  
        cv2.imwrite(output_path, image)  
        print(f"Frame extracted and saved to {output_path}")  
    else:  
        print("Error: Could not extract frame at the given time.")  
      
    # Release the video capture object  
    video.release()  

  
def extract_frames_with_fixed_interval(video_path, interval=60):  
    # Check if the video file exists  
    if not os.path.isfile(video_path):  
        print("File not found:", video_path)  
        return  
      
    # Create a directory to store the extracted frames  
    base_path, _ = os.path.splitext(video_path)  
    frame_dir = base_path + '_frames' + '_interval' + str(interval)
    if not os.path.exists(frame_dir):  
        os.makedirs(frame_dir)  
  
    # Capture the video  
    video = cv2.VideoCapture(video_path)  
      
    # Get video properties  
    fps = video.get(cv2.CAP_PROP_FPS)  
      
    # Calculate the frame number after every 30 seconds  
    frame_interval = int(fps * interval)  
      
    frame_count = 0  
    image_count = 0  
      
    while True:  
        success, frame = video.read()  
        if not success:  
            break  
          
        # Check if we are at the interval to save the frame  
        if frame_count % frame_interval == 0:  
            image_path = os.path.join(frame_dir, f"frame_{image_count:04d}.jpg")  
            cv2.imwrite(image_path, frame)  
            print(f"Extracted frame at {frame_count/fps:.2f} seconds, saved to {image_path}")  
            image_count += 1  
          
        frame_count += 1  
      
    video.release()  
    print("Done extracting frames.")  



#The function below can adopt two possible methods: "LLM" or "CV"
def extract_distince_slides_from_video(video_path, output_dir, interval=60, method = "LLM"):
    slides_with_time_info = []
    # Check if the video file exists  
    if not os.path.isfile(video_path):  
        print("File not found:", video_path)  
        return  
      
    # Create a directory to store the extracted frames  
    slide_dir = output_dir
    if not os.path.exists(slide_dir):  
        os.makedirs(slide_dir)  
  
    # Capture the video  
    video = cv2.VideoCapture(video_path)  
      
    # Get video properties  
    fps = video.get(cv2.CAP_PROP_FPS)  
      
    # Calculate the frame number after every 60 seconds  
    frame_interval = int(fps * interval)      
    frame_count = 0  
    image_count = 0  
    firstFlag = True
    prev_image_path = ""
    print("start processing")
    frame_gray = None
    pre_frame_gray = None

    while True:  
        success, frame = video.read()  
        if not success:  
            break  
          
        # Check if we are at the interval to save the frame  
        if frame_count % frame_interval == 0: 
            
            file_name = f"slide_{image_count:04d}.jpg"
            image_path = os.path.join(slide_dir, file_name)  
            cv2.imwrite(image_path, frame)  
            print(f"saved to {image_path}")
            if method == 'CV':
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if firstFlag == True:                
                firstFlag = False
                prev_image_path = image_path
                if method == 'CV':
                    pre_frame_gray = frame_gray
                image_count += 1
                slide_info = {}
                slide_info['file'] = file_name
                slide_info['time'] = [f"{frame_count/fps:.2f}"]
                slides_with_time_info.append(slide_info)                
            else:
                print(prev_image_path, image_path)
                similar_level = -1
                if method == 'CV':
                    similar_level = image_comparison_with_cv(pre_frame_gray, frame_gray, 0.868, 0.8)
                else: #method == "LLM"
                    similar_level = identical_slide_detection_with_llm(prev_image_path, image_path)
                print("comparision done, with similar_level: ", similar_level)
                if similar_level == -1: #if definitly different slide
                    slide_info = {}
                    slide_info['file'] = file_name
                    slide_info['time'] = [f"{frame_count/fps:.2f}"]
                    slides_with_time_info.append(slide_info)
                    image_count += 1
                    prev_image_path = image_path
                    if method == 'CV':
                        pre_frame_gray = frame_gray
                else:  #if being the same slide or mostly similar. To be further improved.
                    slides_with_time_info[-1]['time'].append(f"{frame_count/fps:.2f}")
                    os.remove(image_path)
                                        
        frame_count += 1  

    video.release()  

    summary_path = os.path.join(slide_dir, "slides_time_info.txt")
    with open(summary_path, 'w') as out:
        for slide_info in slides_with_time_info:
            out.write(slide_info['file'] + ": " + str(slide_info['time']) + '\n')

    print("Done extracting slides.")

def get_video_duration(video_path):  
    """  
    Returns the duration of the video in seconds.  
  
    Parameters:  
    video_path (str): The file path to the video.  
      
    Returns:  
    float: Duration of the video in seconds.  
    """  
    if not os.path.isfile(video_path):  
        print("File not found:", video_path)  
        return -1 

    # Capture the video  
    video = cv2.VideoCapture(video_path)  
      
    # Get frames per second  
    fps = video.get(cv2.CAP_PROP_FPS)
         
    # Get number of frames  
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  
    
    # Release the video capture object  
    video.release()  
      
    # Calculate duration  
    duration = frames / fps  
    return duration  #return duration in seconds


#The function below can adopt two possible methods: "LLM" or "CV". More efficent if only sampling every 30 seconds or more
def sparse_extract_distince_slides_from_video(video_path, output_dir, interval=60, method = "LLM"):
    slides_with_time_info = []
    # Check if the video file exists  
    if not os.path.isfile(video_path):  
        print("File not found:", video_path)  
        return  

    # Create a directory to store the extracted frames  
    slide_dir = output_dir
    if not os.path.exists(slide_dir):  
        os.makedirs(slide_dir)  
  
    image_count = 0
    prev_image_path = ""
    frame_gray = None
    pre_frame_gray = None
    print("start processing")
    
    duration = get_video_duration(video_path)
    time_cur = 0.0
    firstFlag = True
    while time_cur <= duration:
        file_name = f"slide_{image_count:04d}.jpg"
        image_path = os.path.join(slide_dir, file_name)
        extract_frame_at_given_time(video_path, time_cur, image_path)
        print(f"saved to {image_path}")
        frame = cv2.imread(image_path)
        if method == 'CV':
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if firstFlag == True:                
            firstFlag = False
            prev_image_path = image_path
            if method == 'CV':
                pre_frame_gray = frame_gray
            image_count += 1
            slide_info = {}
            slide_info['file'] = file_name
            slide_info['time'] = [f"{time_cur:.2f}"]
            slides_with_time_info.append(slide_info)                
        else:
            print(prev_image_path, image_path)
            similar_level = -1
            if method == 'CV':
                similar_level = image_comparison_with_cv(pre_frame_gray, frame_gray, 0.868, 0.8)
            else: #method == "LLM"
                similar_level = identical_slide_detection_with_llm(prev_image_path, image_path)
            print("comparision done, with similar_level: ", similar_level)
            if similar_level == -1: #if definitly different slide
                slide_info = {}
                slide_info['file'] = file_name
                slide_info['time'] = [f"{time_cur:.2f}"]
                slides_with_time_info.append(slide_info)
                image_count += 1
                prev_image_path = image_path
                if method == 'CV':
                    pre_frame_gray = frame_gray
            else:  #if being the same slide or mostly similar. To be further improved.
                slides_with_time_info[-1]['time'].append(f"{time_cur:.2f}") 
                os.remove(image_path)
        time_cur += interval
    
    summary_path = os.path.join(slide_dir, "slides_time_info.txt")
    with open(summary_path, 'w') as out:
        for slide_info in slides_with_time_info:
            out.write(slide_info['file'] + ": " + str(slide_info['time']) + '\n')

    print("Done extracting slides.")


if __name__ == "__main__":

    video_file_path = "video\\Kapitel5.mp4"

    '''
    time_interval = 60
    extract_frames_with_fixed_interval(video_file_path, time_interval)  
    print('interval based extraction done')
    
    screenshot_time = 4.0  # Time in seconds  
    output_image_path = 'video\\test_image_s1.jpg'  # Replace with your desired output image path  
    extract_frame_at_given_time(video_file_path, screenshot_time, output_image_path)  
    print('specific time based extraction done')    
    '''

    interval = 60
    output_dir = "video\\Kapitel5_slides_test"    
    #extract_distince_slides_from_video(video_file_path, output_dir, interval, "LLM") 
    #extract_distince_slides_from_video(video_file_path, output_dir, interval, "CV")

    #print(get_video_duration(video_file_path))
    sparse_extract_distince_slides_from_video(video_file_path, output_dir, interval, "LLM") #recommended solution for distict slide extraction
    #sparse_extract_distince_slides_from_video(video_file_path, output_dir, interval, "CV")
