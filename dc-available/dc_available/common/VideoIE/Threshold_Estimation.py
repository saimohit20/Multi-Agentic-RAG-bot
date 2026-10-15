from skimage.metrics import structural_similarity as ssim  
import cv2
import os

from .ExtractScreenShotsFromVideo import extract_frame_at_given_time

def automatic_threhold_selection_for_slide_mapping(distinct_slides_path, video_path):  
    slides_summary_path = os.path.join(distinct_slides_path, 'slides_time_info.txt')

    slides = []
    with open(slides_summary_path, 'r') as file:  
        for line in file:  
            slide_name, slide_times = line.strip().split(': ')  
            times_list = eval(slide_times)
            slide = {}
            slide['file'] = os.path.join(distinct_slides_path, slide_name)
            slide['times']= times_list
            slides.append(slide)
    
    similar_scores = []
    for slide in slides:
        if len(slide['times']) > 1:
            imageA = cv2.imread(slide['file'])
            imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            for time_stamp in slide['times'][1:]:
                extract_frame_at_given_time(video_path, float(time_stamp), 'tmp.jpg')
                imageB = cv2.imread('tmp.jpg')
                imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                score, _ = ssim(imageA_gray, imageB_gray, full=True)
                similar_scores.append(score)
        
    similar_scores = sorted(similar_scores)
    print("scores of similar images (ranked)", similar_scores)
                    
    diff_scores = []
    for i in range(len(slides)-1):
        imageA = cv2.imread(slides[i]['file'])
        imageB = cv2.imread(slides[i+1]['file'])
        imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(imageA_gray, imageB_gray, full=True)
        diff_scores.append(score)

    diff_scores = sorted(diff_scores)
    print("scores of different images (ranked)", diff_scores)

    threshold = (similar_scores[0] + diff_scores[-1]) / 2  #to be improved. 
    print("selected threshold: ", threshold)
    return threshold


if __name__ == "__main__":

    distinct_slides_path = 'video\\Kapitel5_slides'
    video_path = 'video\\Kapitel5.mp4'
    threshold = automatic_threhold_selection_for_slide_mapping(distinct_slides_path, video_path)