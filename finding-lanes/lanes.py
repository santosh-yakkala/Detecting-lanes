import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #convert the RGB image to grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0) #cv2.GaussianBlur(name of the image,kernel,deviation)
    canny = cv2.Canny(blur,50,150) #cv2.Canny(name of the image,lower_threshold,upper_threshold) upper : low = 1 to 3
    return canny

def region_of_intrest(image):
    height = image.shape[0]
    polygon = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    cropped_image = cv2.bitwise_and(mask,image)
    return cropped_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    slope , intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = np.polyfit((x1,x2),(y1,y2),1)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# image = cv2.imread('test_image.jpeg') #convert the image into an array
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_intrest(canny_image)
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap = 5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image,averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result',combo_image) #display the image
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()): #cap.isOpened() returns true when the video is initialised
    _ , frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_intrest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap = 5)
    #cv2.HoughLinesP(image,pixels,degree precision,threshold(number of intersections to accept a line),
    #place holder array,lenght of line to be accepted,gap between the lines to consider as a single line)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image) #display the image
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
