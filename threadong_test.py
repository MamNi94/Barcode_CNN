import cv2
import threading
import time
import numpy as np
from pyzbar.pyzbar import decode

#AI Barcode detection
import torch
from torchvision import transforms

#open cbar
import zbarlight
from PIL import Image
#Sound Imports
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

#load model
yolo = True
if yolo == True:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')


# Load the sound file
sound_file_true =  './beep.mp3'
audio_true = AudioSegment.from_file(sound_file_true)

sound_file_false =  './wrong.mp3'
audio_wrong = AudioSegment.from_file(sound_file_false)


def beep(audio,start_time = 60 , end_time = 100 ):


    part_audio = audio[start_time:end_time]

    
    play_obj = sa.play_buffer(part_audio.raw_data, 
                            num_channels=part_audio.channels, 
                            bytes_per_sample=part_audio.sample_width, 
                            sample_rate=part_audio.frame_rate)

class VideoCaptureThread:
    def __init__(self, src=0):
        x_resolution = 2600
        y_resolution = 2600
        self.capture = cv2.VideoCapture(src)
        self.capture .set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, x_resolution)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, y_resolution)

        #check actual Data
        actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"Resolution: {actual_width} x {actual_height}")
        print(f"Frame Rate: {actual_fps}")
        self.ret, self.frame = self.capture.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()
        
def scanner(frame, K_pre, K):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, K_pre,0)
        
    # Compute the gradient in the x and y direction
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    #v2.imshow('gradient', gradient)
    # Blur the gradient image
    blurred = cv2.GaussianBlur(gradient, K_post, 0)
    #cv2.imshow('blurr', blurred)
    # Apply a binary threshold to the blurred image
    _,  thresh = cv2.threshold(blurred, threshold_value , 255, cv2.THRESH_BINARY)
    #cv2.imshow('thresh', thresh)
    # Construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, K_morph)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Perform a series of erosions and dilations to remove small blobs
    closed = cv2.erode(closed, None, iterations=10)
    closed = cv2.dilate(closed, None, iterations=10)

    #cv2.imshow('closed', closed)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area, keeping only the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    barcodeContour = None

    if yolo == True:
        yolo_frame = gray.copy()
        results = model(yolo_frame)
        
        results.render()  # This modifies the frame in-place

        scale_factor = 0.3  # This will scale the image to half its original size

        # Calculate the new dimensions of the image
        new_width = int(yolo_frame.shape[1] * scale_factor)
        new_height = int(yolo_frame.shape[0] * scale_factor)
        resized_image = cv2.resize(yolo_frame, (new_width, new_height))
        cv2.imshow('YOLO Barcode', resized_image)
        
    if contours:
        # Assume the largest contour is the barcode
        barcodeContour = contours[0]

        # Compute the bounding box of the barcode region and draw it on the image
        rect = cv2.minAreaRect(barcodeContour)
        box = cv2.boxPoints(rect)
        
        box = np.int32(box)
    
        #cut reigon
        width = int(rect[1][0])
        height = int(rect[1][1])

        # Get the rotation matrix   
        angle = rect[2]
      
        if width < height:
            angle = angle + 90

  
        M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        # Rotate the entire image
        rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))

        ###Rotate Box#############
        rotated_box = cv2.transform(np.array([box]), M)[0]

        # Convert the rotated box points to integer coordinates
        rotated_box = np.int32(rotated_box)
        #cv2.drawContours(rotated, [rotated_box], -1, (0, 255, 0), 2)

        scale_factor = 0.3  # This will scale the image to half its original size

        # Calculate the new dimensions of the image
        new_width = int(rotated.shape[1] * scale_factor)
        new_height = int(rotated.shape[0] * scale_factor)
        resized_image = cv2.resize(rotated, (new_width, new_height))
       


        #########Test#############3
        # Extract the rotated bounding box coordinates
        x, y, w, h = cv2.boundingRect(rotated_box)
        #dim= max(w,h) #+ 100
        cropped = rotated[y-50:y+h+50, x-50:x+w+50]
        
        if cropped.size > 0:
            cv2.imshow('Cropped Image', cropped)
            # Crop the region of interest
            #cropped = sr.upsample(cropped)
            #blurred = cv2.GaussianBlur(cropped, (5,5), 0)
            barcodes = decode(cropped)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 20)
            if barcodes:
                #beep(audio_true)
                for barcode in barcodes:
                    print('Detected Barcode no processing ',barcode.data.decode('utf-8'), time.time()) 
                    
                    
                return
            
            K_cropped = 1
            cropped = cv2.GaussianBlur(cropped, (K_cropped, K_cropped), 0)
            cropped = cv2.equalizeHist(cropped)
            
            #_, cropped = cv2.threshold(cropped, 180, 255, cv2.THRESH_BINARY)
            cv2.imshow('Detected Barcode', cropped)
            barcodes = decode(cropped)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 20)
            if barcodes:
                #beep(audio_true)
                for barcode in barcodes:
                    print('Detected Barcpde processing',barcode.data.decode('utf-8'), time.time()) 
                    
    return

K= 7
K_pre = (3,3)

K_post = (K,K)
threshold_value = 255/2
K_morph = (21,21)

def main():
    cap_thread = VideoCaptureThread()

    while True:
        start = time.time()
        ret, frame = cap_thread.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        scanner(frame, K_pre, K)
        scale_factor = 0.3 # This will scale the image to half its original size

        # Calculate the new dimensions of the image
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        resized_image = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Frame', resized_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        end = time.time()
        print(end-start)

    cap_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
