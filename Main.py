'''

Project: Handwritten Digit Recognizer Using Softmax Regression

Author: Priyanka Kasture | pkasture2010@gmail.com

-> Live Webcam Support Only
-> Algorithms: 
    1. Softmax Regression - Multi-class Classification [0 to 9 i.e 10 classes]
    2. Gradient Descent - Optimization
    3. Back-Propagation - Adjustment of the weights by calculating the gradient of the loss function

-> Training accuracy percentage: 91.91
-> Testing accuracy percentage: 92.85

'''

''' Importing Libraries '''
import numpy as np
import inputs 
import cv2
import Softmax_Reg

''' Our operations on the frame come here '''

def OnFrameOperations(img):
    x, y, w, h = 0, 0, 300, 300
    
    # Converting the captured picture to gray-scale image and storing it into another variable named 'gray'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Using the Gaussian Kerel for bluring the edges, removing noises, and removing high frequency content 
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    
    # If pixel value is greater than a threshold value, it is assigned one value, else it is assigned another value.
    # Here THRESH_BINARY_INV is used, that means the background will appear - black and the potential digit - white
    # THRESH_OTSU is used to calculate the threshold of bimodal images, or images whose histogram has two peaks
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh1 = thresh1[y:y + h, x:x + w]
    
    # cv2.findContours function gives us a list of contours that it has found
    # cv2.RETR_TREE tells OpenCV to compute the hierarchy (relationship) between contours 
    # We tell OpenCV to compress the contours to save space using cv2.CV_CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    return img, contours, thresh1

''' The main function '''
def main():
    
    '''Placeholders for Training Sets'''
    mnist = inputs.read_data_sets("MNIST_data/", one_hot=False)
    data = mnist.train.next_batch(8000) # randomly picking 8000 samples from the MNIST dataset
    # Function returns 'data' which is a tuple of two elements; containing flattened images [0] and their corresponding labels [1]
    
    train_x = data[0] # Flattened Images
    Y = data[1] # labels, actual numbers, vector
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int) # encoded array
    
    '''Placeholders for Testing Sets'''
    mnist = inputs.read_data_sets("MNIST_data/", one_hot=False)
    tb = mnist.train.next_batch(2000)
    Y_test = tb[1]
    X_test = tb[0]
    
    directory = Softmax_Reg.model(train_x.T, train_y.T, Y, X_test.T, Y_test, num_iters=2000, alpha=0.07, print_cost=True)
    # Returns a dictionary of Costs, Weights and Biases recorded after each Iteration
    
    weights = directory["w"] # Final Weights
    biases = directory["b"] # Final Biases

    ''' WebCam Support '''

    ''' Creating a VideoCapture object. Its argument can be either the device index or the name of a video file '''
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        
        # Capturing frame-by-frame
        ret, img = cap.read()
        
        img, contours, thresh = OnFrameOperations(img)
        answer = ''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                Captured_Image = thresh[y:y + h, x:x + w]
                
                # Resizing the captured image into 28X28
                Captured_Image = cv2.resize(Captured_Image, (28, 28))
                
                # Converting the captured image into a numpy array
                Captured_Image = np.array(Captured_Image)
                
                # Flattening the captured image
                Captured_Image = Captured_Image.flatten()
                
                Captured_Image = Captured_Image.reshape(Captured_Image.shape[0], 1)
                answer = Softmax_Reg.predict(weights, biases, Captured_Image)

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Predicted Digit is " + str(answer), (30, 320),cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
        
        # imshow - Displaying an images being captured in the specified window
        # Parameters are 1 - Name of the Window and 2 - Image to be Shown
        cv2.imshow("Window",img)
        cv2.imshow("Contours",thresh)
        k=cv2.waitKey(10)
        if k==20:
            break
        
main()
