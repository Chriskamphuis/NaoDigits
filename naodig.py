import sys
import numpy as np
import cv2
from naoqi import ALProxy
from matplotlib import pyplot as plt

class NaoCvML:

    def __init__(self, IP="131.174.106.197", PORT=9559):
        self.IP = IP
        self.PORT = PORT
        self.video = None
        try:
            self.video = ALProxy('ALVideoDevice', self.IP, self.PORT)
        except Exception as e:
            print e
            print "Can not connect with NAO"

    def snap(self):

        # subscribe top camera
        AL_kTopCamera = 0
        AL_kQVGA = 1           
        AL_kBGRColorSpace = 13
        captureDevice = self.video.subscribeCamera(
            "video_stream", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

        # create image
        width = 320 
        height = 240 
        image = np.zeros((height, width, 3), np.uint8)

        result = self.video.getImageRemote(captureDevice);

        if result == None:
            print 'cannot capture.'
        elif result[6] == None:
            print 'no image data string.'
        else:
            # translate value to mat
            values = map(ord, list(result[6]))
            i = 0
            for y in range(0, height):
                for x in range(0, width):
                    image.itemset((y, x, 0), values[i + 0])
                    image.itemset((y, x, 1), values[i + 1])
                    image.itemset((y, x, 2), values[i + 2])
                    i += 3

            # show image
            cv2.imwrite("nao-top-camera-320x240-snap.png", image)
        self.video.unsubscribe(captureDevice)
        return image

    def invert(self, image):
        return (255-image)

    def edit_image(self, image): 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray_image, 140, 255, 0)
        gray1 = gray.copy()
        gray2 = gray.copy() 
        contours, hier = cv2.findContours(gray,cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)

        x, y, w, h = 0, 0, 0, 0
        for cnt in contours:
            if 2000<cv2.contourArea(cnt)<10000:
                (x,y,w,h) = cv2.boundingRect(cnt)
                cv2.rectangle(gray2,(x,y),(x+w,y+h),0,-1)
        mask = cv2.addWeighted(gray1,0.5,gray2,0.5,0)
        pnts1 = np.float32([[x, y],[x+w, y],[x, y+h],[x+w, y+h]])
        pnts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        M = cv2.getPerspectiveTransform(pnts1, pnts2)
        dst = cv2.warpPerspective(mask, M, (300,300))
        ret, fin = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
        resized_image = cv2.resize(fin, (20, 20))

        cv2.imwrite("nao-top-camera-320x240-snap_e.png", gray1)
        cv2.imwrite("nao-top-camera-320x240-snap_e2.png", gray2)
        cv2.imwrite("nao-top-camera-320x240-snap_e3.png", mask)
        cv2.imwrite("nao-top-camera-320x240-snap_e4.png", dst)
        cv2.imwrite("nao-top-camera-320x240-snap_e5.png", fin)
        cv2.imwrite("nao-top-camera-320x240-snap_e6.png", resized_image)
        return resized_image 

    def learn_digits(self):
    
        img = cv2.imread('digits.png')
        img = self.invert(img)
        cv2.imwrite("invert.png", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        # Now we split the image to 5000 cells, each 20x20 size
        cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
          
        # Make it into a Numpy array. It size will be (50,100,20,20)
        x = np.array(cells)

        # Now we prepare train_data and test_data.
        train = x[:,:50].reshape(-1,400).astype(np.float32)
        test = x[:,50:100].reshape(-1,400).astype(np.float32)

        # Create labels for train and test data
        k = np.arange(10)
        train_labels = np.repeat(k,250)[:,np.newaxis]
        test_labels = train_labels.copy()

        # Initiate kNN, train the data, then test it with test data for k=1
        knn = cv2.KNearest()
        knn.train(train,train_labels)
        ret,result,neighbours,dist = knn.find_nearest(test,k=5)
        
        number = self.edit_image(self.snap())
        number = number.reshape(-1, 400).astype(np.float32)
        nparray = np.array(number)
        ret2, result2, neighbours2, dist2 = knn.find_nearest(nparray,k=5)
        print result2


        # Now we check the accuracy of classification
        # For that, compare the result with test_labels
        # and check which are wrong
        matches = result==test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct*100.0/result.size
        print accuracy
