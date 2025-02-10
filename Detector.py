import cv2
import numpy as np
import time
from kalmanfilter import KalmanFilter

def ballBox(bboxs, bboxIdx, classLabelIDs, classesList):
    res = []
    for i in range(0, len(bboxIdx)):
        #Determine the bbox
        classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
        classLabel = classesList[classLabelID]
        bbox = bboxs[np.squeeze(bboxIdx[i])]
        
        # target class is sports ball which is number 37 in coco dataset
        if classLabelID == 37:
            res.append(bbox)
                        
    return res

#Setup video detector class
class Detector:
    def __init__(self, video, config, model, classes):
        self.video = video
        self.config = config
        self.model = model
        self.classes = classes
        
        # SSD (Single Shot MultiBox Detector) MobileNet v3 utilizes depthwise separable convolutions, which reduce the number of parameters and computation
        self.net = cv2.dnn_DetectionModel(self.model, config)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    #Read in coco objects
    def readClasses(self):
        with open(self.classes, 'r') as f:
            self.classesList = f.read().splitlines()
        
        self.classesList.insert(0, 'temp')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

    def onVideo(self):
        #Attempt opening file
        cap = cv2.VideoCapture(self.video)

        #If file cannot be opened give error
        if(cap.isOpened()==False):
            print("Error opening file")
            return
        
        (ret,image) = cap.read()

        #Get video size
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        #Determine write file name
        filename = ""
        if(self.video == "videos/input/ball.mp4"):
            filename = "videos/output/singleball.mp4"
        elif(self.video == "videos/input/multiObject.avi"):
            filename = "videos/output/multiball.avi"

        res = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,size)

        #Initialize kalman filter (set up with a 4-state vector (position and velocity) and 2 measurement variables (only position)
        kf = KalmanFilter()

        while ret:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.2)

            # Bounding Box
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            # Non-Maximum Suppression (NMS) is applied to eliminate redundant bounding boxes and retain the most confident ones
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.2, nms_threshold = 0.2)

            # Variables used for bounding box coordinates and center positions
            x, y, a, b, cx, cy = 0, 0, 0, 0, 0, 0
                # x, y: Coordinates of the top-left corner of the bounding box around the detected object
                #   - 'x' is the horizontal (x-axis) position
                #   - 'y' is the vertical (y-axis) position

                # a, b: These variables are used for the dimensions of the bounding box
                #   - 'a' represents the width of the bounding box (horizontal dimension)
                #   - 'b' represents the height of the bounding box (vertical dimension)

                # cx, cy: Coordinates of the center of the bounding box
                #   - 'cx' is the x-coordinate of the center
                #   - 'cy' is the y-coordinate of the center

            if len(bboxIdx) != 0:
  
                bbox = ballBox(bboxs, bboxIdx, classLabelIDs, self.classesList)
                displayText = "{}".format("sports ball", thickness = 1)
                if bbox: # bbox exists (visible in the video) 
                    for i in bbox:  # Loop through each bounding box in the 'bbox' list
                        x, y, a, b = i  # Unpack the coordinates (x, y) and dimensions (width 'a', height 'b') of the bounding box
                        
                        # Calculate the center of the bounding box (cx, cy)
                        # cx = (x + (x + a)) / 2 and cy = (y + (y + b)) / 2
                        cx, cy = int((2 * x + a) / 2), int((2 * y + b) / 2)
                        
                        # process: correct/update → predict next state(position)
                        predicted = kf.visible_predict(cx, cy)

                        # Visualize the bounding box
                        # (x, y) for the top-left corner, and (x + a, y + b) for the bottom-right corner
                        cv2.rectangle(image, (x, y), (x + a, y + b), (255, 255, 0), thickness=2)
                        cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
                        cv2.circle(image, (predicted[0], predicted[1]), 5, (0, 0, 0), 4)

                else:  # no bounding box is detected (occlusion)
                    predicted = kf.hidden_predict()  # Predict where the object might be based on previous motion
                    cv2.circle(image, (predicted[0], predicted[1]), 5, (255, 0, 0), 4)


            #Write the bounding box onto the video
            res.write(image)
            cv2.imshow("Result",image)

            #Quit video if q keyis pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (ret, image) = cap.read()

        cap.release()
        res.release()
        cv2.destroyAllWindows()