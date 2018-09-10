from styx_msgs.msg import TrafficLight
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import rospy

from yolo.utils import get_yolo_boxes, makedirs
from yolo.bbox import draw_boxes
from keras.models import load_model
import keras as kf

#Minimum probability for object detection correctness
MIN_SCORE = 0.90

class YoloDetector(object):

    def __init__(self):
        self.net_h, self.net_w = 416, 416 # a multiple of 32, the smaller the faster
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        self.labels = ["green","red","yellow"]
        self.colors_bgr = [(0,255,0), (0,0,255),(0,255,255)]

        self.light_value = [TrafficLight.GREEN, TrafficLight.RED, TrafficLight.YELLOW]

        self.anchors = [55,69, 75,234, 133,240, 136,129, 142,363, 203,290, 228,184, 285,359, 341,260]

        CWD_PATH = os.getcwd()

        yolo_model_path = os.path.join(CWD_PATH,'yolo','trafficlight.h5')
        self.infer_model = load_model(yolo_model_path)
        self.graph = tf.get_default_graph()

    def predict(self, image):
        with self.graph.as_default():
            boxes = get_yolo_boxes(self.infer_model, [image], self.net_h, self.net_w, self.anchors, self.obj_thresh, self.nms_thresh)[0]
            label_idx = -1
            max_score = 0.
            target_box = None
            for box in boxes:
                if box.get_score() > max_score:
                    max_score = box.get_score()
                    label_idx = box.get_label()
                    target_box = box
            if label_idx == -1:
                return TrafficLight.UNKNOWN, image
            cv2.rectangle(image,(target_box.xmin, target_box.ymin),(target_box.xmax, target_box.ymax),self.colors_bgr[label_idx],3)
            #return self.light_value[label_idx], draw_boxes(image, boxes, self.labels, self.obj_thresh)
            return self.light_value[label_idx], image
            

class TFDetector(object):
    def __init__(self):
        
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph'

        # Number of classes the object detector can identify
        NUM_CLASSES = 3

        self.labels = ['red', 'yellow', 'green']
        self.colors_bgr = [(0,0,255), (0,255,255),(0,255,0)]
        self.label_idx_map = [1, 2, 3]
        self.light_value = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN]

         # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph_sym.pb')

        #if is_site == True:
        #    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph_real.pb')

        rospy.loginfo("TL_C: Load graph from: " + PATH_TO_CKPT)
        
         # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            self.sess = tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options))

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        '''
        
        '''
        self.net_h, self.net_w = 416, 416 # a multiple of 32, the smaller the faster
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        
    def predict(self, image):
          #if image is None or image.shape[0]!=600 or image.shape[1]!=800:
        #    return TrafficLight.UNKNOWN, None

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        #print(image.shape)
        #Find best score
        if boxes is not None and len(boxes) > 0:
            scores = scores[0]
            boxes = boxes[0]
            classes = classes[0]
            max_score_idx = np.argmax(scores)
            #print(classes)
            #print(scores)
            #print("**********************")
            if max_score_idx!=-1 and scores[max_score_idx] >= MIN_SCORE:
                cl = int(classes[max_score_idx])
                score =int(scores[max_score_idx]*100)
                label_idx = self.label_idx_map.index(cl)
                label = self.labels[label_idx]
                rospy.loginfo("TL_C: Found light: "+label + " " + str(score)+"%")
                #rospy.loginfo(boxes[max_score_idx])
                box = boxes[max_score_idx]
                box[0] = int(box[0]*image.shape[0])
                box[1] = int(box[1]*image.shape[1])
                box[2] = int(box[2]*image.shape[0])
                box[3] = int(box[3]*image.shape[1])

                #Add rect and score on the input image
                cv2.rectangle(image,(box[1],box[0]),(box[3], box[2]),self.colors_bgr[label_idx],3)
                #cv2.putText(image, label + " " + str(score)+"'%'",(box[1],int(box[0]-20)), cv2.FONT_HERSHEY_SIMPLEX , 0.5,colors_bgr[label_idx],1,cv2.LINE_AA)
                #print("Ret:"+str(light_value[label_idx]) + str(TrafficLight.RED))
                return self.light_value[label_idx], image
            else:
                rospy.loginfo("TL_C: no light found")

        return TrafficLight.UNKNOWN, image

class TLClassifier(object):
    def __init__(self, is_site):

        #load classifier
        if is_site == True:
            self.detector = YoloDetector()
        else: 
            self.detector = TFDetector()

        print("TF:"+tf.__version__)
        print("Keras:"+kf.__version__)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        return self.detector.predict(image)

