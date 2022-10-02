# importing the libraries needed
import os
import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import speech_recognition as sr
import cv2
import pyttsx3
from utils import label_map_util
from utils import visualization_utils as vis_util
import threading
import time


# choosing model and dataset to use
MODEL = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_FOLDER = 'http://download.tensorflow.org/models/object_detection/'
GRAPH_FILE = MODEL + '/frozen_inference_graph.pb'
LABEL_FILE = 'mscoco_label_map.pbtxt'
LABEL_PATH = os.path.join(os.getcwd(), 'data', LABEL_FILE)

# downloading the model
if not os.path.exists(GRAPH_FILE):
    print('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_FOLDER + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        if 'frozen_inference_graph.pb' in os.path.basename(file.name):
	        tar_file.extract(file, os.getcwd())
else:
	print('Model already exists :D')

# loading the label
label_map = label_map_util.load_labelmap(LABEL_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, use_display_name=True, max_num_classes = 90)
category_index = label_map_util.create_category_index(categories)

# loading the model
graph = tf.Graph()
with graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(GRAPH_FILE, 'rb') as f:
        serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

# setting object priority
object_dict = {}
object_names = []
vehicles = ['car', 'bus', 'truck']
for i in range(len(categories)):
    object_names.append(categories[i]['name'])
    name = categories[i]['name']
    if name in vehicles:
        categories[i]['priority'] = 1
    elif name == 'person':
        categories[i]['priority'] = 2
    else:
        categories[i]['priority'] = 3
    new_object = {'name':categories[i]['name'],'id':categories[i]['id'],'priority':categories[i]['priority']}
    object_dict[categories[i]['id']] = new_object
    
# utility function to speak messages using a seperate thread
def create_thread(message, INIT_NUM_THREAD, engine, wait = 0.1):
    if len(threading.enumerate()) <= INIT_NUM_THREAD:
        say = threading.Thread(target = engine.say, args = (message,))
        run = threading.Thread(target = engine.runAndWait)
        wait = threading.Thread(target = time.sleep(wait))
        say.start()
        run.start()
        wait.start()

# utility function to perform main operation
def calc_dist(image_np, object, boxes, scores, detected, mode, confident_cutoff = 0.6):
    if scores[0][i] >= confident_cutoff:
        # append 'detected' list
        if mode == 'aware' or mode == 'search':
            detected.append(object)
        
        # calculate distance
        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
        
        # display text (mode) on screen
        cv2.putText(image_np, '{}'.format(apx_distance),
                    (int(mid_x * 800), int(mid_y * 450)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # display warning if object is in a very close distance
        if apx_distance <= 0.5:
            if mid_x > 0.3 and mid_x < 0.7:
                if mode == 'warn':
                    detected.append(object)
                cv2.putText(image_np, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    return image_np

# utility function to gently close the program
def quit(cap):
    cv2.destroyAllWindows()
    cap.release()

# utility class to handle speech recognition
class SpeechRecognizer():
    default_msg = "start speaking and stop once you are done..."
    
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.response = {
            "error": None,
            "transcription": "",
        }
        
    def hear(self) -> dict[str, str]:
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        try:
            self.response["transcription"] = self.recognizer.recognize_google(audio)
        except sr.RequestError:
            self.response["error"] = "Cannot reach API"
        except sr.UnknownValueError:
            self.response["error"] = "Speech unrecognizable"
        return self.response
    
    def get_input(self, message: str = default_msg) -> str:
        print(message)
        response = self.hear()
        return response["transcription"]


# running the model...
url = "http://10.27.234.91:8080/video"
cap = cv2.VideoCapture(url)
engine = pyttsx3.init()

AVAILABLE_MODES = ['aware', 'warn', 'search']
INIT_NUM_THREAD = len(threading.enumerate())
USE_SPEECH = True # toggle this to use speech recognition

# utility function to choose mode based on speech recognition
def choose_mode():
    chosen_mode = None
    if USE_SPEECH:
        while chosen_mode == None:
            mode_input = SpeechRecognizer().get_input()
            print(mode_input)
            chosen_mode = [m for m in AVAILABLE_MODES if m in mode_input]
            chosen_mode = chosen_mode[0] if chosen_mode else None
    else:
        chosen_mode = input("Enter mode: ")
        mode_input = ""
        while chosen_mode.strip().lower() not in AVAILABLE_MODES:
            chosen_mode = input("Enter mode: ")
    return chosen_mode, mode_input

with graph.as_default():
    with tf.compat.v1.Session(graph=graph) as sess:
        create_thread('Starting camera, choose your mode', INIT_NUM_THREAD, engine) 
        
        # initialize looping variables
        iterator, temp = 80, 0
        input_search = True
        search_item = None
        detected = []
        mode = None
        
        # choose mode
        create_thread('Choose your mode', INIT_NUM_THREAD, engine)
        mode, transcripts = choose_mode()
        create_thread(f'Entering {mode} mode...', INIT_NUM_THREAD, engine)

        while True:
            # stopping the program when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit(cap)
                break
            
            # choose mode when 'c' is pressed
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("Switch mode")
                create_thread(f'Switch mode...', INIT_NUM_THREAD, engine)
                mode, transcripts = choose_mode()

            # initialize the graph
            return_value, image_np = cap.read()
            ret,image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes = graph.get_tensor_by_name('detection_boxes:0')
            scores = graph.get_tensor_by_name('detection_scores:0')
            classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # visualizing the detection
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index ,
                use_normalized_coordinates=True,
                line_thickness=8)

            # list objects detected
            detected.clear()
            for i, b in enumerate(boxes[0]):
                image_np = calc_dist(image_np, object_dict[classes[0][i]], boxes, scores, detected, mode = mode)
                detected_object_list = [d['name'] for d in sorted(detected, key = lambda x: x['priority'])]

            # add mode title
            cv2.putText(image_np, f'{mode.capitalize()} Mode', (400,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)

            # mode selection
            if mode == 'aware':
                NO_TOP_PICKS = 3
                print(detected_object_list)
                create_thread(' '.join(detected_object_list[:NO_TOP_PICKS]), INIT_NUM_THREAD, engine) 

            elif mode == 'warn':
                #say = threading.Thread(target = engine.say, args = (message,))
                if len(detected_object_list) == 1:
                    create_thread(f'Warning, {detected_object_list[0]} very close', INIT_NUM_THREAD, engine)

            elif mode == 'search':
                # determine search item
                if input_search == True:
                    # search based on the object that the user has said
                    object_names = [val["name"] for _, val in object_dict.items()]
                    for name in object_names:
                        if name in transcripts:
                            search_item = name
                    
                    # prompt the user for what item to search (if user does not mention it yet)
                    if not search_item:
                        search_item = input("Enter search item: ")
                        while search_item not in object_names:
                            search_item = input("Enter search item: ")
                        create_thread(f"Searching for {search_item}", INIT_NUM_THREAD, engine)
                input_search = False
                
                # check if search_item is in the detected_object_list, exit program when found
                found = search_item in detected_object_list
                if found: temp += 1
                elif temp >= 1:
                    create_thread(f"{search_item} found! Exiting search mode.", INIT_NUM_THREAD, engine)
                    temp += 1
                if temp == 50:
                    quit(cap)
                    break
                
                # periodically say the item currently being searched every LOOP_FRAMES times
                LOOP_FRAMES = 100
                iterator += 1
                if iterator == LOOP_FRAMES:
                    create_thread(f"Still searching for {search_item}", INIT_NUM_THREAD, engine)
                    iterator = 0

            # Show the image
            cv2.imshow('image', cv2.resize(image_np, (1024, 768)))
