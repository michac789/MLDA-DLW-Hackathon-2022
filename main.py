# importing the libraries needed
import os
import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import pyttsx3
from utils import visualization_utils as vis_util
import threading
import time
import speech_recognition as sr


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
def read_label_map(label_map_path, default_priority = 3, default_width = 0.1):
    items = {}
    with open(label_map_path, "r") as file:
        
        item_id = None
        item_name = None
        item_priority = default_priority
        item_width = default_width

        for line in file:
            line.replace(" ", "")
            if "item{" in line:
                pass
            elif  "}" in line:
                items[item_id] = {'id':item_id,
                                  'name':item_name,
                                  'priority':item_priority,
                                  'width':item_width}
                item_id = None
                item_name = None
                item_priority = default_priority
                item_width = default_width
            elif "id:" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name:" in line:
                item_name = line.split(":", 1)[1].replace('"', "").strip()
            elif "priority:" in line:
                item_priority = int(line.split(":", 1)[1].strip())
            elif "width:" in line:
                item_width = float(line.split(":", 1)[1].strip())
    return items

object_dict = read_label_map(LABEL_PATH)
object_names = []
for e in object_dict:
    object_names.append(object_dict[e]['name'])

# loading the model
graph = tf.Graph()
with graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(GRAPH_FILE, 'rb') as f:
        serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

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
    
# creating a new thread for the engine voice message
def create_thread(message, INIT_NUM_THREAD, engine, wait = 0.1):
    if len(threading.enumerate()) <= INIT_NUM_THREAD:
        say = threading.Thread(target = engine.say, args = (message,))
        run = threading.Thread(target = engine.runAndWait)
        end = threading.Thread(target = engine.endLoop)
        wait = threading.Thread(target = time.sleep(wait))
        say.start()
        run.start()
        end.start()
        wait.start()

# depth perception calculations
def calc_dist(image_np, index, object, boxes, scores, detected, mode, confident_cutoff = 0.6, dist_0 = 1):
    if scores[0][index] >= confident_cutoff:
        mid_x = (boxes[0][index][1]+boxes[0][index][3]) / 2
        mid_y = (boxes[0][index][0]+boxes[0][index][2]) / 2
        perceived_width = abs(boxes[0][index][3] - boxes[0][index][1])
        scale = object['width'] / perceived_width
        apx_distance = dist_0 * scale
        apx_distance = round(apx_distance, 1)
        object['distance'] = apx_distance

        if mode == 'aware' or mode == 'search':
            detected.append(object)

        cv2.putText(image_np, '{} m'.format(apx_distance), (
            int(mid_x * 800), int(mid_y * 450)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if apx_distance <= 1:
            if mode == 'warn':                    
                detected.append(object)
                cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    return image_np

# utility function to gently close the program
def quit(cap):
    cv2.destroyAllWindows()
    cap.release()


# running the model...
IP = '0.0.0.0' # for mobile connection, change ip here
url = 'http://' + IP + ':8080/video'
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()

AVAILABLE_MODES = ['aware','warn','search']
INIT_NUM_THREAD = len(threading.enumerate())
USE_SPEECH = False # toggle this to use speech recognition
SEARCH_LOOP_FRAMES = 100
QUIT_DELAY = 50

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

mode, transcripts = choose_mode()

with graph.as_default():
    with tf.compat.v1.Session(graph=graph) as sess:
        
        # Initialize looping variables
        iterator = 0
        input_search = True
        search_item = None
        detected = []
        quit_counter = 0

        while True:
            # stopping the program when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit(cap)
                break

            # initialize the graph
            return_value, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes = graph.get_tensor_by_name('detection_boxes:0')
            scores = graph.get_tensor_by_name('detection_scores:0')
            classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict = {image_tensor: image_np_expanded})

            # visualizing the detection
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                object_dict,
                use_normalized_coordinates=True,
                line_thickness=5)

            # list objects detected
            detected.clear()
            for i, b in enumerate(boxes[0]):
                image_np = calc_dist(image_np, i, object_dict[classes[0][i]], boxes, scores, detected, mode = mode)
                object_distance = {d['name']:d['distance'] for d in sorted(detected, key = lambda x: x['priority'])}
            
            object_found = []
            for e in object_distance:
                object_found.append(e)

            # add mode title
            cv2.putText(image_np, f'{mode.capitalize()} Mode', (400,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)

            # mode selection
            if mode == 'aware':
                NO_TOP_PICKS = 3
                create_thread(' '.join(object_found[:NO_TOP_PICKS]), INIT_NUM_THREAD, engine) 

            elif mode == 'warn':
                if len(object_found) != 0:
                    create_thread(f'Warning, {object_found[0]} very close', INIT_NUM_THREAD, engine)

            elif mode == 'search':
                # determine search item
                if input_search == True:
                    if USE_SPEECH:
                        # search based on the object that the user has said
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
                
                # check if search_item is in the object_found, exit program when found
                # quit counter to delay the quitting
                found = search_item in object_found
                if found and quit_counter==0: 
                    create_thread(f"{search_item} found {object_distance[search_item]} meters away! Exiting search mode.", INIT_NUM_THREAD, engine)
                    quit_counter = 1
                elif quit_counter >= 1:
                    quit_counter += 1
                if quit_counter == 50:
                    quit(cap)
                    break

                # periodically say the item currently being searched every LOOP_FRAMES times
                iterator += 1
                if iterator == SEARCH_LOOP_FRAMES:
                    create_thread(f"Still searching for {search_item}", INIT_NUM_THREAD, engine)
                    iterator = 0

            # Show the image
            cv2.imshow('image',cv2.resize(image_np,(1024,768)))
