import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
import random

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(
                tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={
                                "Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[
            :, :, (2, 1, 0)]  # RGB -> BGR

        with tf.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(
                output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]

def game(player):
    possble = ["Scissor", "Paper", "Rock"]
    comp = possble[random.randint(0,2)]
    print(comp)
    result = ""
    if player == comp:
        result = "Draw"
    elif player == "Scissor":
        if comp == "Rock":
            result = "Computer Win"
        elif comp == "Paper":
            result = "Player Win"
    elif player == "Rock":
        if comp == "Scissor":
            result = "Player Win"
        elif comp == "Paper":
            result = "Computer Win"
    elif player == "Paper":
        if comp == "Rock":
            result = "Player Win"
        elif comp == "Scissor":
            result = "Computer Win"
    print(result)
    return result

def main(webcam_id):
    # Load a TensorFlow model
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())
    print(cv2.getWindowImageRect('Frame'))
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)

    cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        sys.exit("Failed to open camera")


    while True:

        if cv2.getWindowProperty("test", 0) < 0:
            print("Display window closed by user, Exiting...")
            break

        ret_val, img = cap.read()

        if not ret_val:
            print("VideoCapture.read() failed, Exiting...")
            break
        


        cv2.imshow("test", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            predictions = od_model.predict_image(image)

            if predictions != []:
                biggest_probability = 0
                for i in range(len(predictions)):
                    if predictions[i]['probability'] > predictions[biggest_probability]['probability']:
                        biggest_probability = i
                        
                ret = predictions[biggest_probability]
                print(ret)
                h, w, _ = img.shape
                prob = ret['probability']
                tagName = ret['tagName']
                bbox = ret['boundingBox']
                left = bbox['left']
                top = bbox['top']
                width = bbox['width']
                height = bbox['height']
                x1 = int(left*w)
                y1 = int(top*h)
                x2 = x1 + int(width * w)
                y2 = y1 + int(height * h)
                p0 = (max(x1, 15), max(y1, 15))

                info = "{:.2f}:-{}".format(prob, tagName)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, info, p0, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
                result = game(tagName)
                cv2.putText(img, result, (150, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
                cv2.imshow("test", img)
                    
                cv2.waitKey(0)               

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} webcam id'.format(sys.argv[0]))
    else:
        main(int(sys.argv[1]))