import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Note: required! to initialize pycuda
import tensorrt as trt
import numpy as np
from PIL import Image
import cv2
trt_engine_path = "model.engine"
onnx_path = "best.onnx"
INPUT_SIZE = 640
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
new_width = INPUT_SIZE
new_height = INPUT_SIZE
new_dimensions = (new_width, new_height)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
    
def post_process(input_image, outputs):
      # Lists to hold respective values while unwrapping.
      class_ids = []
      confs = []
      boxes = []
      # Rows.
      rows = outputs[0].shape[1]
      image_height, image_width = input_image.shape[:2]
      # Resizing factor.
      x_factor = image_width / INPUT_WIDTH
      y_factor =  image_height / INPUT_HEIGHT
      print(x_factor,y_factor)
      # Iterate through detections.
      for r in range(rows):
            row = outputs[0][0][r]
            conf = row[4]
            # Discard bad detections and continue.
            if conf >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confs.append(conf)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        print((cx,cy),(w,h))
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
      indices = cv2.dnn.NMSBoxes(boxes, confs, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
          box = boxes[i]
          left = box[0]
          top = box[1]
          width = box[2]
          height = box[3]
          cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
          label = "{}:{:.2f}".format(classes[class_ids[i]], confs[i])
          draw_label(input_image, label, left, top)
      return input_image

def pre_process(input_image, net):
      # Create a 4D blob from a frame.
      
      image = cv2.resize(input_image,(INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
      blob = cv2.dnn.blobFromImage(image, 1/255,(INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
 
      # Sets the input to the network.
      net.setInput(blob)
 
      # Run the forward pass to get output of the output layers.
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_io_tensors
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)

        self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device_buffers[name] = device_mem
            self.bindings[i] = int(device_mem)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.host_outputs[name] = np.empty(size, dtype=dtype)
                
    def infer(self, image):
        #input_data = preprocess(image)
        input_data = np.ascontiguousarray(image, dtype=np.float32)
        original_shape = image.shape[:2]

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_htod_async(self.device_buffers[name], input_data, self.stream)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        self.context.execute_async_v3(self.stream.handle)

        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)

        self.stream.synchronize()

        outputs = []
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs.append(self.host_outputs[name].reshape(shape))

        return outputs, original_shape
      
    
        
        
if __name__ == "__main__":
    classesFile = 'classes.txt'
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    model = TRTInference(trt_engine_path)
    imag = cv2.imread('atest.jpg')
    '''----------onnx----------------'''
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    detections = pre_process(imag, net)
    frame = post_process(imag, detections)
    '''--------engine------------'''
    img = cv2.resize(imag, new_dimensions, interpolation=cv2.INTER_LINEAR)
    img_array = np.array(img)
    inputs = img_array.transpose(2, 0, 1)  # (3, 224, 224)
    out,val = model.infer(inputs)
    #print(out[0][0][0],val)
    im = post_process(imag,out)
    #t = 10.0
    #label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
    #cv2.putText(im, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
    #cv2.imshow('Output', im)
    #cv2.waitKey(0)
