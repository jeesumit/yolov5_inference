import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Note: required! to initialize pycuda
import tensorrt as trt
import cv2
trt_engine_path = "model.engine"
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


class TensorRTInference:
    def __init__(self, engine_path):
        # initialize
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        # setup
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(
            self.engine
        )

    def load_engine(self, engine_path):
        # loads the model from given filepath
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem, shape):
            # keeping track of addresses
            self.host = host_mem
            self.device = device_mem
            # keeping track of shape to un-flatten it later
            self.shape = shape

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem, shape))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem, shape))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )

        # run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # transfer predictions back
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(
                self.outputs[i].host, self.outputs[i].device, self.stream
            )

        # synchronize the stream
        self.stream.synchronize()

        # un-flatten the outputs
        outputs = []
        for i in range(len(self.outputs)):
            output = self.outputs[i].host
            output = output.reshape(self.outputs[i].shape)
            outputs.append(output)

        return outputs
        
if __name__ == "__main__":
    classesFile = 'classes.txt'
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    model = TensorRTInference(trt_engine_path)
    imag = cv2.imread('atest.jpg')
    img = cv2.resize(imag, new_dimensions, interpolation=cv2.INTER_LINEAR)
    img_array = np.array(img)
    img_array = img_array.transpose(2, 0, 1)  # (3, 224, 224)
    #print(img.shape)
    val = model.infer(img_array)
    im = post_process(imag,val)
    t = 10.0
    label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
    cv2.putText(im, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
    cv2.imshow('Output', im)
    cv2.waitKey(0)
    #img_array = np.array(img)
    #inputs = img_array.transpose(2, 0, 1)  # (3, 224, 224)
    #print(inputs.shape)
    '''
    print(out,val)
    im = post_process(imag,out)
    t = 10.0
    label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
    cv2.putText(im, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
    cv2.imshow('Output', im)
    cv2.waitKey(0)
    '''
