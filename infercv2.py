import cv2
import argparse
import numpy as np

# Colors.
# Constants.
INPUT_WIDTH =550
INPUT_HEIGHT =550
INPUT_SIZE=550
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
onnx_path = "yolact_base_54_800000.onnx"
with open("classes.txt") as f:
    class_names = f.read().strip().split("\n")

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
      # Create a 4D blob from a frame.
      
      image = cv2.resize(input_image,(INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
      blob = cv2.dnn.blobFromImage(image, 1/255,(INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
 
      # Sets the input to the network.
      net.setInput(blob)
 
      # Run the forward pass to get output of the output layers.
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs


def generate_priors():
    feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]
    aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
    scales = [24, 48, 96, 192, 384]
    priors = []
    for idx, fsize in enumerate(feature_map_sizes):
        scale = scales[idx]
        for y in range(fsize[0]):
            for x in range(fsize[1]):
                cx = (x + 0.5) / fsize[1]
                cy = (y + 0.5) / fsize[0]
                for ratio in aspect_ratios[idx]:
                    r = np.sqrt(ratio)
                    w = scale / INPUT_SIZE * r
                    h = scale / INPUT_SIZE / r
                    priors.append([cx, cy, w, h])
    return np.array(priors, dtype=np.float32)

def decode_boxes(loc, priors, variances=[0.1, 0.2]):
    boxes = np.zeros_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes



def post_process(frame, outputs):
    proto,loc,mask,conf=outputs
    

        
    loc = np.squeeze(loc, 0)
    conf = np.squeeze(conf, 0)
    mask = np.squeeze(mask, 0)
    proto = np.squeeze(proto, 0)


    scores = np.max(conf[:, 1:], axis=1)
    classes = np.argmax(conf[:, 1:], axis=1)
    keep = scores > 0.1
    
    if not np.any(keep):
        return frame

    loc, scores, classes, mask = loc[keep], scores[keep], classes[keep], mask[keep]
    priors = generate_priors()[keep]

    boxes = decode_boxes(loc, priors)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                               CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) == 0:
        return frame

    indices = indices.flatten()
    boxes, scores, classes, mask = boxes[indices], scores[indices], classes[indices], mask[indices]


    h, w = frame.shape[:2]
    for i in range(0,len(boxes)):

        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = map(int, [x1 * w, y1 * h, x2 * w, y2 * h])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 2)
        cv2.putText(frame, f"{class_names[int(classes[i])]} {scores[i]:.2f}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)

    return frame


if __name__ == '__main__':
      # Load class names.
      classesFile = 'classes.txt'
      classes = None
      with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
      # Load image.
      # Give the weight files to the model and load the network using       them.
      net = cv2.dnn.readNetFromONNX(onnx_path)
      net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
      net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
      #cap = cv2.VideoCapture("/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4")
      cap = cv2.VideoCapture(0)
      if not cap.isOpened():
          print(f"Error: Could not open video source '{source}'")
      # Process image.
      while True:
          ret, frame = cap.read()
          if not ret:
              print("End of video stream or error reading frame.")
              break
          detections = pre_process(frame, net)
          img = post_process(frame.copy(), detections)
          """
          Put efficiency information. The function getPerfProfile returns       the overall time for inference(t) 
          and the timings for each of the layers(in layersTimes).
          """
          t, _ = net.getPerfProfile()
          label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
          print(label)
          cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
          cv2.imshow('Output', img)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      cv2.destroyAllWindows()
      cap.release()
