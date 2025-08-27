import onnxruntime
import numpy as np
import cv2
import time

# Constants
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
INPUT_SIZE = 550
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    img = (img - MEANS) / STD
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

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

def decode(loc, priors, variances=[0.1, 0.2]):
    boxes = np.zeros_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    return np.array(indices).flatten() if len(indices) > 0 else np.array([], dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(output, original_shape):
    loc, conf, mask, _, proto = output
    loc = np.squeeze(loc, axis=0)
    conf = np.squeeze(conf, axis=0)
    mask = np.squeeze(mask, axis=0)
    proto = np.squeeze(proto, axis=0)

    scores = np.max(conf[:, 1:], axis=1)
    classes = np.argmax(conf[:, 1:], axis=1)
    keep = scores > 0.5

    if not np.any(keep):
        return [], [], [], []

    scores = scores[keep]
    classes = classes[keep]
    mask = mask[keep]
    loc = loc[keep]
    priors = generate_priors()[keep]

    boxes = decode(loc, priors)
    keep_nms = nms(boxes, scores, iou_threshold=0.5)

    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    classes = classes[keep_nms]
    mask = mask[keep_nms]

    masks = proto @ mask.T
    masks = sigmoid(masks)
    masks = np.transpose(masks, (2, 0, 1))

    resized_masks = []
    for m in masks:
        resized = cv2.resize(m, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_masks.append(resized > 0.5)

    return np.array(resized_masks, dtype=bool), classes, scores, boxes

class YolactONNX:
    def __init__(self, model_path):
        #self.session = onnxruntime.InferenceSession(model_path)
        self.session=onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def infer(self, image):
        input_tensor = preprocess(image)
        return self.session.run(self.output_names, {self.input_name: input_tensor})

if __name__ == "__main__":
    model_path = "yolact_base_54_800000.onnx"
    yolact = YolactONNX(model_path)

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Could not open webcam.")
        exit()

    frame_count = 0
    total_inference_time = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        orig_h, orig_w = frame.shape[:2]
        outputs = yolact.infer(frame)
        masks, classes, scores, boxes = postprocess(outputs, (orig_h, orig_w))

        inference_time = time.time() - frame_start
        total_inference_time += inference_time
        frame_count += 1
        
        for i, mask in enumerate(masks):
            color = COLORS[int(classes[i]) % len(COLORS)]
            frame[mask] = frame[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5

            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = map(int, [x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h])
            label = f"{class_names[int(classes[i])]} {scores[i]:.2f}"

            cv2.putText(frame, label, (x1 + 10, max(y1 - 5, 0) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display FPS on frame
        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #print(fps)
        cv2.imshow("YOLACT - Webcam", frame.astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    avg_inference_time = total_inference_time / frame_count
    avg_fps = frame_count / (end_time - start_time)

    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

