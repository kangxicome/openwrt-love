import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
import urllib.request

# MJPEG stream dimensions
STREAM_WIDTH = 640
STREAM_HEIGHT = 512

# YOLO model configuration
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CONF_THRESH = 0.3  # Lowered from 0.5
NMS_THRESH = 0.45  # Slightly increased from 0.4

# Load COCO class names
with open('coco.names', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Global variables for image dimensions
img_height = None
img_width = None

class TensorRTInference:
    def __init__(self, engine_path):
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input and output binding details directly from engine
        self.num_bindings = self.engine.num_bindings
        self.input_idx = None
        self.output_idx = []
        
        # Find input and output bindings
        for i in range(self.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
                input_shape = self.engine.get_tensor_shape(self.engine.get_binding_name(i))
                print("Input binding: {}, shape: {}, name: {}".format(i, input_shape, self.engine.get_binding_name(i)))
            else:
                self.output_idx.append(i)
                output_shape = self.engine.get_tensor_shape(self.engine.get_binding_name(i))
                print("Output binding: {}, shape: {}, name: {}".format(i, output_shape, self.engine.get_binding_name(i)))
        
        if self.input_idx is None:
            raise RuntimeError("No input binding found in TensorRT engine")
        if not self.output_idx:
            raise RuntimeError("No output bindings found in TensorRT engine")
        
        # Allocate memory for inputs and outputs
        self.dtype = np.float32
        self.host_input = cuda.pagelocked_empty(trt.volume(self.engine.get_tensor_shape(self.engine.get_binding_name(self.input_idx))), dtype=self.dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        
        # Initialize host and device outputs
        self.host_outputs = []
        self.device_outputs = []
        
        for idx in self.output_idx:
            output_shape = self.engine.get_tensor_shape(self.engine.get_binding_name(idx))
            host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=self.dtype)
            device_output = cuda.mem_alloc(host_output.nbytes)
            self.host_outputs.append(host_output)
            self.device_outputs.append(device_output)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
    
    def preprocess_image(self, img):
        # Ensure input is the expected size
        if img.shape[1] != STREAM_WIDTH or img.shape[0] != STREAM_HEIGHT:
            print(f"Warning: Input image size {img.shape[1]}x{img.shape[0]} differs from expected {STREAM_WIDTH}x{STREAM_HEIGHT}")
        
        # Resize to exactly what the model expects
        resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        
        # Normalize
        normalized = resized / 255.0
        
        # HWC to NCHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.ascontiguousarray(transposed[np.newaxis, ...])
        
        return batched
    
    def infer(self, img):
        # Preprocess image
        preprocessed = self.preprocess_image(img)
        
        # Copy input to device
        np.copyto(self.host_input, preprocessed.ravel())
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        
        # Prepare bindings
        bindings = [int(self.device_input)]
        for device_output in self.device_outputs:
            bindings.append(int(device_output))
        
        # Run inference
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy outputs back to host
        for i, output_idx in enumerate(self.output_idx):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.device_outputs[i], self.stream)
        
        self.stream.synchronize()
        
        # Process output - only use the first output for simplicity
        # (You may need to adjust this based on your specific model)
        return self.process_output(self.host_outputs[0])
    
    def process_output(self, output):
        # YOLOv4-tiny output processing - you may need to adjust this
        # based on your specific model configuration
        
        # Get output dimensions from engine
        output_name = self.engine.get_binding_name(self.output_idx[0])
        output_shape = self.engine.get_tensor_shape(output_name)
        
        # Reshape the output according to the expected format
        # This may vary depending on your specific YOLOv4-tiny model
        try:
            # For ONNX YOLOv4-tiny model, the output is typically:
            # [batch, boxes, 5+classes] where 5 is for x,y,w,h,confidence
            num_classes = len(CLASSES)
            output = output.reshape((output_shape[0], output_shape[1], num_classes + 5))
        except Exception as e:
            print("Error reshaping output: {}".format(e))
            print("Output shape from engine: {}".format(output_shape))
            print("Output size: {}".format(output.size))
            print("Attempting alternative reshape...")
            
            # Try alternative reshape based on output size
            try:
                # Estimate number of boxes
                num_boxes = output.size // (5 + num_classes)
                output = output.reshape((1, num_boxes, 5 + num_classes))
            except Exception as e:
                print("Alternative reshape failed: {}".format(e))
                return []
        
        # Parse detections
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each detection
        for b in range(output_shape[0]):  # batch size
            for n in range(output.shape[1]):  # number of detections
                detection = output[b, n]
                
                # Check if we have a valid detection
                confidence = detection[4]
                if confidence >= CONF_THRESH:
                    # Get class scores
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    
                    if class_confidence >= CONF_THRESH:
                        # Get bounding box coordinates
                        x, y, w, h = detection[0:4]
                        
                        # Convert normalized coordinates to pixel coordinates
                        x1 = max(0, int((x - w/2) * img_width))
                        y1 = max(0, int((y - h/2) * img_height))
                        x2 = min(img_width, int((x + w/2) * img_width))
                        y2 = min(img_height, int((y + h/2) * img_height))
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Only add valid boxes
                        if width > 0 and height > 0:
                            boxes.append([x1, y1, width, height])
                            confidences.append(float(confidence * class_confidence))
                            class_ids.append(class_id)
        
        # Apply non-maximum suppression
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
            
            results = []
            if len(indices) > 0:
                # Handle different OpenCV versions
                if isinstance(indices, list):
                    # OpenCV 4.5.4+
                    for i in indices:
                        if isinstance(i, (list, tuple)):
                            i = i[0]
                        box = boxes[i]
                        confidence = confidences[i]
                        class_id = class_ids[i]
                        results.append((box, confidence, class_id))
                else:
                    # Older OpenCV
                    for i in indices.flatten():
                        box = boxes[i]
                        confidence = confidences[i]
                        class_id = class_ids[i]
                        results.append((box, confidence, class_id))
            
            return results
        return []

# MJPEG stream handling
def get_mjpeg_stream(url):
    # Use OpenCV's built-in video capture for MJPEG streams
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise Exception("Cannot open MJPEG stream at {}".format(url))
    return cap

def main():
    # Initialize TensorRT inference
    try:
        print("Loading TensorRT engine...")
        tensorrt_inference = TensorRTInference("yolov4-tiny-fp16.trt")
        print("TensorRT engine loaded successfully")
    except Exception as e:
        print("Error loading TensorRT engine: {}".format(e))
        return
    
    # Initialize MJPEG stream
    try:
        print("Connecting to MJPEG stream...")
        cap = get_mjpeg_stream("http://localhost:8080")
        
        # Read first frame to get dimensions
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to get initial frame from stream")
            return
            
        global img_height, img_width
        img_height, img_width = first_frame.shape[:2]
        print(f"Stream dimensions: {img_width}x{img_height}")
        
        print("Connected to MJPEG stream")
    except Exception as e:
        print("Error connecting to MJPEG stream: {}".format(e))
        return
    
    # Create window
    cv2.namedWindow("YOLOv4-tiny Detection", cv2.WINDOW_NORMAL)
    
    # For motion detection
    prev_frame = first_frame.copy()
    motion_mask = None
    
    print("Starting detection loop...")
    while True:
        try:
            # Read frame from MJPEG stream
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from stream")
                break
            
            # Update global dimensions in case stream changes
            img_height, img_width = frame.shape[:2]
            
            # Simple motion detection
            # Calculate absolute difference between current and previous frame
            frame_diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                  cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
            # Threshold the difference
            _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            # Apply morphological operations to remove noise
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)
            
            # Optional: Apply motion mask as an overlay
            motion_highlight = frame.copy()
            motion_highlight[:,:,2] = cv2.add(motion_highlight[:,:,2], motion_mask)  # Add to red channel
            # Blend with original frame
            frame = cv2.addWeighted(frame, 0.7, motion_highlight, 0.3, 0)
            
            # Store current frame for next iteration
            prev_frame = frame.copy()
            
            # Start timer
            start_time = time.time()
            
            # Run inference
            detections = tensorrt_inference.infer(frame)
            
            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            
            # Count detections by class
            person_count = 0
            total_count = len(detections)
            
            # Draw detections on frame
            for box, confidence, class_id in detections:
                x, y, w, h = box
                
                # Special handling for person class (usually class_id 0 in COCO)
                if class_id == 0:  # Person class
                    person_count += 1
                    color = (0, 0, 255)  # Red for people
                else:
                    color = COLORS[int(class_id) % len(COLORS)]
                
                # Draw thicker bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Add a filled background for better text visibility
                label = "{}: {:.2f}".format(CLASSES[int(class_id)], confidence)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - 25), (x + text_size[0], y), color, -1)  # Filled rectangle
                
                # Draw label with white text
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw a dot at the center of the object
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Display status information
            #cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.putText(frame, "Objects detected: {}".format(total_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.putText(frame, "People detected: {}".format(person_count), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add dimensions indicator
            #dim_text = f"Stream: {img_width}x{img_height}"
            #cv2.putText(frame, dim_text, (10, img_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("YOLOv4-tiny Detection", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print("Error in main loop: {}".format(e))
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Detection loop ended")

if __name__ == "__main__":
    main()