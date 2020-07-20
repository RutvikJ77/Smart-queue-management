

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    """
    Class for dealing with queues.
    
    Performs basic operations for queues like adding to a queue, getting the queues 
    and checking the coordinates for queues.
    
    Attributes:
        queues: A list containing the queues data
    """
    
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        """
        Add points to the queue.

        Args:
            points: A list of points to be added.

        Raises:
            TypeError: points is None.
        """
        
        self.queues.append(points)

    def get_queues(self, image):
        """
        Get queues from images.

        Args:
            image: A list of the image.

        Yields:
            A list containing each frame.
        """
            
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords, init_w, init_h):
        """
        Check coordinates for queues.

        Args:
            coords: A list of the coordinates.
        """
        
        d={k+1:0 for k in range(len(self.queues))}
        
        dum = ['0', '1' , '2', '3']
        
        for coord in coords:
            xmin = int(coord[3] * init_w)
            ymin = int(coord[4] * init_h)
            xmax = int(coord[5] * init_w)
            ymax = int(coord[6] * init_h)
            
            dum[0] = xmin
            dum[1] = ymin
            dum[2] = xmax
            dum[3] = ymax
            
            for i, q in enumerate(self.queues):
                if dum[0]>q[0] and dum[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    """
    Class for the Person Detection Model.
    
    Performs person detection and preprocessing.
    
    Attributes:
        model_weights: A string containing model weights path.
        model_structure: A string conatining model structure path.
        device: A string conatining device name.
        threshold: A floating point number containing threshold value.
        input_name: A list of input names.
        input_shape: A tuple of the input shape.
        output_name: A list of output names.
        output_shape: A tuple of the output shape.
        core: IECore object.
        net: Loaded net object.
    """

    def __init__(self, model_name, device, threshold=0.60):
        """
        Inits PersonDetect class with model_name, device, threshold.
        """
        
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model = self.core.read_network(model = self.model_structure,weights = self.model_weights)
            
            #self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        print('Model creation...')
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        """
        Loads the model.
        """
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print('Network Loading...')
        
    def predict(self, image):
        """
        Make asynchronous predictions from images.

        Args:
            image: List of the image data.

        Returns:
            The outputs and the image.
        """
        
        input_name = self.input_name
        input_img = self.preprocess_input(image)      
        input_dict={input_name: input_img}  
        
        # Start asynchronous inference for specified request.

        infer_request = self.network.start_async(request_id=0, inputs=input_dict)
        infer_stat = infer_request.wait()
        if infer_stat == 0:
            outputs = infer_request.outputs[self.output_name]
            
        return outputs, image
    
    def draw_outputs(self, coords, frame, init_w, init_h):
        """
        Draws outputs or predictions on image.

        Args:
            coords: The coordinates of predictions.
            image: The image on which boxes need to be drawn.

        Returns:
            the frame
            the count of people
            bounding boxes above threshold
        """
        
        cur_count = 0
        det = []
        
        for ob in coords[0][0]:
            
            # Draw bounding box for the detected object when it's probability 
            # is more than the specified threshold
            if ob[2] > self.threshold:
                start = (int(ob[3] * init_w),int(ob[4] * init_h))
                end = (int(ob[5] * init_w),int(ob[6] * init_h))
                cv2.rectangle(frame, start, end, (0, 255, 0), 1)
                cur_count = cur_count + 1
                
                det.append(ob)
                
        return frame, cur_count, det

    def preprocess_outputs(self, outputs):
        """
        Preprocess the outputs.

        Args:
            outputs: The output from predictions.

        Returns:
            Preprocessed dictionary.
        """
        #out_dict = {self.output_name:out for out in outputs}
        out_dict = {}
        for out in outputs:
            out_name = self.output_name
            out_img = out
            out_dict[output_name] = out_img
        
        return out_dict
    
        return out
        

    def preprocess_input(self, image):
      
        in_img = image
        
        # Preprocessing input
        n, c, h, w = self.input_shape
        
        in_img=cv2.resize(in_img, (w, h), interpolation = cv2.INTER_AREA)
        
        # Change image from HWC to CHW
        in_img = in_img.transpose((2, 0, 1))
        in_img = in_img.reshape((n, c, h, w))

        return in_img


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            coords, image= pd.predict(frame)
            frame, current_count, coords = pd.draw_outputs(coords, image, initial_w, initial_h)
        
            num_people = queue.check_coords(coords, initial_w, initial_h)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                print(k, v)
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40

            out_video.write(image)
            
        total_time=time.time()-start_inference_time    
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)