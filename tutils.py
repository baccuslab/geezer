from engine.models import ellipsoid
from engine.engine import Engine
from engine.processor import Shape
from skimage import measure
import numpy as np
from globals import pupil_params
import cv2
from multiprocessing import Queue

## initialize the pupil contour parameters##
diagonal_size = 2**10

#boolean identity matrix
main_diagonal = np.eye(diagonal_size, diagonal_size, dtype=bool)

#setting up boolean matrices all False
half_diagonal = np.full((diagonal_size, diagonal_size), False, dtype=bool)
third_diagonal = half_diagonal.copy()
fourth_diagonal = half_diagonal.copy()
onethird = 1/3
onefourth = 1/4

invhalf_diagonal = half_diagonal.copy()
invthird_diagonal = third_diagonal.copy()
invfourth_diagonal = fourth_diagonal.copy()

#creating conditional boolean matrices
for i, _ in enumerate(half_diagonal):
    half_diagonal[int(i/2), i] = True
    third_diagonal[int(i/3), i] = True
    fourth_diagonal[int(i/4), i] = True

    invfourth_diagonal[i, int(i/2)] = True
    invthird_diagonal[i, int(i/3)] = True
    invfourth_diagonal[i, int(i/4)] = True

#vectors for radius storage
rr_stock = np.zeros((32), dtype=np.float64)
rr_2d = np.zeros((32, 2), dtype=np.float64)

rx_multiply = np.ones((32), dtype=np.float64)
ry_multiply = rx_multiply.copy()

crop_stock = np.zeros((32), dtype=int)

#constants that can be optimized based on the video input
onehalf_ry_add = [8,10,12,14]
onehalf_rx_add = [8,11,12,15]
onehalf_rx_subtract = [9,10,13,14]
onehalf_ry_subtract = [9,11,13,15]
onehalf_ry_multiplier = [8,9,10,11]
onehalf_rx_multiplier = [12,13,14,15]

onefourth_ry_add = [16,19,20,21]
onefourth_rx_add = [16,17,20,23]
onefourth_rx_subtract = [18,19,21,22]
onefourth_ry_subtract = [17,18,22,23]
onefourth_ry_multiplier = [16,17,18,19]
onefourth_rx_multiplier = [20,21,22,23]

onethird_ry_add = [24,25,28,29]
onethird_rx_add = [24,27,28,31]
onethird_rx_subtract = [25,26,29,30]
onethird_ry_subtract = [26,27,30,31]
onethird_ry_multiplier = [24,25,26,27]
onethird_rx_multiplier = [28,29,30,31]

#for multiplying operations
rx_multiplied = np.array(np.concatenate((onehalf_rx_multiplier, onefourth_rx_multiplier, onethird_rx_multiplier)), dtype=int)
ry_multiplied = np.array(np.concatenate((onehalf_ry_multiplier, onefourth_ry_multiplier, onethird_ry_multiplier)), dtype=int)
ones_ = np.ones(4, dtype=np.float64)
rx_multiply = np.array(np.concatenate((ones_ * .5, ones_ * onefourth, ones_*onethird)))
ry_multiply = np.array(np.concatenate((ones_ * .5, ones_ * onefourth, ones_ * onethird)))

#for add operations
ry_add = np.array(np.concatenate(([0,2,4], onehalf_ry_add, onefourth_ry_add, onethird_ry_add)), dtype=int)
rx_add = np.array(np.concatenate(([1,2,5], onehalf_rx_add, onefourth_rx_add, onethird_ry_add)), dtype=int)

#for subtract oprtations
ry_subtract = np.array(np.concatenate(([3,5,7], onehalf_ry_subtract, onefourth_ry_subtract,onethird_ry_subtract)))
rx_subtract = np.array(np.concatenate(([3,4,6], onehalf_rx_subtract, onefourth_ry_subtract,onethird_rx_subtract)))

#pupil max and min radii
min_radius = 2
max_radius = 100


class fit_ellipse:#gotta change the dependecy on image_proc_tab_instace and instead just pass the variables themselves to the tutils.py
    def __init__(self, pf, pt, ff, ft, model): # , pupil_co #, image_proc_tab_instance
        self.img = pt
        self.pf, self.ff ,self.ft = pf, ff, ft
        self.model = model
        self.r = np.zeros((32,2),dtype=np.float64)
        self.frame_count = 0
        self.threshold = len(crop_stock) * min_radius * 1.05
        self.cond = self.cond_
        self.save_mp4 = False
        self.min_radius = min_radius
        self.max_radius = max_radius
        # self.pupil_co = pupil_co

    # def iterator(self):

    #     frames = self.img
    #     # i think since the pupil locator will correct the original prediction
    #     #then it may be okay to put in the same original pup_co

    #     for frame in frames:
    #         if self.frame_count == 0:
    #             center = self.pupil_co
    #             center = np.round(center).astype(int)
    #         yield self.pupil_locator(frame, center)
    #         self.frame_count += 1

    def cond_(self, r, crop_list):

        ### return the r within the normalized difference ###
        
        dists = np.linalg.norm(np.mean(r, axis = 0, dtype=np.float64) - r, axis = 1)

        mean_ = np.mean(dists)
        std_ = np.std(dists)
        lower, upper = mean_ - std_, mean_ + std_ *.8
        cond_ = np.logical_and(np.greater_equal(dists, lower),np.less(dists, upper))

        return r[cond_]

    def pupil_locator(self, frame, center):

        self.center = center

        canvas = np.array(frame,dtype=int)
        canvas[-1,:] = canvas[:, -1] = canvas[0,:] = canvas[:, 0] = 0

        r = self.r
        crop_list = np.zeros((32), dtype=int)

        # this crops the square area around the pupil
        # flips the upper left part of the canvas and zeros the edges
        canvas_ =  canvas[center[1]:, center[0]:]
        canv_shape0, canv_shape1 = canvas_.shape
        crop_canvas = np.flip(canvas[:center[1], :center[0]])
        crop_canv_shape0, crop_canv_shape1 = crop_canvas.shape

        #flips the upper left part of the canvas left to right
        #keeping the lower right part unchanged
        crop_canvas2 = np.fliplr(canvas[center[1]:, :center[0]])
        crop_canv2_shape0, crop_canv2_shape1 = crop_canvas2.shape

        #flips the upper left part of te canvas veritically
        #keeping the lower right part unchanged
        crop_canvas3 = np.flipud(canvas[:center[1], center[0]:])
        crop_canv3_shape0, crop_canv3_shape1 = crop_canvas3.shape

        canvas2 = np.flip(canvas)

        crop_list=np.array([
        np.argmax(canvas_[:, 0][self.min_radius:self.max_radius] == 0), np.argmax(canvas_[0, :][self.min_radius:self.max_radius] == 0), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(canvas2[-center[1], -center[0]:][self.min_radius:self.max_radius] == 0), np.argmax(canvas2[-center[1]:, -center[0]][self.min_radius:self.max_radius] == 0),
        np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
        np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0)
        ], dtype=int) + self.min_radius



        if np.sum(crop_list) < self.threshold:
            #origin inside corneal reflection?
            offset_list = np.array([
            np.argmax(canvas_[:, 0][1:] == 255), np.argmax(canvas_[0, :][1:] == 255), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
            np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
            np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas2[-center[1], -center[0]:][1:] == 255), np.argmax(canvas2[-center[1]:, -center[0]][1:] == 255),
            np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
            np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
            np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
            np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
            np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
            np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
            np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
            np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
            np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255)
            ], dtype=int) + 1


            crop_list=np.array([
            np.argmax(canvas_[:, 0][offset_list[0]:] == 0), np.argmax(canvas_[0, :][offset_list[1]:] == 0), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][offset_list[2]:] == 0),
            np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[3]:] == 0), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[4]:] == 0),
            np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[5]:] == 0), np.argmax(canvas2[-center[1], -center[0]:][offset_list[6]:] == 0), np.argmax(canvas2[-center[1]:, -center[0]][offset_list[7]:] == 0),
            np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][offset_list[8]:] == 0), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[9]:] == 0), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[10]:] == 0),
            np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[11]:] == 0), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][offset_list[12]:] == 0),
            np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[13]:] == 0), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[14]:] == 0),
            np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[15]:] == 0), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[16]:] == 0), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[17]:] == 0),
            np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[18]:] == 0), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[19]:] == 0), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[20]:] == 0),
            np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[21]:] == 0), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[22]:] == 0), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[23]:] == 0),
            np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][offset_list[24]:] == 0), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[25]:] == 0), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[26]:] == 0),
            np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[27]:] == 0), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][offset_list[28]:] == 0), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[29]:] == 0),
            np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[30]:] == 0), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[31]:] == 0)
            ], dtype=int) + offset_list


            if np.sum(crop_list) < self.threshold:
                raise IndexError("Lost track, do reset")

        r[:8,:] = center
        r[ry_add, 1] += crop_list[ry_add]
        r[rx_add, 0] += crop_list[rx_add]
        r[ry_subtract, 1] -= crop_list[ry_subtract] #
        r[rx_subtract, 0] -= crop_list[rx_subtract]
        r[rx_multiplied, 0] *= rx_multiply
        r[ry_multiplied, 1] *= ry_multiply
        r[8:,:] += center

        return self.cond(r, crop_list)
    
    def fit(self):

        r = self.iterator #returns 

        self.center = self.model.fit(r)

        params =  self.model.params

        pupil_params["pupil"] = params

    def draw_ellipse(self):

        ellipse_params = pupil_params["pupil"]

        center, width, height, phi = ellipse_params

        canvas = np.array(self.pf)

        return self.draw_ellipse_on_canvas(canvas, center, width, height, phi)

    def draw_ellipse_on_canvas(self, canvas, center, width, height, phi):
        # Create a copy of the canvas as an OpenCV image
        canvas_copy = canvas.copy().astype(np.uint8)

        # Convert width and height to integers
        width = int(width)
        height = int(height)

        # Convert the canvas to a BGR image if it's grayscale
        if len(canvas_copy.shape) == 2:
            canvas_copy = cv2.cvtColor(canvas_copy, cv2.COLOR_GRAY2BGR)

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, phi, 1.0)

        # Draw the ellipse on the OpenCV image
        cv2.ellipse(canvas_copy, center, (width, height), phi, 0, 360, (0, 0, 255), -1)

        # Convert the OpenCV image back to a NumPy array
        canvas_with_ellipse = np.array(canvas_copy)

        return canvas_with_ellipse

    def save_mp4(frames, output_path, frame_rate=30):
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
        height, width, _ = frames[0].shape  # Assuming all frames have the same shape
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # Write each frame to the video
        for frame in frames:
            out.write(frame)

        # Release the VideoWriter
        out.release()

