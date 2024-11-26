import numpy as np

i = 1 # person index
j = 1 # viewpoint index
gender = 'm' # gender
root_dir = './imm_face_db/'

# load all facial keypoints/landmarks
file = open(root_dir + '{:02d}-{:d}{}.asf'.format(i,j,gender))
points = file.readlines()[16:74]
landmark = []

for point in points:
    x,y = point.split('\t')[2:4]
    landmark.append([float(x), float(y)])

# the nose keypoint
nose_keypoint = np.array(landmark).astype('float32')[-6]
print(nose_keypoint, len(landmark))