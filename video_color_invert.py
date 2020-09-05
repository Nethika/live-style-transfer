# USAGE
# python neural_style_transfer_video.py --models models

# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import imutils
import time
import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps  

def color_invert(image):

    #image.save('new_name4.png')
    #print("inverting.... image.mode =",image.mode)
    if image.mode == 'RGBA':
        r,g,b,a = image.split()
        rgb_image = Image.merge('RGB', (r,g,b))

        inverted_image = PIL.ImageOps.invert(rgb_image)

        #r2,g2,b2 = inverted_image.split()

        #final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

        #final_transparent_image.save('new_file.png')

    else:
        inverted_image = PIL.ImageOps.invert(image)
        #inverted_image.save('new_name3.png')
    
    return(inverted_image)



# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths = paths.list_files('models', validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each of the model paths, then combine the
# two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# use the cycle function of itertools that can loop over all model
# paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
#print(type(modelIter))
#print(modelIter)
(modelID, modelPath) = next(modelIter)

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()




time.sleep(2.0)
print("[INFO] {}. {}".format(modelID + 1, modelPath))

# loop over frames from the video file stream
nexttime = time.time() + 30

while True:

	# grab the frame from the threaded video stream
	frame = vs.read()

	#(h1, w1) = frame.shape[:2]

	#print("size:",h1,w1)
	

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=700)
	orig = frame.copy()
	(h, w) = frame.shape[:2]

	# construct a blob from the frame, set the input, and then perform a
	# forward pass of the network
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()
	
	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3]))

	output = output.transpose(1,2,0)
	
	"""
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output = output.transpose(1, 2, 0)
	pil_im = output

	"""
	
	#output /= 255.0
	

	#Invert Colors
	#output = cv2.bitwise_not(output)
	#print("Mode:",)
	#output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)  #CV_8UC1
	#output = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_HSV)
	# You may need to convert the color.

	
	"""
	img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	cv2.imwrite('test2.png',img)
	
	img = img.astype(np.uint8)
	"""
	im = Image.fromarray(output.astype(np.uint8))
	#print(pil_im.astype(np.uint8))
	im_pil = color_invert(im)
	# For reversing the operation:
	output2 = np.asarray(im_pil)
	

	# show the original frame along with the output neural style
	# transfer
	#cv2.imshow("Input", frame)
	cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.imshow("window", output2)
	#cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# grab the next nueral style transfer model model and load it
	if nexttime < time.time():
		#print(nexttime)
		nexttime += 30
		(modelID, modelPath) = next(modelIter)
		#print("[INFO] {}. {}".format(modelID + 1, modelPath))
		net = cv2.dnn.readNetFromTorch(modelPath)
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()