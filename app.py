import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk

###############################################3
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
import scipy.spatial as ss
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
##################################################

##################################################
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = ss.distance.euclidean(eye[1], eye[5])
	B = ss.distance.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = ss.distance.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# print(C)
	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

def mainfunc(): 
	builder = Gtk.Builder()
	builder.add_from_file("blinker.glade")
	handlers = { "onQuit" : onQuit,
				 "onSwitch" : onSwitch
	}
	builder.connect_signals(handlers)

	window = builder.get_object("windows")
	window.show_all()

	blinks = builder.get_object("leftlabel")

	switch = builder.get_object("switch")

	Gtk.main()

	

##################################################

def onQuit():
	Gtk.main_quit()

def onSwitch(switch,data):
	if(data==True):
		print("on")

		# define two constants, one for the eye aspect ratio to indicate
		# blink and then a second constant for the number of consecutive
		# frames the eye must be below the threshold
		EYE_AR_THRESH = 0.3
		EYE_AR_CONSEC_FRAMES = 3

		# initialize the frame counters and the total number of blinks
		COUNTER = 0
		TOTAL = 0

		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		print("[INFO] loading facial landmark predictor...")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(args["shape_predictor"])

		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# start the video stream thread
		print("[INFO] starting video stream thread...")
		#vs = FileVideoStream(args["video"]).start()
		#fileStream = True
		vs = VideoStream(src=0).start()
		# vs = VideoStream(usePiCamera=True).start()
		fileStream = False
		time.sleep(1.0)

		i = 0
		min_ear = 100
		max_ear = 0
		ear = 0
		# loop over frames from the video stream
		while (True):
			# if this is a file video stream, then we need to check if
			# there any more frames left in the buffer to process
			if fileStream and not vs.more():
				break

			# grab the frame from the threaded video file stream, resize
			# it, and convert it to grayscale
			# channels)
			frame = vs.read()
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# detect faces in the grayscale frame
			rects = detector(gray, 0)

			# loop over the face detections
			for rect in rects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# print(shape)
				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)

				# average the eye aspect ratio together for both eyes
				ear = (leftEAR + rightEAR) / 2.0

				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				# check to see if the eye aspect ratio is below the blink
				# threshold, and if so, increment the blink frame counter
				if ear < EYE_AR_THRESH:
					COUNTER += 1

				# otherwise, the eye aspect ratio is not below the blink
				# threshold
				else:
					# if the eyes were closed for a sufficient number of
					# then increment the total number of blinks
					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						TOTAL += 1
					# reset the eye frame counter
					COUNTER = 0

				#print(COUNTER)
				# draw the total number of blinks on the frame along with
				# the computed eye aspect ratio for the frame
				cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# show the frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			
			if i<50:
				if ear<min_ear:
					min_ear = ear		
				elif ear>max_ear:
					max_ear = ear
			elif i == 50 or key == ord("r"):
				EYE_AR_THRESH = (min_ear+max_ear)/2
			# print(ear)

			i+=1
		
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()
		switch.set_active(False)
	else:
		print("off")

mainfunc()