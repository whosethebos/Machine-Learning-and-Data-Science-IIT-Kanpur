from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
#python opencv_Tracking.py --camera_or_vedio c
#python opencv_Tracking.py --camera_or_vedio v
#python opencv_Tracking.py --video vedios/chaplin.mp4
#python opencv_Tracking.py --tracker tld
#Space to hold, mouse to draw box and space to track opbject
#q to exit vedio/camera


def main():
	boxcord = None

	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"`mosse`": cv2.TrackerMOSSE_create
	}


	# arguments to be passed
	ap = argparse.ArgumentParser()
	ap.add_argument("-camera_or_vedio", "--camera_or_vedio", type=str,default="v", help="Choose between camera or vedio")
	ap.add_argument("-video", "--video", type=str, default="vedios/chaplin.mp4", help="please give vedio")
	ap.add_argument("-tracker", "--tracker", type=str, default="kcf",	help="OpenCV object tracker type")
	args = vars(ap.parse_args())
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
	# initialize the bounding box coordinates of the object to be tracked


	if args["camera_or_vedio"] == "c":
		print("Starting webCam..")
		vediostream = VideoStream(src=0).start()
		time.sleep(1.0)

	else:
		vediostream = cv2.VideoCapture(args["video"])
	# initialize the FPS throughput estimator
	fps = None

	# loop over frames from the video stream
	while True:
		# grab the current frame
		frame = vediostream.read()
		frame = frame[1] if args["camera_or_vedio"] == "v" else frame
		if frame is None:
			break

		# resize the frame
		frame = imutils.resize(frame, width=1000)
		(Height, Width) = frame.shape[:2]

		# check to see if we are currently tracking an object
		if boxcord is not None:
			# grab the new bounding box coordinates of the object
			(success, box) = tracker.update(frame)
			# check to see if the tracking was a success
			if success:
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
				cv2.putText(frame,"Object",(x, y),cv2.FONT_HERSHEY_SIMPLEX ,1,(0, 255, 0),2)

			# update the FPS counter
			fps.update()
			fps.stop()

			# initialize the set of information we'll be displaying on the frame
			info = [
				("Camera or Vedio", "WebCam" if args["camera_or_vedio"] =="c" else "Vedio Recording"),
				("Tracker", args["tracker"]),
				("Success", "Yes" if success else "No"),
				("FPS", "{:.2f}".format(fps.fps())),
			]

			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, Height - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the 's' key is selected, we are going to "select" a bounding
		# box to track
		if key == ord(" ") or key == ord("s"):
			# select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
			boxcord = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)

			# start OpenCV object tracker using the supplied bounding box
			# coordinates, then start the FPS throughput estimator as well
			tracker.init(frame, boxcord)
			fps = FPS().start()

		# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			break

	# if we are using a webcam, release the pointer
	if args["camera_or_vedio"] =="c" :
		vediostream.stop()

	# otherwise, release the file pointer
	else:
		vediostream.release()

	# close all windows
	cv2.destroyAllWindows()

main()