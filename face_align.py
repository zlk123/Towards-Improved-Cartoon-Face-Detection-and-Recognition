import sys
import os
import glob
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import cv2

imagesPath = sys.argv[1]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

for imageFile in glob.glob(os.path.join(imagesPath, "*.jp*g")):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imageFile)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        x = max(x, 0)
        y = max(y, 0)
        w = max(w, 0)
        h = max(h, 0)

        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

        import uuid

        f = str(uuid.uuid4())
        tokensInPath = imageFile.split("/")
        finalFile = tokensInPath[len(tokensInPath) - 1]
        cv2.imwrite("cartoonFaces/alignedFaces/" + finalFile, faceAligned)
        # display the output images
        # cv2.imshow("faceAligned", faceAligned)
        # cv2.imshow("Original", faceOrig)
        # cv2.waitKey(0)