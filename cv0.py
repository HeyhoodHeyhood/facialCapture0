import cv2
import dlib
import numpy as np

# For face_positions func
detector = dlib.get_frontal_face_detector()
# For getMainPoints func
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Get the largest face, avoid misdirect to other faces
def face_positions(img):
    dets = detector(img, 0)

    # If there is no face detected
    if not dets:
        return None

    # Iterate and compute the areas, and get the largest one
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))


# Get the main points on the facial features
def getMainPoints(img, facePos):
    # No face detected, no points given
    if facePos is None:
        return []

    # Return a model of facial features
    landmark_shape = predictor(img, facePos)

    # Trying to interpret them into np arrays
    mainPoints = []

    # Iterate the 68 main points from the dlib
    for i in range(68):
        pos = landmark_shape.part(i)
        mainPoints.append(np.array([pos.x, pos.y], dtype=np.float32))
    return mainPoints


# Main executable func
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        myFace = face_positions(frame)
        mainPoints = getMainPoints(frame, myFace)

        for i, (px, py) in enumerate(mainPoints):
            cv2.putText(frame, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))

        cv2.imshow("Camera", frame)

        # This is important
        cv2.waitKey(1)