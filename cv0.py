import cv2
import dlib
import numpy as np

# For face_positions func
detector = dlib.get_frontal_face_detector()
# For getMainPoints func
predictor = dlib.shape_predictor('res/shape_predictor_68_face_landmarks.dat')


# Get the largest face, avoid misdirect to other faces
def face_positions(img):
    dets = detector(img, 0)

    # If there is no face detected
    if not dets:
        return None

    # Iterate and compute the area of the faces, and get the largest face
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


# Get the main points deciding the status of your face
def getConstructionPoints(keyPoints):
    # This func: get the average position of important facial features
    def center(indexArray):
        return sum([keyPoints[i] for i in indexArray]) / len(indexArray)

    # The important positions, list them in arrays
    eyebrowL = [18, 19, 20, 21]
    eyebrowR = [22, 23, 24, 25]
    chin = [6, 7, 8, 9, 10]
    nose = [29, 30]
    return center(eyebrowL + eyebrowR), center(chin), center(nose)


# Reproduce the facial features' status
def generateFeatures(constructionPoints):
    eyebrowM, chinM, noseM = constructionPoints

    # Consider the 3 points as a triangle
    median = eyebrowM - chinM
    hypotenuse = eyebrowM - noseM

    horizontalAngle = np.cross(median, hypotenuse) / np.linalg.norm(median) ** 2
    verticalAngle = median @ hypotenuse / np.linalg.norm(median) ** 2
    return np.array([horizontalAngle, verticalAngle])


#Draw the vtuber
def Draw(horizontalAngle, verticalAngle):
    img = np.ones([512, 512], dtype=np.float32)
    height = 200
    center = 256, 256
    eyeL = int(220 + horizontalAngle * height), int(249 + verticalAngle * height)
    eyeR = int(292 + horizontalAngle * height), int(249 + verticalAngle * height)
    mouth = int(256 + horizontalAngle * height / 2), int(310 + verticalAngle * height / 2)
    cv2.circle(img, center, 100, 0, 1)
    cv2.circle(img, eyeL, 15, 0, 1)
    cv2.circle(img, eyeR, 15, 0, 1)
    cv2.circle(img, mouth, 5, 0, 1)
    return img


# Get features from a picture
def getPicFeature(img):
    facePos = face_positions(img)

    # If face is not recognized
    if not facePos:
        cv2.imshow('self', img)
        cv2.waitKey(1)
        return None

    # Start analyze the face
    KeyPoints = getMainPoints(img, facePos)
    consPoints = getConstructionPoints(KeyPoints)
    features = generateFeatures(consPoints)

    # Show the camera
    cv2.imshow('self', img)

    return features


# Main executable func
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Read the sample picture features
    sampleF = getPicFeature(cv2.imread('res/sample.png'))
    # Initial: no feature seen (no angle given)
    features = sampleF - sampleF

    while True:
        # Use camera to get the picture
        ret, frame = cap.read()

        # Get the new picture features
        newF = getPicFeature(frame)

        # Get the difference
        # if newFeature is not None:
        #     thisFeature = newFeature - sampleFeature

        # #This is for show the numbers that representing your features on your face
        # mainPoints = getMainPoints(frame, newFeature)
        # for i, (px, py) in enumerate(mainPoints):
        #     cv2.putText(frame, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))

        # If current face is detected
        if newF is not None:
            features = newF - sampleF

        # If current face is not detected, use the previous data, and vtuber holds
        horizontalAngle, verticalAngle = features

        # Draw and Show vtuber
        cv2.imshow("Vtuber", Draw(horizontalAngle, verticalAngle))

        # This is important
        cv2.waitKey(1)
