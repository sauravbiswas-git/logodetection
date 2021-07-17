from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)
trainingPath = head + "/" + "train"
testPath = head + "/" + "testdata"


hists = [] 
labels = [] 

for imagePath in glob.glob(trainingPath + "/*/*.*"):
    
    label = imagePath.split("/")[-2]
    
    image = cv.imread(imagePath)
    try:
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Calculate Low and Up value to extract Edges
        md = np.median(gray)
        sigma = 0.35
        low = int(max(0, (1.0 - sigma) * md))
        up = int(min(255, (1.0 + sigma) * md))
        # Create Edged Image from Gray Scale
        edged = cv.Canny(gray, low, up)

        # extract only shape in image
        (x, y, w, h) = cv.boundingRect(edged) 
        logo = gray[y:y + h, x:x + w]
        logo = cv.resize(logo, (200, 100))

        # Calculate histogram
        hist = feature.hog(
                logo, 
                orientations=9, 
                pixels_per_cell=(10, 10),cells_per_block=(2, 2),
                transform_sqrt=True,
                block_norm="L1"
            )
        # Add value into Lists
        hists.append(hist)
        labels.append(label)
    except cv.error:
        # If Image couldn't be Read
        print(imagePath)
        print("Training Image couldn't be read")


model = KNeighborsClassifier(n_neighbors=1)
model.fit(hists, labels)


for (imagePath) in glob.glob(testPath + "/*.*"):
    # Read Images
    image = cv.imread(imagePath)
    try:
        # Convert to Gray and Resize
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        logo = cv.resize(gray, (200, 100))

        # Histogram
        hist =  feature.hog(
                    logo, 
                    orientations=9,
                    pixels_per_cell=(10, 10),
                    cells_per_block=(2, 2), 
                    transform_sqrt=True, 
                    block_norm="L1"
                )
        # Predict 
        predict = model.predict(hist.reshape(1, -1))[0]
        height, width = image.shape[:2]
        reWidth = int((300/height)*width)
        image = cv.resize(image, (reWidth, 300))
        cv.putText(image, predict.title(), (10, 30), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0 ,255, 0), 4)

        # Get Image name and show Image
        imageName = imagePath.split("/")[-1]
        cv.imshow(imageName, image)
        cv.waitKey(0)
        
        cv.destroyAllWindows()
    except cv.error:
        print("Test Image couldn't be read")

