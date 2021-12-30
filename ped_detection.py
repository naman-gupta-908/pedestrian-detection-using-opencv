import cv2

video_src = 'pedestrians.avi'

# Read the input video
cap = cv2.VideoCapture(video_src)

# Load cascade
peds_cascade = cv2.CascadeClassifier('pedestrian.xml')

while True:
    _, img = cap.read()
    
    if (type(img) == type(None)):
        break
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect peds
    peds = peds_cascade.detectMultiScale(gray,1.2,4)

    # Draw rectangle around the peds
    for(a,b,c,d) in peds:
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,210),4)
    
    # Display the output
    cv2.imshow('Pedestrian detection', img)
    
    # Stop if escape key or 'q' key is pressed
    if cv2.waitKey(30) == ord("q") or cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
