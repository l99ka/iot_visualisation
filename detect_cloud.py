import cv2
import numpy as np

cap = cv2.VideoCapture("videos/video.mov")
previous_frame = None
onceIndex = 0
detectIndex = 0
locker = 0

while True:
    success, img_rgb = cap.read()

    if onceIndex == 0:
        onceIndex += 1
        img_rgbSun = cv2.GaussianBlur(src=img_rgb, ksize=(7, 7), sigmaX=0)
        img_hsvSun = cv2.cvtColor(img_rgbSun, cv2.COLOR_BGR2HSV)

        highLevelSun = np.array([255, 255, 255])
        lowerLevelSun = np.array([15, 15, 15])
        sunMask = cv2.inRange(img_hsvSun, lowerLevelSun, highLevelSun)

        circles = cv2.HoughCircles(sunMask, cv2.HOUGH_GRADIENT, 1, 2000, param1=30, param2=24, minRadius=10,maxRadius=900)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print("Sun locate:  " + str(circles))

    # 1. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=9)

    # Detect zone
    prepared_frameMask = np.zeros(prepared_frame.shape[:2], dtype='uint8')
    circle = cv2.circle(prepared_frameMask.copy(), (circles[0][0], circles[0][1]), circles[0][2] + 100, 255, -1)
    prepared_frame = cv2.bitwise_and(prepared_frame, prepared_frame, mask=circle)

    # 2. Calculate the difference
    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = prepared_frame
        continue

    # 3. Set previous frame and continue if there is None
    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = prepared_frame
        continue

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=3, maxval=255, type=cv2.THRESH_BINARY)[1]

    # 6. Find and optionally draw contours
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Comment below to stop drawing contours
    cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    if circles is not None:
        cv2.putText(img_rgb, 'Sun', (circles[0, 0] - 65, circles[0, 1] - 170), cv2.FONT_HERSHEY_DUPLEX, 2, (36, 255, 12), thickness=3)  # текст
        for (x, y, r) in circles:
            cv2.circle(img_rgb, (x, y), r+20, (36, 255, 12), 5)

    if len(contours) > 10 and detectIndex < 100:
        # print("contours:  " + str(len(contours)))
        detectIndex += 10
        # print("+detectIndex:  " + str(detectIndex))
    elif detectIndex > 0:
        detectIndex -= 1
        # print("-detectIndex:  " + str(detectIndex))

    if detectIndex > 60 and locker == 0:
        print("Команда:  ВКЛЮЧИТЬ   (detectIndex:" + str(detectIndex) + ")")
        locker = 1
    elif detectIndex < 50 and locker == 1:
        print("Команда:  ВЫКЛЮЧИТЬ   (detectIndex:" + str(detectIndex) + ")")
        locker = 0


    cv2.imshow('Motion detector', img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()