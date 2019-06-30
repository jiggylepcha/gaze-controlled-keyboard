import cv2
import numpy as np
import dlib
import time
from math import hypot
import string
import pyglet

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left_sound = pyglet.media.load("left.wav", streaming=False)
right_sound = pyglet.media.load("right.wav", streaming=False)
up_sound = pyglet.media.load("up.wav", streaming=False)
blink_sound = pyglet.media.load("sound.wav", streaming=False)

keyboard = np.zeros((300, 1000, 3))

ALL_KEYBOARD_CHARACTERS=[]
CURRENT_KEYBOARD_POS=0
UP_POSITIONS_DICT={0:19, 1:20, 2:21, 3:22, 4:23, 5:24, 6:25, 7:17, 8:8, 9:18, 10:0, 11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:9, 19:10, 20:11, 21:12, 22:13, 23:14, 24:15, 25:16}
LEFT_POSITIONS_DICT={0:8, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:17, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:25, 19:18, 20:19, 21:20, 22:21, 23:22, 24:23, 25:24}
RIGHT_POSITIONS_DICT={0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:0, 9:10, 10:11, 11:12, 12:13, 13:14, 14:15, 15:16, 16:17, 17:9, 18:19, 19:20, 20:21, 21:22, 22:23, 23:24, 24:25, 25:18}

def put_keys(letter, x, y, light):
    # print(light)
    height = width = 100
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 5
    font_th = 4
    text_size = cv2.getTextSize(letter, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if(light==True):
        # print('light')
        cv2.rectangle(keyboard, (x, y), (x + width, y + height), (255, 255, 255))
        cv2.putText(keyboard, letter, (text_x, text_y), font_letter, font_scale, (151, 151, 151), font_th)
    else:
        # print('no light')
        cv2.rectangle(keyboard, (x, y), (x + width, y + height), (151, 151, 151))
        cv2.putText(keyboard, letter, (text_x, text_y), font_letter, font_scale, (0, 255, 255), font_th)        

def change_keyboard(letter_index):
    alphabets=list()
    for c in string.ascii_uppercase:
        alphabets.append(c)
    x = x_val = 0
    y = y_val = 0
    count = 0
    for i in range(len(string.ascii_uppercase)):
        count += 1
        if (count == 10):
            x = 0
            y += 100
            x_val = x
            y_val += y
            count = 1
        else:
            x += 100
            x_val = x
            y = 0
        if(i==letter_index):
            put_keys(alphabets[i], x_val, y_val, True)
        else:
            # pass
            put_keys(alphabets[i], x_val, y_val, False)

def midpoint(p1 ,p2):
    return int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)

def to_numpy(landmarks):
    numpy_array = np.zeros((68, 2))
    for i in range(68):
        numpy_array[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return numpy_array.astype(int)

def rectangle_to_bounding_box(rectangle):
    return rectangle.left(), rectangle.top(), rectangle.right(), rectangle.bottom()

def get_blink_ratio(coords, landmarks):
    horizontal_distance = np.sum(np.abs((landmarks[coords[0]] - landmarks[coords[3]])) ** 3)
    vertical_distance_1 = np.sum(np.abs((landmarks[coords[1]] - landmarks[coords[5]])) ** 3)
    vertical_distance_2 = np.sum(np.abs((landmarks[coords[2]] - landmarks[coords[4]])) ** 3)
    return int(horizontal_distance / vertical_distance_1), int(horizontal_distance/vertical_distance_2)

def get_circle(image):
    return cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 1)

def get_centroid(image):
    args = np.argwhere(image == 0)
    if args.shape[0] >= 1:
        args = args.mean(axis = 0).astype(int)
        if args[0] < 0 or args[1] > 0:
            return args[0], args[1]
    return int(image.shape[0]/2), int(image.shape[1]/2)

def get_blinking_ratio(eye_points, facial_landmarks_numpy):
    left_point = facial_landmarks_numpy[eye_points[0]]
    right_point = facial_landmarks_numpy[eye_points[3]]
    center_top = midpoint(facial_landmarks_numpy[eye_points[1]], facial_landmarks_numpy[eye_points[2]])
    center_bottom = midpoint(facial_landmarks_numpy[eye_points[5]], facial_landmarks_numpy[eye_points[4]])

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_current_position(movement):
    global CURRENT_KEYBOARD_POS
    if(movement=='left'):
        CURRENT_KEYBOARD_POS=LEFT_POSITIONS_DICT[CURRENT_KEYBOARD_POS]
    elif(movement=='right'):
        CURRENT_KEYBOARD_POS=RIGHT_POSITIONS_DICT[CURRENT_KEYBOARD_POS]
    elif(movement=='up'):
        CURRENT_KEYBOARD_POS=UP_POSITIONS_DICT[CURRENT_KEYBOARD_POS]

    return CURRENT_KEYBOARD_POS

old_pupil_left_x = None
old_pupil_left_y = None
old_pupil_right_x = None
old_pupil_right_y = None
counter = 0
previous_nav = None
previous_time = time.time()
previous_blink = None
previous_blink_time = time.time()

change_keyboard(0)
text = ""

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = None
    right_eye = None
    left_eye_equalized = None
    right_eye_equalized = None
    left_eye_thresholded = None
    right_eye_thresholded = None

    text_image = np.zeros((300, 500, 3))

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        numpy_landmarks = to_numpy(landmarks)
        x1, y1, x2, y2 = rectangle_to_bounding_box(face)

        left_eye_center = np.mean(numpy_landmarks[42:48, :], axis = 0).astype(int)
        right_eye_center = np.mean(numpy_landmarks[36:42, :], axis = 0).astype(int)
        offset_x = (x2-x1)//12
        offset_y = (y2-y1)//20
        scale = 5
        left_eye = frame[left_eye_center[1] - offset_y: left_eye_center[1]+offset_y, left_eye_center[0] - offset_x: left_eye_center[0]+offset_x]
        right_eye = frame[right_eye_center[1] - offset_y: right_eye_center[1]+offset_y, right_eye_center[0] - offset_x: right_eye_center[0]+offset_x]
        left_eye = cv2.resize(left_eye, None, fx = scale, fy = scale)
        right_eye = cv2.resize(right_eye, None, fx = scale, fy = scale)
        left_eye_equalized = cv2.equalizeHist(cv2.GaussianBlur(cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY), (7, 7), 0))
        right_eye_equalized = cv2.equalizeHist(cv2.GaussianBlur(cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY), (7, 7), 0))

        left_eye_thresholded = cv2.threshold(left_eye_equalized, 60, 255, cv2.THRESH_BINARY_INV)[1]
        right_eye_thresholded = cv2.threshold(right_eye_equalized, 60, 255, cv2.THRESH_BINARY_INV)[1]
        
        kernel = np.ones((7, 7), np.uint8) 
        left_eye_thresholded = cv2.erode(left_eye_thresholded, kernel, iterations=3) 
        right_eye_thresholded = cv2.erode(right_eye_thresholded, kernel, iterations=3) 

        _, left_contours, _ = cv2.findContours(left_eye_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, right_contours, _ = cv2.findContours(right_eye_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(left_contours) <= 0 or len(right_contours) <= 0:
            continue
        left_contour = sorted(left_contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        left_contour = left_contour.reshape((-1, 2))
        right_contour = sorted(right_contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        right_contour = right_contour.reshape((-1, 2))

        left_pupil_x, left_pupil_y = left_contour.mean(axis = 0).astype(int)
        right_pupil_x, right_pupil_y = right_contour.mean(axis = 0).astype(int)

        ratio = 0.8
        if old_pupil_left_x is None:
            old_pupil_left_x = left_pupil_x
        else:
            left_pupil_x = int(ratio*left_pupil_x + (1-ratio)*old_pupil_left_x)
            old_pupil_left_x = left_pupil_x
        
        if old_pupil_left_y is None:
            old_pupil_left_y = left_pupil_y
        else:
            left_pupil_y = int(ratio*left_pupil_y + (1-ratio)*old_pupil_left_y)
            old_pupil_left_y = left_pupil_y
        
        if old_pupil_right_x is None:
            old_pupil_right_x = right_pupil_x
        else:
            right_pupil_x = int(ratio*right_pupil_x + (1-ratio)*old_pupil_right_x)
            old_pupil_right_x = right_pupil_x
        
        if old_pupil_right_y is None:
            old_pupil_right_y = right_pupil_y
        else:
            right_pupil_y = int(ratio*right_pupil_y + (1-ratio)*old_pupil_right_y)
            old_pupil_right_y = right_pupil_y

        cv2.circle(left_eye, (left_pupil_x, left_pupil_y), 1, (0, 255, 0), 5) 
        cv2.circle(right_eye, (right_pupil_x, right_pupil_y), 1, (0, 255, 0), 5)

        height, width, _ = left_eye.shape

        cv2.line(left_eye, (width//4, 0), (width//4, height), (0, 0, 255), 2)
        cv2.line(left_eye, ((width * 3)//4, 0), ((width * 3)//4, height), (0, 0, 255), 2)
        cv2.line(left_eye, (0, height//5), (width, height//5), (0, 0, 255), 2)
        cv2.line(left_eye, (0, (height * 3)//4), (width, (height * 3)//4), (0, 0, 255), 2)

        cv2.line(right_eye, (width//4, 0), (width//4, height), (0, 0, 255), 2)
        cv2.line(right_eye, ((width * 3)//4, 0), ((width * 3)//4, height), (0, 0, 255), 2)
        cv2.line(right_eye, (0, height//5), (width, height//5), (0, 0, 255), 2)
        cv2.line(right_eye, (0, (height * 3)//4), (width, (height * 3)//4), (0, 0, 255), 2)

        if left_pupil_x < width/4 and right_pupil_x < width/4:
            if previous_nav != "r":
                print("right")
                right_sound.play()
                time.sleep(1)
                pos=get_current_position("right")
                change_keyboard(pos)
                previous_nav = "r"
                previous_time = time.time()
        elif left_pupil_x > width*3/4 and right_pupil_x > width*3/4:
            if previous_nav != "l":
                print("left")
                left_sound.play()
                time.sleep(1)
                pos=get_current_position("left")
                change_keyboard(pos)
                previous_nav = "l"
                previous_time = time.time()
        elif left_pupil_y < height/5 and right_pupil_y < height/5:
            if previous_nav != "u":
                print("up")
                up_sound.play()
                time.sleep(1)
                pos=get_current_position("up")
                change_keyboard(pos)
                previous_nav = "u"
                previous_time = time.time()
        elif left_pupil_y > height*3/4 and right_pupil_y > height*3/4:
            if previous_nav != "d":
                print("down")
                pos=get_current_position("down")
                change_keyboard(pos)
                previous_nav = "d"
                previous_time = time.time()
        else:
            if time.time() - previous_time > 1:
                previous_nav = None
                previous_time = time.time()

            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], numpy_landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], numpy_landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            if blinking_ratio > 5.7:
                if previous_blink != "b":
                    alphabets=list()
                    for c in string.ascii_uppercase:
                        alphabets.append(c)
                    print("blink")
                    blink_sound.play()
                    text += alphabets[CURRENT_KEYBOARD_POS]
                    if len(text) % 7 == 0:
                        text += "\n"
                    print(text)
                    previous_blink = "b"
                    previous_blink_time = time.time()
                elif time.time() - previous_blink_time > 1:
                    previous_blink = None
                    previous_blink_time = time.time()
        # ==================================================================================================
        
        # left_pupil_y = left_eye_center[1] - offset_y + left_pupil_y // scale
        # left_pupil_x = left_eye_center[0] - offset_x + left_pupil_x // scale
        
        # right_pupil_y = right_eye_center[1] - offset_y + right_pupil_y // scale
        # right_pupil_x = right_eye_center[0] - offset_x + right_pupil_x // scale

        # cv2.circle(frame, (left_pupil_x, left_pupil_y), 1, (0, 255, 0), 2) 
        # cv2.circle(frame, (right_pupil_x, right_pupil_y), 1, (0, 255, 0), 2)
        # cv2.circle(frame, (int(left_eye_center[0]), int(left_eye_center[1])), 1, (0, 0, 255), 2)
        # cv2.circle(frame, (int(right_eye_center[0]), int(right_eye_center[1])), 1, (0, 0, 255), 2)

        # for x, y in numpy_landmarks:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)
        # cv2.rectangle(frame, (left_eye_center[0] - offset_x, left_eye_center[1] - offset_y), (left_eye_center[0] + offset_x, left_eye_center[1] + offset_y), (0, 255, 0), 2)
        # cv2.rectangle(frame, (right_eye_center[0] - offset_x, right_eye_center[1] - offset_y), (right_eye_center[0] + offset_x, right_eye_center[1] + offset_y), (0, 255, 0), 2)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # if left_eye_thresholded is not None:
    #     cv2.imshow("Left Eye thresholded", left_eye_thresholded)
    # if right_eye_thresholded is not None:
    #     cv2.imshow("Right Eye thresholded", right_eye_thresholded)

    # if left_eye_equalized is not None:
    #     cv2.imshow("Left Eye equalized", left_eye_equalized)
    # if right_eye_equalized is not None:
    #     cv2.imshow("Right Eye equalized", right_eye_equalized)

    cv2.imshow("keyboard", keyboard)
    # print(iter)

    if left_eye is not None:
        cv2.imshow("Left Eye", left_eye)
    if right_eye is not None:
        cv2.imshow("Right Eye", right_eye)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(text_image, text,(30, 100), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Board", text_image)

    # cv2.imshow("Frame", frame)

    key = cv2.waitKey(5)
    if key == 27:
        break
    # time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
