import cv2
import numpy as np
import string

keyboard = np.zeros((300, 1000, 3))

def put_keys(letter, x, y):
    height = width = 100
    cv2.rectangle(keyboard, (x, y), (x + width, y + height), (255, 0, 0))
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(letter, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, letter, (text_x, text_y),
                font_letter, font_scale, (255, 0, 0), font_th)



def main():
    alphabets = list()
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
        put_keys(alphabets[i], x_val, y_val)
    cv2.imshow("keyboard", keyboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
