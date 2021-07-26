import sys

import cv2
import numpy as np


def crop_chessboard_from_image(image, corners):
    output_img_size = image.shape[0]

    pts1 = np.float32(corners)
    pts2 = np.float32(
        [[0, 0], [0, output_img_size], [output_img_size, output_img_size], [output_img_size, 0]]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, matrix, (output_img_size, output_img_size))

    return perspective


def detect_chessboard_frame(image, img_log_list):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mx = 0
    my = 0
    mw = 0
    mh = 0
    out_approx = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(img_log_list[0], cnt, -1, (255, 0, 0), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w > mw and h > mh:
                    mx = x
                    my = y
                    mw = w
                    mh = h
                    out_approx = approx

    x, y, w, h = cv2.boundingRect(out_approx)
    cv2.rectangle(img_log_list[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(img_log_list[0], out_approx, -1, (0, 0, 255), 25)

    if len(out_approx) == 4:
        # TODO sorting clockwise
        from_approx = [
            out_approx[0][0],
            out_approx[3][0],
            out_approx[2][0],
            out_approx[1][0]
        ]

        return from_approx
    else:
        # Frame version
        p1 = [mx, my]
        p2 = [mx, my + mh]
        p3 = [mx + mw, my + mh]
        p4 = [mx + mw, my]
        bounders = [p1, p2, p3, p4]

        return bounders


# Try detect chessboard field size
def detect_field_size(image, img_log_list, min_num_of_field):
    num_of_found_fields = 0
    field_size = 0
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area:
            cv2.drawContours(img_log_list[1], cnt, -1, (255, 0, 0), 25)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            num_cor = len(approx)

            if num_cor == 4:

                # Dummy solution
                field_width = abs(approx[0] - approx[2])[0][0]
                field_height = abs(approx[1] - approx[3])[0][1]
                subs = abs(field_width - field_height)

                if (55 < field_width <= 80 and 55 < field_height <= 80) and subs < 10:
                    n = min(field_width, field_height)
                    field_size = max(field_size, n)

                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(img_log_list[1], (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.drawContours(img_log_list[1], approx, -1, (0, 0, 255), 10)

                    num_of_found_fields += 1
                    if num_of_found_fields >= min_num_of_field:
                        break

    if num_of_found_fields < min_num_of_field:
        raise Exception("Not found min numbers of fields")

    if field_size < 25:
        raise Exception('Can not detect field size')

    return field_size


def crop_playground_from_chessboard_image(source, field_size):
    img_size = source.shape[0]

    tmp = field_size * 1.08
    if tmp * 8.0 < img_size:
        field_size = tmp

    start = int((img_size - field_size * 8) / 2)
    end = img_size - start

    pts1 = np.float32([[start, start], [start, end], [end, end], [end, start]])
    pts2 = np.float32(
        [[0, 0], [0, img_size], [img_size, img_size], [img_size, 0]]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    playground = cv2.warpPerspective(source, matrix, (img_size, img_size))
    return playground


def main(args):
    print('\ncv_chessboard_playground.main')
    img = args["gray"]
    img_log_list = []

    print('Crop chessboard from image')
    g_log_img = args["gray"]
    try:
        img = cv2.blur(img, (2, 2))
        thresh, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img = ~img
        # img = cv2.Canny(img, 200, 200)
        # img = cv2.dilate(img, (300, 300), iterations=6)
        # img = cv2.erode(img, (200, 200), iterations=2)
        img_log_list.append(args["gray"])
        corners = detect_chessboard_frame(img, img_log_list)
        chessboard_img = crop_chessboard_from_image(args["gray"], corners)
        img = chessboard_img

    except Exception:
        cv2.imwrite('./tmp/err.tmp.png', g_log_img)
        raise Exception("(1): CHESSBOARD EXCEPTION")

    print('Cropped chessboard from image')

    print('Getting field size')
    g_log_img = args["gray"]
    try:
        img = cv2.blur(img, (3, 3))
        img = cv2.Canny(img, 200, 200)
        img = cv2.dilate(img, (200, 200), iterations=5)

        img_log_list.append(chessboard_img)
        field_size = detect_field_size(img, img_log_list, 3)

        print('Correct getting field size')

        img = crop_playground_from_chessboard_image(chessboard_img, field_size)

    except Exception as e:
        cv2.imwrite('./tmp/err.tmp.png', g_log_img)
        raise Exception("(2): PLAYGROUND EXCEPTION")

    return {
        'playground': img,
        'chessboard_log': img_log_list[0],
        'playground_log': img_log_list[1]
    }


if __name__ == '__main__':
    main(sys.argv[1:])
