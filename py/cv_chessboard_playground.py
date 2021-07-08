import os
import sys

import cv2
import numpy as np

import global_configuration
from py.function import gen_stack_images


def crop_chessboard_from_image(image, corners):

    output_img_size = image.shape[0]

    pts1 = np.float32(corners)
    pts2 = np.float32(
        [[0, 0], [0, output_img_size], [output_img_size, output_img_size], [output_img_size, 0]]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, matrix, (output_img_size, output_img_size))

    return perspective


def detect_chessboard_frame(image, img_source):
    img_contour = img_source.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mx = 0
    my = 0
    mw = 0
    mh = 0
    out_approx = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 5)
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
    cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(img_contour, out_approx, -1, (0, 0, 255), 25)

    if len(out_approx) == 4:
        # TODO sorting clockwise
        from_approx = [
            out_approx[0][0],
            out_approx[3][0],
            out_approx[2][0],
            out_approx[1][0]
        ]
        # print(from_approx)
        return img_contour, from_approx
    else:
        # Frame version
        p1 = [mx, my]
        p2 = [mx, my + mh]
        p3 = [mx + mw, my + mh]
        p4 = [mx + mw, my]
        bounders = [p1, p2, p3, p4]
        # print(bounders)
        return img_contour, bounders


# Try detect chessboard field size
def detect_field_size(image, img_source, min_num_of_field):
    num_of_found_fields = 0
    field_size = 0
    img_contour = img_source.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 25)
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
                    cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.drawContours(img_contour, approx, -1, (0, 0, 255), 10)

                    num_of_found_fields += 1

                    if num_of_found_fields >= min_num_of_field:
                        break

    if num_of_found_fields < min_num_of_field:
        raise Exception("Not found min numbers of fields")

    if field_size < 25:
        raise Exception('Can not detect field size')

    return img_contour, field_size


def crop_playground_from_chessboard_image(source, field_size):
    img_size = source.shape[0]

    tmp = field_size * 1.08
    if tmp * 8.0 < img_size:
        field_size = tmp

    start = int((img_size - field_size * 8) / 2)
    end = img_size - start
    # playground = source[start:end, start:end]
    # playground = cv2.resize(playground, (img_size, img_size))

    pts1 = np.float32([[start, start], [start, end], [end, end], [end, start]])
    pts2 = np.float32(
        [[0, 0], [0, img_size], [img_size, img_size], [img_size, 0]]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    playground = cv2.warpPerspective(source, matrix, (img_size, img_size))
    return playground


def throw_exception(err):
    print(err['title'])
    cv2.imwrite(os.path.join(err["img_log_dir"], err['output']), err['img'])

    if err["DEBUG_MODE"]:
        cv2.imshow(err['title'], err['img'])
        cv2.waitKey(0)
    raise Exception(err['title'])


def log_detection_playground(info):
    cv2.imwrite(os.path.join(info["img_log_dir"], info['output']), info['img'])

    if info["DEBUG_MODE"]:
        cv2.imshow(info['title'], info['img'])
        cv2.waitKey(0)


def main(args):
    CONFIGURATION = global_configuration.get()

    print('\nInitialize variable: {}'.format(args['img_name']))

    sts_results = {
        'source': args["img"],
        'blank': np.zeros((CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"], 3), np.uint8),
        'gray': cv2.cvtColor(args["img"], cv2.COLOR_BGR2GRAY),
        'result': None
    }

    print('Generating chessboard image mask')
    sts_results['smoothing'] = cv2.blur(sts_results['gray'], (2, 2))
    sts_results['thresh'], sts_results['binary'] = cv2.threshold(sts_results['smoothing'], 100, 255, cv2.THRESH_BINARY)
    sts_results['binary'] = ~sts_results['binary']
    # sts_results['canny'] = cv2.Canny(sts_results['binary'], 200, 200)
    # sts_results['dilate'] = cv2.dilate(sts_results['canny'], (300, 300), iterations=6)
    # sts_results['erode'] = cv2.erode(sts_results['dilate'], (200, 200), iterations=2)

    print('Generated chessboard image mask')

    print('Crop chessboard from image')
    try:
        sts_results['frame_contours'], corners = \
            detect_chessboard_frame(sts_results['binary'], sts_results['source'])
        sts_results['chessboard'] = crop_chessboard_from_image(sts_results['source'], corners)
        sts_results['chessboard_gray'] = cv2.cvtColor(sts_results["chessboard"], cv2.COLOR_BGR2GRAY)
        sts_results['chessboard_smoothing'] = cv2.blur(sts_results['chessboard_gray'], (3, 3))
        sts_results['chessboard_canny'] = cv2.Canny(sts_results['chessboard_smoothing'], 200, 200)
        sts_results['chessboard_dilate'] = cv2.dilate(sts_results['chessboard_canny'], (200, 200), iterations=5)
    except Exception as e:
        print(e)
        throw_exception({
            'img': gen_stack_images(0.35, (
                [
                    sts_results['source'],
                    sts_results['gray'],
                    sts_results['smoothing'],
                    sts_results['binary'],
                    # sts_results['canny'],
                    # sts_results['dilate'],
                    # sts_results['erode']
                ],
            )),
            'output': 'fail_detection_chessboard.png',
            'title': "Can't detect possibility chessboard",
            "img_log_dir": args["img_log_dir"],
            "DEBUG_MODE": CONFIGURATION["DEBUG_MODE"]
        })
    print('Cropped chessboard from image')

    print('Getting field size')
    try:
        sts_results['field_size_contours'], field_size = detect_field_size(
            sts_results['chessboard_dilate'], sts_results['chessboard'], 3
        )
        print('Correct getting field size')

        sts_results['result'] = crop_playground_from_chessboard_image(sts_results['chessboard'], field_size)

    except Exception as e:
        print(e)
        throw_exception({
            'img': gen_stack_images(0.35, (
                [
                    sts_results['source'],
                    sts_results['gray'],
                    sts_results['smoothing'],
                    sts_results['binary'],
                    # sts_results['canny'],
                    # sts_results['dilate'],
                    # sts_results['erode'],
                    sts_results['frame_contours'],
                ],
                [
                    sts_results['chessboard'],
                    sts_results['chessboard_gray'],
                    sts_results['chessboard_smoothing'],
                    sts_results['chessboard_canny'],
                    sts_results['chessboard_dilate'],
                ],
            )),
            'output': 'fail_playground.png',
            'title': "Can't get playgrounds",
            "img_log_dir": args["img_log_dir"],
            "DEBUG_MODE": CONFIGURATION["DEBUG_MODE"]
        })

    log_detection_playground({
        'img': gen_stack_images(0.35, (
            [
                sts_results['source'],
                sts_results['gray'],
                sts_results['smoothing'],
                sts_results['binary'],
                # sts_results['canny'],
                # sts_results['dilate'],
                # sts_results['erode'],
            ],
            [
                sts_results['frame_contours'],
                sts_results['chessboard'],
                sts_results['chessboard_gray'],
                sts_results['chessboard_smoothing'],
            ],
            [
                sts_results['chessboard_canny'],
                sts_results['chessboard_dilate'],
                sts_results['field_size_contours'],
                sts_results['result'],
            ]
        )),
        'output': 'ok_playground.png',
        'title': "Playground",
        "img_log_dir": args["img_log_dir"],
        "DEBUG_MODE": CONFIGURATION["DEBUG_MODE"]
    })

    return sts_results['result']


if __name__ == '__main__':
    main(sys.argv[1:])
