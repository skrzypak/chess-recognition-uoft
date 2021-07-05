import os
import sys

import cv2
import numpy as np

import global_configuration
from py.function import gen_stack_images


def crop_chessboard_from_image(image, p1, p2, p3, p4):

    output_img_size = image.shape[0]

    pts1 = np.float32([p1, p2, p3, p4])
    pts2 = np.float32(
        [[0, 0], [0, output_img_size], [output_img_size, output_img_size], [output_img_size, 0]]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, matrix, (output_img_size, output_img_size))

    # cv2.imshow("crop_chessboard_from_image()", perspective)
    # cv2.waitKey(0)

    return perspective


def detect_chessboard_frame(image, img_source):
    img_contour = img_source.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(img_contour, approx, -1, (0, 0, 255), 25)

    p1 = [x, y]
    p2 = [x, y + h]
    p3 = [x + w, y + h]
    p4 = [x + w, y]

    return img_contour, p1, p2, p3, p4


# Try detect chessboard field size
def detect_field_size(image, img_source, min_num_of_field):
    # cv2.imshow("detect_field_size(image)", image)

    num_of_found_fields = 0
    field_size = 0
    img_contour = img_source.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    # cv2.imshow("detect_field_size(image_contour), field size {}".format(field_size), img_contour)
    # cv2.waitKey(0)

    if num_of_found_fields < min_num_of_field:
        raise Exception("Not found min numbers of fields")

    if field_size < 25:
        raise Exception('Can not detect field size')

    return img_contour, field_size


def crop_playground_from_chessboard_image(source, field_size):
    img_size = source.shape[0]
    half_crop_size = int((img_size - field_size * 8) / 2)
    end = img_size - half_crop_size
    playground = source[half_crop_size:end, half_crop_size:end]
    playground = cv2.resize(playground, (img_size, img_size))

    # cv2.imshow("crop_playground_from_chessboard_image()", playground)
    # cv2.waitKey(0)

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
    sts_results['thresh'], sts_results['binary'] = cv2.threshold(sts_results['gray'], 100, 255, cv2.THRESH_BINARY)
    sts_results['bin_neg'] = ~sts_results['binary']
    print('Generated chessboard image mask')

    print('Crop chessboard from image')
    try:
        sts_results['frame_contours'], p1, p2, p3, p4 = \
            detect_chessboard_frame(sts_results['bin_neg'], sts_results['source'])
        sts_results['chessboard'] = crop_chessboard_from_image(sts_results['source'], p1, p2, p3, p4)
        sts_results['chessboard_gray'] = cv2.cvtColor(sts_results["chessboard"], cv2.COLOR_BGR2GRAY)
        sts_results['chessboard_smoothing'] = cv2.blur(sts_results['chessboard_gray'], (3, 3))
        sts_results['chessboard_canny'] = cv2.Canny(sts_results['chessboard_smoothing'], 200, 200)
        sts_results['chessboard_dilate'] = cv2.dilate(sts_results['chessboard_canny'], (200, 200), iterations=1)
    except Exception as e:
        print(e)
        throw_exception({
            'img': gen_stack_images(0.35, (
                [
                    sts_results['source'],
                    sts_results['gray'],
                    sts_results['binary'],
                    sts_results['bin_neg']
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
                    sts_results['binary'],
                    sts_results['bin_neg'],
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
                    sts_results['blank'],
                    sts_results['blank']
                ]
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
                sts_results['binary'],
                sts_results['bin_neg'],
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
                sts_results['result']
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