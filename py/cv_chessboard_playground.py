import os
import sys

import cv2
import numpy as np

import global_configuration
from function import gen_stack_images


def crop_chessboard_from_image(img_source, approx):

    output_img_size = img_source.shape[0]

    p1, p2, p3, p4 = approx
    p1x, p1y = p1[0]
    p2x, p2y = p2[0]
    p3x, p3y = p3[0]
    p4x, p4y = p4[0]

    minX = min(p1x, p2x, p3x, p4x)
    maxX = max(p1x, p2x, p3x, p4x)
    minY = min(p1y, p2y, p3y, p4y)
    maxY = max(p1y, p2y, p3y, p4y)

    pts1 = np.float32([[minX, minY], [minX, maxY], [maxX, maxY], [maxX, minY]])
    pts2 = np.float32(
        [[0, 0], [0, output_img_size], [output_img_size, output_img_size], [output_img_size, 0]]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(img_source, matrix, (output_img_size, output_img_size))
    return perspective


# Try detect possibility chessboard
def get_contours_chessboard(img_erode, img_source):
    image_contours_all = img_source.copy()
    images_contour = []
    images_approx = []

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        img_contour = img_source.copy()
        if area > 1500:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            cv2.drawContours(image_contours_all, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            num_cor = len(approx)
            if num_cor >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.rectangle(image_contours_all, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.drawContours(img_contour, approx, -1, (0, 0, 255), 25)
                cv2.drawContours(image_contours_all, approx, -1, (0, 0, 255), 25)
                images_contour.append(img_contour)

                # TODO filter inside points

                images_approx.append(approx)

    return images_contour, images_approx, image_contours_all


# Try detect chessboard field size
def detect_chessboard_field_size(img_filter, img_source):

    field_size = 0
    img_contour = img_source.copy()
    contours, hierarchy = cv2.findContours(img_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 5)
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

                    break

    return img_contour, field_size


# Getting chessboard playground from image
def get_playground_from_chessboard_image(img_chessboard, field_size):

    img_size = img_chessboard.shape[0]
    half_crop_size = int((img_size - field_size * 9) / 2)
    end = img_size - half_crop_size
    playground = img_chessboard[half_crop_size:end, half_crop_size:end]
    return cv2.resize(playground, (img_size, img_size))


def throw_exception(err):
    print(err['title'])
    log_invalid_detection_playground(err)
    raise Exception(err['title'])


def log_invalid_detection_playground(err):
    cv2.imwrite(os.path.join(err["img_log_dir"], err['output']), err['img'])

    if err["DEBUG_MODE"]:
        cv2.imshow(err['title'], err['img'])
        cv2.waitKey(0)


# Return all steps images
# Playground image has attempt gray and gauss filter
def main(args):
    kernel = np.ones((5, 5), np.uint8)
    CONFIGURATION = global_configuration.get()
    curr_loop_data = args

    print('\ncv_chessboard_playground: {}'.format(curr_loop_data['img_name']))

    img_steps_results = {
        'blank': np.zeros((CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"], 3), np.uint8),
        'gray': cv2.cvtColor(curr_loop_data["img"], cv2.COLOR_BGR2GRAY),
    }

    img_steps_results['gauss'] = cv2.GaussianBlur(img_steps_results['gray'], (3, 3), 10)
    img_steps_results['canny'] = cv2.Canny(img_steps_results['gauss'], 300, 300)
    img_steps_results['dilation'] = cv2.dilate(img_steps_results['canny'], kernel, iterations=12)
    img_steps_results['erode'] = cv2.erode(img_steps_results['dilation'], kernel, iterations=2)

    # Detection possibility chessboard
    img_contours_array, image_approx_array, img_steps_results['image_contours_all'] = get_contours_chessboard(
        img_steps_results['erode'], curr_loop_data['img']
    )

    if len(image_approx_array) == 0:
        throw_exception({
            'img': gen_stack_images(0.35, (
                [
                    curr_loop_data['img'], img_steps_results['gray'], img_steps_results['gauss'],
                    img_steps_results['canny']
                ],
                [
                    img_steps_results['dilation'], img_steps_results['erode'], img_steps_results['image_contours_all'],
                    img_steps_results['blank']
                ],
            )),
            'output': 'fail_detection_chessboard.png',
            'title': "Can't detect possibility chessboard",
            "img_log_dir": curr_loop_data["img_log_dir"],
            "DEBUG_MODE": CONFIGURATION["DEBUG_MODE"]
        })

    print('Detecting possibility some chessboard')

    img_steps_results['playground'] = None

    # If detect more than one possibility chessboard
    for idx, possibility_chessboard in enumerate(img_contours_array):

        img_steps_results['contours'] = possibility_chessboard
        image_approx = image_approx_array[idx]

        # TODO
        if len(image_approx) != 4:
            continue

        print('Possibility chessboard {}'.format(idx))

        # Cropping chessboard from image
        img_steps_results['chessboard'] = \
            crop_chessboard_from_image(img_steps_results["gray"], image_approx)

        # Detect chessboard field size
        (img_steps_results['thresh'], img_steps_results['binary']) = \
            cv2.threshold(img_steps_results['chessboard'], 100, 255, cv2.THRESH_BINARY)

        img_steps_results['contours_field_size'], field_size = \
            detect_chessboard_field_size(img_steps_results['binary'], img_steps_results['chessboard'])

        print('Detecting field size')

        # Remove additional frame from chessboard (getting playground)
        try:
            img_steps_results['playground'] = \
                get_playground_from_chessboard_image(img_steps_results['chessboard'], field_size)
            break
        except:
            log_invalid_detection_playground({
                'img': gen_stack_images(0.35, (
                    [
                        curr_loop_data['img'], img_steps_results['gray'], img_steps_results['gauss'],
                        img_steps_results['canny']
                    ],
                    [
                        img_steps_results['dilation'], img_steps_results['erode'], img_steps_results['contours'],
                        img_steps_results['chessboard']
                    ],
                    [
                        img_steps_results['binary'], img_steps_results['contours_field_size'],
                        img_steps_results['blank'], img_steps_results['blank']
                    ],
                )),
                'output': 'fail_playground_{}.png'.format(idx),
                'title': "Can't get chessboard playground {}".format(idx),
                "img_log_dir": curr_loop_data["img_log_dir"],
                "DEBUG_MODE": CONFIGURATION["DEBUG_MODE"]
            })

    if img_steps_results['playground'] is None:
        throw_exception({
            'img': gen_stack_images(0.35, (
                [
                    curr_loop_data['img'], img_steps_results['gray'], img_steps_results['gauss'],
                    img_steps_results['canny']
                ],
                [
                    img_steps_results['dilation'], img_steps_results['erode'], img_steps_results['image_contours_all'],
                    img_steps_results['blank']
                ],
            )),
            'output': 'fail_playground.png',
            'title': "Can't get chessboard playground",
            "img_log_dir": curr_loop_data["img_log_dir"],
            "DEBUG_MODE": CONFIGURATION["DEBUG_MODE"]
        })

    return img_steps_results


if __name__ == '__main__':
    main(sys.argv[1:])
