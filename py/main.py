import os
import cv2
import datetime

import global_configuration
import cv_chessboard_playground

if __name__ == '__main__':

    # if not os.path.isdir('../tf/models/chess-piece-0.001-2conv-basic.model'):
    #     print("Directory with .model file not found")
    #     print(quit)
    #     quit()

    CONFIGURATION = global_configuration.get()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    CONFIGURATION["ROOT_LOG_DIR"] = '{}/{}/'.format(CONFIGURATION["RESULT_DIR"], timestamp)

    os.mkdir(CONFIGURATION["ROOT_LOG_DIR"])

    for img_full_name in os.listdir(CONFIGURATION['SOURCE_DIR']):

        # Setup current image necessary configuration
        try:
            curr_loop_data = {
                'img': cv2.imread(os.path.join(CONFIGURATION["SOURCE_DIR"], img_full_name)),
                'img_name': img_full_name.split('.')[0],
            }

            curr_loop_data['img'] = cv2.resize(
                curr_loop_data['img'],
                (CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"])
            )
        except Exception as e:
            print(e)
            continue

        curr_loop_data["img_log_dir"] = \
            '{}/{}/'.format(CONFIGURATION["ROOT_LOG_DIR"], curr_loop_data["img_name"])

        os.mkdir(curr_loop_data["img_log_dir"])

        if not CONFIGURATION['MANUAL_CROPPED']:
            try:
                playground = cv_chessboard_playground.main(curr_loop_data)
            except:
                continue
        else:
            playground = curr_loop_data["img"]

        # Getting chessboard matrix from chessboard playground
        cv2.imshow("PLAYGROUND", playground)
        cv2.waitKey(0)

        # Getting chessboard FEN notation from matrix

        # Generate SVG result from FEN notation
