import os
import cv2
import datetime

CONFIGURATION = {
    "SOURCE_DIR": 'assets/source/manual',
    "RESULT_DIR": 'assets/result',
    "PIECES_CATEGORIES": ["em", "bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"],
    "PROCESS_IMG_SIZE": 640,
    "FIELD_IMG_SIZE": 50,
    "MANUAL_CROPPED": False
}


if __name__ == '__main__':

    # if not os.path.isdir('../tf/models/chess-piece-0.001-2conv-basic.model'):
    #     print("Directory with .model file not found")
    #     print(quit)
    #     quit()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    CONFIGURATION["ROOT_LOG_DIR"] = '{}/{}/'.format(CONFIGURATION["RESULT_DIR"], timestamp)

    os.mkdir(CONFIGURATION["ROOT_LOG_DIR"])

    for img_full_name in os.listdir(CONFIGURATION['SOURCE_DIR']):

        # Setup current image necessary configuration
        try:
            curr_process_config = {
                'img': cv2.imread(os.path.join(CONFIGURATION["SOURCE_DIR"], img_full_name)),
                'img_name': img_full_name.split('.')[0],
            }
        except Exception as e:
            print(e)
            continue

        curr_process_config["img_log_dir"] = \
            '{}/{}/'.format(CONFIGURATION["ROOT_LOG_DIR"], curr_process_config["img_name"])

        os.mkdir(curr_process_config["img_log_dir"])

        if not CONFIGURATION['MANUAL_CROPPED']:
            pass

            # Detection chessboard from image

            # Getting chessboard from image

            # Remove additional frame from chessboard

        else:
            pass

        # Getting chessboard matrix from chessboard playground

        # Getting chessboard FEN notation from matrix

        # Generate SVG result from FEN notation
