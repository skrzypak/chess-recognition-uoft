import io
import os
import cv2
import datetime

import numpy as np

import global_configuration
import cv_chessboard_playground

import chess
import chess.svg

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

import tensorflow as tf
from py.function import gen_stack_images

chessboard_rows_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
chessboard_cols_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]


# Input only RGB image source
def recognition_chessboard_position(playground_img_source, curr_log_dir):
    result_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    playground_img_gray = cv2.cvtColor(playground_img_source, cv2.COLOR_BGR2GRAY)

    w, h = playground_img_gray.shape
    w = int(w / 8)
    h = int(h / 8)

    curr_log_dir += '/fields'
    os.mkdir(curr_log_dir)

    for i in range(8):
        y = i * h
        for j in range(8):
            x = j * w

            img_field = playground_img_gray[y:y+h, x:x+w]
            img_field = cv2.medianBlur(img_field, 5)
            img_field = cv2.resize(img_field, (CONFIGURATION["FIELD_IMG_SIZE"], CONFIGURATION["FIELD_IMG_SIZE"]))
            ret, img_field = cv2.threshold(img_field, 127, 255, cv2.THRESH_TRUNC)

            field_data = np\
                .array(img_field)\
                .reshape(-1, CONFIGURATION["FIELD_IMG_SIZE"], CONFIGURATION["FIELD_IMG_SIZE"], 1)
            field_data = field_data / 255.0

            # AI RECOGNITION
            model_out = model.predict(field_data)[0]
            arg = np.argmax(model_out)
            piece_category = CONFIGURATION["PIECES_CATEGORIES"][arg]
            result_matrix[i][j] = piece_category

            # Save analysed field to debugs
            if CONFIGURATION["DEBUG_MODE"] or CONFIGURATION["DEBUG_FIELD"]:
                field_name = str(chessboard_cols_labels[j]) + str(chessboard_rows_labels[i]) + '_' + str(piece_category)
                cv2.imwrite(os.path.join(curr_log_dir, field_name + '.png'), img_field)

    return result_matrix


def get_fen_notation(chessboard):
    with io.StringIO() as s:
        for row in chessboard:
            empty = 0
            for cell in row:
                c = cell[0]
                if c in ('w', 'b'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell[1].upper() if c == 'w' else cell[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' w KQkq - 0 1')
        return s.getvalue()


def fen_2_svg(fen, curr_log_dir):
    _path_svg = os.path.join(curr_log_dir, "result.svg")

    board = chess.Board(fen)
    board_svg = chess.svg.board(board=board, size=2500)

    with open(_path_svg, "w") as file:
        file.write(board_svg)

    return _path_svg


def svg_2_png(_path_svg, curr_log_dir):
    _path_png = os.path.join(curr_log_dir, "result.png")

    drawing = svg2rlg(_path_svg)
    renderPM.drawToFile(drawing, _path_png, fmt="PNG")

    return cv2.resize(cv2.imread(_path_png), (CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"]))


if __name__ == '__main__':

    if not os.path.isdir('tf/models/chess-piece-0.001-2conv-basic.model'):
        print("Directory with .model file not found")
        print(quit)
        quit()

    CONFIGURATION = global_configuration.get()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    CONFIGURATION["ROOT_LOG_DIR"] = '{}/{}/'.format(CONFIGURATION["RESULT_DIR"], timestamp)

    model = tf.keras.models.load_model('tf/models/chess-piece-0.001-2conv-basic.model')
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

        print('Getting matrix from chessboard playground')
        matrix = recognition_chessboard_position(playground, curr_loop_data["img_log_dir"])

        print('Getting FEN notation from matrix')
        FEN = get_fen_notation(matrix)

        print('Generate SVG chessboard position from FEN notation')
        path_svg = fen_2_svg(FEN, curr_loop_data["img_log_dir"])

        print('Convert SVG to PNG')
        result_position_png = svg_2_png(path_svg, curr_loop_data["img_log_dir"])

        print('Generate result image')
        result_output_path = os.path.join(curr_loop_data["img_log_dir"], "done.png")

        result_all_images_process = cv2.imwrite(result_output_path, gen_stack_images(0.35, (
            [playground, result_position_png],
        )))

        if CONFIGURATION["DEBUG_MODE"]:
            cv2.imshow("Successfully", gen_stack_images(
                0.35, ([
                    curr_loop_data['img'], playground, result_position_png
                ])
            ))
            cv2.waitKey(0)
