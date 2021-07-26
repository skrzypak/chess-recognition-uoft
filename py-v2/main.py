import io
import os
import shutil

import cv2

import numpy as np

import global_configuration
import cv_chessboard_playground

import chess
import chess.svg

import time
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

import tensorflow as tf
from py.function import gen_stack_images

chessboard_rows_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
chessboard_cols_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]


# Input only RGB image source
def recognition_chessboard_position(playground):
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

    w, h = playground.shape
    w = int(w / 8)
    h = int(h / 8)

    logs = './tmp/fields'

    for i in range(8):
        y = i * h
        for j in range(8):
            x = j * w

            img_field = playground[y:y+h, x:x+w]
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
            # piece_category = CONFIGURATION["PIECES_CATEGORIES"][0]
            result_matrix[i][j] = piece_category

            # Save analysed field to debugs
            if CONFIGURATION["DEBUG_MODE"] or CONFIGURATION["DEBUG_FIELD"]:
                field_name = str(chessboard_cols_labels[j]) + str(chessboard_rows_labels[i]) + '_' + str(piece_category)
                cv2.imwrite(os.path.join(logs, field_name + 'tmp.png'), img_field)

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


def fen_2_svg(fen):
    _path_svg = os.path.join('./tmp/', "result.tmp.svg")

    board = chess.Board(fen)
    board_svg = chess.svg.board(board=board, size=2500)

    with open(_path_svg, "w") as file:
        file.write(board_svg)

    return _path_svg


def svg_2_png(result_svg_path):
    path_png = os.path.join('./tmp/', "result.tmp.png")

    drawing = svg2rlg(result_svg_path)
    renderPM.drawToFile(drawing, path_png, fmt="PNG")

    return cv2.resize(cv2.imread(path_png), (CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"]))


if __name__ == '__main__':

    if not os.path.isdir('../tf/models/chess-piece-0.001-2conv-basic.model'):
        print("Directory with .model file not found")
        print(quit)
        quit()

    CONFIGURATION = global_configuration.get()
    model = tf.keras.models.load_model('../tf/models/chess-piece-0.001-2conv-basic.model')

    if os.path.isdir('./tmp/'):
        shutil.rmtree('./tmp/')

    os.mkdir('./tmp/')
    os.mkdir('./tmp/fields')

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    blank_img = np.zeros((CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"], 1), np.uint8)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (CONFIGURATION["ROOT_IMG_SIZE"], CONFIGURATION["ROOT_IMG_SIZE"]))

        try:
            images = cv_chessboard_playground.main({"gray": img})
            img = images['playground']

            print('Getting matrix from chessboard playground')
            matrix = recognition_chessboard_position(img)

            print('Getting FEN notation from matrix')
            FEN = get_fen_notation(matrix)

            print('Generate SVG chessboard position from FEN notation')
            path_svg = fen_2_svg(FEN)

            print('Convert SVG to PNG')
            result = svg_2_png(path_svg)

            cv2.imshow("AI CHESS DETECTION", gen_stack_images(0.75, (
                [
                    cv2.resize(frame, (365, 365)),
                    cv2.resize(result, (365, 365)),
                ],
                [
                    cv2.resize(images['chessboard_log'], (365, 365)),
                    cv2.resize(images['playground_log'], (365, 365)),
                ]
            )))

        except Exception as e:
            print(e)
            img = cv2.imread('./tmp/err.tmp.png')
            cv2.imshow("AI CHESS DETECTION", gen_stack_images(1, ([
                cv2.resize(frame, (365, 365)),
                cv2.resize(img, (365, 365)),
            ])))

        time.sleep(1)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    shutil.rmtree('./tmp/')

