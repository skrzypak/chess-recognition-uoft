def get():
    return {
        "SOURCE_DIR": 'assets/source',
        "RESULT_DIR": 'assets/result',
        "PIECES_CATEGORIES": ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR', 'EM', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR'],
        "ROOT_IMG_SIZE": 640,
        "FIELD_IMG_SIZE": 120,
        "MANUAL_CROPPED": False,
        "DEBUG_MODE": False,
        "DEBUG_FIELD": True
    }


def get_tf():
    return {
        "TRAIN_DIR": '../../assets/chess_dataset/train',
        "TEST_DIR": '../../assets/chess_dataset/test',
        "PIECES_CATEGORIES": ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR', 'EM', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR'],
        "FIELD_IMG_SIZE": 120,
        "MODEL_NAME": 'chess-piece.model',
        "LR": 1e-3
    }
