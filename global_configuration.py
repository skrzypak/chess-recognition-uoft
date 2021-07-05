def get():
    return {
        "SOURCE_DIR": 'assets/source',
        "RESULT_DIR": 'assets/result',
        "PIECES_CATEGORIES": ["em", "bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"],
        "ROOT_IMG_SIZE": 640,
        "FIELD_IMG_SIZE": 50,
        "MANUAL_CROPPED": False,
        "DEBUG_MODE": True
    }


def get_tf():
    return {
        "TRAIN_DIR": '../../assets/chess_dataset/train',
        "TEST_DIR": '../../assets/chess_dataset/test',
        "PIECES_CATEGORIES": ["em", "bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"],
        "FIELD_IMG_SIZE": 50,
        "MODEL_NAME": 'chess-piece-{}-{}.model'.format(1e-3, '2conv-basic'),
        "LR": 1e-3
    }
