import os
import cv2
import numpy as np


def get_contents_in_dir(dir_path, notstartswith, endswith):
    contents = os.listdir(dir_path)
    for e in notstartswith:
        contents = [c for c in contents if not c.startswith(e)]
    res = []
    for e in endswith:
        res += [c for c in contents if c.endswith(e)]
    if len(res) == 0:  # for directory
        res = contents

    res = [os.path.join(dir_path, r) for r in res]

    return sorted(res)


def is_matrix_similar(mat1, mat2, threshold=0.9):
    mat1 = scale_matrix_to_image(mat1)
    mat2 = scale_matrix_to_image(mat2)

    hist_1 = cv2.calcHist([mat1], [0], None, [256], [0, 255])
    hist_1 = cv2.normalize(hist_1, hist_1, 0, 1, cv2.NORM_MINMAX, -1)
    hist_2 = cv2.calcHist([mat2], [0], None, [256], [0, 255])
    hist_2 = cv2.normalize(hist_2, hist_2, 0, 1, cv2.NORM_MINMAX, -1)

    similar_score = cv2.compareHist(hist_1, hist_2, 0)
    # print(f'mat1 and mat2 similar score: {similar_score}')
    if similar_score > threshold:
        return True
    return False


def scale_matrix_to_image(mat):
    mat = mat / mat.max()
    mat *= 255
    mat = mat.astype(np.uint8)
    return mat


if __name__ == '__main__':
    print(get_contents_in_dir('/XXX', ['.'], []))
