import numpy as np
'''
    Select a sub matrix from an matrix
'''
def select_sub_matrix(image, i, j, w, h, boxPointer_x=0, boxPointer_y=0):
    sub_matrix = []
    width, height = image.shape

    for x in range(0, h):
        calcI = i + x - boxPointer_x
        
        hasRow = calcI <= height - 1 - boxPointer_x

        sub_matrix.append([])

        for y in range(0, w):
            calcJ = j + y - boxPointer_y
            hasCol = calcJ <= width - 1 - boxPointer_y

            if (hasRow and hasCol and (calcI >= 0 and calcJ >= 0)):
                sub_matrix[x].append(image[calcI][calcJ])
            else: # what about borders?
                sub_matrix[x].append(0)

    return sub_matrix



# maps and applies function to each of sub matrices of matrix (returns new matrix)
def map_matrix_parts(image)->np.ndarray:

    def _map_matrix_parts(fn, sub_w, sub_h, padding=1, boxPointer_x=0, boxPointer_y=0):
        image_copy = np.copy(image)

        for i in range(0, len(image) - 1, padding):
            for j in range(0, len(image[i]) - 1, padding):
                sub_matrix = select_sub_matrix(image, i, j, sub_w, sub_h, boxPointer_x, boxPointer_y)

                image_copy[i][j] = fn(sub_matrix)

        return image_copy

    return _map_matrix_parts