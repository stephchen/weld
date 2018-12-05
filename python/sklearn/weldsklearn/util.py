import numpy as np
from weld.weldobject import *


# only valid for floats
def dot_mv(matrix, vector):
    """
    Computes the dot product between a matrix and a vector.

    Args:
        matrix (WeldObject / Numpy.ndarray): 2-d input matrix
        vector (WeldObject / Numpy.ndarray): 1-d input vector
        ty (WeldType): Type of each element in the input matrix and vector

    Returns:
        A WeldObject representing this computation
    """
    matrix_ty = WeldFloat()
    vector_ty = WeldFloat()

    weld_obj = WeldObject(NumPyEncoder(), NumPyDecoder())

    matrix_var = weld_obj.update(matrix)
    if isinstance(matrix, WeldObject):
        matrix_var = matrix.obj_id
        weld_obj.dependencies[matrix_var] = matrix

    vector_var = weld_obj.update(vector)
    loopsize_annotation = ""
    if isinstance(vector, WeldObject):
        vector_var = vector.obj_id
        weld_obj.dependencies[vector_var] = vector
    if isinstance(vector, np.ndarray):
        loopsize_annotation = "@(loopsize: %dL)" % len(vector)

    weld_template = """
       map(
         %(matrix)s,
         |row: vec[%(matrix_ty)s]|
           result(
             %(loopsize_annotation)s
             for(
               result(
                 %(loopsize_annotation)s
                 for(
                   zip(row, %(vector)s),
                   appender,
                   |b2, i2, e2: {%(matrix_ty)s, %(vector_ty)s}|
                     merge(b2, f64(e2.$0 * %(matrix_ty)s(e2.$1)))
                 )
               ),
               merger[f64,+],
               |b, i, e| merge(b, e)
             )
           )
       )
    """
    weld_obj.weld_code = weld_template % {"matrix": matrix_var,
                                          "vector": vector_var,
                                          "matrix_ty": matrix_ty,
                                          "vector_ty": vector_ty,
                                          "loopsize_annotation": loopsize_annotation}
    return weld_obj


def dot_vv(v1, v2):
    ty = WeldFloat()
    weld_obj = WeldObject(NumPyEncoder(), NumPyDecoder())

    assert(len(v1) == len(v2))

    v1_var = weld_obj.update(v1)
    if isinstance(v1, WeldObject):          # @@@@ how to transfer weld objects from one to another
        v1_var = v1.obj_id
        weld_obj.dependencies[v1_var] = v1

    v2_var = weld_obj.update(v2)
    if isinstance(vector, WeldObject):
        v2_var = v2.obj_id
        weld_obj.dependencies[v2_var] = v2

    if isinstance(vector, np.ndarray):
        loopsize_annotation = "@(loopsize: %dL)" % len(vector)

    weld_template = """
        result(
            %(loopsize_annotation)s
            for(
                zip(%(v1)s, %(v2)s),
                merger[%(v_ty)s, +],
                |b, i, e| merge(b, e.$0 * e.$1)
            )
        )
    """


    weld_obj.weld_code = weld_template % {"v1": v1_var,
                                          "v2": v2_var,
                                          "v_ty": ty,
                                          "loopsize_annotation": loopsize_annotation}
    return weld_obj





