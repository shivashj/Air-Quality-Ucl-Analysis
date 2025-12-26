"""
ONNX Model Inference
"""

import numpy as np
import onnxruntime as ort


def load_onnx_model(onnx_path):
    """
    Load ONNX model.

    Returns
    -------
    ort_session : onnxruntime.InferenceSession
    """
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    return ort_session


def predict_with_onnx(ort_session, X):
    """
    Run inference using ONNX model.

    Parameters
    ----------
    X : numpy.ndarray
        Input features (n_samples, n_features)

    Returns
    -------
    predictions : numpy.ndarray
    """

    input_name = ort_session.get_inputs()[0].name

    predictions = ort_session.run(
        None,
        {input_name: X.astype(np.float32)}
    )

    return predictions
