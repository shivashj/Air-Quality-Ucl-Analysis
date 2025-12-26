"""
Export trained sklearn model to ONNX format
"""

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(model, n_features, output_path):
    """
    Export sklearn model to ONNX format.

    Parameters
    ----------
    model : sklearn model
        Trained model
    n_features : int
        Number of input features
    output_path : str
        Path to save ONNX model
    """

    initial_type = [
        ("float_input", FloatTensorType([None, n_features]))
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model exported to ONNX: {output_path}")
