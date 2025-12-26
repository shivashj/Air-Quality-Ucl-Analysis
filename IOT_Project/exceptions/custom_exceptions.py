class DataNotFoundError(Exception):
    """Raised when input data file is missing."""
    pass


class EmptyDatasetError(Exception):
    """Raised when dataframe is empty after processing."""
    pass


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


class ONNXExportError(Exception):
    """Raised when exporting model to ONNX fails."""
    pass
