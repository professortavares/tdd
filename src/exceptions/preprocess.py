class PreprocessException(Exception):
    # Exception raised when the preprocess fails
    def __init__(self, message):
        super().__init__(message)