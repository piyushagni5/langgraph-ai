import os

def load_instructions_file(filename: str, default: str = ""):
    """
    Load instructions from a file. If the file does not exist, return the default instructions.
    
    Args:
        filename (str): The path to the instructions file.
        default (str): Default instructions to return if the file does not exist.
    
    Returns:
        str: The content of the instructions file or the default instructions.
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8") as file:
            return file.read()
    return default