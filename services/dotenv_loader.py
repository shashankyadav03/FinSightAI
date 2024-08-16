from dotenv import load_dotenv
import os

def load_environment():
    """
    Loads environment variables from a .env file.

    - Ensures that the environment is correctly set up for running the application.
    - Fixes the OpenMP initialization error by setting a specific environment variable.
    """
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        load_dotenv()
    except Exception as e:
        raise RuntimeError(f"Failed to load environment variables: {e}")
