import time


def get_elapsed_time(start_time: float) -> str:
    """Calculate and format elapsed time.

    Args:
        start_time (float): Time when processing started (from time.time())

    Returns:
        str: Formatted time string (HH:MM:SS)
    """
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    elapsed_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return elapsed_time
