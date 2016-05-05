def sec_to_timestring(seconds):
    """Transform seconds into a formatted time string.

    Parameters
    -----------
    seconds : int
        Seconds to be transformed.

    Returns
    -----------
    time : string
        A well formatted time string.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)