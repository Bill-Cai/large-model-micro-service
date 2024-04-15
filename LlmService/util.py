from typing import Optional


def stream_out(data, stream: Optional[None] = None, verbose: bool = False):
    if stream is None:
        if verbose:
            print(data)
    else:
        pass
