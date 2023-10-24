import sys

def get_commandline_args():
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [
        arg.replace("'", "'\\''")
        if all(char not in arg for char in extra_chars)
        else "'" + arg.replace("'", "'\\''") + "'"
        for arg in sys.argv
    ]

    return sys.executable + " " + " ".join(argv)
