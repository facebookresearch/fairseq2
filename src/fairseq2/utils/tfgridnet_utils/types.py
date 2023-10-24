from distutils.util import strtobool
from typing import Optional, Tuple, Union

def str2bool(value: str) -> bool:
    return bool(strtobool(value))

def str_or_none(value: str) -> Optional[str]:
    """str_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str_or_none)
        >>> parser.parse_args(['--foo', 'aaa'])
        Namespace(foo='aaa')
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return value

def str2triple_str(value: str) -> Tuple[str, str, str]:
    """str2triple_str.

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    value = remove_parenthesis(value)
    a, b, c = value.split(",")

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove quotes from it.
    return remove_quotes(a), remove_quotes(b), remove_quotes(c)
