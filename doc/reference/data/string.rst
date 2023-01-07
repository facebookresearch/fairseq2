String
======

.. currentmodule: fairseq2

.. class:: fairseq2.data.String
    :final:

    Bases: :class:`object`

    Represents an immutable UTF-8 string that supports zero-copy marshalling
    between Python and native code.

    .. method:: __init__()

        Constructs an empty :class:`String`.

    .. method:: __init__(s)
        :noindex:

        Constructs a :class:`String` by copying ``s``.

        :param s:
            The Python string to copy.

        :type s:
            str

    .. method:: lstrip()

        Returns a copy of the string with no whitespace at the beginning.

        :rtype:
            String

    .. method:: rstrip()

        Returns a copy of the string with no whitespace at the end.

        :rtype:
            String

    .. method:: to_py()

        Converts to :class:`str`.

        :rtype:
            str
