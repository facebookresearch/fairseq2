import func_argparse as fap

from . import commands

if __name__ == "__main__":

    parsers = {
        "train": fap.func_argparser(commands.train),
        "evaluate": fap.func_argparser(commands.evaluate),
        "inference": fap.func_argparser(commands.inference),
    }
    # TODO: push this to func_argparse
    with_overrides = []
    for name, parser in parsers.items():
        # Make script a positional argument.
        script_action = [a for a in parser._actions if "--script" in a.option_strings]
        if len(script_action) == 1:
            script_action[0].option_strings = ()

        # Handle overrides separately, I'm not sure why nargs="*" doesn't work as expected
        override_action = [
            a for a in parser._actions if "--overrides" in a.option_strings
        ]
        if len(override_action) == 1:
            parser._actions.remove(override_action[0])
            with_overrides.append(name)

    # TODO: add beam search args to inference

    main_parser = fap.multi_argparser(description=__doc__, **parsers)

    known_args, overrides = main_parser.parse_known_args()
    parsed_args = vars(known_args)
    if not parsed_args:
        # Show help for multi argparser receiving no arguments.
        main_parser.print_help()
        main_parser.exit()
    command = parsed_args.pop("__command")

    if command.__name__ in with_overrides:
        parsed_args["overrides"] = overrides
        typo_in_command = any(o.startswith("-") for o in overrides)
    else:
        typo_in_command = len(overrides) > 0

    if typo_in_command:
        # Redo the parsing so that we have the normal error message for unk args
        main_parser.parse_args()

    command(**parsed_args)
