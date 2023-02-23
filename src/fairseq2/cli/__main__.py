import func_argparse as fa

from fairseq2.cli import commands


def main():
    parsers = {
        "train": fa.func_argparser(commands.train),
        "evaluate": fa.func_argparser(commands.evaluate),
        "inference": fa.func_argparser(commands.inference),
        "grid": fa.func_argparser(commands.grid),
        "eval_server": fa.func_argparser(commands.eval_server),
    }
    # TODO: push this to func_argparse
    with_overrides = []
    for name, parser in parsers.items():
        # Promote the first argument to positional argument
        if len(parser._actions) < 2:
            continue
        if parser._actions[1].default is not None:
            continue
        parser._actions[1].option_strings = ()

        # Handle overrides separately, I'm not sure why nargs="*" doesn't work as expected
        override_action = [
            a for a in parser._actions if "--overrides" in a.option_strings
        ]
        if len(override_action) == 1:
            parser._actions.remove(override_action[0])
            with_overrides.append(name)

    main_parser = fa.multi_argparser(description=__doc__, **parsers)

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


if __name__ == "__main__":
    main()
