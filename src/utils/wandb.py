import argparse


def get_wandb_name(args: argparse.Namespace, argparser: argparse.ArgumentParser):
    nondefault_values = []
    for name, value in vars(args).items():
        default_value = argparser.get_default(name)
        if value != default_value and "wandb" not in name:
            nondefault_values.append((name, value))

    if len(nondefault_values) == 0:
        return None

    name = "_".join(f"{name}:{value}" for name, value in nondefault_values)
    return name
