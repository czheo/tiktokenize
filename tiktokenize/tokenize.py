import tiktoken
import argparse
import sys
from termcolor import colored
import re


COLORS = [
    "dark_grey",
    "red",
    "light_yellow",
    "green",
    "light_blue",
    "yellow",
    "light_green",
    "blue",
    "light_red",
    "magenta",
    "cyan",
    "light_grey",
    "light_magenta",
    "light_cyan",
]


def tokenize(text, encoding):
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text, allowed_special='all')
    lb = enc.decode_tokens_bytes(tokens)
    return zip(lb, tokens)


def print_tokens(tokens):
    prev_color = None
    for i, (b, tok) in enumerate(tokens):
        text = b.decode('utf-8')
        color = COLORS[tok % len(COLORS)]
        if color == prev_color:
            color = COLORS[(tok + 1) % len(COLORS)]
        text = colored(text, color=color, attrs=['underline'])
        print(text, end="")
        prev_color = color


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoding', type=str, choices=tiktoken.list_encoding_names(), default='cl100k_base')
    args = parser.parse_args()
    tokens = tokenize(sys.stdin.read(), args.encoding)
    print_tokens(tokens)
