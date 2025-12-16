"""Main module - entry point."""

from .helpers import helper_function
from .utils import format_output


def main():
    result = helper_function(42)
    output = format_output(result)
    print(output)


if __name__ == "__main__":
    main()

# move helper_function to a new file called helpers.py
# update the import in this file
