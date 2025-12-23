"""Language-specific code generators."""

try:
    from .python import PythonCodeGen
except ImportError:
    from python import PythonCodeGen

GENERATORS = {
    "python": PythonCodeGen,
}


def get_generator(language: str):
    if language not in GENERATORS:
        raise ValueError(f"No code generator for language: {language}")
    return GENERATORS[language]()
