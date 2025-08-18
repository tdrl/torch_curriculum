# PyTorch Curriculum

_Heather Lane \<[heather@illation.net](mailto:heather@illation.net)\>, 2025_

This is a personal learning exercise: a series of small programs of increasing complexity to refresh myself on my now somewhat rusty PyTorch API fluidity. (I last did PyTorch professionally in 2020.) None of the exercises are intended to do anything "real" - they're just to work my way back through key bits of the API. Hence, many of them use synthetic data or toy/small models, make no effort to be optimal or to incorporate all the tricks of the trade, etc. I _do_ try to maintain decent code quality, though - factoring, testing, docco, etc.

# Contents

- [src/torch_playground/](src/torch_playground/): Demo programs and support functions.
  - [util.py](src/torch_playground/util.py): Support functions. Things like a standard Application class, standardized    logging, standardized command-line flag handling, standard train loop, etc.
  - d[0-9]+.py: Demo programs for different bits of the torch API and different learning algorithms.
- [tests/](tests): Unit and integration test code for the above.

# A Note on Versions

This is all written in the now-ancient PyTorch 2.2 because my equally ancient laptop is an Intel Mac, and PyTorch dropped support for Intel Macs after 2.2. This code _should_ be forward-compatible, albeit not with the newest bells-and-whistles. But I haven't tested it against a newer torch version, so YMMV. :shrug:
