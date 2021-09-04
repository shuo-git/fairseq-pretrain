# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from transformers import T5TokenizerFast


SPACE_NORMALIZER = re.compile(r"\s+")

# Here includes several versions of the "tokenize_line" function

# The original version
# def tokenize_line(line):
#     line = SPACE_NORMALIZER.sub(" ", line)
#     line = line.strip()
#     return line.split()

# Use mT5 tokenizer to deal with raw text
fast_tok=T5TokenizerFast.from_pretrained("./mT5_tokenizer")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return fast_tok.tokenize(line)
