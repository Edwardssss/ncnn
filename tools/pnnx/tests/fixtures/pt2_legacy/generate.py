# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Reproduce the pt2 legacy(<2.8) fixtures exported with torch 2.7.1.
# Run with a torch 2.7.x interpreter, e.g.
#   python3 -m venv /tmp/pt27 && /tmp/pt27/bin/pip install \
#       --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"
#   /tmp/pt27/bin/python generate.py

import os
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))


class M1(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Parameter(torch.randn(8, 8))
        self.b = nn.Parameter(torch.randn(8))
        self.register_buffer("pbuf", torch.randn(8))
        self.register_buffer("nbuf", torch.randn(8), persistent=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.w, self.b) + self.pbuf + self.nbuf


class M2(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.fc = nn.Linear(8, 8)
        self.c = torch.arange(8, dtype=torch.float32)

    def forward(self, x):
        return self.fc(x) + self.c


def main():
    torch.manual_seed(7)
    x = torch.randn(4, 8)
    for tag, m in (("linear_params", M1()), ("linear_const", M2())):
        m.eval()
        ep = torch.export.export(m, (x,))
        path = os.path.join(OUT, tag + "_pt2_7.pt2")
        torch.export.save(ep, path)
        print("wrote", path, os.path.getsize(path), "bytes; torch", torch.__version__)


if __name__ == "__main__":
    main()
