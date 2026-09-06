# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Expected pt2 (torch.export) outcome per pnnx test tag.
#
# Every call to test_pnnx() is audited against this table so a test that
# silently drops off the pt2 channel (returns None) is a visible regression
# instead of a quiet skip.
#
#   unlisted tag => "pass"        : torch.export must succeed, conversion must
#                                   match (test_pnnx -> True)
#   "export-skip"                 : torch.export is pinned to reject the model
#                                   (test_pnnx -> None, e.g. dynamic shapes /
#                                   control flow the exporter cannot handle)
#
# Regeneration workflow (after a torch version / suite change):
#   PNNX_PT2_RESULT_LOG=/tmp/r.log PNNX_PT2_RECORD_ONLY=1 \
#       python ../../tests/test_<tag>.py
# then move every None row of /tmp/r.log into EXPECT as "export-skip", and
# move every row that changed to True back to "pass".

EXPECT = {
    # torch.export rejects these models (dynamic indexing / control flow /
    # dynamic-arange shapes); recorded so a future torch that starts exporting
    # them is noticed instead of silently re-covering them
    "test_Tensor_index": "export-skip",
    "test_quantization_shufflenet_v2_x1_0": "export-skip",
    "test_torch_arange": "export-skip",
}
