# pt2 legacy(<2.8) fixtures

These `.pt2` files were exported with **torch 2.7.1+cpu** and pin the legacy
container layout that `torch.export.save` produced before torch 2.8 switched to
the pt2-archive format:

- flat records: `serialized_exported_program.json` (JSON graph, same
  `graph_module` schema the 2.8+ loader parses)
- `serialized_state_dict.pt` / `serialized_constants.pt`: nested `torch.save`
  zips whose `archive/data.pkl` pickle holds the state dict, with storage
  shards in `archive/data/<n>` and a `byteorder` marker
- `serialized_example_inputs.pt`, `version`

They are checked in so the CI (which has no torch < 2.8) can still exercise the
legacy decode path; `torch.load` is backward compatible, so the tests read the
pickled weights with the CI's own torch.

## reproducer

```bash
python3 -m venv /tmp/pt27 && /tmp/pt27/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"
# then run tools/pnnx/tests/fixtures/pt2_legacy/generate.py
```

`generate.py` (checked in next to this README) reproduces the two fixtures with
fixed seeds:

| fixture | model | covers |
|---|---|---|
| `linear_params_pt2_7.pt2` | `F.linear` + persistent buffer + non-persistent buffer | weights + constants payloads, parameter/buffer classification |
| `linear_const_pt2_7.pt2` | `nn.Linear` + tensor attribute | tensor_constant payload |
