# LeHome dataset

The training VM should download the selected source variants into this directory:

```bash
./datasets/pull_lehome.sh
```

The downloader requires the `hf` command from `huggingface_hub`. It retrieves variants
`001` through `005` for the four folding tasks from
`lehome/dataset_challenge`. These complete variants contain the 100 selected training
episodes and the 12 held-out validation episodes described in
[`lehome_selection.json`](./lehome_selection.json).

Downloaded files live under `datasets/lehome/` and are intentionally ignored by Git.
The dataset adapter must ignore the source `action` column and form 60-step,
q0-anchored joint-delta chunks from `observation.state`.
