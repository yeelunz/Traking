# Tools

The ad-hoc utilities that previously lived in this directory have been retired and
replaced by a single consolidated viewer that understands the richer `metadata.json`
artifacts produced by the pipeline.  The new entry point is:

```bash
python -m tools.experiment_viewer --help
```

This command scans the `results/` folder, loads experiment metadata, and presents a
summary table (or an interactive HTML dashboard if `--html` is provided).  It
combines information from detection, segmentation, and classification stages,
including the enhanced dataset insights and stage status records.

The legacy scripts are kept as thin stubs that simply emit a notice pointing to
this viewer.  If you still need their original behaviour, refer to version control
history prior to this change.
