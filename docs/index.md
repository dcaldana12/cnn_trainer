# cnn-trainer documentation

`cnn-trainer` builds reproducible dataset and training pipelines from YAML.
Start with the root [README](../README.md) for installation and a quick start.

| Page | What it covers |
| --- | --- |
| [Pipeline format](pipeline.md) | YAML fields, path resolution, run outputs, complete examples |
| [Source formats](sources.md) | CVAT XML, COCO JSON, YOLO directories, and label conventions |
| [Blocks](blocks.md) | Every block, its parameters, and its aliases |
| [Dataset quality](quality.md) | Auditing a dataset and removing redundant images |
| [Development](development.md) | Local setup with uv, tests, adding blocks, release checklist |

## The model in one paragraph

A **source adapter** loads one dataset format into an in-memory `Dataset` with
absolute `xyxy` boxes, class names, optional split tags, and a flag for missing
label files. Every **block** receives that dataset and returns it, possibly
modified. Export and training blocks materialize the result on disk. Because the
blocks are independent, a pipeline can stop after cleaning labels and never train
a model.

```text
source adapter -> Dataset -> block -> block -> ... -> export / train
```
