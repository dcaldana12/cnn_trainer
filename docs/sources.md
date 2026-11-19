# Source formats and label conventions

A source adapter is the first stage of every pipeline. It reads one dataset
format, validates its structure, and produces the in-memory `Dataset` consumed by
all blocks.

```yaml
source:
  format: cvat_xml   # cvat_xml | coco | yolo
  root: /data/my-dataset
```

Supported aliases for `format`:

| Alias | Canonical format |
| --- | --- |
| `cvat`, `cvat-images`, `cvat_images` | `cvat_xml` |
| `coco_json` | `coco` |
| `ultralytics` | `yolo` |

## Internal representation

| Concept | Convention |
| --- | --- |
| Box geometry | Absolute pixels, `x1, y1, x2, y2` |
| Coordinates | Top-left origin, `x2 > x1`, `y2 > y1` |
| Class identity | Integer id plus a name in the dataset class list |
| Split tag | `train`, `val`, `test`, or `None` |
| Label presence | `has_label=False` when the label file is missing |
| Image order | Deterministic, sorted by relative path |

YOLO export converts boxes to normalized
`class x_center y_center width height`.

## CVAT for images (`cvat_xml`)

Expected input is a **CVAT for images 1.1** export: an `annotations.xml` file plus
the image folder.

```yaml
source:
  format: cvat_xml
  root: /data/cvat-export
  annotations: annotations.xml
  images: images
  class_names: [car, person]     # optional; otherwise read from <meta>
  skip_outside: true             # skip boxes marked outside="1"
  allow_unknown_labels: false    # append labels missing from class_names
```

| Option | Default | Meaning |
| --- | --- | --- |
| `annotations` | `annotations.xml` | XML file, relative to `root` |
| `images` | `images` | Image directory, relative to `root` |
| `class_names` | from XML `<meta>` | Explicit class names in id order |
| `skip_outside` | `true` | Ignore `<box outside="1">` |
| `allow_unknown_labels` | `false` | Append labels absent from `class_names` |

Details:

- The XML is authoritative. Only images listed in the XML are loaded.
- Boxes only (`<box>`) are supported. Polygons, masks, and tracks are ignored.
- If the XML has no label metadata, provide `class_names`.
- `skip_outside: true` matches CVAT's "outside the frame" flag. Set it to `false`
  if you deliberately keep those boxes.
- With `allow_unknown_labels: false`, a label missing from the configured class
  list is a load error. Use this to catch schema drift.

## COCO (`coco`)

Expected input is a detection JSON file with `images`, `annotations`, and
`categories`.

```yaml
source:
  format: coco
  root: /data/coco-export
  annotations: annotations/instances_default.json  # optional; auto-detected
  images: images                                   # optional; defaults to root
  class_names: [car, person]                       # optional; otherwise categories
  skip_crowd: true                                 # skip iscrowd=1 annotations
```

| Option | Default | Meaning |
| --- | --- | --- |
| `annotations` | auto-detected | JSON file, relative to `root` |
| `images` | `root` | Image directory, relative to `root` |
| `class_names` | from `categories` | Explicit class names in id order |
| `skip_crowd` | `true` | Ignore annotations with `iscrowd=1` |

Details:

- If `annotations` is omitted, the loader looks in `annotations/*.json`, then in
  `*.json` under the root.
- COCO `bbox` values are `x, y, width, height`; they are converted to
  `x1, y1, x2, y2`.
- Category ids are mapped to dense class indices in category order. If you need a
  specific ordering, pass `class_names`.
- Images with no annotations are loaded, so background images are preserved.

## YOLO (`yolo`)

Expected input is an Ultralytics-style directory:

```text
dataset/
  data.yaml            # optional
  images/
    train/...
    val/...
    test/...
  labels/
    train/...
    val/...
    test/...
```

```yaml
source:
  format: yolo
  root: /data/yolo-dataset
  images: images          # optional; defaults to root/images or root
  labels: labels          # optional; default mirrors the images path
  data_yaml: data.yaml    # optional; auto-detected
  class_names: [car, person]  # optional; overrides data.yaml names
  infer_splits: true      # infer train/val/test from the first path component
```

| Option | Default | Meaning |
| --- | --- | --- |
| `images` | `root/images` or `root` | Image directory |
| `labels` | derived from `images` | Label directory |
| `data_yaml` | auto-detected | Class names source |
| `class_names` | from `data.yaml`/`nc` | Explicit class names in id order |
| `infer_splits` | `true` | Read the split from the first path component |

Details:

- The labels directory is derived by replacing the `images` path component with
  `labels`. Both `root/images/train` and `root/labels/train` work.
- Class names come from `class_names`, then `data.yaml:names`, then `data.yaml:nc`
  (as `class_0`, `class_1`, ...).
- Missing label files are reported and can be fixed with
  `create_missing_labels`.
- Label lines that cannot be parsed become load errors, which `validate` reports.
- `infer_splits: false` leaves every image unassigned so a `split` block can
  reassign them.

## Label conventions

- One label file per image, paired by stem.
- Label files live beside their images under a mirrored directory tree.
- Empty label files mean "background image" (no objects).
- A missing label file is different from an empty one. Use
  `create_missing_labels` only when missing really means background.
- YOLO pairs images and labels by stem, so `a.jpg` and `a.png` cannot coexist in
  the same directory and split. `validate` reports these collisions.
- Label files in the wrong mirrored location are reported as orphans. The
  `validate` block can ignore, warn, or fail on them.

## Validation before conversion

Run `validate` immediately after loading when changing formats. It checks:

- image/label pairing and missing labels;
- class ids inside the class list;
- zero-area or reversed boxes;
- boxes outside the image bounds;
- duplicate image paths;
- label-stem collisions;
- orphan label files.

See [Blocks](blocks.md#validate-alias-check_dataset) for parameters.
