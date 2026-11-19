# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Use GitHub's private vulnerability reporting:

<https://github.com/dcaldana12/cnn_trainer/security/advisories/new>

Include the affected version, a minimal reproduction, and the impact you expect.
You should receive an acknowledgement within a few days. Once a fix is ready, a
security advisory and a patched release will be published.

## Scope

`cnn-trainer` reads datasets from paths supplied by the user. The most relevant
areas are:

- path traversal or symlink escapes while loading or exporting datasets;
- unsafe archive extraction or file overwrite behavior;
- untrusted YAML or JSON causing code execution rather than a configuration error;
- dependency vulnerabilities in optional extras.

Reports about the training library itself (Ultralytics) or the Python
dependencies should go to those projects, but a note here is welcome if
`cnn-trainer` can mitigate the issue.

## Safe usage

- Review a pipeline before running it against a valuable dataset.
- Run with `--dry-run` to inspect resolved blocks and parameters.
- Keep `max_fraction` and `dry_run` enabled until you have reviewed the
  redundancy report.
- Do not load YAML or JSON files from untrusted sources. The loader uses
  `yaml.safe_load`, but dataset paths and labels are still user-controlled input.
