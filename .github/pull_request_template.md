## Summary

<!-- What changes and why? Link the issue if there is one. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New block or source format
- [ ] Documentation
- [ ] Refactor / internal cleanup
- [ ] Breaking change

## Testing

<!-- Commands you ran and the relevant result. -->

```bash
uv run ruff check src tests
uv run pytest
```

## Checklist

- [ ] Tests cover the new behavior or regression.
- [ ] Documentation is updated (`README.md`, `docs/`).
- [ ] New blocks have a clear name, aliases, and parameter validation.
- [ ] The block does not modify source dataset files.
- [ ] Optional dependencies were added to the correct extra.
- [ ] No private dataset paths or secrets are included.
