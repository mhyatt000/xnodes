# Antipatterns

## 1) Hiding import errors with local fallback classes

Bad pattern:

```python
try:
    import mymodule
except ModuleNotFoundError:
    class MyStub:
        def step(self, *_args, **_kwargs):
            raise NotImplementedError
```

Why this is harmful:

- It hides real dependency failures (missing package, wrong module path, bad environment).
- It creates a fake type hierarchy that does not match production behavior.
- It lets tests pass in an invalid environment and fail later in harder-to-debug places.
- It violates explicitness: readers cannot tell if the dependency is required or optional.

Preferred pattern (required dependency):

```python
from mymodule.submodule import ARealClass
```

If a dependency is truly optional, fail explicitly using the natural ModuleNotFoundError at the boundary:

```python
from mymodule.submodule import ARealClass
```

Review checklist:

- avoid `except ModuleNotFoundError` since we dont need to reraise existing errors.
- Never replace missing runtime dependencies with fake compatibility classes.
- Error messages should tell the operator exactly what package is missing and what feature needs it.
