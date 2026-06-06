"""Optional progress reporting for the heavy CV drivers.

The TSI / fusion drivers wrap their costly outer loops with :func:`progress_iter`.
It is a strict **no-op unless ``progress=True``**, so the test suite and the
example generator (which never pass ``progress``) stay silent and behaviourally
identical. When enabled it uses ``tqdm.auto`` (HTML bars in Colab/Jupyter, text
bars in a terminal); if ``tqdm`` is not installed it degrades to a plain iterator
rather than failing — keeping ``src/`` free of any hard ``tqdm`` dependency.
"""

from __future__ import annotations

from typing import Iterable, Optional


def progress_iter(
    iterable: Iterable,
    progress: bool = False,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    leave: bool = True,
):
    """Wrap ``iterable`` in a tqdm bar when ``progress`` is True, else return it as-is.

    Parameters
    ----------
    iterable : Iterable
        The sequence to iterate (e.g. outer folds, permutation replicates).
    progress : bool
        Master switch. ``False`` (default) returns ``iterable`` untouched — no
        import, no overhead, identical iteration order.
    desc : str, optional
        Bar label (e.g. ``"gtzan | gains [folds]"``).
    total : int, optional
        Item count for the bar when ``iterable`` has no ``len`` (e.g. ``range``/
        generators); tqdm infers it from a list otherwise.
    leave : bool
        Whether to keep the finished bar on screen. Inner/nested bars pass
        ``leave=False`` so they don't pile up under the outer bar.
    """
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # tqdm optional: fall back to a silent plain iterator
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)
