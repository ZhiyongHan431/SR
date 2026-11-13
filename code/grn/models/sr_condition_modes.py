"""Explicit token layouts for supported super-resolution condition modes."""

from dataclasses import dataclass


_MODE_FLAGS = {
    'prefix': (True, True, False),
    'prefix_text': (True, True, True),
    'spatial_control': (True, False, True),
    'native_adapter': (False, False, True),
}


@dataclass(frozen=True)
class SRTokenLayout:
    prefix_tokens: int
    text_tokens: int
    target_offset: int
    total_tokens: int


def _flags(mode: str) -> tuple[bool, bool, bool]:
    try:
        return _MODE_FLAGS[mode]
    except (KeyError, TypeError) as error:
        raise ValueError(f'unknown SR condition mode: {mode!r}') from error


def uses_continuous_lr(mode: str) -> bool:
    return _flags(mode)[0]


def uses_lr_prefix(mode: str) -> bool:
    return _flags(mode)[1]


def uses_empty_text(mode: str) -> bool:
    return _flags(mode)[2]


def condition_layout(
    mode: str,
    visual_tokens: int,
    text_tokens: int,
) -> SRTokenLayout:
    """Return the packed ``[condition?, HR, text?, time]`` token layout."""
    _, has_prefix, has_text = _flags(mode)
    if (
        not isinstance(visual_tokens, int)
        or isinstance(visual_tokens, bool)
        or visual_tokens <= 0
    ):
        raise ValueError('visual_tokens must be a positive integer')
    if (
        not isinstance(text_tokens, int)
        or isinstance(text_tokens, bool)
        or text_tokens < 0
    ):
        raise ValueError('text_tokens must be a nonnegative integer')
    if has_text and text_tokens == 0:
        raise ValueError(f'{mode} requires positive text_tokens')

    prefix_count = visual_tokens if has_prefix else 0
    text_count = text_tokens if has_text else 0
    return SRTokenLayout(
        prefix_tokens=prefix_count,
        text_tokens=text_count,
        target_offset=prefix_count,
        total_tokens=prefix_count + visual_tokens + text_count + 1,
    )
