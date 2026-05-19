"""
Cross-cutting tensor-shape + pipeline constants for the training pipeline.

Shared across the dataloader, model, and augmentation code. Action-encoding
constants (direction enum, sub-channel layout, flat-index layout) live in
`actions.py` — they're owned by the encoder that defines them.

Padding convention: top-left. The unpadded board occupies rows 0..H-1 and
cols 0..W-1 of the padded grid; the right/bottom margins are filler. The
padded re-index in `actions.py` and the channel-assembly code both assume
this convention.
"""

H_PADDED = 32
W_PADDED = 32

# Drop-filter membership.
ELIGIBLE_PLAYER_COUNT = 8
MAX_BOARD_SIDE = 32  # inclusive; drop games where max(w, h) > MAX_BOARD_SIDE
