"""
Output heads reading off the trunk's `[B, C, H, W]` spatial embedding — one
module per head. `BCModel` composes them and routes each head's output into
its forward dict; the loss code weights each head's term independently.
"""
