"""The argmin toy: a controlled study of how SGD learns "return the alive player
with the smallest army" from one frame, as a function of input encoding and model
architecture. Ground truth is analytic (`fq`'s `army_sim`), so the object of study
is *how* the clean solution is found, not whether the signal is present.

Data comes from one `fq` table (`spec.py`); a small table-reading loop (`train.py`)
trains the A–E scorers (`models.py`) over encodings (`encode.py`), scored by
tie-aware metrics (`metrics.py`). See docs/2026-06/6.18-1 for the plan.
"""
