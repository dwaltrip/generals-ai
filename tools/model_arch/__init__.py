"""SVG diagram generation for the model-architecture doc.

Layered: `svgkit` holds generic drawing primitives (boxes, anchors, arrows)
with no model knowledge; `diagrams` composes them into named diagram builders
that take plain-data specs; spec derivation from the live model lives in its
own module so torch stays out of the drawing code's import graph.

Entry point: `tools/gen_model_arch_doc.py`.
"""
