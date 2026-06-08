




# label outer middle inner
# WIDTHS=(
#   "0.5x:64:64:80"
#   "1x:128:128:160"
#   "1.5x:192:192:240"
#   "2x:256:256:320"
# )

# 1x = widths used by DeepNash
# label outer middle inner
WIDTHS = [
    ("1/2-width", 128, 128, 160),
    ("5/8-wdith", 160, 160, 200),
    ("3/4-width", 192, 192, 240),
]

BATCH_SIZE = [256, 512, 1024]


num_workers = 4


OBS_TENSOR_VARIANTS = [
    "DENSE_HISTORY_N=5",
    "DENSE_HISTORY_N=10",
    "DENSE_HISTORY_N=20",
]
