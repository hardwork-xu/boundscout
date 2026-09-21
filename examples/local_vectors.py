"""A complete offline storage/query round-trip. / 完整离线存储与查询示例。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from boundscout import BoundIndex

rng = np.random.default_rng(7)
points = rng.normal(size=(2048, 16))
queries = rng.normal(size=(3, 16))
with TemporaryDirectory() as temporary:
    path = Path(temporary) / "index"
    with BoundIndex.build(points, leaf_size=128) as built:
        built.save(path)
    with BoundIndex.load(path) as loaded:
        result = loaded.search(queries, k=5)
        print(result.indices)
        print(result.distances)
