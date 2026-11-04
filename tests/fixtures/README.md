# Test fixtures

The synthetic geographic scenarios are generated in code by
`src/worldgen/sources/synthetic.py` (so they stay in sync with the OSM schema the
adapters parse) and are listed with `worldgen fixtures`. Each one can be built as a
complete world: `worldgen build --fixture <name> --output generated/<name>`.

To export a fixture's raw OSM-shaped input for inspection:

```python
import json
from worldgen.sources.synthetic import load_fixture
json.dump(load_fixture("bridge_river").osm.to_overpass_json(), open("bridge_river.osm.json", "w"), indent=1)
```
