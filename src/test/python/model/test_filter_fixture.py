"""Round-trip the ``sample_filter`` fixture to prove the fixture is usable."""

import json

from model.codec import filter_from_json
from model.iir import CompleteFilter


def test_sample_filter_round_trip(sample_filter):
    payload = json.dumps(sample_filter.to_json())
    decoded = filter_from_json(json.loads(payload))
    assert isinstance(decoded, CompleteFilter)
    assert decoded.description == 'sample fixture'
    assert len(decoded.filters) == len(sample_filter.filters)
    for original, restored in zip(sample_filter.filters, decoded.filters):
        assert isinstance(restored, original.__class__)
        assert restored.fs == original.fs
