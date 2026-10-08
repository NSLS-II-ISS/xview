"""Adapt scan metadata across Larch header capitalization changes."""


def scan_metadata(attrs):
    """Preserve header keys and provide a canonical edge energy when present."""
    metadata = dict(attrs)
    fields = {key.casefold(): value for key, value in metadata.items()}
    if "ISS (8-ID)" in metadata.values():
        edge_key = "scan_edge_energy_ev"
    elif "QAS" in metadata.values():
        edge_key = "e0"
    else:
        return metadata
    if edge_key in fields:
        metadata["e0"] = fields[edge_key]
    return metadata
