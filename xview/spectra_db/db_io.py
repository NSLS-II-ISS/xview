from event_model import compose_run
from databroker._drivers.mongo_normalized import BlueskyMongoCatalog
from suitcase.mongo_normalized import Serializer
import time as ttime
import pkg_resources

spectrum_start_path = pkg_resources.resource_filename('xview', 'spectra_db/spectrum_start.json')
import json
import jsonschema
#metadata = {'Sample_name': 'Pt', 'compound': 'Pt', 'Element' : 'Pt', 'Edge' : 'L3', 'E0': 11564}
# data = {'Energy': [1, 2, 3], 'mu_norm': [0.1, 0.2, 0.3]}
# timestamps = {'Energy': 0, 'mu_norm': 0}

def validate_schema(input_dict, schema_path):
    with open(schema_path) as f:
        contents = json.load(f)
    jsonschema.validate(input_dict, contents)


def generate_timestamps(keys):
    timestamps = {}
    current_time = ttime.time()
    for key in keys:
        timestamps[key] = current_time
    return timestamps


def _save_spectrum_to_db(serializer, metadata, data):
    bundle = compose_run(metadata=metadata)
    output_start = bundle.start_doc
    validate_schema(output_start, spectrum_start_path)
    serializer('start', output_start)
    bundle_descriptor = bundle.compose_descriptor(data_keys={'Energy': {'dtype': 'array', 'source': '', 'shape': [-1]},
                                                             'mu_norm': {'dtype': 'array', 'source': '', 'shape': [-1]}},
                                                  name='primary')
    output_descriptor = bundle_descriptor.descriptor_doc
    serializer('descriptor', output_descriptor)
    output_event = bundle_descriptor.compose_event(data=data,
                                                   timestamps=generate_timestamps(data.keys()))
    serializer('event', output_event)
    output_stop = bundle.compose_stop()
    serializer('stop', output_stop)
    return output_start['uid']


uri = "mongodb://xf08id-ca1:27017/dev_analyzed_data"

def save_spectrum_to_db(metadata, data):
    ser = Serializer(uri, uri)
    uid = _save_spectrum_to_db(ser, metadata, data)
    return uid


def get_spectrum_catalog():
    return BlueskyMongoCatalog(uri, uri)



# uri = "mongodb://xf08id-ca1:27017/dev_analyzed_data"
# catalog = BlueskyMongoCatalog(uri, uri)
# catalog
# catalog[-1]
# catalog[-1].primary
# catalog[-1].primary.read()
# len(catalog.search({'Element':'Pt'}))
# len(catalog.search({'Element':'Co'}))
# catalog[-1].metadata
# len(catalog.search({'E0':{'$lt' : 10000}}))
# len(catalog.search({'E0':{'$gt' : 10000}}))