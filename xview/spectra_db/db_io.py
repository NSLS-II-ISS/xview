from bluesky_live.run_builder import RunBuilder



from event_model import compose_run
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

    with RunBuilder(metadata=metadata) as builder:
        # builder = RunBuilder(metadata=metadata)
        run = builder.get_run()
        builder.add_stream(
            "primary",
            data=data
        )
    # builder.close()
    for name, doc in run._document_cache._ordered:
        # TODO Use public API when available.
        serializer(name, doc)
    return run.metadata['start']['uid']




uri = "mongodb://xf08id-ca1:27017/dev_analyzed_data"

def save_spectrum_to_db(metadata, data):
    ser = Serializer(uri, uri)
    uid = _save_spectrum_to_db(ser, metadata, data)
    return uid




# from distutils.version import LooseVersion
# if LooseVersion(databroker.__version__) >= LooseVersion('1.0.0'):
#     from databroker._drivers.mongo_normalized import BlueskyMongoCatalog
#     def get_spectrum_catalog():
#         return BlueskyMongoCatalog(uri, uri)
# else:
def get_spectrum_catalog():
    from databroker import Broker
    return Broker.named("iss_dev_analyzed_data")


        # return BlueskyMongoCatalog(uri, uri)






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