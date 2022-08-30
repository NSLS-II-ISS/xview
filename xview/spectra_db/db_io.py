# from bluesky_live.run_builder import RunBuilder
#
#
#
# from event_model import compose_run
# from databroker._drivers.mongo_normalized import BlueskyMongoCatalog
# from suitcase.mongo_normalized import Serializer
# import time as ttime
# import pkg_resources
# spectrum_start_path = pkg_resources.resource_filename('xview', 'spectra_db/spectrum_start.json')
# import json
# import jsonschema
# from os import listdir
# from os.path import isfile, join, getmtime
# import numpy as np
# import matplotlib.pyplot as plt
# from xas.file_io import load_interpolated_df_from_file
#
#
# #metadata = {'Sample_name': 'Pt', 'compound': 'Pt', 'Element' : 'Pt', 'Edge' : 'L3', 'E0': 11564}
# # data = {'Energy': [1, 2, 3], 'mu_norm': [0.1, 0.2, 0.3]}
#
#
# def validate_schema(input_dict, schema_path):
#     with open(schema_path) as f:
#         contents = json.load(f)
#     jsonschema.validate(input_dict, contents)
#
#
# def generate_timestamps(keys):
#     timestamps = {}
#     current_time = ttime.time()
#     for key in keys:
#         timestamps[key] = current_time
#     return timestamps
#
#
# def _save_spectrum_to_db(serializer, metadata, data):
#
#     with RunBuilder(metadata=metadata) as builder:
#     # builder = RunBuilder(metadata=metadata)
#         run = builder.get_run()
#         builder.add_stream(
#             "primary",
#             data=data
#         )
#     for name, doc in run.documents(fill="no"):
#
#         serializer(name, doc)
#
#     return run.metadata['start']['uid']
#
#
#
#
# uri = "mongodb://xf08id-ca1:27017/dev_analyzed_data"
#
# def save_spectrum_to_db(metadata, data):
#     ser = Serializer(uri, uri)
#     uid = _save_spectrum_to_db(ser, metadata, data)
#     return uid
#
#
# class ISSBlueskyMongoCatalog(BlueskyMongoCatalog):
#
#     def search_foil_data(self, element, edge):
#         return list(self.search({'Sample_name' : f'{element} foil', 'Edge' : edge}))
#
#     def validate_foil_edge(self, element, edge):
#         uids = self.search_foil_data(element, edge)
#         if len(uids) == 0:
#             raise Exception(f'Error: No {element} foil {edge}-edge was found in the database')
#
#     def foil_spectrum(self, element, edge):
#         all_uids = self.search_foil_data(element, edge)
#         uid = all_uids[0] # latest
#         ds = self[uid].primary.read()
#         energy = ds['Energy'].values
#         mu = ds['mu_norm'].values
#         return energy, mu
#
#     def read_spectrum(self, uid):
#         run = self[uid]
#         data = run.primary.read()
#         energy = data.Energy.values
#         mu = data.mu_norm.values
#         name = run.metadata['start']['Sample_name']
#         return energy, mu, name
#
#
#
# def get_spectrum_catalog():
#     return ISSBlueskyMongoCatalog(uri, uri)
#
#
#
# ######################## file/folder handling #############################
#
# def files_in_folder(folder, ext):
#     fpaths =[join(folder, f) for f in listdir(folder) if isfile(join(folder, f))
#                                                     and f.endswith(ext)]
#     fnames = [f.split('.')[0] for f in listdir(folder) if isfile(join(folder, f))
#                                                     and f.endswith(ext)]
#     return fpaths, fnames
#
#
# def folder2db(folder, ext='xdi', names='fnames', element='', energy_idx=0, mu_norm_idx=3):
#     fpaths, fnames = files_in_folder(folder, ext)
#     for path in fpaths:
#         file2db(path, ext)
#
#
# def file2db(path, ext):#, name, element, edge, e0, energy_idx, mu_norm_idx):
#     df, header = load_interpolated_df_from_file(path)
#     # return header
#     e0, element, edge, sample_name, compound = None, None, None, None, None
#     if ext == 'xdi':
#         for line in header.split('\n'):
#             if 'Athena.e0' in line:
#                 e0 = float(line.split(' ')[-1])
#             if 'Element.symbol' in line:
#                 element = line.split(' ')[-1]
#             if 'Element.edge' in line:
#                 edge = line.split(' ')[-1]
#             if 'Sample.name' in line:
#                 sample_name = ' '.join(line.split(' ')[2:])
#             if 'Sample.name' in line:
#                 compound = ' '.join(line.split(' ')[2:])
#         print(e0, element, edge, sample_name, compound)
#
#         energy = df['e'].values
#         mu_norm = df['flat'].values
#         data = {'energy': energy, 'mu_norm': mu_norm}
#         metadata = {'Sample_name': sample_name,
#                     'compound': compound,
#                     'Element' : element,
#                     'Edge' : edge,
#                     'E0': e0}
#         save_spectrum_to_db(metadata, data)
#
#




    # :
    #     energy = df['e'].values
    #     mu_norm = df['flat'].values
    #     data = {'energy' : energy, 'mu_norm' : mu_norm}
    #     metadata = {}
    # else:
    #     return None
    # data = {'energy' : _data[:, energy_idx],
    #         'mu_norm' : _data[:, mu_norm_idx]}
    # metadata = {'Sample_name': name,
    #             'compound': name,
    #             'Element' : element,
    #             'Edge' : edge,
    #             'E0': e0}
    # plt.plot(data['energy'], data['mu_norm'])
# def data2db(energy, mu_norm, comment, element)

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