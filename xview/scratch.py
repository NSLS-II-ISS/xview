import h5py

from pathlib import Path
from xas.file_io import load_binned_df_from_file



results =  list( db.v2.search({'element': 'Co'}))
hfile = h5py.File('/nsls2/xf08id/Sandbox/database/Co_ISS_data.h5', 'w')

for i, result in enumerate(results):
    try:
        print(i, result)
        path = db[result].start['interp_filename']
        fn = Path(path)
        filename = os.path.basename(path)
        nfn = fn.with_suffix('.dat')
        df, metadata = load_binned_df_from_file(nfn)

        energy = df['energy']
        ref = np.log(df['it']/df['ir'])
        trans = np.log(df['i0'] / df['it'])
        fluo = (df['iff'] / df['i0'])

        hfile.create_group(str(i))
        hfile.create_dataset(f'{str(i)}/energy', data = energy)
        hfile.create_dataset(f'{str(i)}/ref', data=ref)
        hfile.create_dataset(f'{str(i)}/trans', data=trans)
        hfile.create_dataset(f'{str(i)}/fluo', data=fluo)
        hfile.create_dataset(f'{str(i)}/filename', data=filename)

    except:
        print(f'{result} failed to load')

hfile.close()

'''


plotting


'''



plt.figure(666)
for k in fff.keys():
    energy = fff[k]['energy'][()]
    fluo = fff[k]['ref'][()]

    plt.plot(energy, fluo)

fff.close()

plt.figure()
for i, result in enumerate(results[1000:1100]):
    print(i, result)
    path = db[result].start['interp_filename']
    fn = Path(path)
    filename = os.path.basename(path)
    nfn = fn.with_suffix('.dat')
    df, metadata = load_binned_df_from_file(nfn)

    energy = df['energy']
    ref = np.log(df['it'] / df['ir'])
    fluo = (df['pips'] / df['i0'])
    trans = np.log(df['i0'] / df['it'])
    plt.plot(energy, trans)



'''

QAS


'''
results =  list( db.v2.search({'element': 'Cobalt ( 27'}))
hfile = h5py.File('/nsls2/xf08id/Sandbox/database/Co_data.h5', 'w')

for i, result in enumerate(results):
    try:
        print(i, result)
        path = db[result].start['interp_filename']
        fn = Path(path)
        filename = os.path.basename(path)
        nfn = fn.with_suffix('.dat')
        df, metadata = load_binned_df_from_file(nfn)

        energy = df['energy']
        ref = np.log(df['it']/df['ir'])
        trans = np.log(df['i0'] / df['it'])
        fluo = (df['pips'] / df['i0'])

        hfile.create_group(str(i))
        hfile.create_dataset(f'{str(i)}/energy', data = energy)
        hfile.create_dataset(f'{str(i)}/ref', data=ref)
        hfile.create_dataset(f'{str(i)}/trans', data=trans)
        hfile.create_dataset(f'{str(i)}/fluo', data=fluo)
        hfile.create_dataset(f'{str(i)}/filename', data=filename)

    except:
        print(f'{result} failed to load')

hfile.close()






