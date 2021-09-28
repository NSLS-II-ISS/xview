import h5py

from pathlib import Path
from xas.file_io import load_binned_df_from_file



bender_current_position = bender.pos.user_readback.get()
bender_positions = bender_current_position + np.arange(-15, 20, 5)
x = xlive_gui.widget_run

for bender_position in bender_positions:
    RE(bps.mv(bender.pos, bender_position))
    RE(bps.sleep(3))
    loading = bender.load_cell.get()
    x.parameter_values[0].setText(f'Cu foil - {loading} N - {bender_position} um')
    x.run_scan()


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


'''
Test RIXS normalization using CIE cuts correction
'''

x = xview_gui.widget_rixs
x.parse_rixs_scan(xes_normalization=False)
plt.figure()
plt.imshow(-x.rixs_dict['pil100_ROI1']/x.rixs_dict['i0'], vmin=0, vmax=2)

rixs = -(x.rixs_dict['pil100_ROI1']/x.rixs_dict['i0'])
energy_in = x.rixs_dict['energy_in']
energy_out = x.rixs_dict['energy_out']


cie_intensity = np.array([i.mu for i in xview_gui.project])
cie_energy_in = np.array([7712, 7708, 7709.5])
cie_energy_out = xview_gui.project[0].energy
cie_intensity = cie_intensity / cie_intensity[:, 70:71] * rixs[[53,37,43], 70:71]

plt.figure()
plt.plot(cie_energy_out, cie_intensity.T)

rixs_cuts = rixs[[53,37,43], :-1]
conc_norm = np.zeros(rixs_cuts.shape[1])

for i in range(conc_norm.size):
    result = np.linalg.lstsq(rixs_cuts[:, i][:, None], cie_intensity[:, i][:, None], rcond=-1)
    conc_norm[i] = result[0]

rixs_norm = rixs[:, :-1]*conc_norm[None, :]

plt.figure()
plt.contourf(energy_out[:-1], energy_in, rixs_norm)


plt.figure()
n_in = 53

plt.plot(energy_out[:-1], rixs_norm[n_in, :], label=f'RIXS 1 @ {np.round(energy_in[n_in],2)} eV')
plt.plot(energy_out[:-1], rixs_norm_[n_in, :], label=f'RIXS 2 @ {np.round(energy_in[n_in],2)} eV')

plt.legend()


################
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







