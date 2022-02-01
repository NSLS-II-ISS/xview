import copy

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

##########





# def merge_scans(scans):
#     merged_scan = copy.deepcopy(scans[0])
#     n_scans = len(scans)
#     for i in range(1, n_scans):
#         scan = scans[i]
#         energy_match = False
#         if merged_scan.energy.size == scan.energy.size:
#             if np.all(np.isclose(merged_scan.energy, scan.energy)):
#                 energy_match = True
#
#         if energy_match:
#             merged_scan.images += scan.images
#             merged_scan.muf += scan.muf
#             merged_scan.rixs += scan.rixs
#         else:

# file_list = [f for f in os.listdir(folder) if (f.startswith('TiO2 overnight 0') and f.endswith('.dat'))]
# file_list = [f for f in os.listdir(folder) if (f.startswith('MIL125-NH2_RXES 0') and f.endswith('.dat')) and (f != 'MIL125-NH2_RXES 0001.dat')]


def get_rixs_plane_for_sample(folder, fname_base, fname_calib, return_calibration=False):

    mycalib = VonHamosCalibration(db, fname_calib)
    mycalib.set_roi(89, 7, 115, 235)
    mycalib.show_roi()
    mycalib.integrate_images()
    mycalib.calibrate()

    file_list = [f for f in os.listdir(folder) if (f.startswith(fname_base) and f.endswith('.dat'))]

    scans = []
    for file in file_list:
        scan = VonHamosScan(db, folder + file)
        scan.set_roi(89, 7, 115, 235)
        scan.show_roi()
        scan.integrate_images()
        scan.append_calibration(mycalib)

        scans.append(scan)




    rixs_mean = np.mean(np.array([scan.rixs for scan in scans]), axis=0)
    energy = scans[0].energy
    emission_energy = scans[0].emission_energy
    rixs_mean /= rixs_mean.max()
    try:
        energy_transfer, rixs_transfer = convert_rixs_to_energy_transfer(energy, emission_energy, rixs_mean)
    except:
        energy_transfer, rixs_transfer = None, None
    if return_calibration:
        return energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer, mycalib
    else:
        return energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer


folder = '/nsls2/xf08id/users/2021/3/308282/'
tio2 = get_rixs_plane_for_sample(folder, 'TiO2 overnight 0', f'{folder}TiO2 overnight calibration 0001.dat')
mil125_2fe_nh2 = get_rixs_plane_for_sample(folder, '2FeMIL125-NH2_RXES 0',  f'{folder}2FeMIL125-NH2_calibration 0001.dat')
mil125_nh2 = get_rixs_plane_for_sample(folder, 'MIL125-NH2_RXES 0',  f'{folder}MIL125-NH2_calibration 0001.dat')
mil177lt = get_rixs_plane_for_sample(folder, 'MIL177LT_RXES 0',  f'{folder}MIL177LT_calibration 0001.dat')

tio2_vtc = get_rixs_plane_for_sample(folder, 'TiO2 VTC 5250eV 0', f'{folder}TiO2 overnight calibration 0001.dat')
mil125_2fe_nh2_vtc = get_rixs_plane_for_sample(folder, '2FeMIL125-NH2_VTC 0',  f'{folder}2FeMIL125-NH2_calibration 0001.dat')
mil125_nh2_vtc = get_rixs_plane_for_sample(folder, 'MIL125-NH2_VTC 0',  f'{folder}MIL125-NH2_calibration 0001.dat')



# fname = r'/nsls2/xf08id/users/2021/3/308282/TiO2 overnight 0060.dat'
#
# myscan = VonHamosScan(db, fname)
#
#
# myscan.set_roi(89, 7, 150, 180)
# myscan.show_roi()
# myscan.integrate_images()
# myscan.append_calibration(mycalib)

# from xas.spectrometer import convert_rixs_to_energy_transfer
# energy_transfer, rixs_transfer = convert_rixs_to_energy_transfer(energy, emission_energy, rixs_mean)

plt.figure(5)
plt.clf()

# plt.contourf(myscan.energy, myscan.emission_energy, myscan.rixs)
# plt.contourf(energy, emission_energy, rixs_mean, 51, vmin=0, vmax=1)
plt.contourf(energy, energy_transfer, rixs_transfer.T, 251, vmin=0, vmax=0.5)
plt.axis('square')

plt.xlim(4965, 4985)
plt.ylim(20, 55)


###

plt.figure(5)
plt.clf()

# plt.contourf(myscan.energy, myscan.emission_energy, myscan.rixs)
# plt.contourf(energy, emission_energy, rixs_mean, 51, vmin=0, vmax=1)
# plt.contourf(tio2[0], tio2[3], tio2[4].T, 251, vmin=0, vmax=0.5)
# plt.contourf(mil125_2fe_nh2[0], mil125_2fe_nh2[3], mil125_2fe_nh2[4].T, 251, vmin=0, vmax=0.3)
# plt.contourf(mil125_nh2[0], mil125_nh2[3], mil125_nh2[4].T, 251, vmin=0, vmax=0.3)
plt.contourf(mil125_nh2[2].T, 251, vmin=0, vmax=0.3)
plt.axis('square')

plt.xlim(4965, 4985)
plt.ylim(20, 55)



def plot_cee_cut(input_tuple, cut_index, label=''):
    energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer = input_tuple
    cut = rixs_mean.T[:, cut_index].copy()
    cut /= np.mean(cut[energy > 4987.5])
    plt.plot(energy, cut, label=label)

def plot_cie_cut(input_tuple, cut_index, cut_cee=134, label=''):
    energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer = input_tuple
    cut_cee = rixs_mean.T[:, cut_cee]
    normalization = np.mean(cut_cee[energy > 4987.5])
    cut_cie = rixs_mean.T[cut_index, :]/normalization
    plt.plot(emission_energy, cut_cie, label=label)


def plot_contourf(input_tuple, subplot_idx):
    energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer = input_tuple
    plt.subplot(subplot_idx)
    plt.contourf(energy, emission_energy, rixs_mean, 251, vmin=0, vmax=0.3)
    plt.axis('square')

def plot_plane_with_cuts(input_tuple, cee_energy, cie_energy, subplot_idx=221, label=''):
    energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer = input_tuple

    cee_index = np.argmin(np.abs(emission_energy - cee_energy))
    cie_index = np.argmin(np.abs(energy - cie_energy))

    # cee_index = np.where(np.isclose(cee_energy, emission_energy, atol=np.min(np.abs(np.diff(emission_energy)))))[0]
    # cie_index = np.where(np.isclose(cie_energy, energy, atol=np.min(np.abs(np.diff(energy)))))[0]

    plot_contourf(input_tuple, subplot_idx)
    XLIM = (4965, 4985)
    YLIM = (4920, 4945)
    # YLIM = (20, 55)


    plt.xlim(XLIM)
    plt.ylim(YLIM)

    # print(energy[cee_index])
    plt.hlines([emission_energy[cee_index]], XLIM[0], XLIM[1], colors='r')
    plt.vlines([energy[cie_index]], YLIM[0], YLIM[1], colors='r')

    plt.subplot(223)

    plot_cee_cut(input_tuple, cee_index, label=label)
    plt.xlim(XLIM)


    plt.subplot(224)
    plot_cie_cut(input_tuple, cie_index, label=label)
    plt.xlim(YLIM)


def plot_comparison_of_many_tuples(input_tuples, cee_energy, cie_energy, labels=None):
    n = len(input_tuples)
    if labels is None: labels = ['']*n
    plt.figure(6)
    plt.clf()
    for i in range(n):
        input_tuple = copy.deepcopy(input_tuples[i])
        subplot_idx = int(f'2{n}{i+1}')
        plot_plane_with_cuts(input_tuple, cee_energy, cie_energy, subplot_idx=subplot_idx, label=labels[i])

        plt.subplot(subplot_idx)
        plt.xlabel('Incident energy, eV')
        plt.ylabel('Emission energy, eV')
        plt.title(labels[i])

    plt.subplot(223)
    plt.xlabel('Incident energy, eV')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(224)
    plt.xlabel('Emission energy, eV')
    plt.ylabel('Intensity')
    plt.legend()


plot_comparison_of_many_tuples([mil125_nh2, mil125_2fe_nh2, mil177lt, tio2], 4932.5, 4969, labels=['MIL125_NH2', 'MIL125_2Fe_NH2', 'MIL177LT', 'TiO2a'])




def plot_vtc_tuple(input_tuples, labels=None):
    n = len(input_tuples)
    if labels is None: labels = [''] * n
    plt.figure(7)
    plt.clf()
    for i in range(n):
        input_tuple = copy.deepcopy(input_tuples[i])
        _, emission_energy, xes, _, _ = input_tuple

        plt.plot(emission_energy, xes.ravel(), label=labels[i])

    plt.legend()

plot_vtc_tuple([mil125_nh2_vtc, mil125_2fe_nh2_vtc, tio2_vtc], labels=['MIL125_NH2', 'MIL125_2fe_NH2', 'TiO2a'])


# plot_plane_with_cuts(mil125_nh2, 134, 46, subplot_idx=221)
# plot_plane_with_cuts(mil125_2fe_nh2, 134, 46, subplot_idx=222)

# plt.figure(6)
# plt.clf()
#
# cut = 134
# plot_cee_cut(mil125_nh2, cut)
# plot_cee_cut(mil125_2fe_nh2, cut)
# plt.contourf(myscan.energy, myscan.emission_energy, myscan.rixs)
# plt.contourf(energy, emission_energy, rixs_mean, 51, vmin=0, vmax=1)
# plt.contourf(tio2[0], tio2[3], tio2[4].T, 251, vmin=0, vmax=0.5)
# plt.contourf(mil125_2fe_nh2[0], mil125_2fe_nh2[3], mil125_2fe_nh2[4].T, 251, vmin=0, vmax=0.3)
# plt.plot(mil125_nh2[0], mil125_nh2[2].T[:, cut])
# plt.plot(mil125_nh2[0], mil125_2fe_nh2[2].T[:, cut])

# plt.axis('square')

plt.xlim(4965, 4985)
# plt.ylim(20, 55)




def output_av_rixs(input_tuple, fpath):
    energy, emission_energy, rixs_mean, energy_transfer, rixs_transfer = input_tuple
    # data = np.hstack(([0], energy))
    # data = np.vstack((emission_energy[None, :], rixs_mean))
    data = np.hstack((np.hstack(([0], energy))[:, None], np.vstack((emission_energy[None, :], rixs_mean.T))))
    print(energy.shape, emission_energy.shape, rixs_mean.T.shape, data.shape)
    np.savetxt(fpath, data, delimiter='\t')


output_av_rixs(tio2, '/nsls2/xf08id/users/2021/3/308282/tio2_rixs_averaged.dat')
output_av_rixs(mil125_nh2, '/nsls2/xf08id/users/2021/3/308282/mil125_nh2_rixs_averaged.dat')
output_av_rixs(mil125_2fe_nh2, '/nsls2/xf08id/users/2021/3/308282/2fe_mil125_nh2_rixs_averaged.dat')
output_av_rixs(mil177lt, '/nsls2/xf08id/users/2021/3/308282/mil177lt_rixs_averaged.dat')



pix_hi = mycalib.pixel_cen + mycalib.pixel_fwhm/2
pix_lo = mycalib.pixel_cen - mycalib.pixel_fwhm/2
en_hi = mycalib.energy_converter.nom2act(pix_hi)
en_lo = mycalib.energy_converter.nom2act(pix_lo)
en_res = np.abs(en_hi - en_lo)

plt.figure(79)
plt.clf()

plt.plot(mycalib.energy, en_res)
plt.plot(mycalib.energy, np.sqrt(en_res**2 - (1.3e-4*mycalib.energy)**2))

