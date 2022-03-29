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

##############

x = xview_gui.project
k = x[-1].k
chi = x[-1].chi
window = x[-1].kwin

plt.figure(1, clear=True)
plt.subplot(211)
plt.plot(k, k**2 * chi)
plt.plot(k, window)

r = np.fft.fftfreq(k.size*10, d=(k[1] - k[0])/np.pi)
chi_r = np.fft.fft(k**2 * chi * window, n=k.size*10) / (np.sqrt(k.size-2)*2)

plt.subplot(212)
plt.plot(r, np.abs(chi_r))
plt.plot(x[-1].r, x[-1].chir_mag)


I = np.eye(chi.size)
Afft = np.fft.fft(I * window[:, None], n=k.size*10) / (np.sqrt(k.size-2)*2)

r_mask = (r>0) & (r<10)
Afft = Afft[:, r_mask]

Afft_re = np.real(Afft)
Afft_im = np.imag(Afft)

Afft_reim = np.hstack((Afft_re, Afft_im))

Ainv = np.linalg.pinv(Afft_reim)

chi_r_inv = Ainv @ (k**2 * chi)

N = np.sum(r_mask)
chi_r_inv_re = chi_r_inv[:N]
chi_r_inv_im = chi_r_inv[N:]

chi_r_inv_abs = np.sqrt(chi_r_inv_re**2 + chi_r_inv_im**2)


plt.plot(r[r_mask], chi_r_inv_abs)


##############

def cov2corr(C):
    err = np.sqrt(np.diag(C))
    return np.diag(1 / err) @ C @ np.diag(1 / err)

x = xview_gui.project

energy = x[0].energy
k = x[0].k
r = x[0].r
e0 = x[0].e0
mus = np.array([ds.flat for ds in x]).T
chis = np.array([ds.chi for ds in x]).T

chirs = np.array([np.hstack((ds.chir_re, ds.chir_im)) for ds in x]).T
chirmags = np.array([ds.chir_mag for ds in x]).T

n = mus.shape[1]

mus_av = np.mean(mus, axis=1)
cov_ee = np.cov(mus)/(n - 1)
mus_err = np.sqrt(np.diag(cov_ee))
cor_ee = cov2corr(cov_ee)

chis_av = np.mean(chis, axis=1)
cov_kk = np.cov(chis)/(n - 1)
chis_err = np.sqrt(np.diag(cov_kk))
cor_kk = cov2corr(cov_kk)

# chirmags_av = np.mean(chis, axis=1)
chirs_av = np.mean(chirs, axis=1)
cov_rr = np.cov(chirs)/(n - 1)
chirs_err = np.sqrt(np.diag(cov_kk))
cor_rr = cov2corr(cov_rr)

def band_matrix(A, d=1):
    A_out = A.copy()
    a, b = A.shape
    for i in range(a):
        for j in range(b):
            if np.abs((j-i)) > d:
                A_out[i, j] = 0

    return A_out


def get_score(x, t, tmin=0, tmax=1):
    x_av = np.mean(x, axis=1)
    _nn = x.shape[1]
    # cov = np.cov(x) / (_nn - 1)
    cov = np.cov(x) / (_nn - 1)
    x_err = np.sqrt(np.diag(cov))
    t_mask = (t > tmin) & (t < tmax)

    x_norm = (x_av / x_err)

    # rho = 0.5
    # cov_cond = (1 - rho) * cov + rho * np.diag(np.diag(cov))
    # L = np.linalg.cholesky(np.linalg.pinv(cov_cond))
    # x_norm = L.T @ x_av

    # cov_cond = band_matrix(cor_kk, d=1)
    # rho = 0.5
    # cov_cond = (1 - rho) * cov_cond + rho * np.diag(np.diag(cov_cond))
    # L = np.linalg.cholesky(np.linalg.pinv(cov_cond))
    # x_norm = L.T @ x_av

    score = np.sqrt(np.sum((x_norm[t_mask]) ** 2) / np.sum(t_mask))
    # score = np.sum((x_norm[t_mask]) ** 2) / np.sum(t_mask)
    # score = np.sum((1/x_err[t_mask]) ** 2) / ((np.sum(t_mask) - 1))
    return score, x_av, x_err

# n_curves = np.arange(3, n, 20)
vv = 10**np.linspace(np.log10(2), np.log10(n), 55)
n_curves = np.array(np.ceil(vv), dtype=int)

# scores_1 = np.zeros(n_curves.shape)
# scores_2 = np.zeros(n_curves.shape)
# scores_3 = np.zeros(n_curves.shape)
# scores_4 = np.zeros(n_curves.shape)


n_tries = 50
scores_1 = np.zeros((n_curves.size, n_tries))
scores_2 = np.zeros((n_curves.size, n_tries))
scores_3 = np.zeros((n_curves.size, n_tries))
scores_4 = np.zeros((n_curves.size, n_tries))


for i, _n in enumerate(n_curves):
    for j in range(n_tries):
        idx_choice = np.random.choice(n, _n)
        scores_1[i, j], _, _ = get_score(chis[:, idx_choice], k, 3, 6)
        scores_2[i, j], _, _ = get_score(chis[:, idx_choice], k, 6, 9)
        scores_3[i, j], _, _ = get_score(chis[:, idx_choice], k, 9, 12)
        scores_4[i, j], _, _ = get_score(chis[:, idx_choice], k, 12, 16)

    # scores_1[i], _, _ = get_score(mus[:, :_n], energy, 9600, 9700)
    # scores_2[i], _, _ = get_score(mus[:, :_n], energy, 9700, 9900)
    # scores_3[i], _, _ = get_score(mus[:, :_n], energy, 9900, 10200)

# _, x_av_040, x_err_040 = get_score(chis[:, :40], k, 9, 12)
# _, x_av_075, x_err_075 = get_score(chis[:, :75], k, 9, 12)
# _, x_av_120, x_err_120 = get_score(chis[:, :120], k, 9, 12)

def fit_power(t, x):
    logt = np.log(t)
    logx = np.log(x)
    p = np.polyfit(logt, logx, 1)
    x_fit = np.exp(np.polyval(p, logt))
    return p, x_fit


bias_cor = (1 -
            1 / (4 * n_curves) -
            7 / (32 * n_curves **2) -
            19 / (128 * n_curves**3))

n_curves_sel = n_curves > 20

n_curves_fit = np.arange(1, n_curves.max())
p1, _ = fit_power(np.tile(n_curves[n_curves_sel], n_tries), scores_1[n_curves_sel, :].T.ravel())
scores_1_fit = np.exp(np.polyval(p1, np.log(n_curves_fit)))
p2, _ = fit_power(np.tile(n_curves[n_curves_sel], n_tries), scores_2[n_curves_sel, :].T.ravel())
scores_2_fit = np.exp(np.polyval(p2, np.log(n_curves_fit)))
p3, _ = fit_power(np.tile(n_curves[n_curves_sel], n_tries), scores_3[n_curves_sel, :].T.ravel())
scores_3_fit = np.exp(np.polyval(p3, np.log(n_curves_fit)))
p4, _ = fit_power(np.tile(n_curves[n_curves_sel], n_tries), scores_4[n_curves_sel, :].T.ravel())
scores_4_fit = np.exp(np.polyval(p4, np.log(n_curves_fit)))




ALPHA = 0.5
plt.figure(5, clear=True)
# plt.loglog(n_curves, scores_1 / bias_cor[:, None], 'k.-', alpha=ALPHA)
plt.loglog(n_curves, scores_1, 'b.-', alpha=ALPHA)
plt.loglog(n_curves, scores_2, 'r.-', alpha=ALPHA)
plt.loglog(n_curves, scores_3, 'm.-', alpha=ALPHA)
plt.loglog(n_curves, scores_4, 'g.-', alpha=ALPHA)


plt.plot(n_curves_fit, scores_1_fit, 'k-')
plt.plot(n_curves_fit, scores_2_fit, 'k-')
plt.plot(n_curves_fit, scores_3_fit, 'k-')
plt.plot(n_curves_fit, scores_4_fit, 'k-')


from xas.file_io import load_interpolated_df_from_file
from xas.bin import bin as bin_func
from xas.xasproject import XASDataSet

df_bin, _ = load_interpolated_df_from_file("/nsls2/xf08id/users/2022/1/309682/s061-TiO2-0p5Vref_5sccmN2_0Msty_25C (pos 021) Rh-K  0001.dat")
df_raw, _ = load_interpolated_df_from_file("/nsls2/xf08id/users/2022/1/309682/s061-TiO2-0p5Vref_5sccmN2_0Msty_25C (pos 021) Rh-K  0001.raw")

def get_e_mu_from_df(df):
    _e = df.energy.values
    _mu = -np.log(df.ir / df.i0).values
    return _e, _mu

e_bin, mu_bin = get_e_mu_from_df(df_bin)
e_raw, mu_raw = get_e_mu_from_df(df_raw)

def bin_process(df_interp):
    return  bin_func(df_interp, 23220, edge_start=-30, edge_end=50, preedge_spacing=5,
                xanes_spacing=-1, exafs_k_spacing=0.04, skip_binning=False)

def sequential_boot(df_interp, n_tries=700):

    mus = np.zeros((x[0].energy.size, n_tries))
    chis = np.zeros((x[0].k.size, n_tries))
    chirs = np.zeros((x[0].r.size*2, n_tries))

    npt_raw = df_interp.shape[0]
    for i in range(n_tries):
        idx = np.sort(np.random.choice(npt_raw, npt_raw))
        df_interp_resampled = df_interp.loc[idx]
        _df_bin = bin_process(df_interp_resampled)
        e_bin_boot, mu_bin_boot = get_e_mu_from_df(_df_bin)
        ds = XASDataSet(name=f'bin_boot_{i}', energy=e_bin_boot, mu=mu_bin_boot, xasdataset=x[0], process=False)
        ds.normalize_force()
        ds.extract_chi_force()
        ds.extract_ft_force(window=None)

        mus[:, i] = ds.flat
        chis[:, i] = ds.chi
        chirs[:, i] = np.hstack((ds.chir_re, ds.chir_im))

    return mus, chis, chirs

mus_boot, chis_boot, chirs_boot = sequential_boot(df_raw)

n_boot = mus_boot.shape[1]
cov_ee_boot = np.cov(mus_boot) / (n_boot - 1)
cov_kk_boot = np.cov(chis_boot) / (n_boot - 1)
cov_rr_boot = np.cov(chirs_boot) / (n_boot - 1)

cor_ee_boot = cov2corr(cov_ee_boot)
cor_kk_boot = cov2corr(cov_kk_boot)
cor_rr_boot = cov2corr(cov_rr_boot)

mus_boot_err =   np.sqrt(np.diag(cov_ee_boot))
chis_boot_err =  np.sqrt(np.diag(cov_kk_boot))
chirs_boot_err = np.sqrt(np.diag(cov_rr_boot))

plt.figure(77, clear=True)
plt.subplot(211)
plt.plot(energy, mus_boot, 'k-', alpha=0.5)
plt.plot(energy, mus_av)

plt.subplot(212)
plt.plot(k, k[:, None]**2 * chis_boot, 'k-', alpha=0.5)
plt.plot(k, k**2 * chis_av)

plt.figure(78, clear=True)
plt.subplot(121)
plt.plot(energy, mus_err)
plt.plot(energy, mus_boot_err)

plt.subplot(122)
plt.plot(k, chis_err)
plt.plot(k, chis_boot_err)


plt.figure(79, clear=True)
plt.subplot(211)
plt.matshow(cor_ee, fignum=0)

plt.subplot(212)
plt.matshow(cor_ee_boot, fignum=0)

plt.figure(80, clear=True)
plt.subplot(211)
plt.matshow(cor_kk, fignum=0)

plt.subplot(212)
plt.matshow(cor_kk_boot, fignum=0)


# chirmags = np.array([ds.chir_mag for ds in x]).T

# n = mus.shape[1]
#
# mus_av = np.mean(mus, axis=1)
# cov_ee = np.cov(mus) / (n - 1)
# mus_err = np.sqrt(np.diag(cov_ee))
# cor_ee = cov2corr(cov_ee)
#
# chis_av = np.mean(chis, axis=1)
# cov_kk = np.cov(chis) / (n - 1)
# chis_err = np.sqrt(np.diag(cov_kk))
# cor_kk = cov2corr(cov_kk)
#
# # chirmags_av = np.mean(chis, axis=1)
# chirs_av = np.mean(chirs, axis=1)
# cov_rr = np.cov(chirs) / (n - 1)
# chirs_err = np.sqrt(np.diag(cov_kk))
# cor_rr = cov2corr(cov_rr)


# plt.plot(n_curves, scores_1, '.-')
# plt.plot(n_curves, scores_2, '.-')
# plt.plot(n_curves, scores_3, '.-')

plt.figure(6, clear=True)
plt.errorbar(k, k**2 * x_av_040, k**2 * x_err_040)
plt.errorbar(k, k**2 * x_av_075, k**2 * x_err_075)
plt.errorbar(k, k**2 * x_av_120, k**2 * x_err_120)

plt.figure(7, clear=True)
plt.plot(k, k**2 * x_err_040)
plt.plot(k, k**2 * x_err_075)
plt.plot(k, k**2 * x_err_120)



# score = chis_av[:, None].T @ np.linalg.pinv(cov_kk_cond) @ chis_av[:, None]/k.size
# np.linalg.matrix_rank(cov_kk)

# uu,ss,vv = np.linalg.svd(mus)


rho = 0.5
cov_kk_cond = (1 - rho) * cov_kk + rho * np.diag(np.diag(cov_kk))
L = np.linalg.cholesky(np.linalg.pinv(cov_kk_cond))


# cov_mask = np.abs(cor_kk) < 0.3
# cov_kk_cond2 = cov_kk.copy()
# cov_kk_cond2[cov_mask] = 0
# L2 = np.linalg.cholesky(np.linalg.pinv(cov_kk_cond2))

# plt.matshow(cov2corr(cov_kk_cond2))

xanes_mask = np.abs(energy -e0) < 30
score_xanes = np.sum(((mus_av / mus_err)[xanes_mask])**2) / (np.sum(xanes_mask) - 1)

exafs_lo = (k > 2.5) & (k < 10)
score_exafs_lo = np.sum(((chis_av / chis_err)[exafs_lo])**2) / (np.sum(exafs_lo) - 1)**2

exafs_hi = (k > 10) & (k < 16)
score_exafs_hi = np.sum(((chis_av / chis_err)[exafs_hi])**2) / ((np.sum(exafs_hi) - 1))

plt.figure(1, clear=True)
plt.plot(k, L.T @ chis_av)
# plt.plot(k, L2.T @ chis_av)
plt.plot(k, chis_av / chis_err)
plt.plot(k, +savgol_filter(np.abs(chis_av / chis_err), 71, 0))
plt.plot(k, -savgol_filter(np.abs(chis_av / chis_err), 71, 0))

# plt.plot(k, chis_av * k**2)
# plt.errorbar(k, chis_av * k**2, chis_err * k**2)
# plt.plot(k, (mus_av[-k.size:]-1) * k**2)
# plt.plot(energy, uu[:, :3])

plt.figure(2, clear=True)

ndiag = 1
plt.plot(k[:-ndiag], np.diag(cor_kk,ndiag))

ndiag = 2
plt.plot(k[:-ndiag], np.diag(cor_kk,ndiag))


plt.figure(3, clear=True)

ndiag = 1
plt.plot(r[:-ndiag], np.diag(cor_rr,ndiag))

ndiag = 2
plt.plot(r[:-ndiag], np.diag(cor_rr,ndiag))

plt.matshow(cov_ee)
plt.matshow(cor_ee)

plt.matshow(np.diag(k**2) @ cov_kk @ np.diag(k**2))
plt.matshow(cor_kk)

plt.matshow(cov_rr)
plt.matshow(cor_rr)



mask_r = (r >= 1.25) & (r <= 2.85)
# mask_r = (r >= 0.25) & (r <= 2.85)
plt.matshow(cov_rr[np.ix_(mask_r, mask_r)])
plt.matshow(cov2corr(cov_rr[np.ix_(mask_r, mask_r)]))


k_dummy = np.linspace(3, 12, 101)
i_dummy = np.eye(k_dummy.size)
A_fft = np.fft.fft(i_dummy)

r_dummy = np.fft.fftfreq(k_dummy.size, k_dummy[1] - k_dummy[0])
r_dummy_mask = r_dummy>=0
r_dummy_pos = r_dummy[r_dummy_mask]
A_fft_imre = np.vstack((np.real(A_fft[r_dummy_mask, :]), np.imag(A_fft[r_dummy_mask, :])))
c_dummy = (A_fft_imre @ A_fft_imre.T)





''' Eli is trying to download Japaese database'''


import urllib.request
plt.figure();
for i in range(1,50):
    print(i)
    s = f'https://www.cat.hokudai.ac.jp/catdb/index.php?action=xafs_dbinpbrowsedetail&opnid=2&resid={i}&r=76'
    fp = urllib.request.urlopen(s)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()


    a=mystr.split('<textarea name')[1]
    b=a.split('>')[1]
    c= b.split(',')[0]
    d=c.split('\n')
    e=dict([i.replace('    ','').split('=') for i in d][:-4])
    print(e)


    s=f'https://www.cat.hokudai.ac.jp/catdb/index.php?action=xafs_dbinpbrowsedetail&opnid=2&resid={i}&d=3'
    fp = urllib.request.urlopen(s)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    try:
        sss= np.array([[float(k) for k in i.split('    \t')] for i in mystr.split('\r\n')])
    except:
        try:
            sss = np.array([[float(k) for k in i.split('  ')] for i in mystr.split('\r\n')])
        except:
            sss = np.array([[float(k) for k in i.split('\t')] for i in mystr.split('\r\n')])

    plt.plot(sss[:,0],sss[:,1])


############################
'''Denis checks calibration'''

from xas.db_io import get_fly_uids_for_proposal
from xas.energy_calibration import get_energy_offset
# from xas.file_io import load_binned_df_from_file
# from xas.xasproject import XASDataSet
from xas.fitting import fit_gaussian_with_estimation
from xas.energy_calibration import compute_shift_between_spectra
from scipy.signal import medfilt

uids = get_fly_uids_for_proposal(db, 2022, 1, 309864)

e0_list = []
time_e0_list = []
# for uid in uids[:50]:
for uid in uids:
    hdr = db[uid]
    if hdr.stop['exit_status']:
        _e_ref, _e_obs = get_energy_offset(uid, db, db_proc, attempts=1, sleep_time=0)
        print(f'_e_obs={_e_obs}')
        if _e_obs:
            e0_list.append(_e_obs)

            _time = hdr.start['time']
            time_e0_list.append(_time)

e0 = np.array(e0_list)
time_e0 = np.array(time_e0_list)

x = xview_gui.project
fwhm_list = []
ecen_list = []
time_xes_list = []
names_xes_list = []
for ds in x:
    _energy = ds.energy
    _intensity = ds.mu
    # ecen_list.append(_energy[np.argmax(_intensity)])
    _ecen, _fwhm, _, _, _ = fit_gaussian_with_estimation(_energy, _intensity)
    # ecen_list.append(_ecen)
    _e_shift, _ = compute_shift_between_spectra(_energy, _intensity, x[0].energy, x[0].mu)
    ecen_list.append(_e_shift)
    fwhm_list.append(_fwhm)
    time_xes_list.append(ds.md['time'])
    names_xes_list.append(ds.name)

ecen = np.array(ecen_list)
fwhm = np.array(fwhm_list)
time_xes = np.array(time_xes_list)
names_xes_sorted = [names_xes_list[i] for i in np.argsort(time_xes_list)]


plt.figure(1)
plt.clf()
# plt.plot(time_obs, e0_obs - 11564, 'k.-')
plt.plot(time_obs, medfilt(e0_obs - 11564), 'k.-')
plt.plot(time_xes, ecen - ecen[1], 'r*')


