from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib
import numpy as np
import sys, os
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import shapiro
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.stats import skewnorm
import scipy.stats as stats
from distfit import distfit


def get_full_path(root, dataset, meth, ang_sep):
    path_filled = os.path.join(root, '{}/{}/{}/'.format(dataset, meth, ang_sep))
    return path_filled

def join_pickles(root, dataset, meth, ang_sep):
    # joins all individual data_dict.pkl into one large pickle file
    path = get_full_path(root, dataset, meth, ang_sep)

    list_dicts = []
    for case_pkl in os.listdir(path):
        if 'df.pkl' in case_pkl:
            continue
        data_dict = pd.read_pickle(os.path.join(path, case_pkl))
        data_dict['case_num'] = int(data_dict['case_num'])
        list_dicts.append(data_dict)

    big_df = pd.DataFrame.from_dict(list_dicts)
    output_saving_direc = os.path.join(path, 'df.pkl')
    big_df.to_pickle(output_saving_direc)

class DataViz:
    try:
        def __init__(self, pkl_direc):
            self.pkl_direc = pkl_direc

        def convert_pickle_to_df(self, overwrite=False):
            if overwrite:
                for dataset in os.listdir(self.pkl_direc):
                    lv1_path = os.path.join(self.pkl_direc, dataset)
                    for meth in os.listdir(lv1_path):
                        lv2_path = os.path.join(lv1_path, meth)
                        for ang_sep in os.listdir(lv2_path):
                            full_path = os.path.join(lv2_path, ang_sep)
                            print('Currently in folder: ', full_path)
                            join_pickles(self.pkl_direc, dataset, meth, ang_sep)

        def plot_volumes(self):
            """
            https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
            """
            def autolabel(rects, ax):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            fig, axs = plt.subplots(1, 3)
            data_types = ['UKBB', 'YHC', 'HFC']
            tickslabels = ['', r'$v_{90}$', '',
                           '', r'$v_{60}$', '']
            alphas = [0.1, 0.3, 0.5]

            for i, data_typ in enumerate(data_types):
                direc_list = [get_full_path(self.pkl_direc, data_typ, 'BOD', '90'),
                              get_full_path(self.pkl_direc, data_typ, 'TBC', '90'),
                              get_full_path(self.pkl_direc, data_typ, 'SBR', '90'),
                              get_full_path(self.pkl_direc, data_typ, 'BOD', '60'),
                              get_full_path(self.pkl_direc, data_typ, 'TBC', '60'),
                              get_full_path(self.pkl_direc, data_typ, 'SBR', '60')
                              ]
                posx = [0, 0.5, 1.0, 2.5, 3.0, 3.5]
                ccs = [plt.cm.Greens([0.8, 0.5]), plt.cm.Reds([0.8, 0.5]), plt.cm.Blues([0.8, 0.5]),
                       plt.cm.Greens([0.8, 0.5]), plt.cm.Reds([0.8, 0.5]), plt.cm.Blues([0.8, 0.5])]

                all_perrs = []
                xs = []  # random xs for scattering all points onto the boxplot
                ws = []
                cmaps = []

                for k, direc in enumerate(direc_list):
                    df = pd.read_pickle(os.path.join(direc, 'df.pkl'))
                    all_perrs.append(df['simp_perr'].values)
                    xs.append(np.random.normal(posx[k], 0.04, all_perrs[-1].shape[0]))

                    vals = np.vstack([xs[-1], all_perrs[-1]])
                    kernel = stats.gaussian_kde(vals)
                    weights = kernel(vals)
                    weights = weights / weights.max()
                    ws.append(weights)

                    cols = ccs[k]
                    cols[:, 3] = [1., 0.005]
                    cmaps.append(LinearSegmentedColormap.from_list("", cols))

                for x, val, weights, cmap in zip(xs, all_perrs, ws, cmaps):
                    axs[i].scatter(x, val, s=13, c=weights, cmap=cmap, edgecolor="none", alpha=alphas[i])

                bp = axs[i].boxplot(all_perrs, notch=False, widths=0.4, positions=posx, showfliers=False)
                axs[i].set_xticks(posx)
                axs[i].set_xticklabels(tickslabels, rotation=0, fontsize=18)
                axs[i].tick_params(axis='x', pad=0.1, bottom=False)

                for k, line in enumerate(bp['medians']):
                    curr_perr = all_perrs[k]
                    curr_x_pos = np.mean(xs[k])
                    m1 = np.mean(curr_perr, axis=0)
                    st1 = np.std(curr_perr, axis=0)
                    x, y = line.get_xydata()[1]
                    text = ' μ={:.2f} \n σ={:.2f}'.format(m1, st1)
                    # axs[i].annotate(text, xy=(curr_x_pos-0.15, -14), rotation=0)
                    if (k % 2) == 0:  # if even
                        axs[i].annotate(text, xy=(curr_x_pos - 0.4, -10), rotation=0, fontweight='bold')
                    else:
                        axs[i].annotate(text, xy=(curr_x_pos - 0.5, -13), rotation=0, fontweight='bold')

                # red line
                low_xlim, upp_xlim = axs[i].get_xlim()
                axs[i].axhline(y=0.0, xmin=low_xlim, xmax=upp_xlim, color='k', linestyle='--', alpha=0.4)

                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], color='g', lw=4, label='BOD'),
                                   Line2D([0], [0], color='r', lw=4, label='TBC'),
                                   Line2D([0], [0], color='b', lw=4, label='SBR')]
                axs[i].legend(handles=legend_elements, loc='upper right')

            axs[0].set_title(r'$UKBB \,\, (n=4113)$', fontsize=18)
            axs[1].set_title(r'$YHC \,\, (n=225)$', fontsize=18)
            axs[1].set_xlabel(r'$Viewing \,\, Angle \,\, v_{\theta} \,\,\, (^{\circ})$', fontsize=18)
            axs[2].set_title(r'$HFC \,\, (n=50)$', fontsize=18)
            axs[0].set_ylabel(r'$Percentage \,\, Error \,\, (\%)$', fontsize=20)

            axs[0].set_ylim(-15, 15)
            axs[1].set_ylim(-15, 15)
            axs[2].set_ylim(-15, 15)
            plt.show()

        def plot_eccentricities(self):

            colu = 'simpson_ellipse_ecc'

            # plot eccentricity
            df_ukbb = pd.read_pickle(get_full_path(self.pkl_direc, 'UKBB', 'BOD', '90') + '/df.pkl')
            df_hc = pd.read_pickle(get_full_path(self.pkl_direc, 'YHC', 'BOD', '90') + '/df.pkl')
            df_hf = pd.read_pickle(get_full_path(self.pkl_direc, 'HFC', 'BOD', '90') + '/df.pkl')

            # case-by-case data (averaged)
            lv_eccents_ukbb = np.array([np.mean(ls) for ls in df_ukbb[colu].values])
            lv_eccents_hc = np.array([np.mean(ls) for ls in df_hc[colu].values])
            lv_eccents_hf = np.array([np.mean(ls) for ls in df_hf[colu].values])

            # slice-by-slice data
            columns = ['{}'.format(v) for v in np.arange(20)]
            ukbb_slicewise = pd.DataFrame(df_ukbb[colu].values.tolist(), columns=columns) # .tolist()
            hc_slicewise = pd.DataFrame(df_hc[colu].values.tolist(), columns=columns) # .tolist()
            hf_slicewise = pd.DataFrame(df_hf[colu].values.tolist(), columns=columns) # .tolist()

            slicewise_means = [ukbb_slicewise.mean(0), hc_slicewise.mean(0), hf_slicewise.mean(0)]
            slicewise_stds = [ukbb_slicewise.std(0), hc_slicewise.std(0), hf_slicewise.std(0)]

            # plot
            fig = plt.figure()
            plt.subplots_adjust(hspace=0.0)
            outer_gs = gridspec.GridSpec(1, 2, height_ratios=[1])
            gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0])
            data_gs1 = [lv_eccents_ukbb, lv_eccents_hc, lv_eccents_hf]
            gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1])
            # data_gs2 = [ukbb_slicewise, hc_slicewise, hf_slicewise]

            colors = ['green', 'blue', 'red']  # 'magenta' for ct
            colors1 = ['#00CC66', '#3399FF', '#FF6666']  # '#9834eb' for ct
            colors2 = ['#99FFCC', '#66B2FF', '#FF9999']  # '#7734eb' for ct
            binning = [60, 30, 10]

            # left side subplot
            for i, cell in enumerate(gs1):
                if i == 0:
                    ax = plt.subplot(cell)
                    ax0 = ax
                    ax0.set_title(r'$\mathrm{Histogram \, of}\ \mu : case-wise$', fontsize=20)
                else:
                    ax = plt.subplot(cell, sharex=ax0)

                (mu, sigma) = norm.fit(data_gs1[i])
                n, bins, patches = ax.hist(data_gs1[i], bins=int(binning[i]), density=True, facecolor=colors[i],
                                           alpha=0.5)  # int(binning[i])
                y = norm.pdf(bins, mu, sigma)
                # pval = norm.cdf(data_gs1[i], mu, sigma)
                W, p = shapiro(data_gs1[i])
                label = r'$\mathcal{N} \ (\mu=%.3f,\ \sigma=%.3f)$' % (mu, sigma)
                l = ax.plot(bins, y, linewidth=2, color=colors[i], label=label)
                ax.axvline(x=1.0, linestyle='--', color='k', alpha=0.7)
                ax.grid(True)
                ax.legend(fontsize=12, loc='upper right')

            ax.set_xlabel(r"Eccentricity $\mu$", ha='center', fontsize=18)
            fig.text(0.09, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)  # ylabel

            # right side plot (slice-wise, vertical plot)
            for i, cell in enumerate(gs2):
                if i == 0:
                    ax = plt.subplot(cell)
                    ax0 = ax
                    ax0.set_title(r'$\mathrm{Histogram \, of}\ \mu : slice-wise$', fontsize=20)
                    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax0.set_xticks(np.arange(21))
                else:
                    ax = plt.subplot(cell, sharex=ax0)

                label = r'$\mu=%.3f,\ \sigma=%.3f$' % (np.mean(slicewise_means[i]), np.mean(slicewise_stds[i]))
                ax.plot(np.arange(len(slicewise_means[i])), slicewise_means[i], color=colors1[i], label=label)
                ax.fill_between(np.arange(len(slicewise_means[i])), slicewise_means[i] - slicewise_stds[i],
                                slicewise_means[i] + slicewise_stds[i],
                                alpha=0.5, edgecolor=colors1[i], facecolor=colors2[i])
                ax.yaxis.tick_right()
                ax.legend(fontsize=12, loc='upper left')
                ax.axhline(y=1.0, linestyle='--', color='k', alpha=0.7)
                ax.grid(True)

            ax.set_xlabel(r"$Slice \,\, \#$", ha='center', fontsize=18)
            fig.text(0.94, 0.5, r"Eccentricity $\mu$", va='center', rotation='vertical', fontsize=18)  # ylabel
            fig.text(0.51, 0.72, r"$\mathrm{UKBB}$", rotation='vertical', fontsize=18)
            fig.text(0.51, 0.48, r"$\mathrm{YHC}$", rotation='vertical', fontsize=18)
            fig.text(0.51, 0.22, r"$\mathrm{HFC}$", rotation='vertical', fontsize=18)
            plt.show()

        def plot_orientation_angles(self):

            def make_white_background_transparent(img):
                img = img.convert("RGBA")
                datas = img.getdata()
                newData = []
                for item in datas:
                    if item[0] == 255 and item[1] == 255 and item[2] == 255:
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)
                img.putdata(newData)
                return img

            def circular_hist(ax, x, bins=16, density=True, offset=0, theta_dir=1, gaps=True):
                """
                Produce a circular histogram of angles on ax.

                Parameters
                ----------
                ax : matplotlib.axes._subplots.PolarAxesSubplot
                    axis instance created with subplot_kw=dict(projection='polar').

                x : array
                    Angles to plot, expected in units of radians.

                bins : int, optional
                    Defines the number of equal-width bins in the range. The default is 16.

                density : bool, optional
                    If True plot frequency proportional to area. If False plot frequency
                    proportional to radius. The default is True.

                offset : float, optional
                    Sets the offset for the location of the 0 direction in units of
                    radians. The default is 0.

                gaps : bool, optional
                    Whether to allow gaps between bins. When gaps = False the bins are
                    forced to partition the entire [-pi, pi] range. The default is True.

                Returns
                -------
                n : array or list of arrays
                    The number of values in each bin.

                bins : array
                    The edges of the bins.

                patches : `.BarContainer` or list of a single `.Polygon`
                    Container of individual artists used to create the histogram
                    or list of such containers if there are multiple input datasets.
                """
                # Wrap angles to [-pi, pi)
                x = (x + np.pi) % (2 * np.pi) - np.pi

                # Force bins to partition entire circle
                if not gaps:
                    bins = np.linspace(-np.pi, np.pi, num=bins + 1)

                # Bin data and record counts
                n, bins = np.histogram(x, bins=bins)

                # Compute width of each bin
                widths = np.diff(bins)

                # By default plot frequency proportional to area
                if density:
                    # Area to assign each bin
                    area = n / x.size
                    # Calculate corresponding bin radius
                    radius = (area / np.pi) ** .5
                # Otherwise plot frequency proportional to radius
                else:
                    radius = n

                # Plot data on ax
                patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                                 edgecolor='C0', fill=False, linewidth=1.0)

                # Set the offset from the 0 angle
                ax.set_theta_offset(offset)

                # Set the direction of the zero angle
                ax.set_theta_direction(theta_dir)  # 1 is clockwise, -1 is anticlockwise

                # Remove ylabels for area plots (they are mostly obstructive)
                if density:
                    ax.set_yticks([])

                return n, bins, patches

            colu = 'angs_a4c_to_major'

            df_ukbb = pd.read_pickle(get_full_path(self.pkl_direc, 'UKBB', 'BOD', '90') + '/df.pkl')
            df_hc = pd.read_pickle(get_full_path(self.pkl_direc, 'YHC', 'BOD', '90') + '/df.pkl')
            df_hf = pd.read_pickle(get_full_path(self.pkl_direc, 'HFC', 'BOD', '90') + '/df.pkl')

            slice_number = 9  # mid-ventricular is around slice 9
            mid_lv_Am_ukbb = df_ukbb[colu].map(lambda x: x[slice_number]).values
            mid_lv_Am_hc = df_hc[colu].map(lambda x: x[slice_number]).values
            mid_lv_Am_hf = df_hf[colu].map(lambda x: x[slice_number]).values

            # make sure plot is from 0 to 180
            ukbb_arr = mid_lv_Am_ukbb
            hc_arr = mid_lv_Am_hc
            hf_arr = mid_lv_Am_hf

            ukbb_arr[ukbb_arr < 0.0] = ukbb_arr[ukbb_arr < 0.0] + 180
            hc_arr[hc_arr < 0.0] = hc_arr[hc_arr < 0.0] + 180
            hf_arr[hf_arr < 0.0] = hf_arr[hf_arr < 0.0] + 180

            datas = [ukbb_arr, hc_arr, hf_arr]
            radius_limits = [500, 80, 16]
            axes_coords = (0.1, 0.1, 0.8, 0.8)
            radius_labels = [r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$', '', '', '', r'$-90^{\circ}$',
                             r'$-45^{\circ}$']

            # open image
            im_axes_coords = [0.25, 0.23, 0.5, 0.5]
            titls = ['UKBB', 'YHC', 'HFC']
            binnings = [60, 30, 30]

            for i, data in enumerate(datas):
                rot_angle = 22
                img = Image.open("D://phdcoding//src_14_nov_2018//Heart_normal_short_axis_section.jpg")
                img = img.rotate(rot_angle, fillcolor='white')
                img = make_white_background_transparent(img)  # make white background transaprent
                # else: # for rounder thicker heart
                #     rot_angle = 22
                #     img = Image.open("D://phdcoding//src_14_nov_2018//Heart_left_ventricular_hypertrophy_sa.jpg")
                #     img = img.rotate(rot_angle, fillcolor='white')
                #     img = make_white_background_transparent(img)  # make white background transaprent

                fig = plt.gcf()
                ax_polar = fig.add_axes(axes_coords, projection='polar', label="polar image")
                x = np.radians(data)  # convert to radians
                circular_hist(ax_polar, x, bins=binnings[i], density=False, theta_dir=-1, offset=np.pi)
                ax_polar.set_rlabel_position(135)  # where to set the radius labels
                ax_polar.set_xticklabels(radius_labels, fontsize=15)

                # a4c and a2c lines and annotation texts
                ax_polar.vlines(np.pi / 2, 0, radius_limits[i], color='b', linestyle='-', lw=2.5, zorder=3)
                ax_polar.vlines(3 * np.pi / 2, 0, radius_limits[i], color='b', linestyle='-', lw=2.5, zorder=3)
                ax_polar.plot((0, np.pi), (0, 0), color='g', linestyle='-', lw=2.5, zorder=3)
                ax_polar.text(0.02, 10, 'A4C', fontweight='bold', fontsize=15)
                ax_polar.text((np.pi / 2.0) + 0.068, 100, 'APLAX', fontweight='bold', fontsize=15, rotation='vertical')

                ax_polar.set_ylim(radius_limits[i], 0)  # ukbb
                ax_polar.patch.set_alpha(0)

                ax_image = fig.add_axes(im_axes_coords, label="ax image")
                ax_image.imshow(img, alpha=0.5, extent=[-1, 1, -1, 1], aspect=1)
                ax_image.axis('off')

                ax_polar.yaxis.set_tick_params(labelsize=12)

                ax_polar.set_title(titls[i], fontsize=20, fontweight='bold')

                plt.show()

        def plot_basal_slanting(self):
            # plot
            df_ukbb = pd.read_pickle(get_full_path(self.pkl_direc, 'UKBB', 'BOD', '90') + '/df.pkl')
            df_hc = pd.read_pickle(get_full_path(self.pkl_direc, 'YHC', 'BOD', '90') + '/df.pkl')
            df_hf = pd.read_pickle(get_full_path(self.pkl_direc, 'HFC', 'BOD', '90') + '/df.pkl')

            # case-by-case data
            a4c_sla_ukbb = df_ukbb['a4c_sla'].values
            a4c_sla_hc = df_hc['a4c_sla'].values
            a4c_sla_hf = df_hf['a4c_sla'].values

            # slice-by-slice data
            a2c_sla_ukbb = df_ukbb['a2c_sla'].values
            a2c_sla_hc = df_hc['a2c_sla'].values
            a2c_sla_hf = df_hf['a2c_sla'].values

            fig = plt.figure()
            plt.subplots_adjust(hspace=0.0)
            outer_gs = gridspec.GridSpec(1, 2, height_ratios=[1])
            gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0])
            data_gs1 = [a4c_sla_ukbb, a4c_sla_hc, a4c_sla_hf]
            gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1])
            data_gs2 = [a2c_sla_ukbb, a2c_sla_hc, a2c_sla_hf]

            colors = ['green', 'blue', 'red']
            colors1 = ['#00CC66', '#3399FF', '#FF6666']
            colors2 = ['#99FFCC', '#66B2FF', '#FF9999']
            dataset = ['UKBB', 'YHC', 'HFC']
            binning = [60, 30, 15]
            quantile_yoffsets1 = [[0.048, 0.068, 0.083],
                                    [0.039, 0.055, 0.065],
                                    [0.027, 0.037, 0.047]]
            quantile_yoffsets2 = [[0.075, 0.1, 0.125],
                                    [0.08, 0.107, 0.137],
                                    [0.023, 0.031, 0.038]]

            #################################################### left side subplot
            for i, cell in enumerate(gs1):
                if i == 0:
                    ax = plt.subplot(cell)
                    ax0 = ax
                    ax0.set_title(r'$\mathrm{A4C}$', fontsize=20)
                else:
                    ax = plt.subplot(cell, sharex=ax0)

                n, bins, patches = ax.hist(data_gs1[i], bins=int(binning[i]), density=True, facecolor=colors[i],
                                           alpha=0.5)

                # find_which_distribution_is_best(data_gs1[i])
                # exponential fitting for UKBB and YHC plots
                if i < 2:
                    P = stats.expon.fit(data_gs1[i], floc=0)
                    # label = r'$E \ (\lambda=%.3f)$' % (1/P[1])
                    rX = np.linspace(0, 35, 100)
                    rP = stats.expon.pdf(rX, *P)
                    # ax.plot(rX, rP, linestyle='--', color=colors[i], label=label)

                    # gamma
                    counts, _ = np.histogram(data_gs1[i], bins=binning[i])
                    bin_width = bins[1] - bins[0]
                    total_count = float(sum(counts))

                    params = stats.exponweib.fit(data_gs1[i], loc=0)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    label = r'$W \ (\alpha=%.3f, \gamma=%.3f)$' % (scale, arg[0])
                    pdf_fitted = stats.exponweib.pdf(bins, *arg, loc=-1, scale=scale)  # * total_count * bin_width
                    ax.plot(bins, pdf_fitted, linestyle='--', lw=2, color=colors[i], label=label)

                # normal distribution fit for HFC plot
                if i == 2:
                    (mu, sigma) = norm.fit(data_gs1[i])
                    label = r'$\mathcal{N} \ (\mu=%.3f,\ \sigma=%.3f)$' % (mu, sigma)
                    y = norm.pdf(bins, mu, sigma)
                    l = ax.plot(bins, y, linewidth=2, linestyle='--', color=colors[i], label=label)

                # calculate percentiles
                quant_25 = np.quantile(data_gs1[i], 0.25)
                quant_50 = np.quantile(data_gs1[i], 0.5)
                quant_75 = np.quantile(data_gs1[i], 0.75)

                quants = [[quant_25, 0.8, 0.26, '25th'], [quant_50, 1, 0.36, '50th'], [quant_75, 0.8, 0.46, '75th']]

                # Plot the lines with a loop
                for k in quants:
                    ax.axvline(k[0], alpha=k[1], ymax=k[2], linestyle=":", color=(0.2, 0.3, 0.3))
                    # ax.text(k[0], 0.03, k[3], size=11, alpha=0.85, fontweight='bold')

                ax.text(quant_25 - 0.5, quantile_yoffsets1[i][0], "25th", size=11, alpha=0.85, fontweight='bold')
                ax.text(quant_50 - 0.5, quantile_yoffsets1[i][1], "50th", size=12, alpha=1, fontweight='bold')
                ax.text(quant_75 - 0.5, quantile_yoffsets1[i][2], "75th", size=11, alpha=0.85, fontweight='bold')

                ax.grid(True)
                ax.set_xlim(0)
                ax.legend(fontsize=12, loc='upper right')

                if i < 2:
                    plt.setp(ax.get_xticklabels(), visible=False)

            fig.text(0.085, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)  # ylabel

            ########################################## right side plot (slice-wise, vertical plot)
            for i, cell in enumerate(gs2):
                if i == 0:
                    ax = plt.subplot(cell)
                    ax0 = ax
                    ax0.set_title(r'$\mathrm{A2C}$', fontsize=20)
                else:
                    ax = plt.subplot(cell, sharex=ax0)

                n, bins, patches = ax.hist(data_gs2[i], bins=int(binning[i]), density=True, facecolor=colors[i],
                                           alpha=0.5)

                # exponential fitting for UKBB and YHC plots
                if i < 2:
                    P = stats.expon.fit(data_gs2[i], floc=0)
                    # label = r'$E \ (\lambda=%.3f)$' % (1/P[1])
                    rX = np.linspace(0, 35, 100)
                    rP = stats.expon.pdf(rX, *P)
                    # ax.plot(rX, rP, linestyle='--', color=colors[i], label=label)

                    # gamma
                    counts, _ = np.histogram(data_gs2[i], bins=binning[i])
                    bin_width = bins[1] - bins[0]
                    total_count = float(sum(counts))
                    params = stats.gamma.fit(data_gs2[i], loc=0)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    label = r'$W \ (\alpha=%.3f, \gamma=%.3f)$' % (scale, arg[0])
                    pdf_fitted = stats.gamma.pdf(bins, *arg, loc=-1, scale=scale)
                    ax.plot(bins, pdf_fitted, linestyle='--', lw=2, color=colors[i], label=label)

                # normal distribution fit for HFC plot
                if i == 2:
                    (mu, sigma) = norm.fit(data_gs2[i])
                    label = r'$\mathcal{N} \ (\mu=%.3f,\ \sigma=%.3f)$' % (mu, sigma)
                    y = norm.pdf(bins, mu, sigma)
                    l = ax.plot(bins, y, linewidth=2, linestyle='--', color=colors[i], label=label)

                # calculate percentiles
                quant_25 = np.quantile(data_gs2[i], 0.25)
                quant_50 = np.quantile(data_gs2[i], 0.5)
                quant_75 = np.quantile(data_gs2[i], 0.75)

                quants = [[quant_25, 0.8, 0.26, '25th'], [quant_50, 1, 0.36, '50th'], [quant_75, 0.8, 0.46, '75th']]

                # Plot the lines with a loop
                for k in quants:
                    ax.axvline(k[0], alpha=k[1], ymax=k[2], linestyle=":", color=(0.2, 0.3, 0.3))
                    # ax.text(k[0], 0.03, k[3], size=11, alpha=0.85, fontweight='bold')

                ax.text(quant_25 - 0.5, quantile_yoffsets2[i][0], "25th", size=11, alpha=0.85, fontweight='bold')
                ax.text(quant_50 - 0.5, quantile_yoffsets2[i][1], "50th", size=12, alpha=1, fontweight='bold')
                ax.text(quant_75 - 0.5, quantile_yoffsets2[i][2], "75th", size=11, alpha=0.85, fontweight='bold')

                ax.grid(True)
                ax.set_xlim(0)
                ax.legend(fontsize=12, loc='upper right')

                if i < 2:
                    plt.setp(ax.get_xticklabels(), visible=False)

            fig.text(0.45, 0.05, r"Basal Slanting $b_{\theta} (^{\circ})$", va='center', fontsize=18)  # ylabel
            fig.text(0.51, 0.75, r"$\mathrm{UKBB}$", rotation='vertical', fontsize=18)
            fig.text(0.51, 0.57, r"$\mathrm{YHC}$", rotation='vertical', fontsize=18)
            fig.text(0.51, 0.38, r"$\mathrm{HFC}$", rotation='vertical', fontsize=18)
            fig.text(0.51, 0.2, r"$\mathrm{CT}$", rotation='vertical', fontsize=18)

            plt.show()


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, exc_obj, fname, exc_tb.tb_lineno)
        sys.exit()



