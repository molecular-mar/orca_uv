# -*- coding: utf-8 -*-
'''
orca-uv
'''

import sys                              # sys files processing
import os                               # os file processing
import re                               # regular expressions
import argparse                         # argument parser
import math                             # for floor/ceil in rounding functions
import numpy as np                      # summation
import matplotlib.pyplot as plt         # plots
import matplotlib.ticker as ticker
from scipy.signal import find_peaks     # peak detection

# global constants
#found_data_section = False
specstring_start = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
specstring_end = 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'
ecd_start = "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
ecd_end = "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
ecdv_start = "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
ecdv_end = "Total run time"
vcd_start = "VCD SPECTRUM CALCULATION"
vcd_end = "throughout the entire PROP-calculation"
ir_start = "IR SPECTRUM"
ir_end = "The epsilon (eps)"
export_delim = " "

# Default line widths (FWHM) in respective units
w_wn = 5323.3                 # cm⁻¹
w_nm = 20                   # nm
w_ev = 0.66                 # eV  ### NEW ###

# Plot configuration
spectrum_title = "Absorption spectrum"
spectrum_title_weight = "bold"
#y_label = "intensity"
x_label_wn = r'Wavenumber (cm$^{-1}$)'
x_label_nm = r'$\lambda$ (nm)'
x_label_ev = r'E (eV)'
normalize_export = False
normalize_factor = 1
figure_dpi = 300

# Options
#ecd_spectra = False
show_single_gauss = True
show_single_gauss_area = False
show_conv_spectrum = True
#show_sticks = True
label_peaks = False
minor_ticks = False
show_grid = False
linear_locator = True

# global lists
#energylist = []
#intenslist = []
#gauss_sum = []

HC_EV_NM = 1239.84193

def cm1_to_mev(x):
     x = np.asarray(x)
     return x * 0.123984193

def mev_to_cm1(x):
     x = np.asarray(x)
     return x / 0.123984193

def nm_to_ev(x_nm):
     x = np.asarray(x_nm)
     x = np.clip(x, 0.1, None)
     return HC_EV_NM / x

def ev_to_nm(x_ev):
     x = np.asarray(x_ev)
     x = np.clip(x, 0.1, None)
     return HC_EV_NM / x

def gauss(a, m, x, w):
    """
    Calculate a height-normalized Gaussian line shape.
      a : amplitude (intensity)
      m : center (position)
      x : array of x values
      w : line width (FWHM)
    """
    return a * np.exp(-4* (np.log(2) * ((m - x) / w) ** 2))


def nGauss(a, m, x, w):
    """
    Calculate an area-normalized Gaussian line shape.
      a : amplitude (intensity)
      m : center (position)
      x : array of x values
      w : line width (FWHM)
    """
    return a * (2 * np.sqrt(np.log(2)) / (w * np.sqrt(np.pi))) * np.exp(-4* (np.log(2) * ((x - m) / w) ** 2))


def roundup(x, unit):
    """Round x up to a nice boundary depending on the chosen unit."""
    if unit == 'nm':
        factor = 10
    elif unit == 'ev':
        factor = 0.1
    else:  # 'wn'
        factor = 50
    return math.ceil(x / factor) * factor

def rounddown(x, unit):
    """Round x down to a nice boundary depending on the chosen unit."""
    if unit == 'nm':
        factor = 10
    elif unit == 'ev':
        factor = 0.1
    else:  # 'wn'
        factor = 100
    return math.floor(x / factor) * factor

parser = argparse.ArgumentParser(prog='orca_uv', description='Easily plot absorption spectra from orca.out')

# Required argument: ORCA output file
parser.add_argument("filename", help="the ORCA output file")

# Second file data
parser.add_argument('-file2', '--second_file', type=str, default='',
                    help='Second output file to be plotted')

# Plot ECD
parser.add_argument('-ecd', '--ecd_spectra', default=0, action='store_true',
                    help='plot electronic circular dichroism')

# Plot ECD-Velocity
parser.add_argument('-ecdv', '--ecdv_spectra', default=0, action='store_true',
                    help='plot electronic circular dichroism - velocity')

# Plot S-ECD
parser.add_argument('-secd', '--secd_spectra', default=0, action='store_true',
                    help='plot electronic circular dichroism from simplified TDDFT')

# Plot VCD
parser.add_argument('-vcd', '--vcd_spectra', default=0, action='store_true',
                    help='plot vibrational circular dichroism')

# Plot IR
parser.add_argument('-ir', '--ir_spectra', default=0, action='store_true',
                    help='plot IR spectra')

# Double axis plot
parser.add_argument('-double', '--double_axis', default=0, action='store_true',
                    help='include second axis in energy units')

# Plot sticks
parser.add_argument('-stick', '--show_sticks', default=0, action='store_true',
                    help='plot signal sticks')

# Label peaks
parser.add_argument('-peak', '--label_peaks', default=0, action='store_true',
                    help='plot signal sticks')

# Y-axis label
parser.add_argument('-ylab', '--y_label', type=str, default='',
                    help='Label for y axis')

# Activate single gaussian plotting
parser.add_argument('-ssg', '--show_single_gauss', default=0, action='store_true',
                    help='show single gaussians')

# Set step size
parser.add_argument('-step', '--step_size', type=float, default=0,
                    help='set step size')

# Set X axis base multiplier
parser.add_argument('-mul', '--x_multiplier', type=float, default=50,
                    help='set x axis scale multiplier')

# Set X axis base multiplier
parser.add_argument('-mul2', '--multi2', type=float, default=1,
                    help='set 2nd x axis scale multiplier')

# Activate colored area under single gaussians
parser.add_argument('-ssga', '--show_single_gauss_area', default=0, action='store_true',
                    help='color single gaussians area')

# Whether to show the matplotlib window
parser.add_argument('-s', '--show', default=0, action='store_true',
                    help='show the plot window')

# Whether to save the figure
parser.add_argument('-n', '--nosave', default=1, action='store_false',
                    help='do not save the spectrum')

# Broadening: nm
parser.add_argument('-wnm', '--linewidth_nm', type=float, default=20,
                    help='line width (FWHM) in nm (for unit nm)')

# Broadening: wave numbers
parser.add_argument('-wwn', '--linewidth_wn', type=float, default=1000,
                    help='line width (FWHM) in cm⁻1 (for unit wn)')

# Broadening: eV  ### NEW ###
parser.add_argument('-wev', '--linewidth_ev', type=float, default=0.2,
                    help='line width (FWHM) in eV (for unit ev)')

# X-range
parser.add_argument('-x0', '--startx', type=float,
                    help='start of x-axis range (depends on --unit)')
parser.add_argument('-x1', '--endx', type=float,
                    help='end of x-axis range (depends on --unit)')
# Y-range
parser.add_argument('-y0', '--starty', type=float,
                    help='start of y-axis range')
parser.add_argument('-y1', '--endy', type=float,
                    help='end of y-axis range')


# Export data
parser.add_argument('-e', '--export', default=0, action='store_true',
                    help='export data to file')

# Choice of unit for x-axis
parser.add_argument('-u', '--unit', choices=['nm', 'wn', 'ev'], default='nm',
                    help='energy unit for the x-axis: nm, wn, or ev')

# Choice of unit for y-axis
parser.add_argument('-yu', '--yunit', choices=['mili', 'mcd'],
                    help='energy unit for the y-axis: mili or mcd')

# Colors
parser.add_argument('-cs','--colors', nargs=2, type=str, default=['black','grey'])

# Usage of normalized gaussian
parser.add_argument('-norm','--norm_gauss', default=0, action='store_true',
                    help='Use area-normalized gaussians instead of heigth-normalized ones.')

args = parser.parse_args()

colors = args.colors

norm_g = args.norm_gauss

spec_type = 'uvvis'
if args.ir_spectra:
    spec_type = 'ir'
elif args.ecd_spectra or args.ecdv_spectra:
    spec_type = 'ecd'
elif args.secd_spectra:
    spec_type = 'secd'
elif args.vcd_spectra:
    spec_type = 'vcd'

if args.ir_spectra or args.vcd_spectra:
    args.unit = 'wn'

# Automatic labeling given spectra type
if not args.y_label:
    if args.ecd_spectra or args.ecdv_spectra or args.secd_spectra:
        args.y_label = 'R ($10^{-40}esu^2cm^2$)'
    elif args.vcd_spectra:
        args.y_label = 'R ($10^{-44}esu^2cm^2$)'
    elif args.ir_spectra:
        args.y_label = 'I (km/mol)'
    else:
        args.y_label = '$f_{obs}$'

if args.yunit:
    if not norm_g:
        print("Warning: Correct unit conversion requieres normalized gaussians.")
    if args.yunit == 'mcd':
        args.y_label = r'$\Delta \epsilon (M^{-1} cm^{-1})$'
    elif args.yunit == 'mili':
        args.y_label = r'$[\theta]$ ($\deg cm^2 dmol^{-1}$)'

show_spectrum = args.show
save_spectrum = args.nosave
export_spectrum = args.export
unit = args.unit

# Validate line widths
if not (1 <= args.linewidth_nm <= 500):
    print("warning! linewidth_nm out of [1..500], reset to 20")
    w_nm = 20
else:
    w_nm = args.linewidth_nm
#w_nm = args.linewidth_nm

if not (100 <= args.linewidth_wn <= 20000) and not (args.vcd_spectra or args.ir_spectra):
    print("warning! linewidth_wn out of [100..20000], reset to 1000")
    w_wn = 1000
else:
    w_wn = args.linewidth_wn

if not (0 < args.linewidth_ev <= 10):
    print("warning! linewidth_ev out of (0..10], reset to 0.2")
    w_ev = 0.2
else:
    w_ev = args.linewidth_ev

# Check x0/x1
if args.startx is not None and args.endx is not None and args.startx == args.endx:
    print("Warning: x0 == x1. Exit.")
    sys.exit(1)
if args.startx is not None and args.startx < 0:
    print("Warning: x0 < 0. Exit.")
    sys.exit(1)
if args.endx is not None and args.endx < 0:
    print("Warning: x1 < 0. Exit.")
    sys.exit(1)

# Read ORCA output
def read_orca_output(filename, spec_type):
    energylist = []
    intenslist = []
    found_data_section = False
    specstring_start = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
    specstring_end = 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'
    ecd_start = "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
    ecd_end = "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
    ecdv_start = "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
    ecdv_end = "Total run time"
    vcd_start = "VCD SPECTRUM CALCULATION"
    vcd_end = "throughout the entire PROP-calculation"
    secd_start = "CD SPECTRUM"
    secd_end = "sTD-DFT done"
    ir_start = "IR SPECTRUM"
    ir_end = "The epsilon (eps)"
    export_delim = " "
    try:
        with open(filename, "r") as input_file:
            for line in input_file:
                # detect program version to pick correct columns
                if "Program Version 6" in line:
                    energy_column = 4
                    intens_column = 6
                elif "Program Version" in line:
                    energy_column = 1
                    intens_column = 3
                if spec_type == "ecd":
                    if ecd_start in line:
                        found_data_section = True
                        for line in input_file:
                            if ecd_end in line:
                                break
                            if re.search(r"\d\s{1,}\d", line):
                                splitted = line.strip().split()
                                energylist.append(float(splitted[energy_column]))
                                intenslist.append(float(splitted[intens_column]))
                elif spec_type == "ecdv":
                    if ecdv_start in line:
                        found_data_section = True
                        for line in input_file:
                            if ecdv_end in line:
                                break
                            if re.search(r"\d\s{1,}\d", line):
                                splitted = line.strip().split()
                                energylist.append(float(splitted[energy_column]))
                                intenslist.append(float(splitted[intens_column]))

                if spec_type == "secd":
                    energy_column = 1
                    intens_column = 3
                    if secd_start in line:
                        found_data_section = True
                        for line in input_file:
                            if secd_end in line:
                                break
                            if re.search(r"\d\s{1,}\d", line):
                                splitted = line.strip().split()
                                energylist.append(float(splitted[energy_column]))
                                intenslist.append(float(splitted[intens_column]))
                elif spec_type == "vcd":
                    freq_column = 1
                    intens_column = 2
                    if vcd_start in line:
                        found_data_section = True
                        for line in input_file:
                            if vcd_end in line:
                                break
                            if re.search(r"\d\s{1,}\d", line):
                                splitted = line.strip().split()
                                energylist.append(float(splitted[freq_column]))
                                intenslist.append(float(splitted[intens_column]))
                elif spec_type == "ir":
                    freq_column = 1
                    intens_column = 3
                    if ir_start in line:
                        found_data_section = True
                        for line in input_file:
                            if ir_end in line:
                                break
                            if re.search(r"\d\s{1,}\d", line):
                                splitted = line.strip().split()
                                energylist.append(float(splitted[freq_column]))
                                intenslist.append(float(splitted[intens_column]))

                elif spec_type == 'uvvis':
                    if specstring_start in line:
                        found_data_section = True
                        for line in input_file:
                            if specstring_end in line:
                                break
                            if re.search(r"\d\s{1,}\d", line):
                                splitted = line.strip().split()
                                #print(splitted)
                                energylist.append(float(splitted[energy_column]))
                                intenslist.append(float(splitted[intens_column]))
    except IOError:
        print(f"'{filename}' not found")
        sys.exit(1)

    if not found_data_section:
        print(f"Searched data not found in '{filename}'")
        sys.exit(1)

    return energylist,intenslist

energylist2 = []
intenslist2 = []

energylist, intenslist = read_orca_output(args.filename, spec_type)
if args.second_file:
    energylist2, intenslist2 = read_orca_output(args.second_file, spec_type)

if args.double_axis:
    if (args.vcd_spectra or args.ir_spectra):
        unit = 'wn'
    else:
        unit = 'ev'

# Convert energies from cm⁻¹ to the chosen unit
if not (args.vcd_spectra or args.ir_spectra):
    if unit == 'nm':
        # cm⁻¹ -> nm
        energylist = [1.0e7 / wn for wn in energylist]
        energylist2 = [1.0e7 / wn for wn in energylist2]
        w = w_nm
    elif unit == 'ev':
        # cm⁻¹ -> eV
        # 1 cm⁻¹ = 1.23984193e-4 eV
        energylist = [wn * 1.23984193e-4 for wn in energylist]
        energylist2 = [wn * 1.23984193e-4 for wn in energylist2]
        w = w_ev
    else:
        # unit == 'wn', no conversion
        w = w_wn
else:
    w=w_wn

fig, ax = plt.subplots()

# Decide on step for the x range
if unit == 'ev':
    step = 0.01
else:
    step = 1.0

if args.step_size:
    step = args.step_size


xmax_val = max(energylist) if energylist else 0
plt_range_x = np.arange(0, xmax_val + w * 3, step)

#for i in range(20):
#    print(energylist[300+i])
#    print(intenslist[300+i])

if args.second_file:
    xmax_val2 = max(energylist2) if energylist2 else 0
    plt_range_x2 = np.arange(0, xmax_val2 + w * 3, step)

# Build the Gaussian sum
def gauss_sum(energylist, intenslist, x_range, w, spec_type,
        ssg=False, ssga=False, double_ax=False,
        ax = 0, colors = ['black', 'black'], norm = False):
    gauss_sum_data = []
    gauss_sum_data_all = []

    if double_ax and (not (spec_type == 'vcd' or spec_type == 'ir')):
        x_range_s = ev_to_nm(x_range)
    else:
        x_range_s = x_range

    for i, en in enumerate(energylist):
        if norm:
            single_gauss = nGauss(intenslist[i], en, x_range, w)
        else:
            single_gauss = gauss(intenslist[i], en, x_range, w)
        if ssg:
            if spec_type in ['ecd', 'ecdv', 'secd', 'vcd'] and norm:
                ax.plot(x_range_s, single_gauss * en,
                        color=colors[0], alpha=0.3)
            else:
                ax.plot(x_range_s, single_gauss,
                        color=colors[0], alpha=0.3)
        if ssga:
            if spec_type in ['ecd', 'ecdv', 'secd', 'vcd'] and norm:
                ax.fill_between(x_range_s, single_gauss * en,
                        color=colors[0], alpha=0.3)
            else:
                ax.fill_between(x_range_s, single_gauss,
                        color=colors[0], alpha=0.3)

        # R strength needs multiplication by the corresponding energy
        if spec_type in ['ecd', 'ecdv', 'secd', 'vcd'] and norm:
            gauss_sum_data_all.append(single_gauss * en)
        else:
            gauss_sum_data_all.append(single_gauss)

    gauss_sum_data = np.sum(gauss_sum_data_all, axis=0)
    return gauss_sum_data, gauss_sum_data_all

plt_range_gauss_sum_y, _ = gauss_sum(energylist, intenslist, plt_range_x,
                        w, spec_type, ssg=args.show_single_gauss,
                        ssga=args.show_single_gauss_area, double_ax=args.double_axis,
                        ax=ax, colors=colors, norm=norm_g)
if args.second_file:
    plt_sum_y2, _ = gauss_sum(energylist2, intenslist2, plt_range_x2,
                        w, spec_type, ssg=args.show_single_gauss,
                        ssga=args.show_single_gauss_area, double_ax=args.double_axis,
                        ax=ax, colors=colors)

if args.double_axis and not (args.vcd_spectra or args.ir_spectra):
    plt_range_x = ev_to_nm(plt_range_x)
    if args.second_file:
        plt_range_x2 = ev_to_nm(plt_range_x2)
    energylist = ev_to_nm(energylist)
    unit = 'nm'

if args.yunit:
    if not norm_g:
        print("Warning: Correct unit conversion requieres normalized gaussians.")
    if args.yunit == 'mcd':
        plt_range_gauss_sum_y = plt_range_gauss_sum_y / 22.94
        args.y_label = r'$\Delta \epsilon (M^{-1} cm^{-1})$'
    elif args.yunit == 'mili':
        plt_range_gauss_sum_y = (plt_range_gauss_sum_y / 22.97) * 32980 #* c * l
        args.y_label = r'$[\theta]$ ($\deg cm^2 dmol^{-1}$)'

# Plot stick spectrum
if args.show_sticks:
    _, stemlines, _ = ax.stem(energylist, intenslist, linefmt=colors[0], markerfmt=" ", basefmt=" ")
    plt.setp(stemlines, alpha=0.4, linewidth=1)

# Plot the convoluted spectrum
if show_conv_spectrum:
    ax.plot(plt_range_x, plt_range_gauss_sum_y, color=colors[0], linewidth=0.8)
    if args.second_file:
        ax.plot(plt_range_x2, plt_sum_y2, color=colors[1], ls='--', linewidth=0.8)


# Peak detection
peaks, _ = find_peaks(plt_range_gauss_sum_y, height=0)

# Label peaks if desired
if show_conv_spectrum and args.label_peaks:
    for peak in peaks:
        ax.annotate(peak,
                    xy=(peak, plt_range_gauss_sum_y[peak]),
                    ha="center", rotation=90, size=8,
                    xytext=(0, 5),
                    textcoords='offset points')

# Axis labels
if unit == 'nm':
    ax.set_xlabel(x_label_nm)
elif unit == 'ev':
    ax.set_xlabel(x_label_ev)
else:
    ax.set_xlabel(x_label_wn)

ax.set_ylabel(args.y_label)
#ax.set_title(spectrum_title, fontweight=spectrum_title_weight)
#ax.get_yaxis().set_ticks([])  # remove y-axis ticks
#plt.tight_layout()
#fig.tight_layout()

# Minor ticks
if minor_ticks:
    ax.minorticks_on()

energylist = np.array(energylist)
# X-limits
if args.startx is not None:
    xlim_autostart = args.startx
else:
    if (args.vcd_spectra or args.ir_spectra):
        xlim_autostart = rounddown(min(energylist) - w * 3, unit) if energylist.any() else 0
    else:
        xlim_autostart = rounddown(min(energylist) - w * 3, unit) if energylist.any() else 0

if args.endx is not None:
    xlim_autoend = args.endx
else:
    xlim_autoend = roundup(max(plt_range_x), unit)

if xlim_autostart < 0:
    plt.xlim(0, xlim_autoend)
else:
    plt.xlim(xlim_autostart, xlim_autoend)

# Y-limits
if args.starty is not None:
    ylim_autostart = args.starty
else:
    ylim_autostart = math.floor(min(plt_range_gauss_sum_y))

if args.endy is not None:
    ylim_autoend = args.endy
else:
    ylim_autoend = math.ceil(max(plt_range_gauss_sum_y))

plt.ylim(ylim_autostart, ylim_autoend)


# Dynamic y-limits
#xmin_plot = int(ax.get_xlim()[0])
#xmax_plot = int(ax.get_xlim()[1])
#if xmin_plot > xmax_plot:
#    ymax = max(plt_range_gauss_sum_y[xmax_plot:xmin_plot]) if xmax_plot < xmin_plot else 1
#    ax.set_ylim(0, ymax + ymax * 0.1)
#else:
#    ymax = max(plt_range_gauss_sum_y[xmin_plot:xmax_plot]) if xmin_plot < xmax_plot else 1
#    ax.set_ylim(0, ymax + ymax * 0.1)
xlimits = ax.get_xlim()  # (xmin, xmax) in float
xlow, xhigh = min(xlimits), max(xlimits)
mask = (plt_range_x >= xlow) & (plt_range_x <= xhigh)
#if np.any(mask):
#    ysub = plt_range_gauss_sum_y[mask]
#    ymax = np.max(ysub)
#    ax.set_ylim(0, ymax + 0.1 * ymax)
#else:
#    ax.set_ylim(0, 1)

# Linear locator
if linear_locator:
    ax.xaxis.set_major_locator(plt.LinearLocator())

# Grid
if show_grid:
    ax.grid(True, which='major', axis='x', color='black', linestyle='dotted', linewidth=0.5)

# Double axis
if args.double_axis:
    if (args.vcd_spectra or args.ir_spectra):
        secax = ax.secondary_xaxis(
                "top",
                functions=(cm1_to_mev, mev_to_cm1),
            )
        secax.set_xlabel("E (meV)")
    else:
        secax = ax.secondary_xaxis(
                "top",
                functions=(nm_to_ev, ev_to_nm),
            )
        secax.set_xlabel("E (eV)")

# Increase figure size by a factor N
#N = 1
#params = plt.gcf()
#plSize = params.get_size_inches()
#params.set_size_inches((plSize[0] * N, plSize[1] * N))

#fig.subplots_adjust(
#    left=0.15,
#    right=0.95,
#    bottom=0.15,
#    top=0.9
#)
# Article params:
plt.rcParams.update({
    'figure.figsize': (3.4, 2.1), # Single column width is typically 3.4 inches
#    'figure.autolayout': True,     # Automatically adjust subplot params for tight layout
    'font.size': 8,                # Match the main text font size
    'axes.labelsize': 10,          # Axis label font size
    'legend.fontsize': 8,          # Legend font size
    'xtick.labelsize': 8,          # X-tick label font size
    'ytick.labelsize': 8,          # Y-tick label font size
    'lines.linewidth': 1.5,        # Line width for plots
    'savefig.dpi': 300,            # High DPI for publication quality
#    'savefig.format': 'pdf',       # Save as vector graphic for best quality
#    'text.usetex': True,           # Use LaTeX for text rendering if available (optional)
#    'font.family': 'serif'         # Match the article's font style
})

# X axis labels frequency
#nm
#xlabels = np.arange(xlow,xhigh+1,50)
#if len(xlabels) < 10:
#    ax.set_xticks(xlabels)
data_range = xhigh - xlow
k = max(1, int(np.ceil(data_range / (args.x_multiplier * 10 - 1 )))) # Ensure < 10 ticks
step_size = args.x_multiplier * k

# 3. Apply the locator
ax.xaxis.set_major_locator(ticker.MultipleLocator(step_size))
if args.double_axis:
    secax.xaxis.set_major_locator(ticker.MultipleLocator(args.multi2))


# Save the plot if requested
if save_spectrum:
    filename, file_extension = os.path.splitext(args.filename)
    plt.savefig(f"{filename}_{spec_type}.png", dpi=figure_dpi, bbox_inches='tight')

# Export data if requested
if export_spectrum and len(ax.lines) > 0:
    # The first line is presumably the convolved spectrum
    plotdata = ax.lines[0]
    xdata = plotdata.get_xdata()
    ydata = plotdata.get_ydata()
    xlimits = plt.gca().get_xlim()

    if normalize_export:
        ymax_norm = max(ydata)
    else:
        ymax_norm = 1
        normalize_factor = 1

    try:
        with open(args.filename + "-mod.dat", "w") as output_file:
            for i in range(len(xdata)):
                if xlimits[0] <= xdata[i] <= xlimits[1]:
                    out_x = xdata[i]
                    out_y = ydata[i] / ymax_norm * normalize_factor
                    output_file.write(f"{out_x}{export_delim}{out_y}\n")
    except IOError:
        print("Write error. Exit.")
        sys.exit(1)

if show_spectrum:
    plt.show()
