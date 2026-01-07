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
from scipy.signal import find_peaks     # peak detection

# global constants
found_uv_section = False
specstring_start = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
specstring_end = 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'
ecd_start = "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
ecd_end = "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
vcd_start = "VCD SPECTRUM CALCULATION"
vcd_end = "throughout the entire PROP-calculation"
ir_start = "IR SPECTRUM"
ir_end = "The epsilon (eps)"
export_delim = " "

# Default line widths (FWHM) in respective units
w_wn = 1000                 # cm⁻¹
w_nm = 20                   # nm
w_ev = 0.2                  # eV  ### NEW ###

# Plot configuration
spectrum_title = "Absorption spectrum"
spectrum_title_weight = "bold"
#y_label = "intensity"
x_label_wn = r'Wavenumber (cm$^{-1}$)'
x_label_nm = r'$\lambda$ (nm)'
x_label_ev = r'Energy (eV)'
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
minor_ticks = True
show_grid = False
linear_locator = True

# global lists
energylist = []
intenslist = []
gauss_sum = []

def gauss(a, m, x, w):
    """
    Calculate a Gaussian line shape.
      a : amplitude (intensity)
      m : center (position)
      x : array of x values
      w : line width (FWHM)
    """
    return a * np.exp(- (np.log(2) * ((m - x) / w) ** 2))

def roundup(x, unit):
    """Round x up to a nice boundary depending on the chosen unit."""
    if unit == 'nm':
        factor = 10
    elif unit == 'ev':
        factor = 0.1
    else:  # 'wn'
        factor = 100
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

# Plot ECD instead of UV-Vis
parser.add_argument('-ecd', '--ecd_spectra', default=0, action='store_true',
                    help='plot electronic circular dichroism')

# Plot VCD
parser.add_argument('-vcd', '--vcd_spectra', default=0, action='store_true',
                    help='plot vibrational circular dichroism')

# Plot IR
parser.add_argument('-ir', '--ir_spectra', default=0, action='store_true',
                    help='plot IR spectra')

# Plot sticks
parser.add_argument('-stick', '--show_sticks', default=0, action='store_true',
                    help='plot signal sticks')

# Label peaks
parser.add_argument('-peak', '--label_peaks', default=0, action='store_true',
                    help='plot signal sticks')

# Plot sticks
parser.add_argument('-ylab', '--y_label', type=str, default='Intensity',
                    help='Label for y axis')

# Activate single gaussian plotting
parser.add_argument('-ssg', '--show_single_gauss', default=0, action='store_true',
                    help='show single gaussians')

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

# Export data
parser.add_argument('-e', '--export', default=0, action='store_true',
                    help='export data to file')

# Choice of unit for x-axis
parser.add_argument('-u', '--unit', choices=['nm', 'wn', 'ev'], default='nm',
                    help='energy unit for the x-axis: nm, wn, or ev')

args = parser.parse_args()

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
try:
    with open(args.filename, "r") as input_file:
        for line in input_file:
            # detect program version to pick correct columns
            if "Program Version 6" in line:
                energy_column = 4
                intens_column = 6
            elif "Program Version" in line:
                energy_column = 1
                intens_column = 3
            if args.ecd_spectra:
                if ecd_start in line:
                    found_uv_section = True
                    for line in input_file:
                        if ecd_end in line:
                            break
                        if re.search(r"\d\s{1,}\d", line):
                            splitted = line.strip().split()
                            energylist.append(float(splitted[energy_column]))
                            intenslist.append(float(splitted[intens_column]))
            elif args.vcd_spectra:
                freq_column = 1
                intens_column = 2
                if vcd_start in line:
                    found_uv_section = True
                    for line in input_file:
                        if vcd_end in line:
                            break
                        if re.search(r"\d\s{1,}\d", line):
                            splitted = line.strip().split()
                            energylist.append(float(splitted[freq_column]))
                            intenslist.append(float(splitted[intens_column]))
            elif args.ir_spectra:
                freq_column = 1
                intens_column = 3
                if ir_start in line:
                    found_uv_section = True
                    for line in input_file:
                        if ir_end in line:
                            break
                        if re.search(r"\d\s{1,}\d", line):
                            splitted = line.strip().split()
                            energylist.append(float(splitted[freq_column]))
                            intenslist.append(float(splitted[intens_column]))

            else:
                if specstring_start in line:
                    found_uv_section = True
                    for line in input_file:
                        if specstring_end in line:
                            break
                        if re.search(r"\d\s{1,}\d", line):
                            splitted = line.strip().split()
                            energylist.append(float(splitted[energy_column]))
                            intenslist.append(float(splitted[intens_column]))

except IOError:
    print(f"'{args.filename}' not found")
    sys.exit(1)

if not found_uv_section and not (args.vcd_spectra or args.ir_spectra):
    print(f"'{specstring_start}' not found in '{args.filename}'")
    sys.exit(1)

# Convert energies from cm⁻¹ to the chosen unit
if not (args.vcd_spectra or args.ir_spectra):
    if unit == 'nm':
        # cm⁻¹ -> nm
        energylist = [1.0e7 / wn for wn in energylist]
        w = w_nm
    elif unit == 'ev':
        # cm⁻¹ -> eV
        # 1 cm⁻¹ = 1.23984193e-4 eV
        energylist = [wn * 1.23984193e-4 for wn in energylist]
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

xmax_val = max(energylist) if energylist else 0
plt_range_x = np.arange(0, xmax_val + w * 3, step)

# Build the Gaussian sum
for i, en in enumerate(energylist):
    if args.show_single_gauss:
        ax.plot(plt_range_x, gauss(intenslist[i], plt_range_x, en, w),
                color="grey", alpha=0.5)
    if args.show_single_gauss_area:
        ax.fill_between(plt_range_x, gauss(intenslist[i], plt_range_x, en, w),
                        color="grey", alpha=0.5)
    gauss_sum.append(gauss(intenslist[i], plt_range_x, en, w))

plt_range_gauss_sum_y = np.sum(gauss_sum, axis=0)

# Plot the convoluted spectrum
if show_conv_spectrum:
    ax.plot(plt_range_x, plt_range_gauss_sum_y, color="black", linewidth=0.8)

# Plot stick spectrum
if args.show_sticks:
    ax.stem(energylist, intenslist, linefmt="dimgrey", markerfmt=" ", basefmt=" ")

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
plt.tight_layout()

# Minor ticks
if minor_ticks:
    ax.minorticks_on()

# X-limits
if args.startx is not None:
    xlim_autostart = args.startx
else:
    xlim_autostart = rounddown(min(energylist) - w * 3, unit) if energylist else 0

if args.endx is not None:
    xlim_autoend = args.endx
else:
    xlim_autoend = roundup(max(plt_range_x), unit)

if xlim_autostart < 0:
    plt.xlim(0, xlim_autoend)
else:
    plt.xlim(xlim_autostart, xlim_autoend)

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

# Increase figure size by a factor N
N = 1
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches((plSize[0] * N, plSize[1] * N))

# Save the plot if requested
if save_spectrum:
    filename, file_extension = os.path.splitext(args.filename)
    plt.savefig(f"{filename}-abs.png", dpi=figure_dpi)

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
