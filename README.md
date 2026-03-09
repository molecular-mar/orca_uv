# orca-uvm
Python code for plotting script for plotting spectra from [ORCA](https://orcaforum.kofo.mpg.de) 
output files.

Currently it manages vibrational (**IR**, **VCD**) and electronic (**UV-Vis**, **ECD**) transitions data.
Build taking as base [orca_uv](https://github.com/radi0sus/orca_uv), this tools goes beyond UV data (uvm stands to "UV and More"),
adding capabilities for other types of spectra. 

It directly uses data from output, allowing major control over the transformations involved for broadening the calculated signals.
While more complex than its predecesor, it remains useful to understand many of the different constants used in other tools.

Documentation still under construction. Detail might still refer to the original *orca_uv* code. 

## Capabilities

Types of spectra:

* UV-Vis
* Electronic Circular Dichroism (ECD)
  * Length or Velocity gauge
  * sTDDFT-calculated
* IR
* Vibrational CD (VCD)

Functions for broadening:

* Gaussian
    * Area-normalized (correct physical interpretation)
    * Height-normalized (visually easier to understand, current default)

Plotting options:

* Signal sticks
* Per-signal plotting of broadening function (lines or area-colored)
* Different units for both axis, according to user and type of spectra
* Double X-axis for $\lambda$/Energy or Wavenumber/Energy dual description
* Double Y-axis, allowing to display Spectra/Signal it two different units
* Two plots on same figure, for comparisson (i.e., plotting two enantiomers CD)
* Peak detection and labeling
* X and Y axis range control
* Color assignment when two curves are present

Generated figure follows the output file base name and includes the type of spectra plotted.

## External modules
 `numpy` 
 `scipy`
 `matplotlib`
 
## Quick start
 Start the script with:
```console
python3 orca-uv.py filename
```
it will save the plot as PNG bitmap:
`filename-abs.png`

Execution without filename will print the full list of options.

## Command-line options
- `filename` , required: filename
- `-s` , optional: shows the `matplotlib` window
- `-n` , optional: do not save the spectrum
- `-e` , optional: export the line spectrum in a csv-like fashion; filename of the export is input filename + "-mod.dat"
- `-u` , optional: specify the energy unit for the x-axis. Allowed choices are:
  - `nm` (wavelength, default),
  - `wn` (wave number in cm<sup>-1</sup>),
  - `ev` ev (electronvolts). (Note: Using -u wn replaces the old -pwn flag for plotting in wave numbers.)
- `-wnm` `N` , optional: line width of the gaussian for the nm scale (default is `N = 20`).
- `-wwn` `N` , optional: line width of the gaussian for the cm<sup>-1</sup> scale (default is `N = 1000`).
- `-wev` `N` , optional: line width of the gaussian for the eV scale (default is `N = 0.2`).
- `-x0`  `N` , optional: start spectrum at N nm or N cm<sup>-1</sup> (`x0 => 0`).
- `-x1`  `N` , optional: end spectrum at N nm or N cm<sup>-1</sup> (`x1 => 0`).

