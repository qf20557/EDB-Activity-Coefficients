# EDB-Activity-Coefficients
An interactive Streamlit analysis pipeline to convert EDB evaporation data to aerosol activity coefficient plots in comparison to AIOMFAC.

The activity coefficient at a given mole fraction is the ratio of two evaporation rates (gradients of radius² against time): the one measured in the EDB, taken from a fit to the cleaned data in time bins, and the one from a simulated ideal evaporation (activity coefficient = 1) at the same composition.

## Before you start
1. Check `compounds.json` and make sure every chemical you are working with is in there.
2. Add all relevant AIOMFAC-web output files to the `AIOMFAC simulations` folder.
3. Run `EDB Pipeline.py`:
   ```powershell
   streamlit run "EDB Pipeline.py"
   ```
   (or `python "EDB Pipeline.py"`).

## Stage 1 · Clean & fit
1. **Add files...**: choose raw EDB files (or cleaned data saved earlier) to add them to the acquisition-order list (use **Move up**, **Move down**, **Remove selected**, or **Clear** to adjust order).
2. **Load / plot**: streams and parses files in parallel with a progress bar.
3. Choose the droplet **Compound** (sets the refractive-index correction of the radius).
4. Clean the data interactively with **Cull** and **Keep box** drag modes directly on the plot (**Undo cull** and **Restore all** step back).
5. **Fit** the whole trace, or drag horizontally in **Fit region** mode.
6. **Save cleaned data** (`.npz` or `.csv`) and **Export fit CSV**, then click **Next: MFS & gradients →**.

## Stage 2 · MFS & gradients
7. Set up the initial wet droplet vial formulation in the boxes at the top (all compounds in `compounds.json` are listed).
8. Set the interval start and end (type them, or drag horizontally on the plot in **Select interval** mode) and click **Calculate intervals**. **Bin width** is `auto` by default (equal bins closest to 1000 s that fill the interval exactly; `0` = single interval).
9. **Export CSV** (binned gradients), then click **Next: activity coefficients →**.

## Stage 3 · Activity coefficients & Publication plot
10. **Load theoretical CSV** from your simulation (binned experimental gradients arrive automatically from stage 2, or **Load experimental CSV (override)**).
11. Choose the relevant AIOMFAC simulation file from the dropdown menu.
12. Switch to **Publication plot...** to customise colours, markers, error bars/bands, axes, and typography, and save at 1200 dpi (`PNG`, `TIFF`) or vector (`PDF`, `SVG`, `EPS`).

## Working with large files
Raw files of several GB (tens of millions of rows) are supported natively:
- **Fast direct loading**: **Add files...** queues file paths directly on disk (no browser upload copy), and clean numeric blocks are parsed zero-copy in parallel using PyArrow C++ (~500+ MB/s), requiring 16 bytes of RAM per row.
- **All points plotted**: **100% of retained points** are always plotted regardless of dataset size. Every point is projected onto a high-DPI raster layer using compiled Numba kernels in milliseconds, keeping isolated single-point spikes clearly visible and zooming sharp at any scale.
- **Instant culling & fitting**: culling updates the boolean keep-mask in place without copying the trace, and re-binning/fitting runs in a single compiled pass.
- **Saving cleaned data**: `NumPy binary (*.npz)` is about a third of the size of CSV and saves/reloads tens of millions of rows in seconds; both stage 1 and stage 2 read `.npz` and `.csv`.
