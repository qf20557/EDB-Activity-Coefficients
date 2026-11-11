# EDB-Activity-Coefficients
An analysis pipeline to convert EDB evaporation data to aerosol activity coefficient plots in comparison to AIOMFAC.

The activity coefficient at a given mole fraction is the ratio of two evaporation rates (gradients of radius² against
time): the one measured in the EDB, taken from a fit to the cleaned data in time bins, and the one from a simulated
ideal evaporation (activity coefficient = 1) at the same composition.

## Before you start
1. Check `compounds.json` and make sure every chemical you are working with is in there.
2. Add all relevant AIOMFAC-web output files to the `AIOMFAC simulations` folder.
3. Run `EDB Pipeline.py` (`uv run "EDB Pipeline.py"`).

## Stage 1 · Clean & fit
1. **Add files...**: raw EDB files (or cleaned data saved earlier), in acquisition order. The list shows each file's size.
2. **Load / plot**. Loading runs in the background with a progress bar and a Cancel button.
3. Choose the droplet **Compound** (sets the refractive-index correction of the radius).
4. Clean the data with the **Cull** and **Keep box** modes (drag a rectangle on the plot; **Undo cull** steps back).
5. **Fit** the whole trace, or drag a time region in **Fit region** mode.
6. **Save cleaned data** and **Export fit CSV** to a directory of your choice, then press **Next**.

## Stage 2 · MFS & gradients
7. Set up the droplet vial formulation in the boxes at the top (everything in `compounds.json` is listed).
8. Set the interval start and end (type them, or drag on the plot) and press **Calculate intervals**. **Bin width** is
   `auto` by default: the interval is split into equal bins, as many as makes each closest to 1000 s, so the bins
   always fill the interval exactly. Type a number of seconds for fixed-width bins instead (0 = one interval).
9. **Export CSV** (the binned gradients), then press **Next**.

## Stage 3 · Activity coefficients
10. Load the ideal evaporation CSV from your own simulation (the binned gradients arrive from stage 2, or load them).
11. Choose the relevant AIOMFAC file from the dropdown menu.
12. Customise the publication style plot and save it to your device!

## Working with large files
Raw files of several GB are expected. A 2 GB file (56 million rows) loads in about 10 seconds and needs a little over 1 GB
of memory: 16 bytes per row, whatever the size of the text file.

- **Plot every Nth point** is `auto` by default: the plot shows at most 40,000 points however many are loaded. Half are
  an even sample and half are the lowest and highest radius in each stretch of the trace, so every spike you may want
  to cull stays visible. Zoom in and the points of that region are drawn in full. Type a number instead of `auto` to
  get a plain every-Nth plot.
- Culling, keeping and fitting always act on **all** the points, not only the ones drawn.
- **Save cleaned data** offers two formats. CSV is readable anywhere but slow to write for large data (minutes for
  tens of millions of rows; it runs in the background and can be cancelled). `NumPy binary (*.npz)` holds the same information, is about a third of the size, and
  saves and reloads in seconds; both stage 1 and stage 2 open either format.

## Tests
`uv run pytest`
