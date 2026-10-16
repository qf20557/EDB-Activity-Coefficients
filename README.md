# EDB-Activity-Coefficients
An analysis pipeline to convert EDB evaporation data to aerosol activity coefficient plots in comparison to AIOMFAC.

READ FIRST:
1. check compounds.json and make sure every chemical you are working with is in there
2. add all relevant AIOMFAC web output files to the 'AIOMFAC simulations' folder
3. Open EDB Pipeline.py
4. Load in raw EDB or CSV data, clean and fit using the cull and fit buttons.
5. Save cleaned EDB data and fit CSV to a directory of your choice
6. Proceed to the second stage
7. Set up the droplet vial formulation in the above boxes (everything in compounds.json should be there)
8. Set bin size and apply to plot
9. Export binned data
10. Proceed to the third stage
11. Upload all relevant data (binned gradients from stage 2, and an ideal evaporation plot from your own simulation)
12. Choose the relevant AIOMFAC file from the dropdown menu
13. Customise publication style plot
14. Save to your device!