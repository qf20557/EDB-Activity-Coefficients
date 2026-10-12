# EDB-Activity-Coefficients
An analysis pipeline to convert EDB evaporation data to aerosol activity coefficient plots in comparison to AIOMFAC.

READ FIRST:
1. check compounds.json and make sure every chemical you are working with is in there
2. Open 'EDB Plotter.py'
	Add EDB Files - plot r2, plot every 1th point, select monotone convex fit
	click 'Load/Plot'
	process plot using Cull and Zoom, drawing boxes on the plot to select the region
	click 'Save altered data'
	click 'Fit'
	Export fit CSV, title appropriately and save
3. Open 'EDB Fit Processing.py'
	select the correct compound in A,B and Solvent
	input the masses in amounts made up into the solution vial
	load the fit CSV from part 2
	load the cleaned EDB CSV from part 2
	interval start ~5s
	plot every 1th point
	set axis limits
	click 'Apply'
	set bin width for data binning
	click 'Calculate intervals'
	click 'Export CSV', name appropriately and save
4. Open 'EDB Activity Coefficients.py'
	load theoretical CSV (DropletLab simulation)
	load experimental CSV from part 3
	load AIOMFAC file
	change label to whatever compound abbreviation you want
	change axes
	click 'Export activity coefficients CSV'
	click 'Publication plot...'
	customise and save
	
