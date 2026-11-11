"""EDB evaporation data -> activity coefficients.

The maths modules (raw_data, trace, culling, optics, fitting, fit_io, mfs, activity, aiomfac, plotting) do not import
Tk, so each can be imported and tested on its own. The Tk interface lives in edb.gui; run it with `EDB Pipeline.py`.

Raw traces can be several GB. They are held as two float arrays plus a keep mask (trace.Trace) and are read, culled,
fitted, drawn and saved block by block, so nothing in the pipeline copies a whole trace.
"""
