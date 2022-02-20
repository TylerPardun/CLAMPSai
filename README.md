# CLAMPSai
The Collaborative Lower Atmospheric Mobile Profiling System Analysis and Imagery repository has a collection of Jupyter Notebooks and python scripts used to analyze CLAMPS data gathered from the University of Oklahoma and National Severe Storms Laboratory.

These scripts have been used to analyze data collected within the convective boundary layer from the VORTEX-SE field campaigns of 2016-2019. Data can be downloaded from https://www.eol.ucar.edu/field_projects/vortex-se. 

# The Setup
Ater data has been downloaded and put into an initial directory (data_dir), there needs to be a sub-directory that organizes the data by year in order to parse through the data using clamps_parser.py. For instance, a thermodynamic retrieval file from the AERI in 2016 shall look like 

data_dir/2016/thermo/[list of files]

For LiDAR data in 2016, specifically holding VAD scans:
data_dir/2016/lidar/vad/[list of files]

For Vertical stares in 2016:
data_dir/2016/lidar/vs/[list of files]

And finally for surface data from the MetTower onbaord CLAMPS in 2016:
data_dir/2016/surface/[list of files]


To then parse through a file or list of files, the user will need to alter the datetime objects listed in clamps_parser.py. Once this is completed, the code will return a list of dictionaries corresponding to the number of datetime objects for each convective mode case and write to a pickle file for analysis. An example is included in the file to play around with.

# Visualization and Analysis
Included is a few jupyter notebooks that vislauizes and analyzes CLAMPS data on a case-by-case basis. Inside each notebook is a detailed description of how things work and what is being plotted.

Should there be any questions, please do not hesitate to reach me! My email is listed below. Have fun with it!

My email: tyler.pardun@noaa.gov
