===========================================================================

                          Source code for the artice 
   
                 Recurrent amplification of grid-cell activity
                    by Tiziano D'Albis and Richard Kempter

============================================================================


REQUIREMENTS
============

All source code is in Python language and was tested in the following environmnent:

- Python 2.7.15
- Matplotlib 2.2.3 
- NumPy 1.15.1
- Scipy 1.2.1
- Pandas 0.24.2
- Psutils 5.6.3
- Pytables 3.5.2

The easiest way to run the code is to create an Anaconda environment using the file 'conda_env.yml' provided in the root folder of this package.
This can be done with the command:

conda env create -f conda_env.yml

This will create a new conda environment named 'grid_amp'. Note that this environment may contain more packages then actually needed (i.e., it provides more than the minimal set of required dependencies).


USAGE
=====

0. Before starting make sure to edit the file "config.json" in the project root.
   The file contains the paths on your local machines where the simulation results (RESULTS_PATH)
   and the figures (FIGURES_PATH) are going to be saved
   
2D MODEL
--------

1. Run the script "amp_paper_2d_main.py". This generates and saves to disk all the required simulation data to be plotted.  Note: if you need only a subset of the result you can easily filter out simulations in the main section of the program.

2. Run the snippet of code that generates the required figure. 
   The script "amp_paper_2d_fig_main.py" contains the code to generate Figure 1
   The script "amp_paper_2d_fig_temporal.py" contains the code to generate Figures 2 and 3
   The script "amp_paper_2d_fig_grid_index.py" contains the code to generate Figure 8
   The script "amp_paper_2d_fig_noise.py" contains the code to generate all the remaining figures of the 2D model

3. Find the generated figures saved as SVG and PNG in the selected target folder (see FIGURES_PATH in config.json)


1D MODEL
--------

1. Run the script "amp_paper_1d_plots.py" to generate all figures related to the 1D model
   
2. Find the generated figures saved as SVG and PNG in the selected target folder (see FIGURES_PATH in config.json)   

TECHNICAL NOTE
==============

This project imports an external git repository (grid_utils) using  ``git subtree''.
For a tutorial see: http://atlassianblog.wpengine.com/2013/05/alternatives-to-git-submodule-git-subtree/
