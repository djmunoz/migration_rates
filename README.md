# migration_rates
Python module for the (semi)analytical estimate of migration rates of close-in planets via Lidov-Kozai oscillations

To add to Python modules, simply include

import migration_rates as mr

in your script. You might need to install other Python packages for this to work.


To create a migration fraction plot (close-in fraction+ disruption fraction), you can run

$] python migration_rates_examples.py $mp $chi $rocky

where $mp is the mass of the planet (in Jupiter masses), chi is the dimensionless time lag, and rocky=0 if
the planet is a gas giant and 1 if it is a rocky planets

For examples,

$] python migration_rates_examples.py 1 10 0

creates the figure: migration_fraction_integrated_mplanet1_chi10.pdf

You can reproduce all the figures in Munoz, Lai & Liu (2016) by running

$] python make_figures_for_munozetal.py

which may make use of data in the data/ directory

Some figures will be created using pre-computed migration fractions, but some less detailed version 
of these figures will be computed from scratch. Using the option PARALLEL or parallel in the scripts
will speed up the Monte-Carlo integration of the migration fractions. For that, you need to have the
module joblib installed.

Figures created by make_figures_for_munozetal.py are

figure1.pdf
figure2.pdf
figure3.pdf
figure4a_light.pdf
figure4a.pdf
figure4b_light.pdf
figure4b.pdf
figure4c_light.pdf
figure4c.pdf
figure5_light.pdf
figure5.pdf


