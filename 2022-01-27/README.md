# How to generate spatial immune profiles

There are two methods for you to choose from: (1) using a [Python Jupyter notebook](#step-3a-python-jupyter-notebook-method) or (2) using [raw Python](#step-3b-raw-python-method). Both methods show you fully how to go from an input datafile to the spatial immune profiles (SIPs) from scratch on your computer. Both methods require you to have a sufficient installation of Python set up on your local computer (instructions [below](#step-2-set-up-python-appropriately)); in both methods you must use this version of Python.

I am happy to help run this code; feel free to reach out to [andrew.weisman@nih.gov](mailto:andrew.weisman@nih.gov).

At various stages the pipeline in both methods will "checkpoint" itself, saving intermediate datafiles that took significant time to generate so that the workflow can be re-run to completion relatively quickly. To remove these checkpoints and to therefore re-run a section, delete the corresponding `.pkl` files in the `results/processed_data/slices_AxB` directory.

## Step 1: Set up the files/directories appropriately

(1) Inside some main directory (called the "project directory" in the code) on your computer (e.g., `C:\Users\weismanal\projects\gmb2` on Andrew's Windows laptop), create three directories: `data`, `repo`, and `results`.

(2) Inside the `data` directory, place the file called `Consolidated_data.txt` that Andrew referenced in his email on 6/23/21. This is the original file that Houssein created. It is a tab-separated text file containing all the data from the original patient cohort.

(3) Clone [this repository](https://github.com/fnlcr-bids-sdsi/time_cell_interaction.git) to the `repo` directory. You can use, e.g., [GitHub Desktop](https://desktop.github.com) to do this, e.g., using the settings:

<img src="./github_desktop_clone_settings.png" width="400" />

**Note:** The "Local path" field in the image above should probably have "\repo" appended to it. Don't worry, you can't go wrong because the main script will check to ensure that all paths are set up correctly; it will give you a warning if you've set it up incorrectly!

If you are using Git on the command line, from the project directory, run something like:

```bash
git clone https://github.com/fnlcr-bids-sdsi/time_cell_interaction.git repo
```

## Step 2: Set up Python appropriately

You can create a Python environment sufficient for generating the SIPs by installing the [Conda environment manager](https://docs.conda.io/en/latest) and then running from the `repo` directory:

```bash
conda env create -f python_environment.yml
conda activate spatial_immune_profiles
```

## Step 3a: Python Jupyter notebook method

Running Python Jupyter notebooks on your local computer is a very common task with ample help online.

To use this method, open and go through the [`main.ipynb` Jupyter notebook](./main.ipynb).

## Step 3b: Raw Python method

In the `repo` directory you created [above](#step-1-set-up-the-filesdirectories-appropriately), run:

```bash
python main.py
```

## Results

The spatial immune profiles (the averaged-over-ROI plots of the P values for the densities) will generate in `results/webpage/slices_AxB/real/average_density_pvals-real-*.png`.

The corresponding numerical data will generate in `results/processed_data/slices_AxB/average_density_pvals-real.csv`.

The plots of the ROIs will generate in `results/webpage/slices_AxB/real/roi_*.png`.
