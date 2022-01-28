class TIMECellInteraction:
    '''
    Instantiation of this class mainly loads Consolidata_data.txt into a Pandas dataframe (or reads in a simulated one in the case of simulated data) and performs some preprocessing on it

    It will create a pickle file (initial_data.pkl) of the read-in and preprocessed data, unless the file already exists, in which case this step is skipped
    '''

    def __init__(self, coord_units_in_microns, min_coord_spacing, project_dir, input_data_filename, allow_compound_species, mapping_dict, nslices=10, thickness_new=4, simulate_data=False, species_equivalents={}, **kwargs):

        # Import relevant module
        import os

        # Set local variables
        thickness = thickness_new / coord_units_in_microns  # we want "thickness" to be in the same units as the coordinates in the input file. Essentially, we're just making sure the units match from the get-go
        dataset_name = 'slices_{}x{}'.format(nslices, thickness_new)
        pickle_dir = os.path.join(project_dir, 'results', 'processed_data', dataset_name)
        csv_file = os.path.join(project_dir, 'data', input_data_filename)
        webpage_dir = os.path.join(project_dir, 'results', 'webpage', dataset_name)

        # Set attributes
        self.dataset_name = dataset_name
        self.project_dir = project_dir
        self.input_data_filename = input_data_filename
        self.webpage_dir = webpage_dir
        self.mapping_dict = mapping_dict
        self.nslices = nslices
        self.thickness = thickness
        self.coord_units_in_microns = coord_units_in_microns
        self.min_coord_spacing = min_coord_spacing
        self.thickness_new = thickness_new

        # These next block isn't technically needed but it helps to set these here to help for linting purposes
        # These are set in this method but not saved in the traditional way (instead, using make_pickle_dict())
        self.pickle_dir = pickle_dir  # directory for storing the processed data, i.e., pickle files
        self.unique_species = []
        self.doubling_type = None
        self.unique_slides = []
        self.is_real_data = None
        self.compound_species_allowed = None
        self.csv_file = csv_file  # input datafile from Houssein... better variable name is input_pathname
        self.plotting_map = []
        self.num_colors = None

        # These are set in other functions in this class but not saved in the traditional way (instead, using make_pickle_dict())
        self.data_by_slide = []
        self.dr = None
        self.k_max = None
        self.min_nvalid_centers = None

        # Assign local variables that aren't the same as those inputted in order to save them later using make_pickle_dict()
        is_real_data = not simulate_data
        compound_species_allowed = allow_compound_species

        # Constant
        pickle_file = 'initial_data.pkl'

        # If the pickle file doesn't exist...
        if not os.path.exists(os.path.join(pickle_dir, pickle_file)):

            # If requesting simulated data...
            if simulate_data:

                # ...generate simulated data
                midpoints = kwargs['midpoints']
                max_real_area = kwargs['max_real_area']
                # min_coord_spacing = kwargs['min_coord_spacing']  # I believe this refers to the minimal spacing between the input coordinates, e.g., 0.5 (original data), 0.1 probably (new data)
                mult = kwargs['mult']
                self.doubling_type = kwargs['doubling_type']
                csv_file = None
                self.data = get_simulated_data(kwargs['doubling_type'], midpoints, max_real_area, min_coord_spacing, mult)

            else:

                # ...otherwise, read in the data from the CSV file
                # self.csv_file = kwargs['csv_file']
                self.csv_file = csv_file
                doubling_type = None
                # self.data = get_consolidated_data(kwargs['csv_file'])
                self.data = get_consolidated_data(csv_file)

            # Preprocess the data, i.e., the pandas dataframe
            self.phenotypes = self.preprocess_dataframe(allow_compound_species)

            # Get the plotting map, number of colors, unique species, and the unique slides
            plotting_map, _, _, unique_slides = get_dataframe_info(self.data, self.phenotypes, mapping_dict)

            # Combine marker combinations into the actual species they represent
            plotting_map, data, num_colors, unique_species = incorporate_species_equivalents(species_equivalents, plotting_map, self.data)
            self.data = data

            # Save the data to a pickle file
            self.make_pickle_dict(['pickle_dir', 'is_real_data', 'compound_species_allowed', 'doubling_type', 'csv_file', 'data', 'phenotypes', 'plotting_map', 'num_colors', 'unique_species', 'unique_slides'], locals(), pickle_file)

        else:

            # Load the data from the pickle file if it already exists
            self.load_pickle_dict(pickle_file, pickle_dir=pickle_dir)

            # However, overwrite the pickle and webpage directories as we should be able to load these same pickle files on different systems
            self.pickle_dir = pickle_dir
            self.webpage_dir = webpage_dir


    def calculate_metrics(self):
        '''
        Calculate the P values (and Z scores) from the coordinates of the species in every ROI in every slide.

        The assumed distributions are Poisson for the densities and binomial for the PMFs (see summary_of_formulas.lyx) for the formula and physical notebook notes on 12-16-20 for the derivations.

        As seen in the Lyx/PDF file, the precise random variable that is Poisson-distributed (S) is the total number of neighbors of a particular species over all n samples (i.e., over all n centers of a particular species). This is converted to "density" by dividing by n, but then note this is not technically Poisson distributed; e.g., the Poisson distribution applies to only discrete random variables. The corresponding "average number of neighbors around the n centers" is what is typically referred to as "density," which is more technically thought of as the average number of neighbors DIVIDED BY an area. Note that <S> = n * lambda, where lambda is basically the number of neighbors (of a certain species) in the ROI, divided by the ROI area, times the area of the slice under consideration. Note that S runs from 0, 1, 2, ..., infinity.

        The precise random variable that is binomial-distributed (Z_j) is the number of centers (of a particular species) having j neighbors (of a particular species). This (divided by n) is the probability mass function (PMF) of the random variable Y, which is the number of neighbors (of a particular species) around a center (of a particular species). (Note that Y is Poisson-distributed with mean lambda and S = Sum[Y_i], where i goes from 1, 2, ..., n, and Y_i is the number of neighbors around center i.) Note that <Z_j> = n * lambda^j*exp(-lambda)/j!. Note that Z_j runs from 0, 1, 2, ..., n. n is, as above, the total number of centers of a particular species.

        Assuming these distributions, the null hypotheses are that means are as defined above. The "left" or "less" alternative hypotheses are that the means are "less" than (i.e., to the "left" of) these stated values. The "right" or "greater" alternative hypotheses are that the means are "greater" than (i.e., to the "right" of) these stated values. The P values are then the probabilities of observing values of S or Z_j as extreme as (in the directions of the alternative hypotheses) the observed instances of S or Z_j under the assumptions of the null hypotheses.

        Notes:

          * Took ~47 (later: 65) minutes on laptop
          * Units here should be in the same units provided in Consolidated_data.txt (which were originally half-microns, i.e., for dr=8, each slice is 4 microns thick)
        '''

        # Import relevant modules
        import os
        import numpy as np

        # Set variables already defined as attributes
        unique_slides = self.unique_slides
        pickle_dir = self.pickle_dir
        plotting_map = self.plotting_map
        nslices = self.nslices
        thickness = self.thickness
        min_coord_spacing = self.min_coord_spacing

        # Set some attributes from the method parameters
        self.k_max = nslices
        self.dr = thickness

        # Constant
        pickle_file = 'calculated_metrics.pkl'

        # Experiment-wide variables
        all_species_list = [x[0] for x in plotting_map]
        nall_species = len(all_species_list)

        # If the pickle file doesn't already exist...
        if not os.path.exists(os.path.join(pickle_dir, pickle_file)):

            # For each slide...
            data_by_slide = []
            for uslide in unique_slides:
                print('On slide ' + uslide + '...')

                # Get the unique ROIs in the current slide
                unique_rois = np.unique(self.data.iloc[np.nonzero((self.data['Slide ID'] == uslide).to_numpy())[0]]['tag'])

                # For each ROI in the slide...
                data_by_roi = []
                for uroi in unique_rois:
                    print('  On ROI ' + uroi + '...')

                    # Get the needed ROI data
                    data_roi = self.data.iloc[np.nonzero((self.data['tag'] == uroi).to_numpy())[0]]
                    data_roi = data_roi.reset_index(drop=True)
                    x_roi = np.array(data_roi['Cell X Position'])
                    y_roi = np.array(data_roi['Cell Y Position'])
                    species_roi = np.array(data_roi['Species int'], dtype='uint64')
                    unique_species_in_roi = np.unique(species_roi)
                    coords_roi = np.c_[x_roi, y_roi]

                    # Run some checks
                    roi_x_range, roi_y_range, roi_min_coord_spacing = roi_checks_and_output(x_roi, y_roi, do_printing=False)  # note I don't believe roi_min_coord_spacing is ever actually used for anything other than saving the value to a text file
                    # if (not ([x_roi.min(),x_roi.max()]==roi_x_range)) or (not ([y_roi.min(),y_roi.max()]==roi_y_range)) or (not (0.5==roi_min_coord_spacing)):
                    if (not ([x_roi.min(), x_roi.max()] == roi_x_range)) or (not ([y_roi.min(), y_roi.max()] == roi_y_range)):
                        print('ERROR: A basic check failed')
                        # print(roi_x_range, roi_y_range, roi_min_coord_spacing)
                        print(roi_x_range, roi_y_range)
                        exit()

                    # Define the arrays of interest holding the metrics and other data
                    real_data = np.empty((nall_species, nall_species, nslices), dtype=object)
                    sim_data = np.empty((nall_species, nall_species, nslices), dtype=object)

                    # For every center and neighbor species in the entire experiment...
                    for icenter_spec, center_species in enumerate(all_species_list):
                        for ineighbor_spec, neighbor_species in enumerate(all_species_list):

                            # If the current center and neighbor species exist in the current ROI...
                            if (center_species in unique_species_in_roi) and (neighbor_species in unique_species_in_roi):

                                # Determine the coordinates of the current center and neighbor species
                                coords_centers = coords_roi[species_roi == center_species, :]
                                coords_neighbors = coords_roi[species_roi == neighbor_species, :]

                                # For every radius/slice...
                                for islice in range(nslices):

                                    # Define the inner and outer radii of the current slice
                                    small_rad = islice * thickness
                                    large_rad = (islice + 1) * thickness

                                    # Count the neighbors, calculate the PMFs, and from these determine the P values of interest for the single set of real data
                                    density_metrics_real, pmf_metrics_real, nexpected_real, nvalid_centers_real, coords_centers_real, coords_neighbors_real, valid_centers_real, edges_real, npossible_neighbors_real, roi_area_used_real, slice_area_used_real = \
                                        calculate_metrics_from_coords(min_coord_spacing, input_coords=(coords_centers, coords_neighbors), neighbors_eq_centers=(neighbor_species == center_species), nbootstrap_resamplings=0, rad_range=(small_rad, large_rad), use_theoretical_counts=False, roi_edge_buffer_mult=1, roi_x_range=roi_x_range, roi_y_range=roi_y_range, silent=False)

                                    # Save the results, plus some other data, into a primary array of interest
                                    real_data[icenter_spec, ineighbor_spec, islice] = (density_metrics_real, pmf_metrics_real, nexpected_real, nvalid_centers_real, coords_centers_real, coords_neighbors_real, valid_centers_real, edges_real, npossible_neighbors_real, roi_area_used_real, slice_area_used_real, center_species, neighbor_species, small_rad, large_rad, islice)

                                    # Count the neighbors, calculate the PMFs, and from these determine the P values of interest for a single set of simulated data with the same properties as the real data
                                    density_metrics_sim, pmf_metrics_sim, nexpected_sim, nvalid_centers_sim, coords_centers_sim, coords_neighbors_sim, valid_centers_sim, edges_sim, npossible_neighbors_sim, roi_area_used_sim, slice_area_used_sim = \
                                        calculate_metrics_from_coords(min_coord_spacing, input_coords=None, neighbors_eq_centers=(neighbor_species == center_species), ncenters_roi=coords_centers.shape[0], nneighbors_roi=coords_neighbors.shape[0], nbootstrap_resamplings=0, rad_range=(small_rad, large_rad), use_theoretical_counts=False, roi_edge_buffer_mult=1, roi_x_range=roi_x_range, roi_y_range=roi_y_range, silent=False)

                                    # Save the results, plus some other data, into a primary array of interest
                                    sim_data[icenter_spec, ineighbor_spec, islice] = (density_metrics_sim, pmf_metrics_sim, nexpected_sim, nvalid_centers_sim, coords_centers_sim, coords_neighbors_sim, valid_centers_sim, edges_sim, npossible_neighbors_sim, roi_area_used_sim, slice_area_used_sim, center_species, neighbor_species, small_rad, large_rad, islice)

                # Save the data to a data structure
                    roi_data_item = [(roi_x_range, roi_y_range, roi_min_coord_spacing, unique_species_in_roi, all_species_list, x_roi, y_roi, species_roi), real_data, sim_data]
                    data_by_roi.append(roi_data_item)
                data_by_slide.append([uslide, unique_rois, data_by_roi])  # save the current slide data and the inputted parameters

            # Create a pickle file saving the data that we just calculated
            make_pickle(data_by_slide, pickle_dir, pickle_file)

        # If the pickle file already exists, load it
        else:
            data_by_slide = load_pickle(pickle_dir, pickle_file)

        # Save the calculated data as a property of the class object
        self.metrics = data_by_slide


    def save_figs_and_corresp_data(self, plot_real_data=True, pval_figsize=(8, 12), log_pval_range=(-40, 0), calculate_empty_bin_pvals=True, max_nbins=None, roi_figsize=(6, 4), marker_size_step=0.80, pval_dpi=150, alpha=1, roi_dpi=200, square=True, yticklabels=2, pickle_dir=None, save_individual_pval_plots=True):
        '''
        Create and save all the figures and the data corresponding to the figures (i.e., the left and right P values for both the densities and PMFs), i.e., the actual data that is plotted.

        This way, I can see exactly what the read-in data actually look like and therefore trust them more.

        * alpha is the transparency for the circles in the ROI plots (0 is fully transparent, 1 is fully opaque)
        * marker_size_step=0.80 means the radius should be 80% larger for cells plotted behind other cells
        '''

        # Antonio's fix to enable plot generation in SLURM's batch mode
        import matplotlib
        matplotlib.use('Agg')

        # Import relevant libraries
        # import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Set a default value for this; note the input parameter to this call can be something like max_nbins=np.max([slices1.max_nbins_over_exp, slices2.max_nbins_over_exp])
        if max_nbins is None:
            max_nbins = self.max_nbins_over_exp
        self.max_nbins_used = max_nbins

        # Set variables already defined as attributes
        plotting_map = self.plotting_map
        metrics = self.metrics
        num_colors = self.num_colors
        if pickle_dir is None:
            webpage_dir = self.webpage_dir
            pickle_dir = self.pickle_dir
        else:
            webpage_dir = pickle_dir
        mapping_dict = self.mapping_dict
        coord_units_in_microns = self.coord_units_in_microns

        # Define the directory holding all the images for the webpage and the filename of the file holding all the corresponding figure data
        webpage_dir = os.path.join(webpage_dir, ('real' if plot_real_data else 'simulated'))
        pickle_file = 'figure_data-{}.pkl'.format(('real' if plot_real_data else 'simulated'))

        # If the pickle file doesn't already exist...
        pickle_pathname = os.path.join(pickle_dir, pickle_file)
        if not os.path.exists(pickle_pathname):

            # Experiment-wide variables
            all_species_list = [x[0] for x in plotting_map]
            nall_species = len(all_species_list)
            if mapping_dict is not None:  # species_names = [x[1] for x in plotting_map]
                species_names = [get_descriptive_cell_label(x[1], mapping_dict)[0] for x in plotting_map]
            else:
                species_names = [phenotypes_to_string(x[1]) for x in plotting_map]

            # Extract the correct number of colors from the default color palette
            ielem = 0
            colors = []
            for elem in matplotlib.rcParams['axes.prop_cycle']():
                color = elem['color']
                colors.append(color)
                ielem = ielem + 1
                if ielem == num_colors:
                    break
            default_marker_size = matplotlib.rcParams['lines.markersize']

            # Define the figures to use for plotting the ROIs and the P values
            fig_roi = plt.subplots(figsize=roi_figsize)[0]
            fig_pvals = plt.subplots(nrows=2, ncols=2, figsize=pval_figsize)[0]

            # Since at this point we're definitely saving some images, ensure their parent directory exists
            os.makedirs(webpage_dir, exist_ok=True)

            # Define the experiment-wide metadata to save and initialize the filedata
            metadata = {
                'webpage_dir': webpage_dir, 'pickle_pathname': pickle_pathname, 'num_slides': len(metrics), 'plot_real_data': plot_real_data, 'all_species_ids': all_species_list, 'all_species_names': species_names, 'nall_species': nall_species,
                'roi_dpi': roi_dpi, 'roi_figsize': roi_figsize,
                'pval_dpi': pval_dpi, 'pval_figsize': pval_figsize, 'log_pval_range': log_pval_range, 'calculate_empty_bin_pvals': calculate_empty_bin_pvals, 'max_nbins': max_nbins
            }
            filedata = []

            # For each slide...
            for islide, slide_data in enumerate(metrics):

                print('On slide {} of {}...'.format(islide + 1, len(metrics)))

                # For each ROI in the slide...
                for iroi, (roi_data, roi_name) in enumerate(zip(slide_data[2], slide_data[1])):  # get the metrics data for the current ROI of the current slide
                    # roi_data = [(roi_x_range, roi_y_range, roi_min_coord_spacing, unique_species_in_roi, _, x_roi, y_roi, species_roi), real_data, sim_data]
                    print('  On ROI {} of {}...'.format(iroi + 1, len(slide_data[2])))

                    # Collect the data needed to plot the ROIs, plot the ROI, and save it to disk
                    x_range = np.array(roi_data[0][0])
                    y_range = np.array(roi_data[0][1])
                    roi_min_coord_spacing = roi_data[0][2]
                    x_roi = roi_data[0][5]
                    y_roi = roi_data[0][6]
                    species_roi = roi_data[0][7]
                    uroi = roi_name
                    spec2plot_roi = [x[0] for x in plotting_map if x[0] in roi_data[0][3]]

                    plot_roi(fig_roi, spec2plot_roi, species_roi, x_roi, y_roi, plotting_map, colors, x_range, y_range, uroi, marker_size_step, default_marker_size, roi_dpi, mapping_dict, coord_units_in_microns, filepath=None, do_plot=True, alpha=alpha)
                    roi_fig_filename = 'roi_{}.png'.format(roi_name)
                    roi_fig_pathname = os.path.join(webpage_dir, roi_fig_filename)
                    fig_roi.savefig(roi_fig_pathname, dpi=roi_dpi, bbox_inches='tight')

                    # Determine which dataset to plot, each of which is (nall_species, nall_species, nslices)
                    # The elements of the dataset ndarray are either a tuple (if both the center and neighbor species are in the ROI) or None
                    if plot_real_data:
                        data_to_plot = roi_data[1]  # this is the single set of real data
                    else:
                        data_to_plot = roi_data[2]  # this is the corresponding single set of simulated data

                    # Determine whether both the center and neighbor species are in the dataset
                    center_and_neighbor_species_in_dataset = ((data_to_plot != None).sum(axis=2)) > 0  # (nall_species, nall_species)... since we want to do element-wise comparisons here, don't listen to linting when it says the right way to do the comparison is "data_to_plot is not None"

                    # For all combinations of centers and neighbors...
                    for icenter_spec in range(nall_species):
                        center_name = species_names[icenter_spec]

                        for ineighbor_spec in range(nall_species):
                            neighbor_name = species_names[ineighbor_spec]

                            # If data exist for at least one of the slices for the current center/neighbor combination...
                            if center_and_neighbor_species_in_dataset[icenter_spec, ineighbor_spec]:

                                # Create and save the P value figure containing four heatmaps as well as their corresponding data
                                print(icenter_spec, ineighbor_spec)
                                figure_title = '{} data\nSlide/ROI: {}\ncenter={}, neighbor={}'.format(('Real' if plot_real_data else 'Simulated'), roi_name, center_name, neighbor_name)
                                nvalid_centers_per_slice, left_log_dens_pvals, right_log_dens_pvals, left_log_pmf_pvals, right_log_pmf_pvals = plot_pvals(fig_pvals, data_to_plot[icenter_spec, ineighbor_spec, :], log_pval_range, name=figure_title, calculate_empty_bin_pvals=calculate_empty_bin_pvals, max_nbins_over_slices=max_nbins, square=square, yticklabels=yticklabels)
                                if nvalid_centers_per_slice.sum(axis=0) == 0:
                                    print('NOTE: Even though there are centers of species {} present in the ROI, none of them are valid; we can eliminate this species as a center (for any neighbor) from any analyses!'.format(center_name))
                                else:
                                    pvals_fig_filename = 'pvals_{}_center-{}_neighbor-{}.png'.format(roi_name, all_species_list[icenter_spec], all_species_list[ineighbor_spec])
                                    pvals_fig_pathname = os.path.join(webpage_dir, pvals_fig_filename)
                                    if save_individual_pval_plots:  # make this an option since it takes significant time in aggregate
                                        fig_pvals.savefig(pvals_fig_pathname, dpi=pval_dpi, bbox_inches='tight')
                                    filedata.append({
                                        'roi_fig_pathname': roi_fig_pathname, 'nrois_in_slide': len(slide_data[2]), 'roi_name': roi_name, 'unique_species_in_roi': spec2plot_roi, 'roi_x_range': x_range, 'roi_y_range': y_range, 'roi_spacing': roi_min_coord_spacing,
                                        'pvals_fig_pathname': pvals_fig_pathname, 'center_species_id': all_species_list[icenter_spec], 'neighbor_species_id': all_species_list[ineighbor_spec], 'center_species_name': species_names[icenter_spec], 'neighbor_species_name': species_names[ineighbor_spec], 'nvalid_centers_per_slice': nvalid_centers_per_slice, 'left_log_dens_pvals': left_log_dens_pvals, 'right_log_dens_pvals': right_log_dens_pvals, 'left_log_pmf_pvals': left_log_pmf_pvals, 'right_log_pmf_pvals': right_log_pmf_pvals
                                    })

            # Save the figure data to disk
            make_pickle((metadata, filedata), pickle_dir, pickle_file)

        else:

            # Read in the figure data from disk
            (metadata, filedata) = load_pickle(pickle_dir, pickle_file)

        self.metadata = metadata
        self.filedata = filedata

        # # Return the figure data
        # return(metadata, filedata)


    def calculate_max_nbins(self):
        '''
        Calculate the maximum number of possible bins (of numbers neighbors) over the entire experiment, for both the real and simulated data.
        '''

        # Set variables already defined as attributes
        plotting_map = self.plotting_map
        metrics = self.metrics

        # Experiment-wide variables
        nall_species = len(plotting_map)

        # Initialize the variable of interest
        max_nbins_over_exp = 0

        # For each slide...
        for slide_data in metrics:

            # For each ROI in the slide...
            for roi_data in slide_data[2]:  # get the metrics data for the current ROI of the current slide

                # Determine which dataset to plot, each of which is (nall_species, nall_species, nslices)
                # The elements of the dataset ndarray are either a tuple (if both the center and neighbor species are in the ROI) or None
                for data_to_plot in roi_data[1:3]:

                    # Determine whether both the center and neighbor species are in the dataset
                    center_and_neighbor_species_in_dataset = ((data_to_plot != None).sum(axis=2)) > 0  # (nall_species, nall_species)... since we want to do element-wise comparisons here, don't listen to linting when it says the right way to do the comparison is "data_to_plot is not None"

                    # For all combinations of centers and neighbors...
                    for icenter_spec in range(nall_species):
                        for ineighbor_spec in range(nall_species):

                            # If data exist for at least one of the slices for the current center/neighbor combination...
                            if center_and_neighbor_species_in_dataset[icenter_spec, ineighbor_spec]:

                                # Create the P value figure containing four heatmaps and return the figure object
                                max_nbins_over_exp = get_max_nbins_for_center_neighbor_pair(data_to_plot[icenter_spec, ineighbor_spec, :], max_nbins_over_exp)

        # Set the maximum number of bins as a property of the object itself and print this result
        self.max_nbins_over_exp = max_nbins_over_exp
        print('Calculated maximum number of bins over the entire experiment: {}'.format(max_nbins_over_exp))

        return(max_nbins_over_exp)


    def load_pickle_class(self, pickle_file, pickle_dir=None):
        '''
        Load some data from a pickle file ("class" just refers to this function being part of the TIMECellInteraction class)
        '''
        if pickle_dir is None:
            pickle_dir = self.pickle_dir
        return(load_pickle(pickle_dir, pickle_file))


    def load_pickle_dict(self, pickle_file, pickle_dir=None):
        '''
        Load a bunch of values to the self object from a pickle file by way of a dictionary
        '''
        dict2load = self.load_pickle_class(pickle_file, pickle_dir=pickle_dir)
        for key in dict2load:
            val = dict2load[key]
            setattr(self, key, val)


    def make_pickle_dict(self, vars2save, local_dict, pickle_file):
        '''
        Make a pickle file of a dictionary of data
        '''
        dict2save = {}
        for key in vars2save:
            if key in local_dict:
                val = local_dict[key]
                setattr(self, key, val)
            else:
                val = getattr(self, key)
            dict2save.update({key: val})
        make_pickle(dict2save, self.pickle_dir, pickle_file)


    def preprocess_dataframe(self, allow_compound_species):
        '''
        Preprocess the initial Pandas dataframe from Consolidata_data.txt (or a simulated one for simulated data) by creating another column (Species int) specifying a unique integer identifying the cell type
        If requested, remove compound species, and return the list of single-protein "phenotypes" contained in the data
        '''

        # Import relevant module
        import numpy as np

        # Preprocess the pandas dataframe in various ways
        data_phenotypes = self.data.filter(regex='^[pP]henotype ')  # get just the "Phenotype " columns
        data_phenotypes = data_phenotypes.reset_index(drop=True)
        phenotype_cols = list(data_phenotypes.columns)  # get a list of those column names
        phenotypes = np.array([x.replace('Phenotype ', '') for x in phenotype_cols])  # extract just the phenotypes from that list
        n_phenotypes = len(phenotypes)  # get the number of possible phenotypes in the datafile
        self.data['Species string'] = data_phenotypes.applymap(lambda x: '1' if (str(x)[-1] == '+') else '0').apply(lambda x: ''.join(list(x)), axis=1)  # add a column to the original data that tells us the unique "binary" species string of the species corresponding to that row/cell
        self.data = self.data.drop(np.nonzero((self.data['Species string'] == ('0' * n_phenotypes)).to_numpy())[0])  # delete rows that all have '...-' as the phenotype or are blank
        self.data = self.data.reset_index(drop=True)  # reset the indices
        self.data['Species int'] = self.data['Species string'].apply(lambda x: int(x, base=2))  # add an INTEGER species column

        # Remove compound species if requested
        if not allow_compound_species:
            self.remove_compound_species()
            self.remove_compound_species()  # ensure nothing happens

        return(phenotypes)


    def remove_compound_species(self):
        '''
        For each compound species ('Species int' not just a plain power of two), add each individual phenotype to the end of the dataframe individually and then delete the original compound entry
        '''

        # Import relevant module
        import numpy as np

        # Get the species IDs
        x = np.array(self.data['Species int'])
        print('Data size:', len(self.data))

        # Determine which are not powers of 2, i.e., are compound species
        powers = np.log2(x)
        compound_loc = np.nonzero(powers != np.round(powers))[0]
        ncompound = len(compound_loc)
        print('  Compound species found:', ncompound)

        # If compound species exist...
        if ncompound > 0:

            print('  Removing compound species from the dataframe...')

            # Get a list of tuples each corresponding to compound species, the first element of which is the row of the compound species, and the second of which is the species IDs of the pure phenotypes that make up the compound species
            compound_entries = [(cl, 2**np.nonzero([int(y) for y in bin(x[cl])[2:]][-1::-1])[0]) for cl in compound_loc]

            # For each compound species...
            data_to_add = []
            for index, subspecies in compound_entries:

                # For each pure phenotype making up the compound species...
                for subspec in subspecies:

                    # You have to put this here instead of outside this loop for some weird reason! Even though you can see the correct change made to series and EVEN TO DATA_TO_ADD by looking at series and data_to_add[-1] below, for some Godforsaken reason the actual data_to_add list does not get updated with the change to 'Species int' when you print data_to_add at the end of both these loops, and therefore the data that gets added to the data dataframe contains all the same 'Species string' values, namely the last one assigned. Thus, we are actually adding the SAME species to multiple (usually 2) spatial points, so that the even-spacing problem arises.
                    series = self.data.iloc[index].copy()

                    series['Species int'] = subspec  # set the species ID of the series data to that of the current phenotype
                    data_to_add.append(series)  # add the data to a running list

            # Add all the data in the list to the dataframe
            self.data = self.data.append(data_to_add, ignore_index=True)
            print('  Added rows:', len(data_to_add))
            print('  Data size:', len(self.data))

            # Delete the original compound species entries
            self.data = self.data.drop(compound_loc)
            self.data = self.data.reset_index(drop=True)
            print('  Deleted rows:', len(compound_loc))
            print('  Data size:', len(self.data))


    def average_over_rois(self, plot_real_data=True, log_pval_range=(-200, 0), figsize=(14, 4), dpi=150, img_file_suffix='', regular_pval_figsize=(8, 12), square=True, yticklabels=2, pval_dpi=150, plot_summary_dens_pvals=True, plot_main_pvals=True, write_csv_files=True, write_pickle_datafile=True, start_slide=None, num_slides=None):
        '''
        Perform a weighted geometric mean of the density and PMF P values over the valid ROIs, saving the corresponding figures as PNG files and data as CSV files.

        Old:

        Read all the data corresponding to the P value plots into a Pandas dataframe, using this structure to write an ndarray holding the left and right density (not PMF) P values (for a single-slice dataset) for every patient (slide) and center/neighbor combination in the entire experiment, performing a weighted average of the P values over all the ROIs for each patient.

        The resulting array can then be read back in to other functions (such as create_summary_plots(), below) to plot these results in heatmaps or to write them to CSV files.

        From def create_summary_plots():
          Read in the summary ndarray from create_summary_array(), above, to plot and save heatmaps of all these data.
          Call like, e.g., create_summary_plots(dataset_name, project_dir, plot_real_data, summary_arr, unique_slides, metadata['all_species_names']).

        From def create_summary_csv_files():
          Write the left and right summary P values to a CSV file.
          Call like:
          tci.create_summary_csv_files(dataset_name, project_dir, plot_real_data, summary_arr, unique_slides, metadata['all_species_names'])
        '''

        # Import relevant libraries
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os

        # Define variables already defined as attributes
        filedata = self.filedata
        metadata = self.metadata
        nslices = self.nslices
        max_nbins = self.max_nbins_used
        pickle_dir = self.pickle_dir
        webpage_dir = self.webpage_dir

        # Assign some variables from the saved metadata
        all_species_names = metadata['all_species_names']
        nall_species = metadata['nall_species']
        all_species_ids = metadata['all_species_ids']

        # Create the Pandas dataframe from the list of dictionaries that were outputted by save_figs_and_corresp_data()
        df = pd.DataFrame()
        df = df.append(filedata)

        # Add a slide column to the dataframe and get a list of the unique slides while maintaining order
        df['slide_name'] = df['roi_name'].apply(lambda x: x.split('_')[0])
        unique_slides = df['slide_name'].unique()
        nunique_slides = len(unique_slides)

        # Determine which unique slides to plot in the main plotting functionality below
        if start_slide is None:
            start_slide = 0
            num_slides = nunique_slides
        stop_slide = start_slide + num_slides
        unique_slides_to_plot = unique_slides[start_slide:stop_slide]

        # Define the directory holding all the images of the averaged data for the webpage and the filename of the file holding all the corresponding data
        webpage_dir = os.path.join(webpage_dir, ('real' if plot_real_data else 'simulated'))
        pickle_file = 'averaged_data-{}.pkl'.format(('real' if plot_real_data else 'simulated'))

        # If the pickle file doesn't already exist...
        if not os.path.exists(os.path.join(pickle_dir, pickle_file)):

            # Initialize the arrays of interest
            nmatches_holder    = np.zeros((nunique_slides, nall_species, nall_species, nslices))
            log_dens_pvals_avg = np.zeros((nunique_slides, nall_species, nall_species, 2, nslices))
            log_pmf_pvals_avg  = np.zeros((nunique_slides, nall_species, nall_species, max_nbins, 2, nslices))

            # For every unique slide...
            for islide, slide in enumerate(unique_slides[:]):

                # For every possible center species...
                for icenter_spec, center_spec in enumerate(all_species_ids[:]):

                    # For every possible neighbor species...
                    for ineighbor_spec, neighbor_spec in enumerate(all_species_ids[:]):

                        # Get a temporary dataframe containing the current slide/center/neighbor combination (this will usually be five rows, one per ROI)
                        df_tmp = df[(((df['slide_name'] == slide) & (df['center_species_id'] == center_spec) & (df['neighbor_species_id'] == neighbor_spec)))]
                        nrows = len(df_tmp)

                        # For every slice...
                        for islice in range(nslices):

                            # Initialize and populate the three temporary arrays
                            nvalid_centers_holder = np.zeros((nrows,))
                            log_dens_pvals = np.zeros((nrows, 2))
                            log_pmf_pvals = np.zeros((nrows, max_nbins, 2))
                            for irow, (nvalid_centers_per_slice, left_log_dens_pvals, right_log_dens_pvals, left_log_pmf_pvals, right_log_pmf_pvals) in enumerate(zip(df_tmp['nvalid_centers_per_slice'], df_tmp['left_log_dens_pvals'], df_tmp['right_log_dens_pvals'], df_tmp['left_log_pmf_pvals'], df_tmp['right_log_pmf_pvals'])):
                                nvalid_centers_holder[irow] = nvalid_centers_per_slice[islice]
                                log_dens_pvals[irow, :] = np.array([left_log_dens_pvals[0, islice], right_log_dens_pvals[0, islice]])  # (2,)
                                log_pmf_pvals[irow, :, :] = np.c_[left_log_pmf_pvals[:, islice], right_log_pmf_pvals[:, islice]]  # (max_nbins,2)

                            # Determine the rows in the temporary dataframe that have at least 1 valid center
                            matches = nvalid_centers_holder >= 1  # (nrows,)
                            nmatches = matches.sum()
                            nmatches_holder[islide, icenter_spec, ineighbor_spec, islice] = nmatches

                            # Perform the weighted averaging over the ROIs
                            if nmatches >= 1:
                                nvalid_centers_tmp = nvalid_centers_holder[matches][:, np.newaxis]  # (nmatches, 1)
                                log_dens_pvals_avg[islide, icenter_spec, ineighbor_spec, :, islice] = (nvalid_centers_tmp * log_dens_pvals[matches, :]).sum(axis=0) / nvalid_centers_tmp.sum(axis=0)  # (2,)
                                nvalid_centers_tmp = nvalid_centers_tmp[:, :, np.newaxis]  # (nmatches, 1, 1)
                                log_pmf_pvals_avg[islide, icenter_spec, ineighbor_spec, :, :, islice] = (nvalid_centers_tmp * log_pmf_pvals[matches, :, :]).sum(axis=0) / nvalid_centers_tmp.sum(axis=0)  # (max_nbins, 2)
                            else:
                                log_dens_pvals_avg[islide, icenter_spec, ineighbor_spec, :, islice] = None
                                log_pmf_pvals_avg[islide, icenter_spec, ineighbor_spec, :, :, islice] = None

            # Set the log of the P value range for the color plotting
            vmin = log_pval_range[0]
            vmax = log_pval_range[1]

            # Set the (negative) infinite values to the darkest color (or else they won't be plotted, as inf values are not plotted)
            log_dens_pvals_avg[np.isneginf(log_dens_pvals_avg)] = vmin
            log_pmf_pvals_avg[np.isneginf(log_pmf_pvals_avg)] = vmin

            # Initialize the figure and axes
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

            if plot_summary_dens_pvals:

                # Plot the average density P values for every slice
                for islice in range(nslices):

                    # For every unique slide...
                    for islide in range(nunique_slides):

                        # Determine the filename of the figure
                        filename = os.path.join(webpage_dir, 'average_density_pvals-{}-{}-slice_{:02d}_of_{:02d}{}.png'.format(('real' if plot_real_data else 'simulated'), unique_slides[islide], islice + 1, nslices, img_file_suffix))

                        # Reset the figure/axes
                        fig.clf()
                        ax = fig.subplots(nrows=1, ncols=2)

                        # Plot the log of the left/less P values
                        sns.heatmap(log_dens_pvals_avg[islide, :, :, 0, islice], vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[0], cbar=True, xticklabels=all_species_names, yticklabels=all_species_names, square=True)
                        ax[0].set_title('log10(\"less\" density pvals)')
                        ax[0].set_xlabel('Neighbor species')
                        ax[0].set_ylabel('Center species')

                        # Plot the log of the right/greater P values
                        sns.heatmap(log_dens_pvals_avg[islide, :, :, 1, islice], vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[1], cbar=True, xticklabels=all_species_names, yticklabels=all_species_names, square=True)
                        ax[1].set_title('log10(\"greater\" density pvals)')
                        ax[1].set_xlabel('Neighbor species')
                        ax[1].set_ylabel('Center species')

                        # Set the figure title to the slide title and ensure the facecolor is white
                        fig.suptitle('Average density P values - {} - {} data - slice {} of {}'.format(unique_slides[islide], ('real' if plot_real_data else 'simulated'), islice + 1, nslices))
                        fig.patch.set_facecolor('white')

                        # Save the figure to disk
                        fig.savefig(filename, dpi=dpi, bbox_inches='tight')

            # Initialize the P value figure
            fig = plt.subplots(nrows=2, ncols=2, figsize=regular_pval_figsize)[0]

            if plot_main_pvals:

                # For every slide, center, and neighbor species...
                # for islide, slide_name in enumerate(unique_slides):
                for islide2, slide_name in enumerate(unique_slides_to_plot):
                    islide = islide2 + start_slide
                    for icenter, center_name in enumerate(all_species_names):
                        for ineighbor, neighbor_name in enumerate(all_species_names):

                            # Initialize the current figure by clearing it and all its axes
                            fig.clf()
                            ax = fig.subplots(nrows=2, ncols=2)

                            # Create the four-axis figure, where the top row has the left and right density P values and the bottom row has the left and right PMF P values

                            # Plot the log10 of the left density P values
                            sns.heatmap(log_dens_pvals_avg[islide, icenter, ineighbor, 0, :][np.newaxis, :], vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[0, 0], cbar=True, yticklabels=True, square=True)
                            ax[0, 0].set_title('log10(\"less\" density pvals)')
                            ax[0, 0].set_xlabel('Slice')

                            # Plot the log10 of the right density P values
                            sns.heatmap(log_dens_pvals_avg[islide, icenter, ineighbor, 1, :][np.newaxis, :], vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[0, 1], cbar=True, yticklabels=True, square=True)
                            ax[0, 1].set_title('log10(\"greater\" density pvals)')
                            ax[0, 1].set_xlabel('Slice')

                            # Plot the log10 of the left PMF P values
                            sns.heatmap(log_pmf_pvals_avg[islide, icenter, ineighbor, :, 0, :], vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[1, 0], cbar=True, yticklabels=yticklabels, square=square)
                            ax[1, 0].set_title('log10(\"less\" PMF pvals)')
                            ax[1, 0].set_xlabel('Slice')
                            ax[1, 0].set_ylabel('Number of neighbors')

                            # Plot the log10 of the right PMF P values
                            sns.heatmap(log_pmf_pvals_avg[islide, icenter, ineighbor, :, 1, :], vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[1, 1], cbar=True, yticklabels=yticklabels, square=square)
                            ax[1, 1].set_title('log10(\"greater\" PMF pvals)')
                            ax[1, 1].set_xlabel('Slice')
                            ax[1, 1].set_ylabel('Number of neighbors')

                            # Place a descriptive title on the figure
                            figure_title = 'Averaged-over-ROI, {} data\nSlide: {}\ncenter={}, neighbor={}'.format(('real' if plot_real_data else 'simulated'), slide_name, center_name, neighbor_name)
                            fig.suptitle(figure_title)

                            # Save the figure
                            pvals_fig_filename = 'averaged_pvals_{}_center-{}_neighbor-{}.png'.format(slide_name, all_species_ids[icenter], all_species_ids[ineighbor])
                            pvals_fig_dirname = os.path.join(webpage_dir, slide_name)
                            pvals_fig_pathname = os.path.join(pvals_fig_dirname, pvals_fig_filename)
                            os.makedirs(pvals_fig_dirname, exist_ok=True)
                            fig.savefig(pvals_fig_pathname, dpi=pval_dpi, bbox_inches='tight')

            # Determine the filename of the CSV files we intend to write
            density_csv_pathname = os.path.join(pickle_dir, 'average_density_pvals-{}.csv'.format(('real' if plot_real_data else 'simulated')))
            pmf_csv_pathname = os.path.join(pickle_dir, 'average_pmf_pvals-{}.csv'.format(('real' if plot_real_data else 'simulated')))

            # Part the slide names into the individual patients and drug treatment
            patient = [int(x.split('-')[0][:-1]) for x in unique_slides]
            pre_or_post = [x.split('-')[0][-1:] for x in unique_slides]
            upatient = []
            upre_or_post = []
            for curr_patient, curr_pre_or_post in zip(patient, pre_or_post):
                if curr_patient not in upatient:
                    upatient.append(curr_patient)
                if curr_pre_or_post not in upre_or_post:
                    upre_or_post.append(curr_pre_or_post)

            # Reshape the average holder for the main arrays to the individual patients and drug treatment, filling in "blanks" if necessary
            impossible_val = 444.4
            log_dens_pvals_avg2 = np.ones((len(upatient), len(upre_or_post), nall_species, nall_species, 2, nslices)) * impossible_val
            log_pmf_pvals_avg2 = np.ones((len(upatient), len(upre_or_post), nall_species, nall_species, max_nbins, 2, nslices)) * impossible_val
            for iunique_slide, (curr_patient, curr_pre_or_post) in enumerate(zip(patient, pre_or_post)):
                upatient_idx = upatient.index(curr_patient)
                upre_or_post_idx = upre_or_post.index(curr_pre_or_post)
                log_dens_pvals_avg2[upatient_idx, upre_or_post_idx, :, :, :, :] = log_dens_pvals_avg[iunique_slide, :, :, :, :]
                log_pmf_pvals_avg2[upatient_idx, upre_or_post_idx, :, :, :, :, :] = log_pmf_pvals_avg[iunique_slide, :, :, :, :, :]
            log_dens_pvals_avg = log_dens_pvals_avg2
            log_pmf_pvals_avg = log_pmf_pvals_avg2

            # Create the Pandas indexes
            index_density = pd.MultiIndex.from_product([upre_or_post, all_species_names, all_species_names, ['left', 'right'], np.arange(nslices) + 1], names=['drug_status', 'center_species', 'neighbor_species', 'pval_type', 'slice_number'])
            index_pmf = pd.MultiIndex.from_product([upre_or_post, all_species_names, all_species_names, np.arange(max_nbins), ['left', 'right'], np.arange(nslices) + 1], names=['drug_status', 'center_species', 'neighbor_species', 'poss_num_neighbors', 'pval_type', 'slice_number'])

            # For the density, Create the Pandas dataframes from the indexes
            df_log_dens_pvals_avg = pd.DataFrame(data=log_dens_pvals_avg.reshape((len(upatient), -1)), index=upatient, columns=index_density).rename_axis('subject')
            df_log_pmf_pvals_avg = pd.DataFrame(data=log_pmf_pvals_avg.reshape((len(upatient), -1)), index=upatient, columns=index_pmf).rename_axis('subject')

            # Write the Pandas dataframes to disk
            if write_csv_files:
                df_log_dens_pvals_avg.to_csv(density_csv_pathname)
                df_log_pmf_pvals_avg.to_csv(pmf_csv_pathname)

            # Save the averaged data to disk
            if write_pickle_datafile:
                make_pickle((nmatches_holder, log_dens_pvals_avg, log_pmf_pvals_avg, df_log_dens_pvals_avg, df_log_pmf_pvals_avg), pickle_dir, pickle_file)

        else:

            # Read in the averaged data from disk
            (nmatches_holder, log_dens_pvals_avg, log_pmf_pvals_avg, df_log_dens_pvals_avg, df_log_pmf_pvals_avg) = load_pickle(pickle_dir, pickle_file)

        return(nmatches_holder, log_dens_pvals_avg, log_pmf_pvals_avg, unique_slides, df_log_dens_pvals_avg, df_log_pmf_pvals_avg)


def incorporate_species_equivalents(species_equivalents, plotting_map, data):
    '''
    Redefine some species as other species since the thresholding that Houssein sets can sometimes be ambiguous, e.g., large CD163 membranes sometimes result in cells positive for CD163 even though they're not.

    Call like: plotting_map, data, num_colors, unique_species = incorporate_species_equivalents(species_equivalents, plotting_map, data)
    '''

    # Import relevant libraries
    import pandas as pd
    import numpy as np

    # # Define some arrays from already-defined attributes
    # plotting_map = self.plotting_map
    # data = self.data

    # Get a dataframe version of the plotting map
    df_plotting_map = pd.DataFrame(data=plotting_map, columns=['species_id', 'positive_markers', 'species_count', 'marker_ids', 'circle_sizes']).set_index('species_id', drop=False)

    # For each key in the species mappings...
    for curr_key in species_equivalents.keys():

        # Determine the corresponding value
        curr_val = species_equivalents.get(curr_key)

        if (curr_key in df_plotting_map['species_id']) and (curr_val in df_plotting_map['species_id']):

            # Print out the species equivalencies that we're asserting
            curr_key_markers = df_plotting_map.loc[curr_key, 'positive_markers']
            curr_val_markers = df_plotting_map.loc[curr_val, 'positive_markers']
            print('Now identifying {} as {}'.format(curr_key_markers, curr_val_markers))

            # Incoporate the key species into the value species and drop the key row
            df_plotting_map.loc[curr_val, 'species_count'] = df_plotting_map.loc[curr_val, 'species_count'] + df_plotting_map.loc[curr_key, 'species_count']
            df_plotting_map = df_plotting_map.drop(labels=curr_key)

            # Redefine the key species labels as the value species labels
            data.loc[data['Species int'] == curr_key, 'Species int'] = curr_val

    # Ensure the attributes of the object are overwritten
    plotting_map = df_plotting_map.to_numpy()
    # self.data = data
    num_colors = (df_plotting_map['circle_sizes'] == 1).sum()
    unique_species = df_plotting_map['species_id'].to_numpy(dtype=np.int64)

    return(plotting_map, data, num_colors, unique_species)


# Get the average spacing on either side of each datapoint in an array
def get_avg_spacing(arr):
    if len(arr) >= 2:
        import numpy as np
        arr2 = np.concatenate(([2 * arr[0] - arr[1]], arr, [2 * arr[-1] - arr[-2]]))
        return((arr2[2:] - arr2[0:-2]) / 2)
    else:
        print('Not actually getting average spacing in arr because len(arr) < 2; returning 1')
        return([1])


# Read in the Consolidated_data.txt TSV file into a Pandas dataframe
def get_consolidated_data(csv_file):
    import pandas as pd
    return(pd.read_csv(csv_file, sep='\t'))  # read in the data


# Given a list of phenotypes in a species, return the A+/B+ etc. string version
def phenotypes_to_string(phenotype_list):
    phenotype_list.sort()
    return('/'.join(phenotype_list))


# Given a list of phenotypes in a species, return the nicely formatted version, if there's a known cell type corresponding to the species
# Note these are purely labels; the species themselves are determined by allow_compound_species as usual
def get_descriptive_cell_label(phenotype_list, mapping_dict):
    # Note: CD163+/CD4+ REALLY ISN'T ANYTHING COMPOUND --> MAKE IT OVERLAPPING SPECIES (i.e., it shouldn't be in the dictionary below)!!!!
    phenotype_string = phenotypes_to_string(phenotype_list)
    descriptive_name = mapping_dict.get(phenotype_string)
    if descriptive_name is None:
        pretty_name = phenotype_string
        is_known = False
    else:
        # pretty_name = phenotype_string + ' (' + descriptive_name + ')'
        pretty_name = descriptive_name
        is_known = True
    return(pretty_name, is_known)


# Obtain the plotting map, total number of unique colors needed for plotting, the list of unique species (in the same order as in plotting_map), and a correctly sorted list of slides (e.g., 1,2,15 instead of 1,15,2)
# Note that individual unique species are specified by the allow_compound_species keyword, which in turn affects which of the 'Species int' columns of the Pandas dataframe are actually unique
# Don't get confused by the mapping_dict variable, which only affects plotting of the species... it doesn't affect what is actually considered a unique species or not!
def get_dataframe_info(data, phenotypes, mapping_dict):

    # Import relevant modules
    import numpy as np
    from operator import itemgetter

    # Create an ndarray containing all the unique species in the dataset in descending order of frequency with columns: integer label, string list, frequency, color(s), circle size(s)
    plotting_map = [[-(list(data['Species int']).count(x)), list(int2list(phenotypes, x)), x] for x in np.unique(data['Species int'])]  # create a list of the unique species in the dataset with columns: -frequency, string list, integer label
    plotting_map.sort(key=itemgetter(0, 1))  # sort by decreasing frequency (since the frequency column is negative) and then by the string list
    plotting_map = [[-x[0], x[1], x[2]] for x in plotting_map]  # make the frequency column positive

    print(plotting_map)

    # Get the colors of the species that are already known to us; use a -1 if the species isn't known
    colors = []
    known_phenotypes = []
    known_colors = []
    icolor = 0
    for item in plotting_map:
        phenotype_list = item[1]
        is_known = get_descriptive_cell_label(phenotype_list, mapping_dict)[1]

        # If the species (each row of plotting_map) is known to us (i.e., in the inputted mapping_dict variable, which simply assigns a cell label to any single or compound species)...
        # ...give that species its own color, and make a note if the species is also a single, non-compound species (i.e., a single phenotype)
        if is_known:
            colors.append(icolor)
            if len(phenotype_list) == 1:
                known_phenotypes.append(phenotype_list[0])
                known_colors.append(icolor)
            icolor = icolor + 1
        else:
            colors.append(-1)

    # Get the colors of the rest of the species using the colors of the already-known single-phenotype species
    # I.e., if the species is not known to us (i.e., not in mapping_dict), do not give the species its own color (unless it contains a phenotype that's not in known_phenotypes)
    # Instead, give each phenotype in the species either the corresponding color in known_phenotypes (if it's in there) or a new color (if it's not in known_phenotypes)
    # Assign the corresponding circle sizes as well
    colors2 = []
    circle_sizes = []
    for item, color in zip(plotting_map, colors):
        phenotype_list = item[1]
        if color == -1:
            curr_colors = []
            for single_phenotype in phenotype_list:
                if single_phenotype in known_phenotypes:
                    curr_colors.append(known_colors[known_phenotypes.index(single_phenotype)])
                else:
                    curr_colors.append(icolor)
                    known_phenotypes.append(single_phenotype)
                    known_colors.append(icolor)
                    icolor = icolor + 1
        else:
            curr_colors = [color]

        # Always have the most prevalent single species (if a lower color number really implies higher prevalence, it should generally at least) first in the color list, and make the corresponding circle size the largest (but in the background of course)
        curr_colors.sort()
        curr_sizes = list(np.arange(start=len(curr_colors), stop=0, step=-1))

        colors2.append(curr_colors)
        circle_sizes.append(curr_sizes)
    colors = colors2

    # Store the total number of unique colors to plot
    num_colors = icolor

    # Finalize the plotting map
    # plotting_map = np.array([ [item[2], item[1], item[0], (color if len(color)!=1 else color[0]), (circle_size if len(circle_size)!=1 else circle_size[0])] for item, color, circle_size in zip(plotting_map, colors, circle_sizes) ])
    plotting_map = np.array([[item[2], item[1], item[0], (color if (len(color) != 1) else color[0]), (circle_size if (len(circle_size) != 1) else circle_size[0])] for item, color, circle_size in zip(plotting_map, colors, circle_sizes)], dtype=object)

    # Use the plotting map to extract just the unique species in the data
    unique_species = np.array([x[0] for x in plotting_map])  # get a list of all the unique species in the dataset in the correct order

    # Get the unique slides sorted correctly
    tmp = [[int(x.split('-')[0][0:len(x.split('-')[0]) - 1]), x] for x in np.unique(data['Slide ID'])]
    tmp.sort(key=(lambda x: x[0]))
    unique_slides = [x[1] for x in tmp]

    return(plotting_map, num_colors, unique_species, unique_slides)


# Given an array of densities, for each of which we will generate a different ROI of the corresponding density, return a Pandas dataframe of the simulated data
# Only a single slide will be returned (but in general with multiple ROIs)
def get_simulated_data(doubling_type, densities, max_real_area, min_coord_spacing, mult):

    # Import relevant modules
    import numpy as np
    import pandas as pd

    # Warn if we're doubling up the data
    if doubling_type != 0:
        print('NOTE: DOUBLING UP THE SIMULATED DATA!!!!')

    # Specify the columns that are needed based on what's used from consolidated_data.txt
    columns = ['tag', 'Cell X Position', 'Cell Y Position', 'Slide ID', 'Phenotype A', 'Phenotype B']

    # Determine a constant number of cells in a simulated ROI using the maximum-sized ROI of the real data and the largest simulated density
    N = int(np.int(densities[-1] * max_real_area) * mult)

    # Initialize the Pandas dataframe
    data = pd.DataFrame(columns=columns)

    # For each ROI density and average either-side density spacing...
    perc_error = []
    for pop_dens, avg_spacing in zip(densities, get_avg_spacing(densities)):
        print('Current population density:', pop_dens)

        # Get the Cartesian coordinates of all the cells from random values populated until the desired density is reached
        tot_area = N / pop_dens
        side_length = np.sqrt(tot_area)
        tmp = np.round(side_length / min_coord_spacing)  # now we want N random integers in [0,tmp]
        coords_A = np.random.randint(tmp, size=(int(2 * N / 3), 2)) * min_coord_spacing
        coords_B = np.random.randint(tmp, size=(int(1 * N / 3), 2)) * min_coord_spacing
        x_A = coords_A[:, 0]
        y_A = coords_A[:, 1]
        x_B = coords_B[:, 0]
        y_B = coords_B[:, 1]

        # Set the ROI name from the current density
        tag = 'pop_dens_{:09.7f}'.format(pop_dens)

        # Add the simulated data to a nested list and then convert to a Pandas dataframe and add it to the master dataframe
        # Number in bracket is number of actually unique dataset (three actually unique datasets)
        # (1) [1] doubling_type=0, allow_compound_species=True:  coords=(A,B), labels=(A,B) # two species with different coordinates
        # (2) [1] doubling_type=0, allow_compound_species=False: coords=(A,B), labels=(A,B)
        # (3) [2] doubling_type=1, allow_compound_species=True:  coords=(A,A), labels=(A,B) # two overlapping species
        # (4) [2] doubling_type=1, allow_compound_species=False: coords=(A,A), labels=(A,B)
        # (5) [3] doubling_type=2, allow_compound_species=True:  coords=(A),   labels=(AB)  # one compound species (AB = compound species)
        # (6) [2] doubling_type=2, allow_compound_species=False: coords=(A,A), labels=(A,B)
        if doubling_type != 2:
            list_set = [[tag, curr_x_A, curr_y_A, '1A-only_slide', 'A+', 'B-'] for curr_x_A, curr_y_A in zip(x_A, y_A)]
            if doubling_type == 0:
                list_set = list_set + [[tag, curr_x_B, curr_y_B, '1A-only_slide', 'A-', 'B+'] for curr_x_B, curr_y_B in zip(x_B, y_B)]
            elif doubling_type == 1:
                list_set = list_set + [[tag, curr_x_A, curr_y_A, '1A-only_slide', 'A-', 'B+'] for curr_x_A, curr_y_A in zip(x_A, y_A)]
        else:
            list_set = [[tag, curr_x_A, curr_y_A, '1A-only_slide', 'A+', 'B+'] for curr_x_A, curr_y_A in zip(x_A, y_A)]
        tmp = pd.DataFrame(list_set, columns=columns)
        data = data.append(tmp, ignore_index=True)

        # Calculate the percent error in the actual density from the desired density
        if doubling_type == 0:
            x_tmp = np.r_[x_A, x_B]
            y_tmp = np.r_[y_A, y_B]
        else:
            x_tmp = x_A
            y_tmp = y_A
        perc_error.append((N / ((x_tmp.max() - x_tmp.min()) * (y_tmp.max() - y_tmp.min())) - pop_dens) / avg_spacing * 100)

    print('Percent error:', perc_error)
    print('Maximum percent error:', np.max(perc_error))

    return(data)


# Convert integer numbers defined in species bit-wise to a string list based on phenotypes
def int2list(phenotypes, species):
    return(phenotypes[[bool(int(char)) for char in ('{:0' + str(len(phenotypes)) + 'b}').format(species)]])


# Load some data from a pickle file
def load_pickle(pickle_dir, pickle_file):
    import pickle
    import os
    filename = os.path.join(pickle_dir, pickle_file)
    print('Reading pickle file ' + filename + '...')
    with open(filename, 'rb') as f:
        data_to_load = pickle.load(f)
    return(data_to_load)


# Write a pickle file from some data
def make_pickle(data_to_save, pickle_dir, pickle_file):
    import pickle
    import os
    filename = os.path.join(pickle_dir, pickle_file)
    print('Creating pickle file ' + filename + '...')
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)


def plot_roi(fig, spec2plot, species, x, y, plotting_map, colors, x_range, y_range, title, marker_size_step, default_marker_size, dpi, mapping_dict, coord_units_in_microns, filepath=None, do_plot=True, alpha=1):
    '''
    For the raw data (coordinates) for a given ROI, plot a circle (scatter plot) representing each species, whether known (in which case get_descriptive_cell_label() is used) or unknown; plot a legend too

    Save the figure to a file as well

    Note that in plotting_map (which is a list made up of five-element lists, one per species in the experiment, in order of decreasing frequency over the entire experiment), each row is what ends up as "plotting_data" below. The fourth and fifth elements (the colors and sizes, respectively) are ordered correctly relative to each other, and (for compound species) are in the order of increasing color, which, due to how plotting_map is defined, means of decreasing frequency. (In general; the first criterion in plotting_map is actually whether the species is "known", i.e., in mapping_dict.) This generally means, for instance, that the more prevalent species is plotting larger and in the back of the less prevalent species in the case of the plotting of a compound species.

    The point is, however, the second element of the plotting_data lists are NOT ordered correctly relative to those two potential-lists, so we take care below to ensure the subspecies are LABELED correctly. After this is done, we do not need the variable icircle, which we are now removing.
    '''

    if do_plot:

        # Import relevant library
        import numpy as np

        # Axis
        ax = fig.axes[0]
        ax.cla()

        # Get the lookup table for the choosing of the correct species for labeling below (where label_lookup_for_color is used). This is what we are referring to in the docstring for this function.
        primary_species = [not isinstance(x[3], list) for x in plotting_map]
        plotting_map_primary_noncompound_species = [x for (x, y) in zip(plotting_map, primary_species) if (y and (len(x[1]) == 1))]
        label_lookup_for_color = [[x[1][0], x[3]] for x in plotting_map_primary_noncompound_species]

        # For each unique species in the current ROI (in the correct order)...
        plotted_colors = []
        plots_for_legend = []
        labels_for_legend = []
        for spec in spec2plot:

            # Get the data for that species
            spec_ind = np.nonzero(species == spec)
            x_pos = x[spec_ind]
            y_pos = y[spec_ind]
            plotting_data = plotting_map[[x[0] for x in plotting_map].index(spec)]

            # Ensure the colors and marker sizes are lists and determine whether a single circle will be plotted for the potentially compound species
            if not isinstance(plotting_data[3], list):
                all_colors = [plotting_data[3]]
                all_sizes = [plotting_data[4]]
                is_primary = True
            else:
                all_colors = plotting_data[3]
                all_sizes = plotting_data[4]
                is_primary = False

            # For each circle to plot within the current species...
            for curr_color, curr_size in zip(all_colors, all_sizes):

                # Obtain the actual color and marker size to plot
                curr_color2 = colors[curr_color]
                curr_size2 = (((curr_size - 1) * marker_size_step + 1) * default_marker_size) ** 2

                # Plot the current species
                # Note: coord_units_in_microns is # of microns per unit (same as coord_units_in_microns, the new input parameter to the entire workflow)
                curr_plt = ax.scatter(x_pos * coord_units_in_microns, y_pos * coord_units_in_microns, s=curr_size2, c=curr_color2, edgecolors='k', alpha=alpha)

                # If we're on a primary species (in which a single circle is plotted for a potentially compound species), add the current plot to the legend
                if is_primary:
                    curr_label = get_descriptive_cell_label(plotting_data[1], mapping_dict)[0]
                    plotted_colors.append(curr_color)  # keep a record of the colors we've plotted so far in order to add a minimal number of non-primary species to the legend
                    plots_for_legend.append(curr_plt)
                    labels_for_legend.append(curr_label)

                # If we're on a non-primary species, only add it to the legend if the color hasn't yet been plotted
                else:

                    # If the color hasn't yet been plotted...
                    if curr_color not in plotted_colors:

                        # Get the correct label for the current phenotype within the non-primary species
                        # curr_label = get_descriptive_cell_label([plotting_data[1][icircle]], mapping_dict)[0] # this assumes the order in plotting_data[1] is consistent with that in plotting_data[3] and plotting_data[4], which is not necessarily true!
                        curr_label = get_descriptive_cell_label([label_lookup_for_color[[x[1] for x in label_lookup_for_color].index(curr_color)][0]], mapping_dict)[0]  # this performs the correct lookup

                        # If the current color to add to the legend was NOT a minimal size, first make a dummy plot of one of the minimal size
                        if not curr_size == 1:
                            curr_plt = ax.scatter(x_range[0] * 2 * coord_units_in_microns, y_range[0] * 2 * coord_units_in_microns, s=(default_marker_size**2), c=curr_color2, edgecolors='k', alpha=alpha)

                        # Add the plot to the legend
                        plotted_colors.append(curr_color)  # keep a record of the colors we've plotted so far in order to add a minimal number of non-primary species to the legend
                        plots_for_legend.append(curr_plt)
                        labels_for_legend.append(curr_label)

        # Complete the plot and save to disk
        ax.set_aspect('equal')
        ax.set_xlim(tuple(x_range * coord_units_in_microns))
        ax.set_ylim(tuple(y_range * coord_units_in_microns))
        ax.set_xlabel('X coordinate (microns)')
        ax.set_ylabel('Y coordinate (microns)')
        ax.set_title('ROI ' + title)
        ax.legend(plots_for_legend, labels_for_legend, loc='upper left', bbox_to_anchor=(1, 0, 1, 1))

        # Save the figure
        if filepath is not None:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


# For the inputted ROI coordinates, print the minimum coordinate spacing, coordinate range, number of cells, ROI area, and density
def roi_checks_and_output(x_roi, y_roi, do_printing=True):
    import numpy as np
    # coord_spacing_check = 0.5
    x_roi2 = x_roi.copy()
    y_roi2 = y_roi.copy()
    x_roi2.sort()
    y_roi2.sort()
    unique_spacing_x = np.unique(x_roi2[1:] - x_roi2[0:-1])[0:2]
    unique_spacing_y = np.unique(y_roi2[1:] - y_roi2[0:-1])[0:2]

    if unique_spacing_x[1] != unique_spacing_y[1]:
        print('NOTE: Minimum spacing in x coordinates ({}) is different from that in y coordinates ({})'.format(unique_spacing_x[1], unique_spacing_y[1]))
    min_coordinate_spacing = unique_spacing_x[1]
    print('Calculated minimum coordinate spacing: {}'.format(min_coordinate_spacing))

    # expected_unique_spacing = [0,coord_spacing_check]
    # if (not (unique_spacing_x==expected_unique_spacing).all()) or (not (unique_spacing_y==expected_unique_spacing).all()): # checks that coord_spacing is coord_spacing_check
    #     print('ERROR: Coordinate spacing is not', coord_spacing_check)
    #     exit()
    x_range = [x_roi2.min(), x_roi2.max()]
    y_range = [y_roi2.min(), y_roi2.max()]
    if do_printing:
        print('x range:', x_range)
        print('y range:', y_range)
    ncells = len(x_roi2)
    area = (x_roi2.max() - x_roi2.min()) * (y_roi2.max() - y_roi2.min())
    if do_printing:
        print('[ncells, area, density]:', [ncells, area, ncells / area])
    # return(area, unique_spacing_x[1])
    return(x_range, y_range, min_coordinate_spacing)


def calculate_metrics_from_coords(min_coord_spacing, input_coords=None, neighbors_eq_centers=False, ncenters_roi=1300, nneighbors_roi=220, nbootstrap_resamplings=0, rad_range=(2.2, 5.1), use_theoretical_counts=False, roi_edge_buffer_mult=1, roi_x_range=(1.0, 100.0), roi_y_range=(0.5, 50.0), silent=False):
    '''
    Given a set of coordinates (whether actual coordinates or ones to be simulated), calculate the P values and Z scores.

    See the calculate_metrics() method of the TIMECellInteraction class for further documentation.
    '''

    # Import relevant libraries
    import numpy as np
    import scipy.spatial
    import scipy.stats

    # Constant
    tol = 1e-8

    # If input_coords is None, then simulate the coordinates
    if input_coords is not None:
        coords_centers, coords_neighbors = input_coords
        ncenters_roi = coords_centers.shape[0]
        if coords_neighbors is not None:
            nneighbors_roi = coords_neighbors.shape[0]
        else:
            nneighbors_roi = None
        simulate_coords = False
    else:
        simulate_coords = True

    # Calculate some properties of the ROI itself
    roi_area_adj = (roi_x_range[1] - roi_x_range[0] + min_coord_spacing) * (roi_y_range[1] - roi_y_range[0] + min_coord_spacing) - min_coord_spacing**2
    ngridpoints_x = int((roi_x_range[1] - roi_x_range[0]) / min_coord_spacing + 1)
    ngridpoints_y = int((roi_y_range[1] - roi_y_range[0]) / min_coord_spacing + 1)
    grid_indices = np.indices((ngridpoints_x, ngridpoints_y)).reshape((2, -1)).T

    # Properties of the slice
    slice_area = np.pi * (rad_range[1]**2 - rad_range[0]**2)
    slice_area_adj = slice_area - min_coord_spacing**2
    if slice_area_adj < 0:
        print('ERROR: Slice area is too small')
        exit()

    # Calculate the expected number of neighbors in the slice; see physical notebook notes on 1/7/21 for details
    roi_area_used = roi_area_adj
    if (not neighbors_eq_centers) and (rad_range[0] > 0):
        npossible_neighbors = nneighbors_roi
        slice_area_used = slice_area
    elif (not neighbors_eq_centers) and (rad_range[0] < tol):
        npossible_neighbors = nneighbors_roi
        slice_area_used = slice_area_adj
    elif (neighbors_eq_centers) and (rad_range[0] > 0):
        npossible_neighbors = ncenters_roi - 1
        slice_area_used = slice_area
    elif (neighbors_eq_centers) and (rad_range[0] < tol):
        # NOTE THAT IN THIS CASE WE MUST SUBTRACT 1 FROM NNEIGHBORS (if not use_simulated_counts)!!
        npossible_neighbors = ncenters_roi - 1
        slice_area_used = slice_area_adj
    else:
        print('ERROR: Impossible situation encountered')
        exit()
    nexpected = npossible_neighbors / roi_area_used * slice_area_used

    # Initialize the random number generator
    rng = np.random.default_rng()

    # Generate random coordinates for the centers and neighbors; by sampling without replacement, we're simulating our real data in that no two species occupy the same point on the grid
    if neighbors_eq_centers:
        if simulate_coords:
            coords_centers = (rng.choice(grid_indices, size=ncenters_roi, replace=False, shuffle=False) * min_coord_spacing) + np.array([roi_x_range[0], roi_y_range[0]])[np.newaxis, :]
        coords_neighbors = coords_centers
    else:
        if simulate_coords:
            coords_tmp = (rng.choice(grid_indices, size=(ncenters_roi + nneighbors_roi), replace=False, shuffle=False) * min_coord_spacing) + np.array([roi_x_range[0], roi_y_range[0]])[np.newaxis, :]
            coords_centers = coords_tmp[:ncenters_roi, :]
            coords_neighbors = coords_tmp[ncenters_roi:, :]

    # Determine and count the centers that are within the larger radius of the ROI boundaries
    if not silent:
        if np.abs(roi_edge_buffer_mult - 1) > tol:
            print('WARNING: roi_edge_buffer_mult should be set to at least the default value of 1 (1 keeps the most data) in order to correctly account for edge effects; it is currently set to {}'.format(roi_edge_buffer_mult))
    valid_centers = (coords_centers[:, 0] <  (roi_x_range[1] + min_coord_spacing / 2 - roi_edge_buffer_mult * rad_range[1])) & \
                    (coords_centers[:, 0] >= (roi_x_range[0] - min_coord_spacing / 2 + roi_edge_buffer_mult * rad_range[1])) & \
                    (coords_centers[:, 1] <  (roi_y_range[1] + min_coord_spacing / 2 - roi_edge_buffer_mult * rad_range[1])) & \
                    (coords_centers[:, 1] >= (roi_y_range[0] - min_coord_spacing / 2 + roi_edge_buffer_mult * rad_range[1]))
    nvalid_centers = valid_centers.sum()

    # As long as there's at least one valid center (i.e., one sample)...
    if nvalid_centers >= 1:

        # Calculate the number of neighbors between the slice radii around all valid centers
        if use_theoretical_counts:
            if not silent:
                print('NOTE: Using artificial distribution')
            nneighbors = scipy.stats.poisson.rvs(nexpected, size=(nvalid_centers,))
        else:
            dist_mat = scipy.spatial.distance.cdist(coords_centers[valid_centers, :], coords_neighbors, 'euclidean')  # calculate the distances between the valid centers and all the neighbors
            nneighbors = ((dist_mat >= rad_range[0]) & (dist_mat < rad_range[1])).sum(axis=1)  # count the number of neighbors in the slice around every valid center
            if (neighbors_eq_centers) and (rad_range[0] < tol):
                nneighbors = nneighbors - 1  # we're always going to count the center as a neighbor of itself in this case, so account for this; see also physical notebook notes on 1/7/21

        # From all possible numbers of neighbors calculate the bin edges
        nbins = nneighbors.max() + 1
        edges = np.arange(nbins + 1) - 0.5

        # Redefine the counts of the neighbors around the centers (i.e., the densities), either by adding a singleton dimension or performing bootstrap resampling
        if nbootstrap_resamplings == 0:
            nneighbors = nneighbors[:, np.newaxis]  # first add a new single-dimension axis so that there's only a single "sample"
        else:
            nneighbors = rng.choice(nneighbors, size=(nvalid_centers, nbootstrap_resamplings), replace=True)  # first perform bootstrap resampling

        # Generate histograms (i.e., the non-normalized PMFs) of the numbers of neighbors
        nsamples = nneighbors.shape[1]
        pmf = np.zeros((nbins, nsamples))
        for isample in range(nsamples):
            pmf[:, isample], _ = np.histogram(nneighbors[:, isample], bins=edges, density=False)  # this quantity (the number of centers having j neighbors) is binomial-distributed and is not technically the PMF since we are not normalizing it here

        # Calculate the Z scores and P values for the densities and PMFs
        density_metrics = calculate_density_metrics(nneighbors, nexpected)
        pmf_metrics = calculate_PMF_metrics(pmf, nexpected, nvalid_centers)

    # If there are no valid centers, don't return anything for the density and PMF metrics
    else:
        density_metrics = None
        pmf_metrics = None
        edges = None

    return(density_metrics, pmf_metrics, nexpected, nvalid_centers, coords_centers, coords_neighbors, valid_centers, edges, npossible_neighbors, roi_area_used, slice_area_used)


def calculate_density_metrics(nneighbors, nexpected):
    '''
    Calculate the Z score (not generally applicable) and the left and right P values for the "density".
    '''

    # Import relevant libraries
    import numpy as np
    import scipy.stats

    # Calculate the number of samples (and number of valid centers) from the input array, which may be 1 particularly in the case where we've decided not to bootstrap prior to calling this function
    nvalid_centers, nsamples = nneighbors.shape

    # Define the array holding the values of interest: the Z score, the "less" or "left" P value, and the "greater" or "right" P value
    metrics = np.zeros((nsamples, 3))

    # Define the variable that is Poisson-distributed
    k = nneighbors.sum(axis=0)  # (nsamples,)

    # Parameter of the Poisson distribution
    mu = nvalid_centers * nexpected  # scalar

    # Properties of the Poisson distribution
    mean = mu  # scalar
    std = np.sqrt(mu)  # scalar

    # Calculate the metrics
    metrics[:, 0] = (k - mean) / std  # z-score; this is meaningless if the distribution is not approximately Gaussian
    metrics[:, 1] = scipy.stats.poisson.cdf(k, mu)  # "less" or "left" P value
    metrics[:, 2] = scipy.stats.poisson.sf(k - 1, mu)  # "greater" or "right" P value

    # Average over all the samples to reduce sensitivity in favor of specificity (if bootstrapping)
    return(metrics.mean(axis=0))  # (3,)


def calculate_PMF_metrics(pmf, nexpected, nvalid_centers):
    '''
    Calculate the Z score (not generally applicable) and the left and right P values for the "PMF".

    Note I found for the random points that the return value never had any nans.
    '''

    # Import relevant libraries
    import numpy as np
    import scipy.stats

    # Calculate some array sizes
    nbins, nsamples = pmf.shape

    # Define the metrics array containing the Z score and the left and right P values
    metrics = np.zeros((nbins, nsamples, 3))

    # Define the variable that is binomial-distributed
    k = pmf  # (nbins, nsamples)

    # Parameters of the binomial distribution
    n = nvalid_centers  # scalar
    p = scipy.stats.poisson.pmf(np.arange(nbins), nexpected)[:, np.newaxis]  # (nbins,1)

    # Properties of the binomial distribution
    mean = n * p  # (nbins,1)
    std = np.sqrt(n * p * (1 - p))  # (nbins,1)

    # Calculate the metrics
    metrics[:, :, 0] = (k - mean) / std  # z-score; this is meaningless if the distribution is not approximately Gaussian
    metrics[:, :, 1] = scipy.stats.binom.cdf(k, n, p)  # "less" or "left" P value
    metrics[:, :, 2] = scipy.stats.binom.sf(k - 1, n, p)  # "greater" or "right" P value

    # Average over all the samples to reduce sensitivity in favor of specificity (if bootstrapping)
    return(metrics.mean(axis=1))  # (nbins,3)


def plot_pvals(fig, data_by_slice, log_pval_range, name=None, calculate_empty_bin_pvals=False, max_nbins_over_slices=None, square=True, yticklabels=2):
    '''
    Plot the four-axis P value heatmaps.

    The top row has the left and right density P values and the bottom row has the left and right PMF P values.

    Here we carefully account for when the number of valid centers is 0, which can be true when this function is called; this function is only not called in the first place when there are no centers of a current species in the ROI... it is thus still called even if there are no VALID centers in the ROI.

    Further, if there are no valid centers for the smallest slice size, then there can't be any valid centers for any larger slice sizes, and therefore there are no valid centers for the current center species period. In this case, both pvals_dens and pvals_pmf below will be None throughout. Furthermore, as the validity of the centers is unaffected by the neighbor species, then there would be no data for this center species as a whole (for any neighbor species), and therefore the entire species as a center could be dropped from any analysis plots. In this case, max_nbins_over_slices would remain 0.

    Of course, if there were some valid centers for the smallest slice size but not for larger slice sizes, then there would indeed be data in these analysis plots, and we should plot the data for this center species as usual. In this case, max_nbins_over_slices would not end up as 0.
    '''

    # Import relevant libraries
    import numpy as np
    import seaborn as sns

    # Calculate some variables
    nslices = len(data_by_slice)  # number of slices from the input data
    (vmin, vmax) = log_pval_range

    # Initialize some variables of interest
    any_valid_centers_over_slices = False  # whether there are any valid centers of all the slices
    nvalid_centers_holder = np.zeros((nslices,), dtype=np.uint32)  # number of valid centers in each slice

    # Define and populate the array holding the density P values for all the slices for the current center/neighbor combination
    pvals_dens = np.zeros((1, nslices, 2))
    if max_nbins_over_slices is None:
        max_nbins_over_slices = 0  # determine the maximum number of bins over all nslices slices, initializing this value to zero
    for islice in range(nslices):  # for every slice...
        if data_by_slice[islice][3] >= 1:  # as long as there is at least 1 valid center...
            pvals_dens[:, islice, :] = data_by_slice[islice][0][1:]  # get the left and right pvals of the islice-th slice; (2,)
            curr_nbins = data_by_slice[islice][1].shape[0]  # get the number of PMF bins for the current slice
            if curr_nbins > max_nbins_over_slices:  # update the maximum number of bins over all the slices
                max_nbins_over_slices = curr_nbins
            any_valid_centers_over_slices = True  # if we've gotten here, then there are SOME valid centers over all the slices
            nvalid_centers_holder[islice] = data_by_slice[islice][3]  # store this number of valid centers for the current slice
        else:
            pvals_dens[:, islice, :] = None

    # Define and populate the array holding the PMF P values for all the slices for the current center/neighbor combination
    if any_valid_centers_over_slices:  # as long as there are some valid centers over all the slices...
        pvals_pmf = np.zeros((max_nbins_over_slices, nslices, 2))
        for islice in range(nslices):  # for every slice...
            if data_by_slice[islice][3] >= 1:  # as long as there is at least 1 valid center...
                curr_nbins = data_by_slice[islice][1].shape[0]  # get the number of PMF bins for the current slice
                pvals_pmf[:curr_nbins, islice, :] = data_by_slice[islice][1][:, 1:]  # get the left and right pvals of the islice-th slice for all the current number of bins; (curr_nbins,2)
                if not calculate_empty_bin_pvals:  # optionally fill in the P values for higher j-values (higher-bins) whose PMFs are zero since these bins aren't even returned in the first place
                    pvals_pmf[curr_nbins:, islice, :] = None  # set the bins that are not set for the current slice (but exist here because over all slices there are bins this large) to None
                else:
                    nexpected, nvalid_centers = data_by_slice[islice][2:4]
                    pvals_pmf[curr_nbins:, islice, :] = calculate_PMF_metrics(np.zeros((max_nbins_over_slices, 1)), nexpected, nvalid_centers)[curr_nbins:, 1:]  # (nbins_remaining,2)
            else:
                pvals_pmf[:, islice, :] = None

    # Initialize the current figure by clearing it and all its axes
    fig.clf()
    ax = fig.subplots(nrows=2, ncols=2)

    # Create the four-axis figure, where the top row has the left and right density P values and the bottom row has the left and right PMF P values
    if any_valid_centers_over_slices:

        # Plot the log10 of the left density P values
        left_log_dens_pvals = np.log10(pvals_dens[:, :, 0])
        sns.heatmap(left_log_dens_pvals, vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[0, 0], cbar=True, yticklabels=True, square=True)
        ax[0, 0].set_title('log10(\"less\" density pvals)')
        ax[0, 0].set_xlabel('Slice')

        # Plot the log10 of the right density P values
        right_log_dens_pvals = np.log10(pvals_dens[:, :, 1])
        sns.heatmap(right_log_dens_pvals, vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[0, 1], cbar=True, yticklabels=True, square=True)
        ax[0, 1].set_title('log10(\"greater\" density pvals)')
        ax[0, 1].set_xlabel('Slice')

        # Plot the log10 of the left PMF P values
        left_log_pmf_pvals = np.log10(pvals_pmf[:, :, 0])
        sns.heatmap(left_log_pmf_pvals, vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[1, 0], cbar=True, yticklabels=yticklabels, square=square)
        ax[1, 0].set_title('log10(\"less\" PMF pvals)')
        ax[1, 0].set_xlabel('Slice')
        ax[1, 0].set_ylabel('Number of neighbors')

        # Plot the log10 of the right PMF P values
        right_log_pmf_pvals = np.log10(pvals_pmf[:, :, 1])
        sns.heatmap(right_log_pmf_pvals, vmin=vmin, vmax=vmax, linewidths=.5, ax=ax[1, 1], cbar=True, yticklabels=yticklabels, square=square)
        ax[1, 1].set_title('log10(\"greater\" PMF pvals)')
        ax[1, 1].set_xlabel('Slice')
        ax[1, 1].set_ylabel('Number of neighbors')

        # Place a descriptive title on the figure
        if name is not None:
            fig.suptitle(name + '\nnum valid centers per slice: {}'.format(nvalid_centers_holder))

    else:
        # Initialize the return variables
        left_log_dens_pvals = None
        right_log_dens_pvals = None
        left_log_pmf_pvals = None
        right_log_pmf_pvals = None

    return(nvalid_centers_holder, left_log_dens_pvals, right_log_dens_pvals, left_log_pmf_pvals, right_log_pmf_pvals)


def get_max_nbins_for_center_neighbor_pair(data_by_slice, max_nbins_over_slices):
    '''
    Get the largest possible number of bins so that we can plot all the experiment's data in the same exact way (having the same number of rows for the PMF P value plots).

    These are calculated from the sizes of the PMF P value arrays and therefore the same sizes (nbins) as the calculated PMFs, where nbins = nneighbors.max() + 1.
    '''

    # Define and populate the array holding the density P values for all the slices for the current center/neighbor combination
    for islice in range(len(data_by_slice)):  # for every slice...
        if data_by_slice[islice][3] >= 1:  # as long as there is at least 1 valid center...
            curr_nbins = data_by_slice[islice][1].shape[0]  # get the number of PMF bins for the current slice
            if curr_nbins > max_nbins_over_slices:  # update the maximum number of bins over all the slices
                max_nbins_over_slices = curr_nbins

    return(max_nbins_over_slices)
