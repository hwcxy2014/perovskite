Module needs to be installed:
 - lhsmdu

If more solvents are added to the model, the following files need to be modified:
 - data_solvent.csv
 
Main function for running BO pipeline:
 - BO_pipeline_be.py 

To plot the result of BO pipeline:
 - plot_BO_process.py 

Description of .py files:
 - stat_model : describes the distributions of solubilities
 - solubility : An object contains the mean function of and the kernel of the GP
 					imposed on the solubitlities of all the n solutions.
				It also contains methods that update the mean and kernel of the GP
					given sample points.
 - BO_pipeline : Builds an object from solutility, determines where to sample next
 					and calls the simulation pipeline to get the next sample.
 - plot_BO_process :  plots the results of BO_pipeline
 - data_parser, read_numbering_solutions : read data from files, organize raw info \
 					into desired format
Description of .csv files:
 - numbering_solutions : a list of all solution combinations
 							a row = #, Ha, Ca, solvent
 - data_solvent : records umbo and polarity of solvents
 							a row = solvent name, umbo, polarity
 - data_iniSample : initial samples for estimating hyperparameters
 						selected by select_initial_sample.py
