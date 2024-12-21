1) 	Install Numpy, Matplotlib, Seaborn, PyTorch, torchvision, scikit-learn
	(or)
	activate the conda statml environment

2) 	Run the python files directly. (Though I used Jupyter while working, I've combined every code to a single executable script for each model)

3) As all the script tunes a large set of hyperparameters using GridSearchCV (or) Random Search for 50 iterations to determine the best set of hyperparams, Expect Neural Networks to take atleast 2hrs and kernel SVM to take 1hr.
	(or) 
	You can adjust the hyperparameter_grid to have the values of C=10, Kernel=RBF and Gamma=auto, which are the best values obtained in my case after tuning for kernel SVM.
	You can make the `n_iterations` parameter in the random_hyperparameter_tuning_function() to 1 and run the codes.
	
