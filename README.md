# Internship-LPSM-IBENS
Non parametric estimation in a cladogenesis model


### Python code ### 

• example.ipynb is a notebook where an example of the optimization procedure on the cetacean phylogeny is detailed. 

• function_optim.py contains all the function used in the optimization procedure (computation of the statistics, loss, penalties, gradient descent...) 

• the folder cross_validation contains an example of cross validation in the file cross_validation.py, while the necessary functions are in cross_valid_score.py. /!\ the file cross_validation.py takes tens of hours to run if not in parallel, I recommend running it on 100 nodes /!\.

### R code ###
 This code we used to extract all the needed data (subtrees for cross validations...) and to simulate phylogenetic trees (in Tree_sim.ipynb). 
