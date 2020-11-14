To simulate neuronal data, need to have the NEST simulator installed

conda create --name ENVNAME -c conda-forge nest-simulator
conda activate ENVNAME

Then, under the NRI_brain/simdata directory, to generate data simply run

python generate_neuron_dataset.py --num-train=TRAINSIZE --num-valid=VALIDSIZE --num-test=TESTSIZE

Then the data will be generated under the directory with name format like:
'spkmat_train_LIF_neurons100_p_exc0.8_epsilon0.1_len15000.npy'
'conn_train_LIF_neurons100_p_exc0.8_epsilon0.1_len15000.npy'
meaning 100 neurons, 80% of which are excitatory (other are inhibitary), and neuron-to-neuron connection probability is 0.1, and a total of 15000ms time series is generated.


spkmat is a matrix of size [number of simulations] x [length of simulation time] x [Number of neurons]

conn is a matrix of size [number of simulations] x [Number of neurons] x [Number of neurons]


The length of simulation can be adjusted by setting:
--length       :length of simulation time series on train/valid set
--length-test  :length of simulation time series on test set

Number of nuerons (N_neurons), percentage of excitatory neurons (p_exc), and connection probability (epsilon) can be adjusted by setting:
--N-neurons    :default is 100
--p-exc		   :default is 0.8
--epsilon      :default is 0.1, but it worth to experiemnt with different sparsity of connection