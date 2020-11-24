
# Simulate the spiking time series of randomly-connected ensembles of excitatory and inhibitory 
# leaky integrate-and-fire neurons with delta synapses.

# The code is modified from 'https://gitlab.com/di. ma/Connectivity_from_event_timing_patterns' 
# from the paper 'Inferring network connectivity from event timing patterns' and need the NEST 
# simulator installed. see:  http://www.nest-simulator.org/py_sample/brunel_delta_nest/index.html

# Xinyue, 11062020

# key Input Parameters:
# - N_neurons: number of neurons
# - epsilon:   connection probability between two neuron
# - p_exc:     percentage of excitatory neurons, the rest are inhibitary
# - T:   total simulation time in ms

# - other parameters of neuronal property etc hard coded as default


# Output:
# - connectivity.npy:   N by N connectivity matrix used in the simulation
# - spktimes.npy:       T by N matrix containing the spike time series of all neurons



import nest
import nest.raster_plot
import time
import numpy as np
import os as os
import matplotlib.pyplot as plt
import pylab as pl
import scipy
import sklearn
from sklearn import metrics


class LIF_neurons_Sim(object):
	def __init__(self,N_neurons=100,p_exc=0.8,epsilon=0.1):
		self.N_neurons = N_neurons
		self.NE = int(N_neurons*p_exc) # number of excitatory neurons
		self.NI = N_neurons - self.NE  # number of inhibitory neurons
		self.CE = round(epsilon * self.NE) # ~number of excitatory synapses per neuron
		self.CI = round(epsilon * self.NI) # ~number of inhibitory synapses per neuron
		self.C_tot = int(self.CI+self.CE) # total number of synapses per neuron

	def simulate_network_and_spike(self, T = 50000.0, Ie_factor=3.0):
		# this function build the random connectivity network of neurons and simulate spiking 
		# time series for each neuron

		# Output:
		# connectivity: N_neuron x N_neuron matrix
		# spk_mat:      T x N_neuron matrix of spiking time series per neuron
        
		tmpfolder ='NestData'+str(np.random.randint(1,10000))
		## Randomization of dynamics
		nest.ResetKernel()
		msd = int(np.ceil(100000*np.random.rand(1)))
		N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
		pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
		nest.SetKernelStatus({'grng_seed' : msd+N_vp})


		## Output files path for NEST tmp file
		if os.path.isdir(tmpfolder):
			os.system("rm -r "+tmpfolder)
		os.system("mkdir "+tmpfolder)

		nest.ResetKernel()
		startbuild = time.time()

		## Defining simulation paramters
		dt = 0.1  # time resolution in ms
		simtime = T  # total simulation time in ms
		delay = 1.5  # synaptic delay in ms
		g = 4.0  # ratio inhibitory weight/excitatory weight
		epsilon = 0.1  # connection probability
		J = 0.3  # postsynaptic amplitude in mV
		J_ex = J  # amplitude of excitatory postsynaptic potential
		J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

		## Grabbing number of neurons and synapses
		NE = self.NE
		NI = self.NI
		N_neurons = self.N_neurons
		CE = self.CE
		CI = self.CI
		C_tot = self.C_tot

		## Defining default properties of neurons
		tauMem = 20.0
		theta  = 20.0
		neuron_params = {"C_m": 1.0,
		         "tau_m": tauMem,
		         "t_ref": 2.0,
		         "E_L": 0.0,
		         "V_reset": 0.0,
		         "V_m": 0.0,
		         "V_th": theta}

		## Creating neurons and spike detectors
		print("Building network model")
		nest.SetKernelStatus({"resolution": dt, "print_time": True,
		                      "overwrite_files": True})

		nest.SetDefaults("iaf_psc_delta", neuron_params)

		nodes_ex = nest.Create("iaf_psc_delta", NE)
		nodes_in = nest.Create("iaf_psc_delta", NI)
		espikes = nest.Create("spike_detector")
		ispikes = nest.Create("spike_detector")


		## Randomizing initial conditions and external inputs of neurons
		# Here using tonic external input current uniformly sampled U(1.2,1.4)
		for neuron in nodes_ex:
			nest.SetStatus([neuron], {"V_m": 0.0+(theta-0.0)*np.random.rand()})
			nest.SetStatus([neuron], {"I_e": Ie_factor*(1.2+(1.4-1.2)*np.random.rand())})

		for neuron in nodes_in:
			nest.SetStatus([neuron], {"V_m": 0.0+(theta-0.0)*np.random.rand()})
			nest.SetStatus([neuron], {"I_e": Ie_factor*(1.2+(1.4-1.2)*np.random.rand())})


		## Defining the tmporary output files of the spikes
		nest.SetStatus(espikes, [{"label": tmpfolder+"/ex_neurons",
		                      "withtime": True,
		                      "withgid": True,
		                      "to_file": True}])

		nest.SetStatus(ispikes, [{"label": tmpfolder+"/in_neurons",
		                      "withtime": True,
		                      "withgid": True,
		                      "to_file": True}])


		## Connecting neurons and spike detectors
		print("Connecting devices")

		nest.CopyModel("static_synapse", "excitatory",
		               {"weight": J_ex, "delay": delay})

		nest.CopyModel("static_synapse", "inhibitory",
		               {"weight": J_in, "delay": delay})

		nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
		nest.Connect(nodes_in, ispikes, syn_spec="excitatory")
		 
		sources_ex = np.random.random_integers(1, NE, (N_neurons, CE))
		sources_in = np.random.random_integers(NE + 1, N_neurons, (N_neurons, CI))

		NE_array = np.arange(1,NE+1)
		NI_array = np.arange(NE+1,N_neurons+1)
		for n in range(N_neurons):
			if np.isin(n+1,sources_ex[n]):
				n_idx, =np.where(n+1==NE_array)
				tmp_NE = np.delete(NE_array,n_idx)
				tmp_NE_p = np.random.permutation(tmp_NE)
				sources_ex[n] = tmp_NE_p[:CE]
			nest.Connect(list(sources_ex[n]), [n + 1], syn_spec="excitatory")
		for n in range(N_neurons):
			if np.isin(n+1,sources_in[n]):
				n_idx, =np.where(n+1==NI_array)
				tmp_NI = np.delete(NI_array,n_idx)
				tmp_NI_p = np.random.permutation(tmp_NI)
				sources_in[n] = tmp_NI_p[:CI]   
			nest.Connect(list(sources_in[n]), [n + 1], syn_spec="inhibitory")

		## Extracting connectivity matrix
		# Connectivity[source,target] = weight
		connectivity=np.zeros((N_neurons,N_neurons))
		conn_ex=nest.GetConnections(nodes_ex)
		conn_ex_source= nest.GetStatus(conn_ex, keys='source')
		conn_ex_target= nest.GetStatus(conn_ex, keys='target')
		conn_ex_weight= nest.GetStatus(conn_ex, keys='weight')

		conn_in=nest.GetConnections(nodes_in)
		conn_in_source= nest.GetStatus(conn_in, keys='source')
		conn_in_target= nest.GetStatus(conn_in, keys='target')
		conn_in_weight= nest.GetStatus(conn_in, keys='weight')

		for i in range(len(conn_ex_source)):
			if conn_ex_source[i]<= N_neurons and conn_ex_target[i]<= N_neurons:
				connectivity[conn_ex_source[i]-1,conn_ex_target[i]-1]=conn_ex_weight[i]
		for i in range(len(conn_in_source)):
			if conn_in_source[i]<=N_neurons and conn_in_target[i]<= N_neurons:
				connectivity[conn_in_source[i]-1,conn_in_target[i]-1]=conn_in_weight[i]				
		#connectivity=connectivity.T


		## Running the simulation
		endbuild = time.time()
		print('Simulating')
		nest.Simulate(simtime)
		endsimulate = time.time()
		events_ex = nest.GetStatus(espikes, "n_events")[0]
		events_in = nest.GetStatus(ispikes, "n_events")[0]
		build_time = endbuild - startbuild
		sim_time = endsimulate - endbuild

		print("Building time     : %.2f s" % build_time)
		print("Simulation time   : %.2f s" % sim_time)


		# Extracting the spikes
		ex_spk=np.loadtxt(tmpfolder+"/ex_neurons-%d-0.gdf"%(N_neurons+1))
		in_spk=np.loadtxt(tmpfolder+"/in_neurons-%d-0.gdf"%(N_neurons+2))

		spk_mat = np.zeros((N_neurons,int(simtime)))

		for i in range(len(ex_spk)):
			if ex_spk[i,1]<simtime:
				spk_mat[int(ex_spk[i,0])-1,int(ex_spk[i,1])] = 1

		for i in range(len(in_spk)):
			if in_spk[i,1]<simtime:
				spk_mat[int(in_spk[i,0])-1,int(in_spk[i,1])] = 1

		spk_mat = spk_mat.T

		os.system("rm -r "+tmpfolder)

		return spk_mat, connectivity

























