
# Simulate the spiking time series of time-varying randomly-connected ensembles of excitatory and inhibitory 
# leaky integrate-and-fire neurons with delta synapses.

# The code is modified from 'https://gitlab.com/di. ma/Connectivity_from_event_timing_patterns' 
# from the paper 'Inferring network connectivity from event timing patterns' and need the NEST 
# simulator installed. see:  http://www.nest-simulator.org/py_sample/brunel_delta_nest/index.html

# Xinyue, 11142020


# Dynamic: The underlying connectivity between neurons changes by phase:
# - in each phase:
# - a random number of neurons is selected to change connection
# - for those selected neurons, their post-synaptic connections are randomly changed, but the number of connections is kept fix

# key Input Parameters:
# - N_neurons: number of neurons
# - epsilon:   connection probability between two neuron
# - p_exc:     percentage of excitatory neurons, the rest are inhibitary
# - T:   	   total simulation time in ms within each phase

# - n_phase:  number of times the connectivity changes
# - p_change: the percentage of neurons selected to change within each phase



# - other parameters of neuronal property etc hard coded as default


# Output:
# - connectivity.npy:   n_phase by N by N connectivity matrix used in the simulation
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

def extract_spk_matrix(N_neurons,n_phase,simtime):
    N = N_neurons
    spk= [[] for x in range(N)]
    ISIs= [[] for x in range(N)]
    ex_spk=np.loadtxt("nestData/ex_neurons-%d-0.gdf"%(N+1))
    in_spk=np.loadtxt("nestData/in_neurons-%d-0.gdf"%(N+2))

    spk_mat = np.zeros((N_neurons,int(simtime*n_phase)+1))

    count = 0

    for i in range(len(ex_spk)):
        spk_mat[int(ex_spk[i,0])-1,int(np.floor(ex_spk[i,1]))] = 1

    for i in range(len(in_spk)):
        spk_mat[int(in_spk[i,0])-1,int(np.floor(in_spk[i,1]))] = 1
    return spk_mat

def extract_conn_matrix(N_neurons, nodes_ex,nodes_in):
###############################################################################
# Extracting the connectivity matrix
###############################################################################
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
    return connectivity



nest.ResetKernel()
class LIF_neurons_dynamic_Sim(object):
	def __init__(self,N_neurons=100,p_exc=0.8,epsilon=0.1):
		self.N_neurons = N_neurons
		self.NE = int(N_neurons*p_exc) # number of excitatory neurons
		self.NI = N_neurons - self.NE  # number of inhibitory neurons
		self.CE = round(epsilon * self.NE) # ~number of excitatory synapses per neuron
		self.CI = round(epsilon * self.NI) # ~number of inhibitory synapses per neuron
		self.C_tot = int(self.CI+self.CE) # total number of synapses per neuron

	def simulate_network_and_spike(self, T = 15000.0, n_phase=3, p_change=0.8,Ie_factor=3.0):
		# this function build the random connectivity network of neurons and simulate spiking 
		# time series for each neuron

		# Output:
		# connectivity: N_neuron x N_neuron matrix
		# spk_mat:      T x N_neuron matrix of spiking time series per neuron
        
		## Output files path for NEST tmp file
		if os.path.isdir('nestData'):
			os.system("rm -r nestData")
		os.system("mkdir nestData")

		## Randomization of dynamics
		nest.ResetKernel()
		msd = int(np.ceil(100000*np.random.rand(1)))
		N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
		pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
		nest.SetKernelStatus({'grng_seed' : msd+N_vp})

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

		n_phase = n_phase

		## Grabbing number of neurons and synapses
		NE = self.NE
		NI = self.NI
		N_neurons = self.N_neurons
		CE = self.CE
		CI = self.CI
		C_tot = self.C_tot

		r_NE = round(NE*p_change)
		r_NI = round(NI*p_change)

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

		for neuron in nodes_ex:
		    nest.SetStatus([neuron], {"V_m": 0.0+(theta-0.0)*np.random.rand()})
		    nest.SetStatus([neuron], {"I_e": Ie_factor*(1.2+(1.4-1.2)*np.random.rand())})

		for neuron in nodes_in:
		    nest.SetStatus([neuron], {"V_m": 0.0+(theta-0.0)*np.random.rand()})
		    nest.SetStatus([neuron], {"I_e": Ie_factor*(1.2+(1.4-1.2)*np.random.rand())})
   

		conn = np.zeros((n_phase,N_neurons,N_neurons))
		for i_phase in np.arange(n_phase):
		    if i_phase==0:
		        # case of the first phase 
		        espikes = nest.Create("spike_detector")
		        ispikes = nest.Create("spike_detector")
		        ###############################################################################
		        # Defining output files
		        ###############################################################################
		        nest.SetStatus(espikes, [{"label": "Data/ex_neurons",
		                                  "withtime": True,
		                                  "withgid": True,
		                                  "to_file": True}])

		        nest.SetStatus(ispikes, [{"label": "Data/in_neurons",
		                                  "withtime": True,
		                                  "withgid": True,
		                                  "to_file": True}])



		        ###############################################################################
		        # Connecting neurons and spike detectors
		        ###############################################################################
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

		        conn[0,:,:] = extract_conn_matrix(N_neurons,nodes_ex,nodes_in)

		        nest.Simulate(simtime)
		    else:
		        r_nodes_ex = np.random.permutation(nodes_ex)[0:r_NE]
		        r_nodes_in = np.random.permutation(nodes_in)[0:r_NI]
		        for s in r_nodes_ex:
		            target_s = []
		            all_target_s = []
		            new_target_s = []
		            target_ss = [i[1] for i in nest.GetConnections(source=[s])]
		            target_s = [i for i in target_ss if i<=N_neurons]
		            if len(target_s)>0:
		                nest.Disconnect([s]*len(target_s),target_s,syn_spec='excitatory')
		                all_target_s = list(range(1,N_neurons+1))
		                all_target_s.remove(s)
		                new_target_s = np.random.permutation(all_target_s)[:len(target_s)]
		                nest.Connect([s],new_target_s.tolist(),syn_spec="excitatory")
		        for s in r_nodes_in:
		            target_s = []
		            all_target_s = []
		            new_target_s = []
		            target_ss = [i[1] for i in nest.GetConnections(source=[s])]
		            target_s = [i for i in target_ss if i<=N_neurons]
		            if len(target_s)>0:
		                nest.Disconnect([s]*len(target_s),target_s,syn_spec='inhibitory')
		                all_target_s = list(range(1,N_neurons+1))
		                all_target_s.remove(s)
		                new_target_s = np.random.permutation(all_target_s)[:len(target_s)]
		                nest.Connect([s],new_target_s.tolist(),syn_spec="inhibitory")
		        
		        nest.Simulate(simtime)
		        conn[i_phase,:,:] = extract_conn_matrix(N_neurons,nodes_ex,nodes_in)
		        
		#extract spiking from simulation
		spk_mat = extract_spk_matrix(N_neurons,T,n_phase)
		spk_mat = spk_mat.T
		os.system("rm -r nestData")
		return spk_mat, conn