from LIF_neurons import LIF_neurons_Sim
from LIF_neurons_dynamic import LIF_neurons_dynamic_Sim
import time
import numpy as np 
import argparse
import scipy.sparse as sparse

parser = argparse.ArgumentParser()

# number of simulations
parser.add_argument('--simulation', type=str, default='LIF',
                    help='What kind of simulation to generate.')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')

# simulation spike time length (in ms)
parser.add_argument('--length', type=int, default=45000,
                    help='Length of training and validation spiking time.')
parser.add_argument('--length-test', type=int, default=45000,
                    help='Length of test set spiking time.')


parser.add_argument('--N-neurons', type=int, default=100,
                    help='Number of neurons in the simulation.')
parser.add_argument('--p-exc', type=float, default=0.8,
                    help='Percentage of excitatory neurons in the simulation.')
parser.add_argument('--epsilon', type=float, default=.1,
                    help='Probability of connection')
parser.add_argument('--Ie_factor', type=float, default=3.0,
                    help='External current amplitude level')


parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')


args = parser.parse_args()

if args.simulation == 'LIF':
    sim = LIF_neurons_Sim(N_neurons = args.N_neurons,p_exc = args.p_exc,epsilon = args.epsilon)
    suffix = '_LIF_neurons'
elif args.simulation == 'LIF_dynamic':
    sim = LIF_neurons_dynamic_Sim(N_neurons = args.N_neurons,p_exc = args.p_exc,epsilon = args.epsilon)
    suffix = '_LIF_dynamic_neurons'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))



suffix += str(args.N_neurons)+'_p_exc'+str(args.p_exc)+'_epsilon'+str(args.epsilon)+'_Ie_factor'+str(args.Ie_factor)
print(suffix)


np.random.seed(args.seed)

def generate_dataset(num_sims, length,Ie_factor):
    spkmat_all = list()
    conn_all = list()

    for i in range(num_sims):
        t = time.time()
        spkmat,conn = sim.simulate_network_and_spike(T=length,Ie_factor=Ie_factor)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        spkmat_all.append(spkmat)
        conn_all.append(conn)

    spkmat_all = np.stack(spkmat_all)
    conn_all = np.stack(conn_all)


    # reshaping to sparse matrix
    sparse_spkmat_all = sparse.csr_matrix(spkmat_all.reshape(num_sims, -1))
    sparse_conn_all   = sparse.csr_matrix(conn_all.reshape(num_sims,-1))

    return sparse_spkmat_all, sparse_conn_all



print("Generating {} training simulations".format(args.num_train))
spkmat_train, conn_train = generate_dataset(args.num_train,
                                                     args.length,args.Ie_factor)
                         

print("Generating {} validation simulations".format(args.num_valid))
spkmat_valid, conn_valid = generate_dataset(args.num_valid,
                                                     args.length,args.Ie_factor)

print("Generating {} test simulations".format(args.num_test))
spkmat_test, conn_test = generate_dataset(args.num_test,
                                                  args.length_test,args.Ie_factor)



sparse.save_npz('conn_train' + suffix +'_len'+str(args.length) +'.npz', conn_train)
sparse.save_npz('spkmat_train' + suffix +'_len'+str(args.length) +'.npz', spkmat_train)

sparse.save_npz('conn_valid' + suffix +'_len'+str(args.length) +'.npz', conn_valid)
sparse.save_npz('spkmat_valid' + suffix +'_len'+str(args.length) + '.npz', spkmat_valid)

sparse.save_npz('conn_test' + suffix +'_len'+str(args.length_test) +'.npz', conn_test)
sparse.save_npz('spkmat_test' + suffix + '_len'+str(args.length_test)+'.npz', spkmat_test)





