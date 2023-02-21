import numpy as np
import matplotlib.pyplot as plt
import glob

plt.style.use(['science','nature'])

# flags
ytype = 'conductance' # acceptable strings are 'resistance' and 'conductance

# device
dev = 'outputs'
# directory
dir = './' + dev + '/'

# files = [
#             '230219_fmnist_SeqNet_poisson500_lr0.0001_tlen60_hch1_och1_config1','230219_fmnist_SeqNet_poisson400_lr0.0001_tlen60_hch1_och1_config1','230219_fmnist_SeqNet_poisson300_lr0.0001_tlen60_hch1_och1_config1','230219_fmnist_SeqNet_poisson200_lr0.0001_tlen60_hch1_och1_config1','230219_fmnist_SeqNet_poisson100_lr0.0001_tlen60_hch1_och1_config1','230219_fmnist_SeqNet_poisson50_lr0.0001_tlen60_hch1_och1_config1',
#             '230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen60_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson400_lr0.0001_tlen60_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson300_lr0.0001_tlen60_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson200_lr0.0001_tlen60_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson100_lr0.0001_tlen60_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson50_lr0.0001_tlen60_hch2_och2_config1',
#             '230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen60_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson400_lr0.0001_tlen60_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson300_lr0.0001_tlen60_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson200_lr0.0001_tlen60_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson100_lr0.0001_tlen60_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson50_lr0.0001_tlen60_hch1_och4_config1',
#          ]

files = [
            '230219_fmnist_SeqNet_poisson500_lr0.0001_tlen10_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen20_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen30_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen40_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen50_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen60_hch1_och1_config1',
            '230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen10_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen20_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen30_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen40_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen50_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen60_hch2_och2_config1',
            '230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen10_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen20_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen30_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen40_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen50_hch4_och1_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen60_hch1_och4_config1',
         ]

# freq = np.array([1/500,1/400,1/300,1/200,1/100,1/50])
freq = np.array([10,20,30,40,50,60])
accs = []
fig1,ax1 = plt.subplots(1,1,figsize=(2.0,1.5))
for idx,fname in enumerate(files):
    data = np.mean(np.load(dir + fname + '/accuracies.npy'),axis=1)
    # print(data)
    # print(fname)
    # variables (add as needed)
    acc_hi = np.max(data)
    accs.append(acc_hi)
    # i_g1 = data['CH2 Current']
    # i_g2 = data['CH3 Current']
    # i_ctl = data['CH4 Current']

    # plotting
    # plt.ylabel('Drain current (mA)')
    # plt.xlabel('Gate voltage (V)')
    # plt.title('Linear transfer curve of PEDOT control: '+dev)
    # plt.subplot(1,2,2)
    # plt.semilogy(v_g,np.abs(i_ds))
    # print(np.min(i_ds))
    # plt.ylabel('Drain current (A)')
    # plt.xlabel('Gate voltage (V)')
    # # plt.ylim([1e-15,1e-1])
    # plt.title('Log transfer curve of PEDOT control: '+dev)
accs = np.array(accs).reshape(-1,len(freq)).T
ax1.plot(freq,accs,'^-')
fig1.savefig('accs_seqlen.svg')
plt.show()