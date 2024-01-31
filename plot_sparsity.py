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

# files = [
#             '230219_fmnist_SeqNet_poisson500_lr0.0001_tlen10_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen15_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen20_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen30_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen40_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen50_hch1_och1_config1','230219_fmnist_SeqNet_poisson500_lr0.0001_tlen60_hch1_och1_config1',
#             '230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen10_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen15_hch1_och4_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen20_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen30_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen40_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen50_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen60_hch2_och2_config1',
#             '230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen10_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen15_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen20_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen30_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen40_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen50_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen60_hch1_och4_config1',
#          ]

files = [
            '230420_fmnist_SeqNet_poisson500_lr0.001_tlen5_hch1_och1_config1','230420_fmnist_SeqNet_poisson500_lr0.001_tlen6_hch1_och1_config1',
            '230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen10_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen15_hch1_och4_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen20_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen30_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen40_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen50_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen60_hch2_och2_config1',
            '230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen10_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen15_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen20_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen30_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen40_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen50_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen60_hch1_och4_config1',
         ]

# files = [
#             '230420_fmnist_SeqNet_poisson500_lr0.001_tlen5_hch1_och1_config1','230420_fmnist_SeqNet_poisson500_lr0.001_tlen6_hch1_och1_config1',
#             '230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen10_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen15_hch1_och4_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen20_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen30_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen40_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen50_hch2_och2_config1','230219_fmnist_DendSeqNet2_poisson500_lr0.0001_tlen60_hch2_och2_config1',
#             '230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen10_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen15_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen20_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen30_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen40_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen50_hch1_och4_config1','230219_fmnist_DendSeqNet3_poisson500_lr0.0001_tlen60_hch1_och4_config1',
#          ]

# freq = np.array([1/500,1/400,1/300,1/200,1/100,1/50])
# freq = np.array([10,15,20,30,40,50,60])
freq = np.linspace(100,900,9)
# freq = np.linspace(5,20,16)
series = 2
accs = []
spks = []
fig1,ax1 = plt.subplots(1,1,figsize=(1.5,1.15))
fig2,ax2 = plt.subplots(1,1,figsize=(1.5,1.15))
colors = ['tab:blue','tab:green','tab:orange']
for i in range(len(freq)):
    for j in range(series):
        if j == 0:
            fname = f'230421_fmnist_SeqNet_poisson{int(freq[i])}_lr0.001_tlen20_hch1_och1_config1'
        elif j == 2:
            fname = f'230421_fmnist_DendSeqNet2_poisson{int(freq[i])}_lr0.001_tlen20_hch2_och2_config2'
        elif j == 1:
            fname = f'230421_fmnist_DendSeqNet2_poisson{int(freq[i])}_lr0.001_tlen20_hch4_och2_config2'

        data_acc = np.mean(np.load(dir + fname + '/accuracies.npy'),axis=1)
        data_spk = np.mean(np.load(dir + fname + '/spk_act.npy'),axis=1)
        acc_hi = np.max(data_acc)
        spk_hi = np.max(data_spk)
        accs.append(acc_hi)
        spks.append(spk_hi)

accs = np.array(accs).reshape(len(freq),-1)
spks = np.array(spks).reshape(len(freq),-1)
freq = freq/10
print(accs > 20)
for i in range(series):
    ax1.plot(freq[accs[:,i] > 20],accs[accs[:,i] > 20,i],'.',color=colors[i])
    ax1.plot(freq[accs[:,i] < 20],accs[accs[:,i] < 20,i],'.',color=colors[i],alpha=0.5)
    ax1.plot(freq,accs[:,i],'--',color=colors[i],linewidth=0.5,alpha=0.5)
    ax1.plot(freq[accs[:,i] > 20],accs[accs[:,i] > 20,i],'-',color=colors[i])

fig1.savefig('./outputs/accs_freq_full.svg')
ax1.set_ylim([80,87])
for i in range(series):
    ax2.plot(freq[accs[:,i] > 20],spks[accs[:,i] > 20,i],'.',color=colors[i])
    ax2.plot(freq[accs[:,i] < 20],spks[accs[:,i] < 20,i],'.',color=colors[i],alpha=0.5)
    ax2.plot(freq,spks[:,i],'--',color=colors[i],linewidth=0.5,alpha=0.5)
    ax2.plot(freq[accs[:,i] > 20],spks[accs[:,i] > 20,i],'-',color=colors[i])
ax1.set_xticks([20,40,60,80])
ax2.set_xticks([20,40,60,80])
ax2.set_yticks([1000,3000,5000,7000])
print(spks[:,1]/spks[:,0])
fig1.savefig('./outputs/accs_freq.svg')
fig2.savefig('./outputs/spks_freq.svg')