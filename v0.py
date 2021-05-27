import numpy as n, matplotlib.pyplot as m
from seaborn import heatmap
m.style.use('seaborn-darkgrid')

N = 1000; pr = 0.01; stim = 200; dt = 0.1; tsteps = int(stim/dt)
refr = 1.; tau = 10.; ref_tstep = int(refr/dt); thresh = 10.; reset = 0.

Ne = int(4*N/5)
Ni = N - Ne
C = n.random.choice([1, 0], size = (N, N), p = [pr, 1 - pr])


# # # # # # # #

def update(v, I, t, r):
    #ids fired
    ids = n.where(v[:, t] >= thresh)[0]
    #reset them and make them refractory
    # this time make it refractory from t itself
    r[ids, t: t + ref_tstep] = True
    v[ids, t+1] = reset
    
    TU = n.where(r[:, t] == False)[0]
    eq = ( -v[TU, t] + I[TU] )*dt/tau
    final = v[TU, t] + eq
    
    spikeid.extend(ids)
    spiketime.extend([t]*len(ids))
    
    v[TU, t+1] = final

    ids_e = ids[ids < Ne]
    ids_i = ids[ids >= Ne]
    
    target_e = C[ids_e].nonzero()[1]
    target_i = C[ids_i].nonzero()[1]
    # todo!! neurons that are refractory shouldn't get updated
    # which of target_e are not refractory?
    # edit, done
    actual_e = target_e[n.where(r[target_e, t+1] == False)[0]]
    actual_i = target_i[n.where(r[target_i, t+1] == False)[0]]
    
    # ufuncs ftw
    n.add.at(v[:, t+1], actual_e, v_add)
    n.subtract.at(v[:, t+1], actual_i, v_sub)
    

# rastor firingrate potentials example 
# TODO: annotate this properly somewhere
def rfpe(t, i, v, idx):
    '''plots rastor w firing rate, potentials and example'''
    
    vex = v[:Ne]
    vin = v[Ne:]
    xdata = n.arange(tsteps)/10
    # i wanna take average of column
    avex = n.mean(vex, axis = 0)
    avin = n.mean(vin, axis = 0)
    avtot = n.mean(v, axis = 0)
    
    ydata = n.concatenate((avex, avin, avtot))
    
    f, ax = m.subplots(4, 1, sharex=True, figsize=(12, 6))
    f.suptitle(f'v_add : {v_add}mV , v_sub : {v_sub}mV, refr : {refr}ms, tau : {tau}ms, N : {N}')
    m.xlabel('time(ms)')
    ax[0].scatter(t/10, i, s = 0.8, color = 'grey')
    ax[0].set_ylabel('# neuron')
    
    unq, ct = n.unique(i, return_counts=True)
    num_spiked = len(unq)
    rate = 5*n.mean(ct)
    frate = n.bincount(spiketime, minlength = tsteps)/0.1
    ax[1].plot(xdata, frate)
    ax[1].set_ylabel('#spikes/dt')
    ax[1].set_yscale('log')
    ax[1].set_title(f'mean f_rate = {n.round(rate, 2)}Hz, total unique neurons fired : {num_spiked}')
    
    # average of averages
    mean_ex = n.round(n.mean(avex),2)
    mean_in = n.round(n.mean(avin), 2)
    mean_total = n.round(n.mean(avtot),2)
    ax[2].plot(xdata, avex, label = 'exc')
    ax[2].plot(xdata, avin, label = 'inh')
    ax[2].plot(xdata, avtot, label = 'total')
    ax[2].set_ylabel('potential (mV)')
    ax[2].set_title(f'<ex.pot>  :{mean_ex}mV, <in.pot> : {mean_in}mV, <tot.pot> : {mean_total}mV')
    ax[2].legend()
    
    ax[3].plot(xdata, v[idx])
    ax[3].set_ylabel(f'pot (mV) for # {idx}')
    
    f.tight_layout()
    

# compute four properties for heatmaps    
def numeric(t, i, v):
    
    ##1.firingrate
    unq, ct = n.unique(i, return_counts=True)
    num_spiked = len(unq)
    # 5 is 200 specific 
    rate = 5*n.mean(ct)
    
    ##2. ISI CV
    ISI = []
    for neuron in range(N):  
        ISI.extend(n.diff(t[n.where(i == neuron)[0]])/10)
    cv = n.std(ISI)/n.mean(ISI)
    
    ##3.mempot
    mean_total = n.mean(v)
    
    return [num_spiked, rate, cv, mean_total]
    
data = []
v_ar = n.linspace(0, 2, 11)
v_sr = n.linspace(0, 2, 11)

for v_add in v_ar:
  print('yo')
  for v_sub in v_sr:

    vals = []
    for _ in range(3):
      state = n.zeros((N, tsteps))
      ref_state = n.full((N, tsteps), fill_value=False)
      curr = n.zeros((N,))

      tofire = n.random.choice(n.arange(N), size = int(0.2*N), replace = False)
      curr[tofire] = 20
      spikeid = []
      spiketime = []

      for t in range(tsteps-1):
          update(state, curr, t, ref_state)

      spikeid = n.asarray(spikeid)
      spiketime = n.asarray(spiketime)

      vals.append(numeric(spiketime, spikeid, state))

    vals = n.asarray(vals)
    data.append(n.mean(vals, axis = 0))
  
data = n.asarray(data)

## heatmaps
f, ((ax1, ax2), (ax3, ax4)) = m.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True, figsize=(12, 8), dpi = 120)
m.xlabel('v_sub')
m.ylabel('v_add')
numfired = heatmap(data = data[:, 0].reshape(11,11), vmin = 200, vmax = 1000, xticklabels=[n.round(i, 2) for i in v_sr], yticklabels=[n.round(i, 2) for i in v_ar], ax = ax1, square=True, linewidths=0.1)
frate    = heatmap(data = data[:, 1].reshape(11,11), xticklabels=[n.round(i, 2) for i in v_sr], yticklabels=[n.round(i, 2) for i in v_ar], ax = ax2, square=True, linewidths=0.1)
isicv    = heatmap(data = data[:, 2].reshape(11,11), center = 1, vmin = 0, vmax = 2, xticklabels=[n.round(i, 2) for i in v_sr], yticklabels=[n.round(i, 2) for i in v_ar], ax = ax3, square=True, linewidths=0.1)
mempot   = heatmap(data = data[:, 3].reshape(11,11), vmin=0, vmax = 5, xticklabels=[n.round(i, 2) for i in v_sr], yticklabels=[n.round(i, 2) for i in v_ar], ax = ax4, square=True, linewidths=0.1)

ax1.set_ylabel('v_add')
ax2.set_xlabel('v_sub')
ax3.set_xlabel('v_sub')
ax3.set_ylabel('v_add')
ax4.set_xlabel('v_sub')

ax1.set_title('# unique neurons fired')
ax2.set_title('avg. firing rate (Hz)')
ax3.set_title('CV of interspike intervals')
ax4.set_title('average membrane potential (mV)')

f.tight_layout()
