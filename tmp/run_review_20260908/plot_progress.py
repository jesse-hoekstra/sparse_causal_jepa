import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path(__file__).resolve().parent
h=json.loads((p/'history_fast.json').read_text())
e=[r for r in h if 'eval/mcc' in r]
x=np.array([r['_step']/1000 for r in e])
def y(k): return np.array([r['eval/'+k] for r in e])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':10,'figure.facecolor':'white','savefig.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(11.6,8.2))
blue='#2864ad'; orange='#d47720'; green='#24846e'; gray='#707985'
ax=axs[0,0]
ax.plot(x,y('position_probe_r2'),color=blue,lw=2,label='Position')
ax.plot(x,y('velocity_probe_r2'),color=orange,lw=2,label='Velocity')
ax.set(title='Physical-state decoding has plateaued',ylabel='Held-out linear-probe R²',ylim=(0,.17))
ax.legend(frameon=False,loc='upper right')
ax=axs[0,1]
for k,c,label in [('target_variance',gray,'Target variance'),('loss_teacher_forcing',blue,'One-step MSE'),('loss_rollout_t2_raw',orange,'Two-step endpoint MSE')]:
 ax.plot(x,y(k),color=c,lw=2,label=label)
ax.set(title='Latent scale and raw errors keep growing',ylabel='Raw latent units (log scale)',yscale='log')
ax.legend(frameon=False,loc='lower right',fontsize=9)
ax=axs[1,0]
ax.plot(x,y('loss_teacher_forcing')/y('target_variance'),color=blue,lw=2,label='One-step MSE / target variance')
ax.plot(x,y('loss_rollout_t2_raw')/y('target_variance'),color=orange,lw=2,label='Two-step MSE / target variance')
ax.plot(x,y('latent_rollout_k30_nrmse'),color=green,lw=1.8,label='30-step latent NRMSE (logged)')
ax.set(title='Short-horizon relative error improves; K30 stays flat',ylabel='Diagnostic ratio / NRMSE',ylim=(0,1.3))
ax.legend(frameon=False,fontsize=8.5,loc='center right',bbox_to_anchor=(1,.52))
ax=axs[1,1]
ax.plot(x,y('slot_centroid_rmse'),color=blue,lw=2)
ax.set(title='Slot localization is essentially unchanged',ylabel='Centroid RMSE (image coordinates)',ylim=(0,.3))
ax.text(.97,.78,f"Latest: {y('slot_centroid_rmse')[-1]:.4f}",transform=ax.transAxes,ha='right',color=blue)
for ax in axs.flat:
 ax.set_xlabel('Training updates (thousands)')
 ax.set_xlim(0,240)
 ax.grid(alpha=.17)
fig.suptitle('causal_jepa · dense visual stage · o376vjsf',fontsize=16,fontweight='bold',y=.985)
fig.text(.5,.935,'47 held-out evaluations through 235,000 / 300,000 updates',ha='center',fontsize=12,color='#555555')
fig.subplots_adjust(left=.075,right=.98,bottom=.105,top=.85,hspace=.39,wspace=.25)
fig.text(.5,.012,'W&B snapshot: 8 Sep 2026, 09:04 UTC · 64 held-out episodes per evaluation · Error/variance ratios are post-hoc diagnostics, not the training loss.',ha='center',fontsize=8,color='#555555')
fig.savefig(p/'progress.png',dpi=180)
fig.savefig(p/'progress.pdf')
print(p/'progress.png')
