import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import numpy as np
import seaborn as sns

sns.set_theme()

keys=np.array(['pred','low','up','m_name'])
df_7=pd.DataFrame(columns=keys)
df_30=pd.DataFrame(columns=keys)
model_names_30=sorted(['AR(30)','DGP_30day_50ip','DGP_30day_100ip','DGP_30day_200ip'])
model_names_7=sorted(['AR(7)','DGP_7day_50ip','DGP_7day_100ip','DGP_7day_200ip'])
model_names_kmeans=sorted(['DGP_7day_200ip_R','DGP_7day_50ip_R','DGP_7day_100ip_R'])
f_names_30=sorted([f for f in os.listdir('predictions2/predictions_30') if not f.startswith('.')])
f_names_7=sorted([f for f in os.listdir('predictions2/predictions_7') if not f.startswith('.')])
f_names_km=sorted([f for f in os.listdir('predictions2/predictions_kmeans') if not f.startswith('.')])

test_y_30 = np.load('y_all_30_test_new.npy').reshape(-1)
test_y_7 = np.load('y_all_7_test_new.npy').reshape(-1)
pred_30=np.zeros((4,335*30))
pred_7=np.zeros((4,(365-7)*7))
pred_km=np.zeros((6,(365-7)*7))
pred_30_low=np.zeros((3,335*30))
pred_7_low=np.zeros((3,(365-7)*7))
pred_30_up=np.zeros((3,335*30))
pred_7_up=np.zeros((3,(365-7)*7))


plot=True

for i,f_name in enumerate(f_names_30):
    a=np.load(os.path.join('predictions2/predictions_30',f_name))
    if f_name[0]=='A':
        pred_30[i]=a.reshape(-1)
    else:
        a=np.squeeze(a)
        if a.shape[0]>4:
            a=np.transpose(a)
        pred_30[i]=a[0]


for i,f_name in enumerate(f_names_7):
    a=np.load(os.path.join('predictions2/predictions_7',f_name))
    if f_name[0]=='A':
        pred_7[i]=a.reshape(-1)
    else:
        a=np.squeeze(a)
        if a.shape[0]>4:
            a=np.transpose(a)
        pred_7[i]=a[0]

for i,f_name in enumerate(f_names_km):
    a=np.load(os.path.join('predictions2/predictions_kmeans',f_name))
    a = np.squeeze(a)
    if a.shape[0] > 4:
        a = np.transpose(a)
    pred_km[i]=a[0]


for i,f_name in enumerate(np.array(f_names_30)[1:]):
        a=np.load(os.path.join('predictions2/predictions_30',f_name))
        a=np.squeeze(a)
        if a.shape[0]>4:
            a=np.transpose(a)
        pred_30_up[i] = a[2]
        pred_30_low[i] = a[1]

for i,f_name in enumerate(np.array(f_names_7)[1:]):
        a=np.load(os.path.join('predictions2/predictions_7',f_name))
        a=np.squeeze(a)
        if a.shape[0]>4:
            a=np.transpose(a)
        pred_7_up[i] = a[2]
        pred_7_low[i] = a[1]

rmse_total_30=np.sqrt(np.mean((pred_30-test_y_30)**2,axis=1))
rmse_total_7=np.sqrt(np.mean((pred_7-test_y_7)**2,axis=1))
rmse_total_km=np.sqrt(np.mean((pred_km-test_y_7)**2,axis=1))

error_30=np.sqrt(np.mean(((pred_30-test_y_30)**2).reshape(4,-1,30),axis=1))
error_7=np.sqrt(np.mean(((pred_7-test_y_7)**2).reshape(4,-1,7),axis=1))

lty=['--bo','--*','--+','--x','-']
color=sns.color_palette()
dpi=500
"""First double plot"""
fig, axs = plt.subplots(2,figsize=(10, 3))
march=np.array([list(range((30+29)*7+i*7*7,(30+29)*7+i*7*7+7)) for i in range(5)]).reshape(-1)[:30]
axs[0].plot(range(1,31),pred_7[1,march],lty[1], label=model_names_7[1],color=color[1])
axs[0].fill_between(range(1,31), pred_7_low[0,march], pred_7_up[0,march], color=color[1], alpha=.1)
axs[0].plot(range(1,31),test_y_7[march],lty[1], label='Observed',color=color[5])
axs[0].legend(loc='lower right',prop={'size': 8})
axs[0].set_xticks(range(1,31))
axs[0].set_xticklabels([])
axs[0].set_ylabel('Energy (kWh)')

march=(30+29)*30
axs[1].plot(range(1,31),pred_30[1,march:march+30],lty[1], label=model_names_30[1],color=color[1])
axs[1].fill_between(range(1,31), pred_30_low[0,march:march+30], pred_30_up[0,march:march+30], color=color[1], alpha=.1)
axs[1].plot(range(1,31),test_y_30[(30+29)*30:(30+29)*30+30],lty[2], label='Observed',color=color[5])
axs[1].legend(loc='lower right',prop={'size': 8})
axs[1].set_xticks(range(1,31))
axs[1].set_ylabel('Energy (kWh)')
axs[1].set_xlabel('days')
if plot==True:
    fig.savefig('plots/rsme_double_plot_1',dpi=dpi)
    plt.show()


"""7 day plots"""

X_axis = np.arange(1)
fig_size_1=(4,3)
plt.figure(figsize=fig_size_1,dpi=dpi)
for i, j in zip(range(4),range(-3,1)):
    plt.bar(X_axis + j * 0.1, rmse_total_7[i], width=0.1, label=model_names_7[i], color=color[i])
    plt.text(X_axis + j * 0.1 - 0.03, rmse_total_7[i] + 10, int(rmse_total_7[i]), color='black', fontweight='bold',fontsize=10)

for i, j in zip(range(3),range(1,5)):
    plt.bar(X_axis + j * 0.1, rmse_total_km[i], width=0.1, label=model_names_kmeans[i], color=color[4+i])
    plt.text(X_axis + j * 0.1 - 0.03, rmse_total_km[i] + 10, int(rmse_total_km[i]), color='black', fontweight='bold',fontsize=10)

plt.xlim((-0.5,0.5))
plt.ylim((0,450))
plt.xticks([])
plt.ylabel('RMSE')
plt.legend(prop={'size': 6, 'weight':'bold'},loc="upper right")
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
if plot==True:
    plt.savefig('plots/rsme_bar_total_7',dpi=dpi)
    plt.show()

plt.figure(figsize=fig_size_1,dpi=dpi)
for i in range(4):
    plt.plot(range(1,8),error_7[i],lty[i], label=model_names_7[i],color=color[i])

plt.legend(loc='lower right',prop={'size': 6,'weight':'bold'})
plt.xticks(range(1,8))
plt.xlabel('forecast horizon in days')
plt.ylabel('RMSE')
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
if plot==True:
    plt.savefig('plots/rsme_line_total_7',dpi=dpi)
    plt.show()

plt.figure(figsize=fig_size_1,dpi=dpi)
for i in range(3):
    a=(test_y_7>pred_7_low[i])&(test_y_7<pred_7_up[i])
    CI=np.mean(a.reshape(-1,7),axis=0)
    plt.plot(range(1,8),100*CI,lty[i+1], label=model_names_7[i+1],color=color[i+1])
plt.plot(range(1,8),[95]*7,lty[4], label='95% CI',color=color[5])
plt.legend(loc='upper right',prop={'size': 6,'weight':'bold'})
plt.xticks(range(1,8))
plt.xlabel('forecast horizon in days')
plt.ylabel('%')
plt.ylim((80,100))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
if plot==True:
    plt.savefig('plots/CI_7',dpi=dpi)
    plt.show()





"""30 day plots"""
fig_size_1=(4,3)
plt.figure(figsize=fig_size_1,dpi=dpi)
color=sns.color_palette()
for i, j in zip(range(4),range(-2,2)):
    plt.bar(X_axis + j * 0.1, rmse_total_30[i], width=0.1, label=model_names_30[i], color=color[i])
    plt.text(X_axis + j * 0.1 - 0.03, rmse_total_30[i] + 10, int(rmse_total_30[i]), color='black', fontweight='bold',fontsize=10)

plt.xlim((-0.5,0.5))
plt.ylim((0,400))
plt.xticks([])
plt.ylabel('RMSE')
plt.legend(loc='upper right',prop={'size': 6,'weight':'bold'})
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
if plot==True:
    plt.savefig('plots/rsme_bar_total_30',dpi=dpi)
    plt.show()



"""Second double plot"""

fig, axs = plt.subplots(2,figsize=(10, 3))
for i in range(4):
    axs[0].plot(range(1,31),error_30[i],lty[i], label=model_names_30[i],color=color[i])
axs[0].legend(loc='center right',prop={'size': 8})
axs[0].set_xticks(range(1,37))
axs[0].set_xticklabels([])
axs[0].set_ylabel('RMSE')
fig.tight_layout()
for i in range(3):
    a=(pred_30_low[i]<test_y_30)&(pred_30_up[i]>test_y_30)
    CI=np.mean(a.reshape(-1,30),axis=0)
    axs[1].plot(range(1,31),CI*100,lty[i+1], label=model_names_30[i+1],color=color[i+1])
axs[1].plot(range(1,31),[95]*30,lty[4], label='95% CI',color=color[5])
axs[1].legend(loc='center right',prop={'size': 8})
axs[1].set_ylim((60,100))
axs[1].set_ylabel('%')
axs[1].set_xticks(range(1,37))
xticks = axs[1].xaxis.get_major_ticks()
for i in range(7):
    xticks[-i].label1.set_visible(False)

if plot==True:
    plt.savefig('plots/double_plot_second',dpi=dpi)
    plt.show()






















































