import os
import torch
import tqdm
import gpytorch
import numpy as np
import pandas as pd
from torch.nn import Linear
from scipy.cluster.vq import kmeans2
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
import random

random.seed(0)
torch.manual_seed(0)

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, inducing_points=None, linear_mean=True):
        if inducing_points is None:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape, inducing_points, num_inducing_pts):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            num_inducing=num_inducing_pts,
            inducing_points=inducing_points,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            num_inducing=num_inducing_pts,
            inducing_points=None,
            linear_mean=False
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a multitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)

def eval_ci(lb,ub,yt):
    score = []
    for i,y in enumerate(yt):
        if (y > lb[i]) & (y < ub[i]):
            score.append(1)
        else:
            score.append(0)
    return np.mean(score)

def unfold(lists,data):
    l = []
    for w in lists:
        for d in list(w):
            for i in list(d):
                l.append(float(i*data.std().numpy()+data.mean().numpy()))
    return l

def unfold_z(lists):
    l = []
    for w in lists:
        for d in list(w):
            for i in list(d):
                l.append(float(i))
    return l

def visualize_predictions(preds,lb,ub,yt,df):
    # Plot results for first three months in 2019
    num_months = 3
    days = [31,28,31]
    months = ["January", "February", "March"]
    for i in range(num_months):
        plt.figure(i+1)
        plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[(-365+sum(days[:i])):(-334+sum(days[:i]))], yt[(sum(days[:i])):(31+sum(days[:i]))], 'g',linewidth=2)
        plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[(-365+sum(days[:i])):(-334+sum(days[:i]))], preds[(sum(days[:i])):(31+sum(days[:i]))], 'b',linewidth=2)
        plt.fill_between(pd.to_datetime(df['Start Date']).to_numpy()[(-365+sum(days[:i])):(-334+sum(days[:i]))], lb[(sum(days[:i])):(31+sum(days[:i]))],
                         ub[(sum(days[:i])):(31+sum(days[:i]))], alpha=0.5)
        plt.legend(['Observed data','Mean','Confidence'])
        plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[(-365+sum(days[:i])):(-334+sum(days[:i]))], yt[(sum(days[:i])):(31+sum(days[:i]))], 'go')
        plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[(-365+sum(days[:i])):(-334+sum(days[:i]))], preds[(sum(days[:i])):(31+sum(days[:i]))], 'bo')
        plt.title("Predictions on total energy consumption in {}".format(months[i]))
        plt.savefig(os.path.join('New_plots',model_name)+'_'+months[i])
        plt.show()


    # Plot results for the first week in Feb
    plt.figure(4)
    plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[-334:-327], yt[31:38], 'g',linewidth=2)
    plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[-334:-327], preds[31:38], 'b',linewidth=2)
    plt.fill_between(pd.to_datetime(df['Start Date']).to_numpy()[-334:-327], lb[31:38],
                     ub[31:38], alpha=0.5)
    plt.legend(['Observed data','Mean','Confidence'])
    plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[-334:-327], yt[31:38], 'go')
    plt.plot(pd.to_datetime(df['Start Date']).to_numpy()[-334:-327], preds[31:38], 'bo')
    plt.title("Predictions on total energy consumption start February")
    plt.savefig(os.path.join('New_plots',model_name) + '_' + 'week')
    plt.show()

df = pd.read_csv(r'../df_sum_ports.csv')

num_inducing_pts = 200          # Number of inducing points in each hidden layer
num_hidden_dgp_dims = 3         # Number of GPs (i.e., the width) in the hidden layer.
batch_size = 10                 # Size of minibatch
use_induce=False
if use_induce:
    if num_inducing_pts==200:
        file_name = 'DGP_7_200_state.pth'
        model_name = 'DGP_200_ip_km'
    elif num_inducing_pts==100:
        file_name = 'DGP_7_100_state.pth'
        model_name = 'DGP_100_ip_km'
    else:
        file_name = 'DGP_7_50_state.pth'
        model_name = 'DGP_50_ip_km'
else:
    if num_inducing_pts==200:
        file_name = 'DGP_7_200_R_state.pth'
        model_name = 'DGP_200_ip'
    elif num_inducing_pts==100:
        file_name = 'DGP_7_100_R_state.pth'
        model_name = 'DGP_100_ip'
    else:
        file_name = 'DGP_7_50_R_state.pth'
        model_name = 'DGP_50_ip'

def new_unfold(lists):
    new_lists = []
    for m in lists:
        for l in m:
            new_lists.append(l)
    matrix = np.array([c for c in new_lists])
    return matrix

path = r'C:\Users\johan\iCloudDrive\DTU\MMC\Semester 1\Deep learning\Project - Deep GP\Data\New data'
train_path = r'C:\Users\johan\iCloudDrive\DTU\MMC\Semester 1\Deep learning\Project - Deep GP\Data'
train_data_X = np.load(os.path.join(train_path, 'X_all_7_train.npy'))
train_data_y = np.load(os.path.join(train_path, 'y_all_7_train.npy'))
test_data_X = np.load(os.path.join(path, 'X_all_7_test_new.npy'))
test_data_y = np.load(os.path.join(path, 'y_all_7_test_new.npy'))
train_data_y_old = np.load(os.path.join(train_path, 'y_all_7_train.npy'))
test_data_y_old = np.load(os.path.join(train_path, 'y_all_7_test.npy'))

train_stats = train_stats = train_data_y_old[:,0]
test_stats = new_unfold(test_data_y_old)

X_train = torch.Tensor(train_data_X)
y_train = torch.Tensor(train_data_y)
X_test = torch.Tensor(test_data_X)
y_test = torch.Tensor(test_data_y)
y_test_old = torch.Tensor(test_data_y_old)
data_X = torch.concat([X_train,X_test])
data_Y = torch.concat([y_train,y_test_old])
train_n = train_data_y.shape[0]

# Map features to [-1,1]
train_x = X_train.contiguous()
X_test = X_test-data_X.min(0)[0]
X_test = 2.0 * (X_test / data_X.max(0)[0]) - 1.0
y_test -= data_Y.mean()
y_test /= data_Y.std()
test_x = X_test.contiguous()
test_y = y_test.contiguous()

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_tasks = test_y.size(-1)

# Use k-means to initialize inducing points (only helpful for the first layer)
inducing_points = (train_x[torch.randperm(min(1000 * 100, train_n))[0:num_inducing_pts], :].squeeze(-1))
inducing_points = inducing_points.clone().data.cpu().numpy()
inducing_points = torch.tensor(kmeans2(train_x.squeeze(-1).data.cpu().numpy(),
                               inducing_points,minit='matrix')[0])
inducing_points = inducing_points.unsqueeze(0).expand((num_hidden_dgp_dims,) + inducing_points.shape)
inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)


state_dict = torch.load(os.path.join('Models',file_name))
model = MultitaskDeepGP(train_x.shape,inducing_points=inducing_points,num_inducing_pts=num_inducing_pts)
model.load_state_dict(state_dict)
model.eval()
means = []
lowers = []
uppers = []
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for x_batch, y_batch in test_loader:
        mean, var = model.predict(x_batch)
        means.append(list(mean.detach().numpy()))
        lower = mean - 2 * var.sqrt()
        upper = mean + 2 * var.sqrt()
        lowers.append(lower.numpy())
        uppers.append(upper.numpy())
print(file_name)
print(model_name)

def new_unfold(lists):
    new_lists = []
    for m in lists:
        for l in m:
            new_lists.append(l)
    matrix = np.array([c for c in new_lists])
    return matrix

preds = new_unfold(means)
lb = new_unfold(lowers)
ub = new_unfold(uppers)
preds2 = unfold(lists=means,data=data_Y)
lb2 = unfold(lists=lowers,data=data_Y)
ub2 = unfold(lists=uppers,data=data_Y)
yt2 = []
for y in test_data_y:
    for e in y:
        yt2.append(e)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(yt2,preds2,squared=False)
print("RMSE: ", rmse)
matrix = np.array([preds2,lb2,ub2])
np.save(model_name,matrix)

visualize_predictions(preds=preds2[:357],lb=lb2[:357],ub=ub2[:357],yt=yt2[:357],df=df)

#np.save(model_name+'_R_lb',lb)
#np.save(model_name+'_R_ub',ub)