# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:52:46 2020
# TODO
# - Put all user-input parameters from main() to the next cell (if __name__ == '__main__':)
# - Implement dropout
# - Add class for dealing with importing datasets (if a new dataset if to be imported
#   one could define a child of that class and re-define the method 'import_dataset')
# - create a more universal progress bar that could also work for importing datasets
# - organize folders 'results' and 'saved_model_params' similar to 'logs' and 'plots'
# - iprove saveing/loading so that there would be no need specifying architecture: the model would be recreated from a checkpoint
# - add the number of layers as a hyperparameter for optuna to tune 
@author: ZumbiAzul 
"""
#%% Import modules etc.
import os
import sys
import signal
from datetime import date, datetime
import timeit
import re
import gc
import psutil
import multiprocessing
from inspect import currentframe, getframeinfo
# from icecream import ic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import optuna # version 2.3.0

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib as mtpltlb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb # Python debugger

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from IPython.display import display, clear_output
import time
import json

from collections import OrderedDict
from collections import namedtuple

from itertools import product

import math

# import seaborn for pretty plots
import seaborn as sns
sns.set_style('ticks')
sns.set_context('notebook')

# Create a modified Jet colormap with white as zero
jet = cm.get_cmap('jet', 20)
r = []
g = []
b = []
npoints = 10
for ii in range(0, npoints):
    val = ii / (npoints - 1.0)
    color = jet(val)
    r.append(color[0])
    g.append(color[1])
    b.append(color[2])
r[0] = 1.0
g[0] = 1.0
b[0] = 1.0

ll = [(r[ii], g[ii], b[ii]) for ii in range(0, len(r))]
modified_jet = mtpltlb.colors.LinearSegmentedColormap.from_list('2D_colormap', ll)

torch.set_printoptions(linewidth=150)

gc.collect()
torch.cuda.empty_cache()

for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass
 
# print current line
print(getframeinfo(currentframe()).lineno)
    
#%% new cell (show_gpu_memory(), dump_cpu_memory_info(), progressBar())
def show_gpu_memory():
    deviceid = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%deviceid
    total, used = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[deviceid].split(',')
    total = int(total)
    used = int(used)
    print('Device: ',deviceid, '. Total GPU memory: ', total, 'Used memory: ', used)

def dump_cpu_memory_info():
    print("Number of logical CPUs: ",psutil.cpu_count())
    print("Number of usable CPUs: ",len(psutil.Process().cpu_affinity()))
    print()
    print('System-wide CPU utilization in percents')
    print(psutil.cpu_percent(interval=1,percpu=True))
    print()
    print(psutil.cpu_times())
    print(psutil.cpu_stats())
    print()
    print("CPU frequency")
    print(psutil.cpu_freq(percpu=True))
    print()
    print("Average system load (last 1,5,15 min):",psutil.getloadavg())
    print()
    print('Virtual memory:',psutil.virtual_memory())
    print('Swap memory:',psutil.swap_memory())
    print()

    p = psutil.Process()
    with p.oneshot():
        print("p.name:",p.name())
        print("p.cpu_times:",p.cpu_times())
        print("p.cpu_percent:",p.cpu_percent())
        print("p.create_time:",p.create_time())
        print("p.ppid:",p.ppid())
        print("p.status:",p.status())
        print()
        print("p.memory_info:",p.memory_info())
        print()
        print("p.memory_full_info:",p.memory_full_info())
        print()
        print("p.num_threads:",p.num_threads())
        print("p.cpu_affinity:",p.cpu_affinity())

show_gpu_memory()
print()
dump_cpu_memory_info()

def progressBar(nnn_, batch_size_, n_train_, barLength = 20):
    percent = (int((nnn_+1)/(n_train_/batch_size_)*100))
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('\rBatches processed (%d): [%s%s] %d %%' % (nnn_+1,arrow,spaces,percent), end='\r'),
    
def getDateTime():
    date_str = str(date.today())
    now = datetime.now()
    time_str = now.strftime("%H-%M-%S")
    return date_str + "_" + time_str


    
def CtrlC_event_handler(m_):
    L = ['\n\nTerminated by user (Ctrl+C).']
    def signal_handler(sig, frame):
        print('\n\nCtrl+C was pressed, wrapping up...')
        m_.log_file.writelines(str(line) + '\n' for line in L)
        m_.log_file.close()
        print('Done.')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

#%% new cell (RunBuilder())
class RunBuilder():
    @staticmethod
    def get_runs(params___):

        # This line creates a new tuple subclass called Run that has named fields. This Run class is used to encapsulate the data for each of our runs. The field names of this class are set by the list of names passed to the constructor. First, we are passing the class name. Then, we are passing the field names, and in our case, we are passing the list of keys from our dictionary.
        Run = namedtuple('Run', params___.keys())

        # Now that we have a class for our runs, we are ready to create some.
        # First we create a list called runs. Then, we use the product() function from itertools to create the Cartesian product using the values for each parameter inside our dictionary. This gives us a set of ordered pairs that define our runs. We iterate over these adding a run to the runs list for each one.
        # For each value in the Cartesian product we have an ordered tuples. The Cartesian product gives us every ordered pair so we have all possible order pairs of learning rates and batch sizes. When we pass the tuple to the Run constructor, we use the * operator to tell the constructor to accept the tuple values as arguments opposed to the tuple itself.
        runs = []
        for v in product(*params___.values()):
            runs.append(Run(*v))

        return runs
    
show_gpu_memory()

#%% new cell (RunManager())
class RunManager():
    def __init__(self,log_folder,log_subfolder):
        self.epoch_count = 0 # for tracking the number of epochs
        self.epoch_cost = 0 # for tracking the running loss for an epoch
        self.epoch_cost_dev = 0 # for tracking the running loss for the dev-set for an epoch
        self.epoch_accuracy = 0
        self.epoch_num_correct = 0 # for tracking the number of correct predictions for an epoch
        self.epoch_start_time = None # for tracking the start time of the epoch
        self.epoch_costs = np.array([]).astype(np.float)
        self.epoch_costs_dev = np.array([]).astype(np.float)
        self.epoch_accuracies = np.array([]).astype(np.float)

        self.run_params = None # It's value will be one of the runs returned by the RunBuilder class.
        self.run_count = 0 # for tracking the run number
        self.run_data = [] # list we'll use to keep track of the parameter values and the results of each epoch for each run
        self.run_data_current = [] # data for the current run
        self.run_start_time = None # will be used to calculate the run duration
        self.run_loss_values = np.array([]).astype(np.float) # list containing loss-values of all epochs
        self.run_loss_dev_values = np.array([]).astype(np.float) # list containing loss-values for dev-set
        self.run_accuracies = np.array([]).astype(np.float)
        
        # next we will save the network and the data loader that are being used for the run, 
        # as well as a SummaryWriter that we can use to save data for TensorBoard.
        self.network = None
        self.loader = None
        self.loader_dev = None
        self.working_directory = os.getcwd()
        self.results_root = os.path.join(self.working_directory, "results")
        self.params_root = os.path.join(self.working_directory, "saved_model_params")
        self.plots_root = os.path.join(self.working_directory, "plots")
        self.logs_root = os.path.join(self.working_directory, "logs")
            
        # preparing folders for saving data
        self.subfolder = getDateTime()
        self.current_log_folder = os.path.join(log_folder,log_subfolder)
        self.path = os.path.join(self.results_root, self.subfolder)
        self.path_params = os.path.join(self.params_root, self.subfolder)
        self.path_plots = os.path.join(self.plots_root, self.current_log_folder, self.subfolder)
        self.path_logs = os.path.join(self.logs_root, self.current_log_folder)
        self.best_cost_filepath = os.path.join(self.logs_root,log_folder,"best_cost_("+log_folder+").txt")
        self.logfile_static_path = os.path.join(self.path_logs,"0000_activations_parameter_ranges.txt")
        
        if not os.path.exists(self.best_cost_filepath):
            f = open(self.best_cost_filepath,'w').close()
        f = open(self.best_cost_filepath,'r')
        best_cost_ = f.readlines()
        f.close()
        self.current_trial = ''
        if len(best_cost_) != 0:
            self.best_cost_so_far = float(best_cost_[0])
        else:
            self.best_cost_so_far = float('NaN')
        
        if not os.path.exists(self.path_logs):
            os.makedirs(self.path_logs)
        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)
            
        if not os.path.exists(self.best_cost_filepath):
            open(self.best_cost_filepath, 'w').close() # create an emoty file            
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if not os.path.exists(self.path_params):
            os.mkdir(self.path_params)
        self.log_file = open(os.path.join(self.path_logs,(self.subfolder+".txt")),"a")
        
        L = ["\t["+getDateTime()+"]\n\n",
             "Results will be written into: \n"+self.path+"\n",
             "Network parameters will be saved into: \n"+self.path_params+"\n",
             "Plots will be saved into:: \n"+self.path_plots+"\n"]
        
        print('\nResults will be written into:',self.path)
        print('Network parameters will be saved into:',self.path_params)
        print('Plots will be saved into:',self.path_plots)        
        self.log_file.writelines(L)        # logging into the file
        
    def begin_run(self, run, network, loader, loader_dev):
        
        # First, we capture the start time for the run.
        self.run_start_time = time.time()
        
        # Then, we save the passed in run parameters and increment the run count by one.
        self.run_params = run
        self.run_count += 1

        # After this, we save our network and our data loader, and then, we initialize a SummaryWriter for TensorBoard.
        self.network = network
        self.loader = loader
        self.loader_dev = loader_dev

    def end_run(self,plot_=False):        
        # When we end a run, set the epoch count back to zero to be ready for the next run.
        self.epoch_count = 0
        self.run_data_current = []
        
        if plot_:
            fig = plt.figure(figsize=(30,6))
            ax_1 = fig.add_subplot(131)
            ax_2 = fig.add_subplot(132)
            ax_3 = fig.add_subplot(133)
            
            ax_1.plot(self.run_loss_values, color=u'#1f77b4')
            ax2 = ax_1.twinx()
            ax2.plot(self.run_loss_dev_values, color=u'#ff7f0e')
            ax_1.set_ylabel('L (train)',fontsize=30)
            ax2.set_ylabel('L (dev)',fontsize=30)
            ax_1.set_xlabel('Batches',fontsize=30)
            
            ax_2.plot(self.epoch_costs, color=u'#1f77b4')
            ax3 = ax_2.twinx()
            ax3.plot(self.epoch_costs_dev, color=u'#ff7f0e')
            ax_2.set_ylabel('J (train)',fontsize=30)
            ax3.set_ylabel('J (dev)',fontsize=30)
            ax_2.set_xlabel('Epoch',fontsize=30)
            
            ax_3.plot(self.epoch_accuracies, color=u'#ff7f0e')
            ax_3.set_ylabel('Accuracy, %',fontsize=30)
            ax_3.set_xlabel('Epoch',fontsize=30)
            
            ax_1.tick_params(axis='both', which='major', labelsize=30)
            ax_2.tick_params(axis='both', which='major', labelsize=30)
            ax2.tick_params(axis='both', which='major', labelsize=30)
            ax3.tick_params(axis='both', which='major', labelsize=30)
            ax_3.tick_params(axis='both', which='major', labelsize=30)
            
            plt.tight_layout()
            fig.savefig(os.path.join(self.path_plots,self.subfolder + "_r" + str(self.run_count) + "_e" + str(self.epoch_count) + "_plt.png"))
            plt.pause(0.00001)
            
        
    def begin_epoch(self):
        
        # For starting an epoch, we first save the start time.
        self.epoch_start_time = time.time()

        # Then, we increment the epoch_count by one and set the epoch_loss and epoch_number_correct to zero.
        self.epoch_count += 1
        self.epoch_cost = 0
        self.epoch_cost_dev = 0
        self.epoch_accuracy = 0
#         self.epoch_num_correct = 0 # this probably makes sense for classification problems
        
    # Now, let's look at where the bulk of the action occurs which is ending an epoch.
    def end_epoch(self,plot_=False):

        # We start by calculating the epoch duration and the run duration.
        # Since we are at the end of an epoch, the epoch duration is final, 
        # but the run duration here represents the running time of the current run. 
        # The value will keep running until the run ends. However, we'll still save it with each epoch.
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # Next, we compute the epoch_loss and accuracy, and we do it relative to the size of the training set. 
        # This gives us the average loss per sample.
        cost = self.epoch_cost / len(self.loader.dataset)
        cost_dev = self.epoch_cost_dev / len(self.loader_dev.dataset)
        self.epoch_costs = np.append(self.epoch_costs,cost)
        self.epoch_costs_dev = np.append(self.epoch_costs_dev,cost_dev)
        self.epoch_accuracies = np.append(self.epoch_accuracies,self.epoch_accuracy)
#         accuracy = self.epoch_num_correct / len(self.loader.dataset) # this probably makes sense for classification problems
            
        # Here, we are building a dictionary that contains the keys and values we care about for our run.
        results = OrderedDict()
        results["r"] = self.run_count # 'r' = 'run'
        results["e"] = self.epoch_count # 'e' = 'epoch'
        results['J'] = cost  # 'L' = 'cost'
        results['J_dev'] = cost_dev  # 'L_dev' = 'cost for dev_set'
#         results["acc."] = accuracy   # 'acc.' = 'accuracy' # probably not needed for regression task like this (FROG)
        results['e. dur.'] = epoch_duration  # 'e. dur.' = 'epoch duration'
        results['r. dur.'] = run_duration  # 'r. dur.' = 'run duration'
        # Then, we iterate over the keys and values inside our run parameters adding them to the results dictionary. 
        # This will allow us to see the parameters that are associated with the performance results.
        for k,v in self.run_params._asdict().items(): results[k] = v
        # Finally, we append the results to the run_data list.
        self.run_data.append(results)
        self.run_data_current.append(results)
        
        print('\nCost: ',self.epoch_costs[-1],'; Cost (dev): ',self.epoch_costs_dev[-1], '; Accuracy: ',self.epoch_accuracies[-1],'; Best accuracy so far: ',self.best_cost_so_far,'\n')
        L = ['\nCost: '+str(self.epoch_costs[-1]),'Cost (dev): '+str(self.epoch_costs_dev[-1]), 'Accuracy: '+str(self.epoch_accuracies[-1])]
        self.log_file.writelines(str(line) + '\n' for line in L)

        # Once the data is added to the list, we turn the data list into a pandas data frame so we can have formatted output.
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        # The two lines (clear_output and display) are specific to Jupyter notebook. 
        # We clear the current output and display plot and new data frame.
        #clear_output(wait=True)
        
        if plot_:
            fig = plt.figure(figsize=(30,6))
            ax_1 = fig.add_subplot(131)
            ax_2 = fig.add_subplot(132)
            ax_3 = fig.add_subplot(133)
            
            ax_1.plot(self.run_loss_values, color=u'#1f77b4')
            ax2 = ax_1.twinx()
            ax2.plot(self.run_loss_dev_values, color=u'#ff7f0e')
            ax_1.set_ylabel('L (train)',fontsize=30)
            ax2.set_ylabel('L (dev)',fontsize=30)
            ax_1.set_xlabel('Batches',fontsize=30)
            
            ax_2.plot(self.epoch_costs, color=u'#1f77b4')
            ax3 = ax_2.twinx()
            ax3.plot(self.epoch_costs_dev, color=u'#ff7f0e')
            ax_2.set_ylabel('J (train)',fontsize=30)
            ax3.set_ylabel('J (dev)',fontsize=30)
            ax_2.set_xlabel('Epoch',fontsize=30)
            
            ax_3.plot(self.epoch_accuracies, color=u'#ff7f0e')
            ax_3.set_ylabel('Accuracy, %',fontsize=30)
            ax_3.set_xlabel('Epoch',fontsize=30)
            
            ax_1.tick_params(axis='both', which='major', labelsize=30)
            ax_2.tick_params(axis='both', which='major', labelsize=30)
            ax2.tick_params(axis='both', which='major', labelsize=30)
            ax3.tick_params(axis='both', which='major', labelsize=30)
            ax_3.tick_params(axis='both', which='major', labelsize=30)
            
            plt.tight_layout()
            fig.savefig(os.path.join(self.path_plots,self.subfolder + "_r" + self.run_count + "_e" + self.epoch_count + "_plt.png"))
            plt.pause(0.00001)
        
        #display(df)
                
    # This is how the epoch_loss and epoch_num_correct values are tracked.
    # We'll have two methods just below for that.
    def track_loss(self, loss, loss_dev, accuracy):
        self.epoch_cost += loss.item()  # * self.loader.batch_size
        self.epoch_cost_dev += loss_dev.item()  # * self.loader.batch_size
        self.epoch_accuracy = accuracy
        self.run_loss_values = np.append(self.run_loss_values,loss.item())
        self.run_loss_dev_values = np.append(self.run_loss_dev_values,loss_dev.item())
        self.run_accuracies = np.append(self.run_accuracies,accuracy)
    
    # This method tells if the current epoch's result is better than of all
    # previous epochs from the current and previous trials and from
    # all study names within the current database
    def best_cost_changed_state(self, cst,direc):
        f = open(self.best_cost_filepath,'r')
        best_cost_ = f.readlines()
        f.close()
        state_ = False
        if len(best_cost_) != 0:
            if (direc == "minimize" and cst < self.best_cost_so_far) \
                    or \
                        (direc == "maximize" and cst > self.best_cost_so_far):
                state_ = True
                self.best_cost_so_far = float(best_cost_[0])
            else:
                state_ = False
        else:
            state_ = True
        return state_
    
    # This method writes the best cost amongst all epochs in all trials into a file
    # direc = optuna's optimization direction, cst = cost value
    def report_best_cost(self,cst,direc):
        state__ = self.best_cost_changed_state(cst,direc)
        L = [str(cst),
            "Optuna's study name: " + self.current_log_folder]
        if state__:
            last_line = "Trial # " + self.current_trial
            f = open(self.best_cost_filepath,'w')
            f.writelines([line + "\n" for line in [*L,last_line]])
            f.close()
            self.best_cost_so_far = cst                
        return self.best_cost_so_far
        
    # for classification problem
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self.get_num_correct(preds, labels)
        
    # To calculate the number of correct predictions we use the same function as before
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def plot_cost(self,date_,run):
        folder = os.path.join(self.results_root,date_)
        
        # since files are not sorted in accordance with 'run' then we have to search for the proper file given 'run'
        files  = [f for f in os.listdir(folder) if f.startswith('-')]
        run_idx = [pd.read_csv(folder+ff).keys().get_loc("r") for ff in files]
        runs = [pd.read_csv(folder+ff).iloc[0][i] for i,ff in zip(run_idx,files)]
        idx = runs.index(run)
        file = files[idx]
        
        df = pd.read_csv(folder + file)
        
        figure, ax1 = plt.subplots()
        ax1.plot(df["e"],df["J"], color=u'#1f77b4')
        plt.gca().set(xlabel='Epoch', ylabel='Loss (train)')
        ax2 = ax1.twinx()
        ax2.plot(df["e"],df["J_dev"], color=u'#ff7f0e')
        ax2.set_ylabel('Loss (dev)')
        
        return df["J"],df["J_dev"]
    
    def save_run_data(self, comment):
        pd.DataFrame.from_dict(
            self.run_data_current, orient='columns'
        ).to_csv(os.path.join(self.path,comment + '.csv'))
        
    def save_network(self, model, network_parameters, input_depth, conv_layer_activations, linear_layer_activations, comment):
        checkpoint = dict(network_parameters)  # or orig.copy()
        extra = {"input_depth": input_depth,
                 "conv_activation_functions": conv_layer_activations,
                 "fc_activation_functions": linear_layer_activations,
                 "state_dict": model.state_dict()}
        checkpoint.update(extra)
        # print()
        # print()
        # print(checkpoint["state_dict"])
        torch.save(checkpoint, os.path.join(self.path_params,"_r" + str(self.run_count) + "_e" + str(self.epoch_count) + ".pth"))
        return self.subfolder
    
    def load_network(self,date_,run,epoch):
        
        print("Available subfolders:")
        print(np.transpose(next(os.walk(self.params_root))[1]))
        print()

        run_epoch,filenames = self.retrieve_run_epoch_from_filenames(date_)
        
        print("Files in " + "[" + date_ + "]")
        print(filenames)
        
        # find the index of the corresponding filename in filenames-list
        for row, sublist in enumerate(run_epoch):
            if (sublist[0] == run) & (sublist[1] == epoch):
                idx_ = row
                
        filename = filenames[idx_]
        
        folder = os.path.join(self.params_root,date_)
        
        checkpoint = torch.load(os.path.join(folder,filename))
        
        model = Network3(checkpoint["input_depth"], 
                    checkpoint["conv_layer_out_channels"], 
                    checkpoint["conv_layer_kernel_sizes"], 
                    checkpoint["conv_layer_strides"], 
                    checkpoint["conv_layer_paddings"], 
                    checkpoint["max_pool_layer_numbers"], 
                    checkpoint["max_pool_kernel_sizes"], 
                    checkpoint["max_pool_strides"], 
                    checkpoint["max_pool_paddings"], 
                    checkpoint["batch_norm_2d_layer_numbers"],
                    checkpoint["dropout2d_layer_numbers"],
                    checkpoint["dropout2d_probabilities"], 
                    checkpoint["linear_layer_out_features"],                    
                    checkpoint["batch_norm_1d_layer_numbers"],
                    checkpoint["conv_activation_functions"],
                    checkpoint["fc_activation_functions"],
                    checkpoint["dropout1d_layer_numbers"],
                    checkpoint["dropout1d_probabilities"]
                    )
        
        model.load_state_dict(checkpoint['state_dict'])
        print("\nLoaded: ")
        print(filename)
        # model.eval()
        return model
        
    def save_network_params(self,model,comment):
        # print()
        # print()
        # print(model.state_dict())
        torch.save(model.state_dict(), os.path.join(self.path_params,"_r" + str(self.run_count) + "_e" + str(self.epoch_count) + ".pt"))
        # torch.save(model.state_dict(), os.path.join(self.path_params,comment + "_r" + str(self.run_count) + "_e" + str(self.epoch_count) + ".pt"))
        return self.subfolder
        
    def load_network_params(self,date_,model,run,epoch,params__):
        
        print("Available subfolders:")
        print(np.transpose(next(os.walk(self.params_root))[1]))
        print()

        print("Files in " + "[" + date_ + "]")
        print(np.transpose(os.listdir(os.path.join(self.params_root,date_))))
        
        p = list(zip(RunBuilder.get_runs(params__))) # parameters
        
        folder = os.path.join(self.params_root,date_)
        filename = "-"+str(p[run-1][0])+"_r"+str(run)+"_e"+str(epoch)+".pt"
        model.load_state_dict(torch.load(os.path.join(folder,filename)))
        print("\nLoaded: " +str(p[run-1][0]))
        print(filename)
        model.eval()
        
    def load_network_params_2(self,date_,model,run,epoch):
        
        print("Available subfolders:")
        print(np.transpose(next(os.walk(self.params_root))[1]))
        print()

        run_epoch,filenames = self.retrieve_run_epoch_from_filenames(date_)
        
        print("Files in " + "[" + date_ + "]")
        print(filenames)
        
        # find the index of the corresponding filename in filenames-list
        for row, sublist in enumerate(run_epoch):
            if (sublist[0] == run) & (sublist[1] == epoch):
                idx_ = row
                
        filename = filenames[idx_]
        
        folder = os.path.join(self.params_root,date_)
        model.load_state_dict(torch.load(os.path.join(folder,filename)))
        print("\nLoaded: ")
        print(filename)
        model.eval()
        
    def number_of_epochs(self,date__, run):
        n_e = 0
        run_epoch,_ = self.retrieve_run_epoch_from_filenames(date__)
        # find the index of the corresponding filename in filenames-list
        for row, sublist in enumerate(run_epoch):
            if (sublist[0] == run):
                n_e += 1        
        return n_e
    
    def retrieve_run_epoch_from_filenames(self,date__):       
        filenames = np.transpose(os.listdir(os.path.join(self.params_root,date__)))  
        # to list only files wothout directories
        # filenames = (file for file in os.listdir(self.params_root+date__) if os.path.isfile(os.path.join(self.params_root+date__, file)))
        # defining regular expressions to read the end of the filenames to identify which run and epoch they correspond to
        regex_last = re.compile(r'r\d+_e\d+') # for finding endings pof the filenames
        regex_r_e = re.compile(r'\d+') # for extracting numbers from the endings of the filenames        
        run_epoch = np.zeros(shape=(len(filenames), 2), dtype=np.int) # initialise array for storing numbers for 'run' and 'epoch'
        for i in range(len(filenames)):
            ending = regex_last.findall(filenames[i]) # finding the ending of the filename
            numbers = [int(i) for i in regex_r_e.findall(ending[0])] # extracting numbers in into a list
            run_epoch[i,:] = [int(numbers[j]) for j in range(2)] # adding list of two extracted numbers into the combining list
        return run_epoch, filenames

    # This method saves the run_data in two formats, json and csv.
    def save(self, fileName):
        
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(os.path.join(self.path,f'{fileName}.csv'))

        with open(os.path.join(self.path,f'{fileName}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
            
    def show_run_grads(self,date___,model,run):
        n_epochs = self.number_of_epochs(date__=date___, run=run)

        d = OrderedDict()
        for e in range(n_epochs):
            self.load_network_params_2( date_=date___, model=model, run=run, epoch=e+1 )
            #clear_output(wait=True) # specific to the jupyter notebook
            if e==0:
                keys_ = []
                idxes = []
                keys = list(dict(model.named_children()).keys())        
                for i in range(len(model)):
                    try:
                        print(i,keys[i],model[i].weight.reshape(1,-1).shape[1])
                        keys_.append(keys[i])
                        idxes.append(i)
                    except: None        
                for j in range(len(keys_)):
                    # initialising a dictionary which is to contain weights and biases for all epochs
                    d[keys_[j]+"_weight"] = np.zeros((n_epochs,model[idxes[j]].weight.reshape(1,-1).shape[1]))
                    d[keys_[j]+"_bias"] = np.zeros((n_epochs,model[idxes[j]].bias.reshape(1,-1).shape[1]))

            for j in range(len(keys_)):
                # filling in the dictionary with weights and biases for each of the epochs in the current run
                d[keys_[j]+"_weight"][e,:] = model[idxes[j]].weight.detach().numpy().reshape(1,-1)
                d[keys_[j]+"_bias"][e,:] = model[idxes[j]].bias.detach().numpy().reshape(1,-1)

        #clear_output(wait=True) # specific to the jupyter notebook
        for j in range(len(keys_)):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
            if d[keys_[j]+"_weight"].shape[1] < 200000:
                pcm = ax1.pcolor(d[keys_[j]+"_weight"].T,cmap='RdBu_r')
            else:
                pcm = ax1.pcolor(d[keys_[j]+"_weight"][:,0:200000].T,cmap='RdBu_r')
            ax1.set_title(keys_[j]+"_weight")
            fig.colorbar(pcm,ax=ax1)

            if d[keys_[j]+"_bias"].shape[1] < 200000:
                pcm = ax2.pcolor(d[keys_[j]+"_bias"].T,cmap='RdBu_r')
            else:
                pcm = ax2.pcolor(d[keys_[j]+"_bias"][:,0:200000].T,cmap='RdBu_r')
            ax2.set_title(keys_[j]+"_bias")
            fig.colorbar(pcm,ax=ax2)
            plt.tight_layout()

show_gpu_memory()

#%% new cell: Losses (function definitions)
# Losses (function definitions)

# mean absolute error
def MAE(yHat,y):
    size = yHat.size()[0]*yHat.size()[1]
    diff = yHat - y
    return torch.sum(torch.abs(diff)) / size

# mean squared error
def MSE(yHat,y):
    size = yHat.size()[0]*yHat.size()[1]
    diff = yHat - y
    return torch.sum(diff**2) / size

# Huber
def Huber(yHat,y,delta=1.):
    size = yHat.size()[0]*yHat.size()[1]
    diff = yHat - y
    return torch.sum(torch.where(torch.abs(diff) < delta,.5*diff**2 , delta*(torch.abs(diff)-.5*delta**2))) / size

# mean squared error with explicit const and linear terms
def MSE_toOptimize(params_,yHat,y):
    y0,y1 = params_
    x = [i for i in range(yHat.size()[1])]
    x = torch.tensor(x).to('cuda:0')
    size = yHat.size()[0]*yHat.size()[1]
    diff = yHat + y0 + y1*x - y
    res_ = torch.sum(diff**2) / size
    return res_.clone().cpu().numpy()

def optimizedParametersAre(yHat,y):
    guess = [0.,0.]
    res = minimize(MSE_toOptimize, guess, args=(yHat,y))
    params_ = res.x
    return torch.tensor(params_).to('cuda:0')

def MSE_mod(yHat,y):
    params_ = optimizedParametersAre(yHat,y)
    loss__ = MSE_toOptimize(params_,yHat,y)
    loss__ = torch.tensor(loss__).to('cuda:0').requires_grad_(True)
    return loss__
    
    
show_gpu_memory()

#%% new cell (optuna-related activity)
# Suggest parameters here (for the next cell)

# define exceptions
class MissingStepParameter(Exception):
    def __str__(self):
        return "Parameter 'step' must be specified."
    
class MissingChoicesParameter(Exception):
    def __str__(self):
        return "Non-keyword parameter 'choices' must be specified as a list of strings."
    
class LogParameterError(Exception):
    def __str__(self):
        return "Parameter 'log' must NOT be specified."

class StepParameterError(Exception):
    def __str__(self):
        return "Parameter 'step' must NOT be specified."
    
class SuggestTypeError(Exception):
    def __str__(self):
        return "Non-keyword parameter 'suggest_type' must have one of the following values: 'int', 'float', 'uniform', 'loguniform', 'discrete_uniform' or 'categorical'."




class optuna_optimize_callback(object):

    def __init__(self, storage, database, studyname):
        self.studyname = studyname
        self.database = database
        self.storage = storage
        self.logs_root = os.path.join(os.getcwd(),"logs")
        self.folder = os.path.join(self.logs_root,self.database,self.studyname)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def __call__(self, study, trial):
        self.filelist = np.transpose(os.listdir(self.folder)) 
        self.last_filename = self.filelist[-1]
        df_ = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        list_arr_ = list(df_.to_numpy())
        try:
            best_trial_ = study.best_trial
            file_=open(os.path.join(self.folder,self.last_filename),'a')
            L = ["\nBest trial:\n"
                "- number: "+str(best_trial_.number),
                "- direction: "+str(study.direction),
                 "- value: "+str(best_trial_.value),
                 "- date-time (started): "+str(best_trial_.datetime_start),
                 "- date-time (completed): "+str(best_trial_.datetime_complete),
                 "- parameters:",
                 *["    {}: {}".format(key, value) for key, value in study.best_trial.params.items()],
                 "\nBest trial's full info:",
                 str(best_trial_)]
            file_.writelines(str(line) + '\n' for line in L)
            file_.close()
        except:
            print("Best trial is not available.") 
        
        headers = ["  - "+col for col in df_.columns]        
        file_=open(os.path.join(self.folder,"0000_trials_info.txt"),'w')
        L = ["Database path: " + self.storage,
             "Database used: " + self.database,
             "Optuna study name: " + self.studyname,
            "\nDataframe (info on previous trials):\n",
             "The elements in each of the lists below are ordered as follows:",
             *headers,
             "",
             *list_arr_]
        file_.writelines(str(line) + '\n' for line in L)
        file_.close()


# The following function returns network_params as a dictionary.
# It tries to suggest the network architecture within the Optuna-framework.
# Try adding other conditions like:
# - if flattened input to the first linear layer is bigger than 10000 then Prune
# - suggesting parameters starting from lower values and gradually scaling up (i.e. change the ranges dynamically)
# - Add handling 'RuntimeError: Calculated padded input size per channel: (9 x 9). Kernel size: (11 x 11). Kernel size can't be greater than actual input size'
# - Add handling 'ZeroDivisionError: float division by zero', when, I reckon, n_flattened = 0
def optuna_suggestNetworkArchitecture(trial, m_, N_conv, N_fc,
                                      batch_norm_1d_0_frozen,
                                      dropout1d_0_frozen,
                                      dropout1d_prob_0_frozen,
                                      params_ranges,
                                      hyperparams_ranges,
                                      network_params_frozen = {},
                                      network_hyperparams_frozen = {},
                                      conv_activations = [],
                                      fc_activations = [],
                                      input_size=100, # size of images                                      
                                      n_classes__=10, # number of outputs
                                      cnt__=100):
    
    optuna.logging.enable_default_handler()
    
    # create file containing info on activation functions and (hyper)params_ranges
    if not os.path.exists(m_.logfile_static_path):
        f = open(m_.logfile_static_path,'w')
        L = ["Conv-layers' activation functions:",
             str(conv_activations) + "\n",
             "Fc-layers' activation functions:",
             str(fc_activations) + "\n\n",
             "---Parameters' tuning ranges---\n",
                *[str(key)+" : "+str(value) for key,value in params_ranges.items()],
                "\n---Hyperparameters' tuning ranges---\n",
                *[str(key)+" : "+str(value) for key,value in hyperparams_ranges.items()]
                ]
        f.writelines(str(line) + '\n' for line in L)
        f.close()
    
    
    if bool(trial):
        L = ["\nCurrent trial #: "+str(trial.number)]
        m_.current_trial = str(trial.number)
        m_.log_file.writelines(str(line) + '\n' for line in L)
    
    n_current = input_size
    n_ = [] # will store lateral dimensions of feature maps
    n_.append(n_current)
        
    network_params = {}
    network_params['conv_layer_out_channels'] = []#[0 for i in range(N_conv)]
    network_params['conv_layer_kernel_sizes'] = []#[0 for i in range(N_conv)]
    network_params['conv_layer_strides'] = []#[0 for i in range(N_conv)]
    network_params['conv_layer_paddings'] = []#[0 for i in range(N_conv)]
    network_params['max_pool_layer_numbers'] = []#[0 for i in range(N_conv)]
    network_params['max_pool_kernel_sizes'] = []#[0 for i in range(N_conv)]
    network_params['max_pool_strides'] = []#[0 for i in range(N_conv)]
    network_params['max_pool_paddings'] = []#[0 for i in range(N_conv)]
    network_params['batch_norm_2d_layer_numbers'] = []#[0 for i in range(N_conv)]
    network_params['dropout2d_layer_numbers'] = []#[0 for i in range(N_conv)]
    network_params['dropout2d_probabilities'] = []#[0 for i in range(N_conv)] 
    i = 0
    if bool(network_params_frozen):
        Ns_ = []
        # finding the maximum length amongst given parameter-vectors
        for key in network_params:
            Ns_.append(len(network_params_frozen[key]))
        N_conv_given = max(Ns_)
        i = N_conv_given
        for k in range(N_conv_given):
            
            network_params['conv_layer_out_channels'].append(0)
            network_params['conv_layer_kernel_sizes'].append(0)
            network_params['conv_layer_strides'].append(0)
            network_params['conv_layer_paddings'].append(0)
            network_params['max_pool_layer_numbers'].append(0)
            network_params['max_pool_kernel_sizes'].append(0)
            network_params['max_pool_strides'].append(0)
            network_params['max_pool_paddings'].append(0)
            network_params['batch_norm_2d_layer_numbers'].append(0)
            network_params['dropout2d_layer_numbers'].append(0)
            network_params['dropout2d_probabilities'].append(0.)
            
            if k < len(network_params_frozen['conv_layer_out_channels']):
                if network_params_frozen['conv_layer_out_channels'][k] != -1:                    
                    network_params['conv_layer_out_channels'][k] = network_params_frozen['conv_layer_out_channels'][k]
                else:
                    network_params['conv_layer_out_channels'][k] = trial.suggest_int("network_params['conv_layer_out_channels']["+str(k)+"]",
                                  params_ranges['conv_layer_out_channels'][0],
                                  params_ranges['conv_layer_out_channels'][1]
                                  )                    
            else: # if it is not given then suggest some value
                network_params['conv_layer_out_channels'][k] = trial.suggest_int("network_params['conv_layer_out_channels']["+str(k)+"]",
                                  params_ranges['conv_layer_out_channels'][0],
                                  params_ranges['conv_layer_out_channels'][1]
                                  )
                
            if k < len(network_params_frozen['conv_layer_kernel_sizes']):
                if network_params_frozen['conv_layer_kernel_sizes'][k] != -1:                    
                    network_params['conv_layer_kernel_sizes'][k] = network_params_frozen['conv_layer_kernel_sizes'][k]
                else:
                    network_params['conv_layer_kernel_sizes'][k] = trial.suggest_int("network_params['conv_layer_kernel_sizes']["+str(k)+"]",
                                  params_ranges['conv_layer_kernel_sizes'][0],
                                  params_ranges['conv_layer_kernel_sizes'][1]
                                  )
            else:
                network_params['conv_layer_kernel_sizes'][k] = trial.suggest_int("network_params['conv_layer_kernel_sizes']["+str(k)+"]",
                                  params_ranges['conv_layer_kernel_sizes'][0],
                                  params_ranges['conv_layer_kernel_sizes'][1],
                                  step=2
                                  )
                
            if k < len(network_params_frozen['conv_layer_strides']):
                if network_params_frozen['conv_layer_strides'][k] != -1:                    
                    network_params['conv_layer_strides'][k] = network_params_frozen['conv_layer_strides'][k]
                else:
                    network_params['conv_layer_strides'][k] = trial.suggest_int("network_params['conv_layer_strides']["+str(k)+"]",
                                  params_ranges['conv_layer_strides'][0],
                                  params_ranges['conv_layer_strides'][1]
                                  )
            else:
                network_params['conv_layer_strides'][k] = trial.suggest_int("network_params['conv_layer_strides']["+str(k)+"]",
                                  params_ranges['conv_layer_strides'][0],
                                  params_ranges['conv_layer_strides'][1]
                                  )
                
            if k < len(network_params_frozen['conv_layer_paddings']):
                if network_params_frozen['conv_layer_paddings'][k] != -1:                    
                    network_params['conv_layer_paddings'][k] = network_params_frozen['conv_layer_paddings'][k]
                else:
                    network_params['conv_layer_paddings'][k] = trial.suggest_int("network_params['conv_layer_paddings']["+str(k)+"]",
                                  params_ranges['conv_layer_paddings'][0],
                                  (network_params['conv_layer_kernel_sizes'][k]-1)//2
                                  )
            else:
                network_params['conv_layer_paddings'][k] = trial.suggest_int("network_params['conv_layer_paddings']["+str(k)+"]",
                                  params_ranges['conv_layer_paddings'][0],\
                                  (network_params['conv_layer_kernel_sizes'][k]-1)//2
                                  )
                    
            if k < len(network_params_frozen['max_pool_layer_numbers']):
                if network_params_frozen['max_pool_layer_numbers'][k] != -1:                    
                    network_params['max_pool_layer_numbers'][k] = network_params_frozen['max_pool_layer_numbers'][k]
                else:
                    network_params['max_pool_layer_numbers'][k] = trial.suggest_int("network_params['max_pool_layer_numbers']["+str(k)+"]",
                                  params_ranges['max_pool_layer_numbers'][0],
                                  params_ranges['max_pool_layer_numbers'][1]
                                  )
            else:
                network_params['max_pool_layer_numbers'][k] = trial.suggest_int("network_params['max_pool_layer_numbers']["+str(k)+"]",
                                  params_ranges['max_pool_layer_numbers'][0],
                                  params_ranges['max_pool_layer_numbers'][1]
                                  )
                
            if network_params['max_pool_layer_numbers'][k]:    
                if k < len(network_params_frozen['max_pool_kernel_sizes']):
                    if network_params_frozen['max_pool_kernel_sizes'][k] != -1 and network_params_frozen['max_pool_kernel_sizes'][k] != 0: 
                        network_params['max_pool_kernel_sizes'][k] = network_params_frozen['max_pool_kernel_sizes'][k]
                    else:
                        network_params['max_pool_kernel_sizes'][k] = trial.suggest_int("network_params['max_pool_kernel_sizes']["+str(k)+"]",
                                      params_ranges['max_pool_kernel_sizes'][0],
                                      params_ranges['max_pool_kernel_sizes'][1]
                                      )
                else:
                    network_params['max_pool_kernel_sizes'][k] = trial.suggest_int("network_params['max_pool_kernel_sizes']["+str(k)+"]",
                                          params_ranges['max_pool_kernel_sizes'][0],
                                          params_ranges['max_pool_kernel_sizes'][1]
                                          )
            else:
                network_params['max_pool_kernel_sizes'][k] = 0
               
            if network_params['max_pool_layer_numbers'][k]:
                if k < len(network_params_frozen['max_pool_strides']):
                    if network_params_frozen['max_pool_strides'][k] != -1 and network_params_frozen['max_pool_strides'][k] != 0:
                        network_params['max_pool_strides'][k] = network_params_frozen['max_pool_strides'][k]
                    else:
                        network_params['max_pool_strides'][k] = trial.suggest_int("network_params['max_pool_strides']["+str(k)+"]",
                                      params_ranges['max_pool_strides'][0],
                                      network_params['max_pool_kernel_sizes'][k]
                                      )
                else:
                    network_params['max_pool_strides'][k] = trial.suggest_int("network_params['max_pool_strides']["+str(k)+"]",\
                                          params_ranges['max_pool_strides'][0],
                                          network_params['max_pool_kernel_sizes'][k]
                                          )
            else:
                network_params['max_pool_strides'][k] = 0
                    
            if network_params['max_pool_layer_numbers'][k]:
                if k < len(network_params_frozen['max_pool_paddings']):
                    if network_params_frozen['max_pool_paddings'][k] != -1 and network_params_frozen['max_pool_paddings'][k] != 0: 
                        network_params['max_pool_paddings'][k] = network_params_frozen['max_pool_paddings'][k]
                    else:
                        network_params['max_pool_paddings'][k] = trial.suggest_int("network_params['max_pool_paddings']["+str(k)+"]",
                                      params_ranges['max_pool_paddings'][0],
                                      network_params['max_pool_kernel_sizes'][k]//2
                                      )
                else:
                    network_params['max_pool_paddings'][k] = trial.suggest_int("network_params['max_pool_paddings']["+str(k)+"]",\
                                          params_ranges['max_pool_paddings'][0],
                                          network_params['max_pool_kernel_sizes'][k]//2
                                          )
            else:
                network_params['max_pool_paddings'][k] = 0
                    
            if k < len(network_params_frozen['batch_norm_2d_layer_numbers']):
                if network_params_frozen['batch_norm_2d_layer_numbers'][k] != -1:                    
                    network_params['batch_norm_2d_layer_numbers'][k] = network_params_frozen['batch_norm_2d_layer_numbers'][k]
                else:
                    network_params['batch_norm_2d_layer_numbers'][k] = trial.suggest_int("network_params['batch_norm_2d_layer_numbers']["+str(k)+"]",
                                  params_ranges['batch_norm_2d_layer_numbers'][0],
                                  params_ranges['batch_norm_2d_layer_numbers'][1]
                                  )
            else:
                network_params['batch_norm_2d_layer_numbers'][k] = trial.suggest_int("network_params['batch_norm_2d_layer_numbers']["+str(k)+"]",
                                  params_ranges['batch_norm_2d_layer_numbers'][0],
                                  params_ranges['batch_norm_2d_layer_numbers'][1]
                                  )
                
            if k < len(network_params_frozen['dropout2d_layer_numbers']):
                if network_params_frozen['dropout2d_layer_numbers'][k] != -1:                    
                    network_params['dropout2d_layer_numbers'][k] = network_params_frozen['dropout2d_layer_numbers'][k]
                else:
                    network_params['dropout2d_layer_numbers'][k] = trial.suggest_int("network_params['dropout2d_layer_numbers']["+str(k)+"]",
                                  params_ranges['dropout2d_layer_numbers'][0],
                                  params_ranges['dropout2d_layer_numbers'][1]
                                  )
            else:
                network_params['dropout2d_layer_numbers'][k] = trial.suggest_int("network_params['dropout2d_layer_numbers']["+str(k)+"]",
                                  params_ranges['dropout2d_layer_numbers'][0],
                                  params_ranges['dropout2d_layer_numbers'][1]
                                  )
                
            if network_params['dropout2d_layer_numbers'][k]:
                if k < len(network_params_frozen['dropout2d_probabilities']):
                    if network_params_frozen['dropout2d_probabilities'][k] != -1 and network_params_frozen['dropout2d_probabilities'][k] != 0: 
                        network_params['dropout2d_probabilities'][k] = network_params_frozen['dropout2d_probabilities'][k]
                    else:
                        network_params['dropout2d_probabilities'][k] = trial.suggest_float("network_params['dropout2d_probabilities']["+str(k)+"]",
                                      params_ranges['dropout2d_probabilities'][0],
                                      params_ranges['dropout2d_probabilities'][1],
                                      step = 0.02,
                                      log = False
                                      )
                else:
                    network_params['dropout2d_probabilities'][k] = trial.suggest_float("network_params['dropout2d_probabilities']["+str(k)+"]",
                                    params_ranges['dropout2d_probabilities'][0],
                                    params_ranges['dropout2d_probabilities'][1],
                                    step = 0.02,
                                    log = False
                                    )
            else:
                network_params['max_pool_paddings'][k] = 0
                
            n_current = int((n_current+2*network_params['conv_layer_paddings'][k]-\
                         network_params['conv_layer_kernel_sizes'][k])/\
                        network_params['conv_layer_strides'][k]+1)
            
            if n_current < 1: 
                flag_to_accept_params = False
                m_.log_file.close()
                raise optuna.TrialPruned()
                break
            
            if network_params['max_pool_layer_numbers'][k]:
                n_current = int((n_current+2*network_params['max_pool_paddings'][k]-\
                             network_params['max_pool_kernel_sizes'][k])/\
                            network_params['max_pool_strides'][k]+1)
                if n_current <= 1: 
                    flag_to_accept_params = False
                    m_.log_file.close()
                    raise optuna.TrialPruned()
                    break
            n_.append(n_current)
                
    # conv-layers part (encoder-part)
    flag_to_accept_params = False    
    runtime_count_ = 0 # this is to escape very long searches at the current layer
    
    while i < N_conv:
        
        if bool(trial):
            runtime_count_ += 1
            if runtime_count_ == cnt__+1: # if runtime_count_ is out then remove last layer
                i -= 1
                runtime_count_ = 0
                n_current = n_[-2]
                # probably also need to check here for empty lists
                _ = n_.pop()
            
            network_params['conv_layer_out_channels'].append(0)
            network_params['conv_layer_kernel_sizes'].append(0)
            network_params['conv_layer_strides'].append(0)
            network_params['conv_layer_paddings'].append(0)
            network_params['max_pool_layer_numbers'].append(0)
            network_params['max_pool_kernel_sizes'].append(0)
            network_params['max_pool_strides'].append(0)
            network_params['max_pool_paddings'].append(0)
            network_params['batch_norm_2d_layer_numbers'].append(0)
            network_params['dropout2d_layer_numbers'].append(0)
            network_params['dropout2d_probabilities'].append(0.)
            
            network_params['conv_layer_out_channels'][i] = \
                trial.suggest_int("network_params['conv_layer_out_channels']["+str(i)+"]",
                                  params_ranges['conv_layer_out_channels'][0],
                                  params_ranges['conv_layer_out_channels'][1]
                                  )
            network_params['conv_layer_kernel_sizes'][i] = \
                trial.suggest_int("network_params['conv_layer_kernel_sizes']["+str(i)+"]",
                                  params_ranges['conv_layer_kernel_sizes'][0],
                                  params_ranges['conv_layer_kernel_sizes'][1],
                                  step=2
                                  )
            network_params['conv_layer_strides'][i] = \
                trial.suggest_int("network_params['conv_layer_strides']["+str(i)+"]",
                                  params_ranges['conv_layer_strides'][0],
                                  params_ranges['conv_layer_strides'][1]
                                  )
            network_params['conv_layer_paddings'][i] = \
                trial.suggest_int("network_params['conv_layer_paddings']["+str(i)+"]",
                                  params_ranges['conv_layer_paddings'][0],\
                                  (network_params['conv_layer_kernel_sizes'][i]-1)//2
                                  )
            n_current = int((n_current+2*network_params['conv_layer_paddings'][i]-\
                             network_params['conv_layer_kernel_sizes'][i])/\
                            network_params['conv_layer_strides'][i]+1)
        
            if n_current < 1: 
                flag_to_accept_params = False
                m_.log_file.close()
                raise optuna.TrialPruned()
                break
            
            network_params['max_pool_layer_numbers'][i] = \
                trial.suggest_int("network_params['max_pool_layer_numbers']["+str(i)+"]",
                                  params_ranges['max_pool_layer_numbers'][0],
                                  params_ranges['max_pool_layer_numbers'][1]
                                  )
            if network_params['max_pool_layer_numbers'][i]:
                network_params['max_pool_kernel_sizes'][i] = \
                    trial.suggest_int("network_params['max_pool_kernel_sizes']["+str(i)+"]",
                                      params_ranges['max_pool_kernel_sizes'][0],
                                      params_ranges['max_pool_kernel_sizes'][1]
                                      )
                network_params['max_pool_strides'][i] = \
                    trial.suggest_int("network_params['max_pool_strides']["+str(i)+"]",\
                                      params_ranges['max_pool_strides'][0],
                                      network_params['max_pool_kernel_sizes'][i]
                                      )
                network_params['max_pool_paddings'][i] = \
                    trial.suggest_int("network_params['max_pool_paddings']["+str(i)+"]",\
                                      params_ranges['max_pool_paddings'][0],
                                      network_params['max_pool_kernel_sizes'][i]//2
                                      )
                n_current = int((n_current+2*network_params['max_pool_paddings'][i]-\
                                 network_params['max_pool_kernel_sizes'][i])/\
                                network_params['max_pool_strides'][i]+1)
                if n_current <= 1: 
                    flag_to_accept_params = False  
                    m_.log_file.close()
                    raise optuna.TrialPruned()
                    break
                
        
            n_.append(n_current)
            
            network_params['batch_norm_2d_layer_numbers'][i] = \
                trial.suggest_int("network_params['batch_norm_2d_layer_numbers']["+str(i)+"]",
                                  params_ranges['batch_norm_2d_layer_numbers'][0],
                                  params_ranges['batch_norm_2d_layer_numbers'][1]
                                  )
                
            network_params['dropout2d_layer_numbers'][i] = \
                trial.suggest_int("network_params['dropout2d_layer_numbers']["+str(i)+"]",
                                  params_ranges['dropout2d_layer_numbers'][0],
                                  params_ranges['dropout2d_layer_numbers'][1]
                                  )
            
            if network_params['dropout2d_layer_numbers'][i]:
                network_params['dropout2d_probabilities'][i] = \
                    trial.suggest_float("network_params['dropout2d_probabilities']["+str(i)+"]",
                                      params_ranges['dropout2d_probabilities'][0],
                                      params_ranges['dropout2d_probabilities'][1],
                                      step = 0.02,
                                      log = False
                                      )
            
            flag_to_accept_params = True
            if flag_to_accept_params:
                i += 1
                flag_to_accept_params = False
            
    network_params['linear_layer_out_features'] = []
    network_params['batch_norm_1d_layer_numbers'] = []
    network_params['dropout1d_layer_numbers'] = []
    network_params['dropout1d_probabilities'] = []
            
    # fc-layers part (decoder part)
    n_flattened = n_current*n_current*network_params['conv_layer_out_channels'][-1]
    if n_flattened == 0:
        flag_to_accept_params = False
        print('The trial',str(trial.number),'is pruned because "n_flattened == 0"')
        m_.log_file.close()
        raise optuna.TrialPruned()
        return
    n_ = []
    n_.append(n_flattened)
    
    if bool(network_params_frozen):
        Ns_fc_ = [len(network_params_frozen['linear_layer_out_features']),\
                      len(network_params_frozen['batch_norm_1d_layer_numbers']),\
                            len(network_params_frozen['dropout1d_layer_numbers']),\
                                len(network_params_frozen['dropout1d_probabilities'])]
        # finding the maximum length amongst given parameter-vectors
        N_fc_given = max(Ns_fc_)
    else:
        N_fc_given = 0
    
    network_params['linear_layer_out_features'].append(n_flattened)
    network_params['batch_norm_1d_layer_numbers'].append(1)
    network_params['dropout1d_layer_numbers'].append(1)
    network_params['dropout1d_probabilities'].append(0.)
    if bool(trial): # if it is optimization mode
        if len(batch_norm_1d_0_frozen) == 0 or (len(batch_norm_1d_0_frozen) == 1 and batch_norm_1d_0_frozen[0] == -1):
            network_params['batch_norm_1d_layer_numbers'][0] = \
                trial.suggest_int("network_params['batch_norm_1d_layer_numbers'][0]",
                                  params_ranges['batch_norm_1d_layer_numbers'][0],
                                  params_ranges['batch_norm_1d_layer_numbers'][1]
                                  )
        else:
            network_params['batch_norm_1d_layer_numbers'][0] = batch_norm_1d_0_frozen[0]
            
        if len(dropout1d_0_frozen) == 0 or (len(dropout1d_0_frozen) == 1 and dropout1d_0_frozen[0] == -1):
            network_params['dropout1d_layer_numbers'][0] = \
                trial.suggest_int("network_params['dropout1d_layer_numbers'][0]",
                                  params_ranges['dropout1d_layer_numbers'][0],
                                  params_ranges['dropout1d_layer_numbers'][1]
                                  )
        else:
            network_params['dropout1d_layer_numbers'][0] = dropout1d_0_frozen[0]
            
        if network_params['dropout1d_layer_numbers'][0]:    
            if len(dropout1d_prob_0_frozen) == 0 or (len(dropout1d_prob_0_frozen) == 1 and dropout1d_prob_0_frozen[0] == -1):
                network_params['dropout1d_probabilities'][0] = \
                    trial.suggest_float("network_params['dropout1d_probabilities'][0]",
                                      params_ranges['dropout1d_probabilities'][0],
                                      params_ranges['dropout1d_probabilities'][1],
                                      step = 0.02,
                                      log = False
                                      )
            else:
                network_params['dropout1d_probabilities'][0] = dropout1d_prob_0_frozen[0]
            
    else: # if it is training mode (batch_norm_1d_0_frozen[0] must be given beforehand)
        network_params['batch_norm_1d_layer_numbers'][0] = batch_norm_1d_0_frozen[0]
        network_params['dropout1d_layer_numbers'][0] = dropout1d_0_frozen[0]
        network_params['dropout1d_probabilities'][0] = dropout1d_prob_0_frozen[0]
        
        
        # dropout1d_prob_0_frozen
    
    flag_to_accept_params = False
    i = 1
    if bool(network_params_frozen):
        
        for k in range(1,N_fc_given+1):
            # print('k=',k)
            network_params['linear_layer_out_features'].append(0)
            network_params['batch_norm_1d_layer_numbers'].append(1)
            network_params['dropout1d_layer_numbers'].append(1)
            network_params['dropout1d_probabilities'].append(0.)
            if k < len(network_params_frozen['linear_layer_out_features'])+1:
                if network_params_frozen['linear_layer_out_features'][k-1] != -1:                    
                    network_params['linear_layer_out_features'][k] = network_params_frozen['linear_layer_out_features'][k-1]
                else:
                    network_params['linear_layer_out_features'][k] = trial.suggest_int("network_params['linear_layer_out_features']["+str(k)+"]",
                                  params_ranges['linear_layer_out_features'][0],#n_classes__//2,
                                  params_ranges['linear_layer_out_features'][1]
                                  )
            else:
                network_params['linear_layer_out_features'][k] = \
                trial.suggest_int("network_params['linear_layer_out_features']["+str(k)+"]",\
                                  params_ranges['linear_layer_out_features'][0],#n_classes__//2,
                                  params_ranges['linear_layer_out_features'][1]
                                  )
            
            if k < len(network_params_frozen['batch_norm_1d_layer_numbers'])+1:
                if network_params_frozen['batch_norm_1d_layer_numbers'][k-1] != -1:                    
                    network_params['batch_norm_1d_layer_numbers'][k] = network_params_frozen['batch_norm_1d_layer_numbers'][k-1]
                else:
                    network_params['batch_norm_1d_layer_numbers'][k] = trial.suggest_int("network_params['batch_norm_1d_layer_numbers']["+str(k)+"]",
                                  params_ranges['batch_norm_1d_layer_numbers'][0],
                                  params_ranges['batch_norm_1d_layer_numbers'][1]
                                  )
            else:
                network_params['batch_norm_1d_layer_numbers'][k] = \
                trial.suggest_int("network_params['batch_norm_1d_layer_numbers']["+str(k)+"]",
                                  params_ranges['batch_norm_1d_layer_numbers'][0],
                                  params_ranges['batch_norm_1d_layer_numbers'][1]
                                  )
                
            if k < len(network_params_frozen['dropout1d_layer_numbers'])+1:
                if network_params_frozen['dropout1d_layer_numbers'][k-1] != -1:
                    network_params['dropout1d_layer_numbers'][k] = network_params_frozen['dropout1d_layer_numbers'][k-1]
                else:
                    network_params['dropout1d_layer_numbers'][k] = trial.suggest_int("network_params['dropout1d_layer_numbers']["+str(k)+"]",
                                  params_ranges['dropout1d_layer_numbers'][0],
                                  params_ranges['dropout1d_layer_numbers'][1]
                                  )
            else:
                network_params['dropout1d_layer_numbers'][k] = \
                trial.suggest_int("network_params['dropout1d_layer_numbers']["+str(k)+"]",
                                  params_ranges['dropout1d_layer_numbers'][0],
                                  params_ranges['dropout1d_layer_numbers'][1]
                                  )
                
            if network_params['dropout1d_layer_numbers'][k]:
                if k < len(network_params_frozen['dropout1d_probabilities'])+1:
                    if network_params_frozen['dropout1d_probabilities'][k-1] != -1:                    
                        network_params['dropout1d_probabilities'][k] = network_params_frozen['dropout1d_probabilities'][k-1]
                    else:
                        network_params['dropout1d_probabilities'][k] = trial.suggest_float("network_params['dropout1d_probabilities']["+str(k)+"]",
                                      params_ranges['dropout1d_probabilities'][0],
                                      params_ranges['dropout1d_probabilities'][1],
                                      step=0.02,
                                      log=False
                                      )
                else:
                    network_params['dropout1d_probabilities'][k] = \
                    trial.suggest_float("network_params['dropout1d_probabilities']["+str(k)+"]",
                                      params_ranges['dropout1d_probabilities'][0],
                                      params_ranges['dropout1d_probabilities'][1],
                                      step=0.02,
                                      log = False
                                      )
        i = N_fc_given+1
    
    while i < N_fc:
        if bool(trial):
            network_params['linear_layer_out_features'].append(0)
            network_params['linear_layer_out_features'][i] = \
                trial.suggest_int("network_params['linear_layer_out_features']["+str(i)+"]",\
                                  params_ranges['linear_layer_out_features'][0],#n_classes__//2,
                                  params_ranges['linear_layer_out_features'][1]
                                  )
            network_params['batch_norm_1d_layer_numbers'].append(1)
            network_params['batch_norm_1d_layer_numbers'][i] = \
                trial.suggest_int("network_params['batch_norm_1d_layer_numbers']["+str(i)+"]",
                                  params_ranges['batch_norm_1d_layer_numbers'][0],
                                  params_ranges['batch_norm_1d_layer_numbers'][1]
                                  )
            network_params['dropout1d_layer_numbers'].append(1)
            network_params['dropout1d_layer_numbers'][i] = \
                trial.suggest_int("network_params['dropout1d_layer_numbers']["+str(i)+"]",
                                  params_ranges['dropout1d_layer_numbers'][0],
                                  params_ranges['dropout1d_layer_numbers'][1]
                                  )
                
            network_params['dropout1d_probabilities'].append(0.)
            if network_params['dropout1d_layer_numbers'][i]:
                network_params['dropout1d_probabilities'][i] = \
                    trial.suggest_float("network_params['dropout1d_probabilities']["+str(i)+"]",
                                      params_ranges['dropout1d_probabilities'][0],
                                      params_ranges['dropout1d_probabilities'][1],
                                      step=0.02,
                                      log=False
                                      )
        
            n_current = network_params['linear_layer_out_features'][i]
            
            n_.append(n_current)
            
            flag_to_accept_params = True
            if flag_to_accept_params:
                i += 1
                flag_to_accept_params = False
    network_params['batch_norm_1d_layer_numbers'].append(0)
    network_params['batch_norm_1d_layer_numbers'][N_fc] = 0 #\ since it is not used anyways
    network_params['linear_layer_out_features'].append(0)
    network_params['linear_layer_out_features'][N_fc] = n_classes__
    network_params['dropout1d_layer_numbers'].append(0)
    network_params['dropout1d_layer_numbers'][N_fc] = 0
    network_params['dropout1d_probabilities'].append(0.)
    network_params['dropout1d_probabilities'][N_fc] = 0.
    
    optuna.logging.enable_default_handler()
    
    # dealing with hyperparameters (suggesting them or not) starts from here
    network_hyperparams = {}
    network_hyperparams['batch_size'] = []
    network_hyperparams['learning_rate'] = []
    network_hyperparams['learning_rate_decay'] = []
    network_hyperparams['learning_rate_decay_const'] = []
    network_hyperparams['b1'] = []
    network_hyperparams['b2'] = []
    network_hyperparams['number_of_epochs'] = []
    network_hyperparams['number_of_workers'] = []
    network_hyperparams['device'] = []
    network_hyperparams['shuffle'] = []
    network_hyperparams['trainset'] = []
    
    if len(network_hyperparams_frozen['batch_size']) == 0 or (len(network_hyperparams_frozen['batch_size']) == 1 and network_hyperparams_frozen['batch_size'][0] == -1):
        network_hyperparams['batch_size'].append(1)
        network_hyperparams['batch_size'][0] = \
            trial.suggest_int("network_hyperparams['batch_size'][0]",
                                  hyperparams_ranges['batch_size'][0],
                                  hyperparams_ranges['batch_size'][1]
                                  )
    else:
        network_hyperparams['batch_size'].append(network_hyperparams_frozen['batch_size'][0])
    
    if len(network_hyperparams_frozen['learning_rate']) == 0 or (len(network_hyperparams_frozen['learning_rate']) == 1 and network_hyperparams_frozen['learning_rate'][0] == -1):
        network_hyperparams['learning_rate'].append(0.001)
        network_hyperparams['learning_rate'][0] = \
            trial.suggest_float("network_hyperparams['learning_rate'][0]",
                                  hyperparams_ranges['learning_rate'][0],
                                  hyperparams_ranges['learning_rate'][1],
                                  log = True
                                  )
    else:
        network_hyperparams['learning_rate'].append(network_hyperparams_frozen['learning_rate'][0])
    
    if len(network_hyperparams_frozen['learning_rate_decay']) == 0 or (len(network_hyperparams_frozen['learning_rate_decay']) == 1 and network_hyperparams_frozen['learning_rate_decay'][0] == -1):
        network_hyperparams['learning_rate_decay'].append('cnst')
        lrd = trial.suggest_int("lrd",
                                  hyperparams_ranges['learning_rate_decay'][0],
                                  hyperparams_ranges['learning_rate_decay'][1]
                                  )
        if lrd == 0:
            network_hyperparams['learning_rate_decay'][0] = 'cnst'
        elif lrd == 1:
            network_hyperparams['learning_rate_decay'][0] = 'exp'
        elif lrd == 2:
            network_hyperparams['learning_rate_decay'][0] = 'sqrt'
    else:
        network_hyperparams['learning_rate_decay'].append(network_hyperparams_frozen['learning_rate_decay'][0])
    
    if len(network_hyperparams_frozen['learning_rate_decay_const']) == 0 or (len(network_hyperparams_frozen['learning_rate_decay_const']) == 1 and network_hyperparams_frozen['learning_rate_decay_const'][0] == -1):
        network_hyperparams['learning_rate_decay_const'].append(0.95)
        network_hyperparams['learning_rate_decay_const'][0] = \
            trial.suggest_int("network_hyperparams['learning_rate_decay_const'][0]",
                                  hyperparams_ranges['learning_rate_decay_const'][0],
                                  hyperparams_ranges['learning_rate_decay_const'][1]
                                  )
    else:
        network_hyperparams['learning_rate_decay_const'].append(network_hyperparams_frozen['learning_rate_decay_const'][0])
    
    if len(network_hyperparams_frozen['b1']) == 0 or (len(network_hyperparams_frozen['b1']) == 1 and network_hyperparams_frozen['b1'][0] == -1):
        network_hyperparams['b1'].append(0.99)
        network_hyperparams['b1'][0] = \
            trial.suggest_float("network_hyperparams['b1'][0]",
                                  hyperparams_ranges['b1'][0],
                                  hyperparams_ranges['b1'][1]
                                  )
    else:
        network_hyperparams['b1'].append(network_hyperparams_frozen['b1'][0])
            
    if len(network_hyperparams_frozen['b2']) == 0 or (len(network_hyperparams_frozen['b2']) == 1 and network_hyperparams_frozen['b2'][0] == -1):
        network_hyperparams['b2'].append(0.999)
        network_hyperparams['b2'][0] = \
            trial.suggest_float("network_hyperparams['b2'][0]",
                                  hyperparams_ranges['b2'][0],
                                  hyperparams_ranges['b2'][1]
                                  )
    else:
        network_hyperparams['b2'].append(network_hyperparams_frozen['b2'][0])
    
    if len(network_hyperparams_frozen['number_of_epochs']) == 0 or (len(network_hyperparams_frozen['number_of_epochs']) == 1 and network_hyperparams_frozen['number_of_epochs'][0] == -1):
        network_hyperparams['number_of_epochs'].append(50)
        network_hyperparams['number_of_epochs'][0] = \
            trial.suggest_int("network_hyperparams['number_of_epochs'][0]",
                                  hyperparams_ranges['number_of_epochs'][0],
                                  hyperparams_ranges['number_of_epochs'][1]
                                  )          
    else:
        network_hyperparams['number_of_epochs'].append(network_hyperparams_frozen['number_of_epochs'][0])
    
    network_hyperparams['number_of_workers'].append(network_hyperparams_frozen['number_of_workers'][0])
    network_hyperparams['device'].append(network_hyperparams_frozen['device'][0])
    network_hyperparams['shuffle'].append(network_hyperparams_frozen['shuffle'][0])
    network_hyperparams['trainset'].append(network_hyperparams_frozen['trainset'][0])  
    
    
    # Printing all parameters and hyperparameters that will be used
    # print('i:',i,'; network_params:',network_params)
    print('\n -------------------- ')
    print('| Network parameters |')
    print(' -------------------- ')
    for key, value in network_params.items():
        print(key, ' : ', value)
    print()
    print("network_params['batch_norm_1d_layer_numbers'][0]:",network_params['batch_norm_1d_layer_numbers'][0])
    print("network_params['dropout1d_layer_numbers'][0]:",network_params['dropout1d_layer_numbers'][0])
    print("network_params['dropout1d_probabilities'][0]:",network_params['dropout1d_probabilities'][0])
    print()
    print(' ------------------------- ')
    print('| Network hyperparameters |')
    print(' ------------------------- ')
    for key, value in network_hyperparams.items():
        print(key, ' : ', value)
    print()
    
    # logging into a file
    if bool(trial):
        L = ["\n -------------------- ",
                 "| Network parameters |",
                 " -------------------- ",
                 *[str(key)+" : "+str(value) for key,value in network_params.items()],
                 " -------------------- ",
                 "| Network hyperparameters |",
                 " -------------------- ",
                 *[str(key)+" : "+str(value) for key,value in network_hyperparams.items()],
                 ""]
        m_.log_file.writelines(str(line) + '\n' for line in L)
    
    return network_params,network_hyperparams # dictionary

#%% new cell (defining a network dynamically)
# Define a network on the fly (dynamically) from given set of parameters here

# One layer is anythin' starting from conv2d till the next conv2d
# Similar logic is for linear (FC) layers
# The block of conv-layers is separated from the block of FC-layers by flatten-operation

input_depth = 2

# Parameters
# conv-layer-related parameters
conv_layer_out_channels = [6,12]#network_parameters['conv_layer_out_channels']#[6,12]# [5,17,26,26,43,58,67,52,68,86,91]
N_conv_layers = len(conv_layer_out_channels) # number of convolutional layers
conv_layer_kernel_sizes = [5,5]#[network_parameters['conv_layer_kernel_sizes']#[5,5]#[8,6,5,7,8,5,7,4,7,8,6]
conv_layer_strides =      [1,1]#network_parameters['conv_layer_strides']#[1,1]#[3,2,2,1,3,1,5,2,5,4,2]
conv_layer_paddings =     [0,0]#network_parameters['conv_layer_paddings']#[0,0]#[0,3,6,5,10,10,7,8,5,5,10]
assert len(conv_layer_out_channels) == N_conv_layers, "Double-check the entries of 'conv_layer_out_channels'."
assert len(conv_layer_kernel_sizes) == N_conv_layers, "Double-check the entries of 'conv_layer_kernel_sizes'."
assert len(conv_layer_strides) == N_conv_layers, "Double-check the entries of 'conv_layer_strides'."
assert len(conv_layer_paddings) == N_conv_layers, "Double-check the entries of 'conv_layer_paddings'."

# max_pool-related parameters
max_pool_layer_numbers =  [1,1]#network_parameters['max_pool_layer_numbers']#[1,1]#[0,1,0,0,0,1,0,1,0,1,0] # layers with maxpool (boolean mask)
max_pool_kernel_sizes =   [2,2]#network_parameters['max_pool_kernel_sizes']#[2,2]#[4,4,4,4,4,4,4,4,4,4,4] # maxpool kernel sizes corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
max_pool_strides =        [2,2]#network_parameters['max_pool_strides']#[2,2]#[2,1,2,2,2,2,2,1,2,1,1] # maxpool kernel strides corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
max_pool_paddings =       [0,0]#network_parameters['max_pool_paddings']#[0,0]#[1,1,2,2,1,2,1,2,2,1,2] # maxpool paddings corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
assert len(max_pool_layer_numbers) == N_conv_layers, "Double-check the maximum number in 'max_pool_layer_numbers'."
assert len(max_pool_kernel_sizes) == N_conv_layers, "Double-check the entries of 'max_pool_kernel_sizes'."
assert len(max_pool_strides) == N_conv_layers, "Double-check the entries of 'max_pool_kernel_strides'."
assert len(max_pool_paddings) == N_conv_layers, "Double-check the entries of 'max_pool_paddings'."

# batchnorm2d-related parameters
batch_norm_2d_layer_numbers = [0,0]#network_parameters['batch_norm_2d_layer_numbers']#[0,0]#[1,1,1,0,1,1,1,0,0,1,0] #[2,5,8,10]
assert len(batch_norm_2d_layer_numbers) == N_conv_layers, "Double-check the entries of 'batch_norm_2d_layer_numbers'."

dropout2d_layer_numbers = [0,0]
dropout2d_probabilities = [0.2,0.2]
assert len(dropout2d_layer_numbers) == N_conv_layers, "Double-check the entries of 'dropout1d_layer_numbers'."
assert len(dropout2d_probabilities) == N_conv_layers, "Double-check the entries of 'dropout1d_probabilities'."

# linear-layer-related parameters
# --------- for FashionMNIST ------------
n_classes_ = 10 #10 # was 128 before
# ---------------------------------------
linear_layer_out_features = [12*4*4,#network_parameters['linear_layer_out_features']#[12*4*4,#conv_layer_out_channels[-1],
                            120,60,
                            n_classes_]
N_linear_layers = len(linear_layer_out_features) # number of linear layers
assert len(linear_layer_out_features) == N_linear_layers, "Double-check the entries of 'linear_layer_out_features'."

total_number_of_layers = N_conv_layers + N_linear_layers # total number of layers

# batchnorm1d-related parameters
batch_norm_1d_layer_numbers = [1,0,1,0]#network_parameters['batch_norm_1d_layer_numbers']#[0,0,0,0]#,0,0,0]#[1,0,1,0]#[3] # later need to distinguish b/w BatchNorm1d and BatchNorm2d (b/w conv and linear layers)
dropout1d_layer_numbers = [0,0,0,0]
dropout1d_probabilities = [0.2,0.1,0.3,0.1]
assert len(batch_norm_1d_layer_numbers) == N_linear_layers, "Double-check the entries of 'batch_norm_layer_numbers'."
assert len(dropout1d_layer_numbers) == N_linear_layers, "Double-check the entries of 'dropout1d_layer_numbers'."
assert len(dropout1d_probabilities) == N_linear_layers, "Double-check the entries of 'dropout1d_probabilities'."

conv_layer_activations = ['lrelu','relu']
linear_layer_activations = ['relu','relu','None',''] # the last element is never used



# Netowrk architecture (auxiliary function, classes + network itself)
def conv_block(in_f, out_f, ks, s, p, mp, mp_ks, mp_s, mp_p, bn, dp, dp_prob,
               activation = 'relu',#act, 
               *args, **kwargs):    
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['sigmoid', nn.Sigmoid()],
                ['tanh', nn.Tanh()]
    ])
    
    cb_modules = []
    cb_modules.append(nn.Conv2d(in_f, out_f, ks, s, p, *args, **kwargs))
    if mp == 1:
        cb_modules.append(nn.MaxPool2d(mp_ks, mp_s, mp_p, *args, **kwargs))
    if bn == 1:
        cb_modules.append(nn.BatchNorm2d(out_f)) 
    if activation != 'None':
        cb_modules.append(activations[activation])
    if dp == 1:
        cb_modules.append(nn.Dropout2d(dp_prob))
    cb = nn.Sequential(*cb_modules)    
    return cb



def fc_block(in_f, out_f, bn, dp, dp_prob,
             activation = 'relu'#act
             ):
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['sigmoid', nn.Sigmoid()],
                ['tanh', nn.Tanh()]
    ])
    fb_modules = []
    fb_modules.append(nn.Linear(in_f, out_f))
    if bn == 1:
        fb_modules.append(nn.BatchNorm1d(out_f))
    if activation != 'None':
        fb_modules.append(activations[activation])
    if dp == 1:
        fb_modules.append(nn.Dropout(dp_prob))
    fb = nn.Sequential(*fb_modules) 
    return fb


class MyEncoder(nn.Module):
    def __init__(self, out_channels, 
                 cb_kernel_sizes, 
                 cb_strides, 
                 cb_paddings, 
                 mp_layers, 
                 mp_kernel_sizes, 
                 mp_strides, 
                 mp_paddings, 
                 batchnorms2d,
                 conv_activations,
                 dropouts2d,
                 dropout2d_probs
                 ):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, ks, s, p, mp, mp_ks, mp_s, mp_p, bn, dp, dp_prob, activation = act)# kernel_size=3, padding=1) 
                       for in_f, out_f, ks, s, p, mp, mp_ks, mp_s, mp_p, bn, dp, dp_prob, act in zip(out_channels, out_channels[1:], cb_kernel_sizes, cb_strides, cb_paddings, mp_layers, mp_kernel_sizes, mp_strides, mp_paddings, batchnorms2d, dropouts2d, dropout2d_probs, conv_activations)])

    def forward(self, x):
        return self.conv_blocks(x)    
    
  
        
class MyDecoder(nn.Module):
    def __init__(self, out_features, 
                 batchnorms1d,
                 fc_activations,
                 dropouts1d,
                 dropout1d_probs
                 ):
        super().__init__()
        self.dropouts1d = dropouts1d
        self.dropout1d_probs = dropout1d_probs
        self.dropout1d = nn.Dropout(dropout1d_probs[0])
        self.fc_blocks = nn.Sequential(*[fc_block(in_f, out_f, bn, dp, dp_prob, activation = act) 
                       for in_f, out_f, bn, dp, dp_prob, act in zip(out_features, out_features[1:], batchnorms1d, dropouts1d[1:], dropout1d_probs[1:], fc_activations)])
        

    def forward(self, x):
        if self.dropouts1d[0]:
            x = self.dropout1d(x)
        x = self.fc_blocks(x)
        return x

    
# this class is actually used for training the network
class Network3(nn.Module):
    def __init__(self, in_c, 
                 out_channels, 
                 cb_kernel_sizes, 
                 cb_strides, 
                 cb_paddings, 
                 mp_layers, 
                 mp_kernel_sizes, 
                 mp_strides, 
                 mp_paddings, 
                 batchnorms2d,
                 dropouts2d,
                 dropout2d_probs,
                 out_features, 
                 batchnorms1d,
                 conv_activations,
                 fc_activations,
                 dropouts1d,
                 dropout1d_probs
                 ):
        super().__init__()
        self.out_channels = [in_c, *out_channels]
        self.cb_kernel_sizes = cb_kernel_sizes
        self.cb_strides = cb_strides
        self.cb_paddings = cb_paddings
        self.mp_layers = mp_layers
        self.mp_kernel_sizes = mp_kernel_sizes
        self.mp_strides = mp_strides
        self.mp_paddings = mp_paddings
        self.batchnorms2d = batchnorms2d
        self.dropouts2d = dropouts2d
        self.dropout2d_probs = dropout2d_probs
        self.out_features = [*out_features]
        self.batchnorms1d = batchnorms1d
        self.conv_activations = conv_activations
        self.fc_activations = fc_activations
        self.dropouts1d = dropouts1d
        self.dropout1d_probs = dropout1d_probs
        self.encoder = MyEncoder(self.out_channels, 
                                 self.cb_kernel_sizes, 
                                 self.cb_strides, 
                                 self.cb_paddings, 
                                 self.mp_layers, 
                                 self.mp_kernel_sizes, 
                                 self.mp_strides, 
                                 self.mp_paddings, 
                                 self.batchnorms2d,
                                 self.conv_activations,
                                 self.dropouts2d,
                                 self.dropout2d_probs
                                 )
        self.decoder = MyDecoder(self.out_features,
                                 self.batchnorms1d,
                                 self.fc_activations,
                                 self.dropouts1d,
                                 self.dropout1d_probs
                                 )
        
    def forward(self, x):
        # print('1:',np.shape(x))
        x = self.encoder(x)    
        # print('2:',np.shape(x))
        # print(self.out_features[0])
        x = x.reshape(-1, self.out_features[0])
        # print('3:',np.shape(x))
        x = self.decoder(x)
        # print('4:',np.shape(x))
        return x
    

network3 = Network3(input_depth, conv_layer_out_channels, 
                    conv_layer_kernel_sizes, 
                    conv_layer_strides, 
                    conv_layer_paddings, 
                    max_pool_layer_numbers, 
                    max_pool_kernel_sizes, 
                    max_pool_strides, 
                    max_pool_paddings, 
                    batch_norm_2d_layer_numbers,
                    dropout2d_layer_numbers,
                    dropout2d_probabilities, 
                    linear_layer_out_features,                    
                    batch_norm_1d_layer_numbers,
                    conv_layer_activations,
                    linear_layer_activations,
                    dropout1d_layer_numbers,
                    dropout1d_probabilities
                    )

network3.eval()

# network_parameters = {"conv_layer_out_channels": conv_layer_out_channels,
#                       "conv_layer_kernel_sizes": conv_layer_kernel_sizes,
#                       "conv_layer_strides": conv_layer_strides,
#                       "conv_layer_paddings": conv_layer_paddings,
#                       "max_pool_layer_numbers": max_pool_layer_numbers,
#                       "max_pool_kernel_sizes": max_pool_kernel_sizes,
#                       "max_pool_strides": max_pool_strides,
#                       "max_pool_paddings": max_pool_paddings,
#                       "batch_norm_2d_layer_numbers": batch_norm_2d_layer_numbers,
#                       "dropout2d_layer_numbers": dropout2d_layer_numbers,
#                       "dropout2d_probabilities": dropout2d_probabilities,
#                       "linear_layer_out_features": linear_layer_out_features,
#                       "batch_norm_1d_layer_numbers": batch_norm_1d_layer_numbers,
#                       "dropout1d_layer_numbers": dropout1d_layer_numbers,
#                       "dropout1d_probabilities": dropout1d_probabilities
#                       }

# network_loaded_old = network3
# network_loaded_new = network3

# mmm1 = RunManager('test', 'test')

# dat = mmm1.save_network_params(network3, 'comment')
# mmm1.load_network_params_2( date_=dat, model=network_loaded_old, run=0, epoch=0)
# network_loaded_old.eval()

# mmm2 = RunManager('test', 'test')

# dat = mmm2.save_network(network3, network_parameters, input_depth, conv_layer_activations, linear_layer_activations, 'test_save')
# network_loaded_new = mmm2.load_network(dat, 0, 0)
# network_loaded_new.eval()




# tnsr = torch.zeros([3,1,15,15], dtype=torch.int32)

# print(network3)
# print(network_loaded_old)
# print(network_loaded_new)
# print(network3 == network_loaded_new)
# print(network_loaded_old == network_loaded_new)
# print(network3 == network_loaded_old)



show_gpu_memory()

#%% new cell (defining a custom dataset: FROGDataset)

# custom dataset (class FROGDataset)
class FROGDataset(Dataset):
    def __init__(self, images, n, n_cut, labels_=None, transforms_=None):
        self.X = images
        self.y = labels_
        self.n = n
        self.n_cut = n_cut
        self.transforms = transforms_
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :]
#         print(data.shape)
        data = np.asarray(data).astype(np.float).reshape(1,self.n,self.n)
        
        if self.transforms:
            data = np.asarray(self.transforms(data)).astype(np.float).reshape(1,self.n,self.n)
            
        if self.y is not None:
            y = self.y.iloc[i,:]
#             y = np.asarray(y).astype(np.float).reshape(2*n+1,) # for 257-vector of labels
            y = np.asarray(y).astype(np.float).reshape(self.n_cut,)# + 2*np.pi # for 128-vector of labels
            return (data, y)
        else:
            return data
        

# custom dataset (class FROGDataset)
class AutocorrDataset(Dataset):
    def __init__(self, images, labels_=None, transforms_=None, n_channels = None):
        self.X = images
        self.y = labels_
        self.n_x = np.shape(images)[-1]
        self.n_y = np.shape(labels_)[-1]
        self.transforms = transforms_
        self.n_channels = n_channels
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = np.asarray(self.X[i]).astype(np.float).reshape(self.n_channels,self.n_x,self.n_x)
        
        if self.transforms:
            data = np.asarray(self.transforms(data)).astype(np.float).reshape(self.n_channels,self.n_x,self.n_x)
            
        if self.y is not None:
            y = np.asarray(self.y[i]).astype(np.float).reshape(self.n_y,)
            return (data, y)
        else:
            return data
        

def importDimers(root,folder,image_depth,image_size):
    dataset_path = os.path.join(root,folder)
    folders = np.transpose(next(os.walk(dataset_path))[1])
    labels_indices_ = [6,7,8]#,17,18,19,20,21,22]
    N_labels = len(labels_indices_)
    N = len(folders)
    images__ = np.empty((N,image_depth,image_size,image_size))
    labels__ = np.empty((N,N_labels))
    labels_buf_ = np.empty(N_labels)
    
    for i,folder in zip(range(N),folders):
        current_path = os.path.join(dataset_path,folder)
        current_image_Re = np.genfromtxt(os.path.join(current_path,"twod_re.cvs"),delimiter = ",")
        current_image_Im = np.genfromtxt(os.path.join(current_path,"twod_im.cvs"),delimiter = ",")
        ReIm_combined = [current_image_Re,current_image_Im]
        current_label = np.genfromtxt(os.path.join(current_path,"params.cvs"))
        images__[i] = np.asarray(ReIm_combined)
        for j,k in zip(labels_indices_,range(N_labels)):
            labels_buf_[k] = current_label[j]
        labels__[i] = labels_buf_
        labels_buf_ = np.empty(N_labels)
    # normalization of labels
    labels__[:,0] = labels__[:,0] - 11850# - 12000 + 75)/75
    labels__[:,1] = labels__[:,1] - 11850# - 12000 - 75)/75
    labels__[:,2] = labels__[:,2] + 685#/685
    # labels__[:,3:6] = (labels__[:,3:6])/5
    # labels__[:,6:9] = (labels__[:,6:9])/10
    print(np.shape(labels__))
    for i in range(len(labels_indices_)):
        print(str(i)+':',np.max(labels__[:,i]))
        print(str(i)+':',np.min(labels__[:,i]))
    
    return images__,labels__

def importAutocorrTraces(root,image_depth,image_size,N_labels):
    dataset_path = os.path.join(root)
    folders = np.transpose(next(os.walk(dataset_path))[1])
    N = len(folders)
    images_ = np.empty((N,image_depth,image_size,image_size))
    labels__ = np.empty((N,N_labels))
    
    for i,folder in zip(range(N),folders):#zip(range(1),folders[0]):#for debugging
        current_path = os.path.join(dataset_path,folder)
        # inputs
        current_S_1 = np.genfromtxt(os.path.join(current_path,"freq_domain_data_IAs.txt"))[1250-800:1250+800,1] # spectra_1_
        current_S_2 = np.genfromtxt(os.path.join(current_path,"freq_domain_data_IAs.txt"))[1250-800:1250+800,3] # spectra_2_
        current_IA1_1 = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs.txt"))[:,7] # interf_autocorrelations_1st_order
        current_IA2_1 = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs.txt"))[:,8] # interf_autocorrelations_2nd_order
        current_IC1_12 = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs.txt"))[:,10] # interf_crosscorrelations_1st_order
        current_IC2_12 = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs.txt"))[:,11] # interf_crosscorrelations_2nd_order
        
        current_IC1_sio2_2 = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs_sio2.txt"))[:,7] # interf_crosscorrelations_1_sio2_2_
        current_IC2_sio2_2 = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs_sio2.txt"))[:,12] # interf_crosscorrelations_2_sio2_2_
        
        # current_P = np.genfromtxt(os.path.join(current_path,"freq_domain_data_IAs.txt"))[1250-200:1250+200,4]
        current_Et = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs.txt"))[800-500:800+500,5] # amplitudes_2_
        current_phit = np.genfromtxt(os.path.join(current_path,"time_domain_data_IAs.txt"))[800-450:800+450,6] # temporal_phases_2_
        current_field = [*current_Et,*current_phit]
        
        # print(np.shape(current_S_1))
        # print(np.shape(current_S_2))
        # print(np.shape(current_IA1_1))
        # print(np.shape(current_IA2_1))
        # print(np.shape(current_IC1_12))
        # print(np.shape(current_IC2_12))
        
        # plt.figure()
        # plt.plot(current_S_1)
        # plt.figure()
        # plt.plot(current_S_2)
        # plt.figure()
        # plt.plot(current_IA1_1)
        # plt.figure()
        # plt.plot(current_IA2_1)
        # plt.figure()
        # plt.plot(current_IC1_12)
        # plt.figure()
        # plt.plot(current_IC2_12)
        # plt.figure()
        # plt.plot(current_P)
        
        
        
        # current_S_1 = np.reshape([*np.zeros(192),*current_S_1,*np.zeros(192)],(1,40,40))
        # current_S_2 = np.reshape([*np.zeros(192),*current_S_2,*np.zeros(192)],(1,40,40))
        current_S_1 = np.reshape(current_S_1,(1,40,40))
        current_S_2 = np.reshape(current_S_2,(1,40,40))
        current_IA1_1 = np.reshape(current_IA1_1[:],(1,40,40))
        current_IA2_1 = np.reshape(current_IA2_1[:],(1,40,40))
        current_IC1_12 = np.reshape(current_IC1_12[:],(1,40,40)) 
        current_IC2_12 = np.reshape(current_IC2_12[:],(1,40,40))
        current_IC1_sio2_2 = np.reshape(current_IC1_sio2_2[:],(1,40,40)) 
        current_IC2_sio2_2 = np.reshape(current_IC2_sio2_2[:],(1,40,40))
        # current_Et = np.asarray(np.reshape(current_Et[:],(1,40,40)))
        
        # image_combined = np.concatenate((current_S_1, current_S_2, current_IA1_1, current_IA2_1, current_IC1_12, current_IC2_12, current_IC1_12_sio2, current_IC2_12_sio2))
        # image_combined = np.concatenate((current_IA1_1, current_IA2_1, current_IC1_12, current_IC2_12, current_IC1_12_sio2, current_IC2_12_sio2))
        # image_combined = np.concatenate((current_S_1, current_S_2, current_IA2_1, current_IC2_12, current_IC2_sio2_2))
        image_combined = np.concatenate((current_S_1, current_S_2, current_IC2_12, current_IC2_sio2_2))
        # labels
        
        # print("np.shape(current_P)",np.shape(current_P))
        images_[i] = np.asarray(image_combined)
        # images_[i] = np.asarray(np.reshape(current_Et,(1,30,30)))
        
        
        # labels__[i] = current_P
        labels__[i] = current_Et#current_field
        # labels__[i] = current_phit#current_phase
    
    return images_,labels__


def importAutocorrTracesNoisy(root,image_depth,image_size,N_labels):
    dataset_path = os.path.join(root)
    folders = np.transpose(next(os.walk(dataset_path))[1])
    N = len(folders)
    images_0_ = np.empty((N,image_depth,image_size,image_size))
    images_1_ = np.empty((N,image_depth,image_size,image_size))
    images_2_ = np.empty((N,image_depth,image_size,image_size))
    images_3_ = np.empty((N,image_depth,image_size,image_size))
    images_4_ = np.empty((N,image_depth,image_size,image_size))
    images_5_ = np.empty((N,image_depth,image_size,image_size))
    labels___ = np.empty((N,N_labels))
    for i,folder in zip(range(N),folders):#zip(range(1),folders[0]):#for debugging
        current_path = os.path.join(dataset_path,folder)
        
        current_S_1 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,0] # spectra_1_
        current_S_2 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,1] # spectra_2_
        current_IC2_12 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,2] # interf_crosscorrelations_2nd_order
        current_IC2_sio2_2 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,3] # interf_crosscorrelations_2_sio2_2_
        current_S_1 = np.reshape(current_S_1,(1,40,40))
        current_S_2 = np.reshape(current_S_2,(1,40,40))
        current_IC2_12 = np.reshape(current_IC2_12[:],(1,40,40))
        current_IC2_sio2_2 = np.reshape(current_IC2_sio2_2[:],(1,40,40))
        
        current_S_1_noisy1 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,4] # spectra_1_
        current_S_2_noisy1 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,5] # spectra_2_
        current_IC2_12_noisy1 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,6] # interf_crosscorrelations_2nd_order
        current_IC2_sio2_2_noisy1 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,7] # interf_crosscorrelations_2_sio2_2_
        current_S_1_noisy1 = np.reshape(current_S_1_noisy1,(1,40,40))
        current_S_2_noisy1 = np.reshape(current_S_2_noisy1,(1,40,40))
        current_IC2_12_noisy1 = np.reshape(current_IC2_12_noisy1[:],(1,40,40))
        current_IC2_sio2_2_noisy1 = np.reshape(current_IC2_sio2_2_noisy1[:],(1,40,40))
        
        current_S_1_noisy2 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,8] # spectra_1_
        current_S_2_noisy2 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,9] # spectra_2_
        current_IC2_12_noisy2 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,10] # interf_crosscorrelations_2nd_order
        current_IC2_sio2_2_noisy2 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,11] # interf_crosscorrelations_2_sio2_2_
        current_S_1_noisy2 = np.reshape(current_S_1_noisy2,(1,40,40))
        current_S_2_noisy2 = np.reshape(current_S_2_noisy2,(1,40,40))
        current_IC2_12_noisy2 = np.reshape(current_IC2_12_noisy2[:],(1,40,40))
        current_IC2_sio2_2_noisy2 = np.reshape(current_IC2_sio2_2_noisy2[:],(1,40,40))
        
        current_S_1_noisy5 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,12] # spectra_1_
        current_S_2_noisy5 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,13] # spectra_2_
        current_IC2_12_noisy5 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,14] # interf_crosscorrelations_2nd_order
        current_IC2_sio2_2_noisy5 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,15] # interf_crosscorrelations_2_sio2_2_
        current_S_1_noisy5 = np.reshape(current_S_1_noisy5,(1,40,40))
        current_S_2_noisy5 = np.reshape(current_S_2_noisy5,(1,40,40))
        current_IC2_12_noisy5 = np.reshape(current_IC2_12_noisy5[:],(1,40,40))
        current_IC2_sio2_2_noisy5 = np.reshape(current_IC2_sio2_2_noisy5[:],(1,40,40))
        
        current_S_1_noisy10 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,16] # spectra_1_
        current_S_2_noisy10 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,17] # spectra_2_
        current_IC2_12_noisy10 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,18] # interf_crosscorrelations_2nd_order
        current_IC2_sio2_2_noisy10 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,19] # interf_crosscorrelations_2_sio2_2_
        current_S_1_noisy10 = np.reshape(current_S_1_noisy10,(1,40,40))
        current_S_2_noisy10 = np.reshape(current_S_2_noisy10,(1,40,40))
        current_IC2_12_noisy10 = np.reshape(current_IC2_12_noisy10[:],(1,40,40))
        current_IC2_sio2_2_noisy10 = np.reshape(current_IC2_sio2_2_noisy10[:],(1,40,40))
        
        current_S_1_noisy20 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,20] # spectra_1_
        current_S_2_noisy20 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,21] # spectra_2_
        current_IC2_12_noisy20 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,22] # interf_crosscorrelations_2nd_order
        current_IC2_sio2_2_noisy20 = np.genfromtxt(os.path.join(current_path,"noisy_input_data.txt"))[:,23] # interf_crosscorrelations_2_sio2_2_
        current_S_1_noisy20 = np.reshape(current_S_1_noisy20,(1,40,40))
        current_S_2_noisy20 = np.reshape(current_S_2_noisy20,(1,40,40))
        current_IC2_12_noisy20 = np.reshape(current_IC2_12_noisy20[:],(1,40,40))
        current_IC2_sio2_2_noisy20 = np.reshape(current_IC2_sio2_2_noisy20[:],(1,40,40))
        
        current_Et = np.genfromtxt(os.path.join(current_path,"label_for_noisy_data.txt")) # amplitudes_2_
        # print(current_Et)
        
        image_combined_0 = np.concatenate((current_S_1, current_S_2, current_IC2_12, current_IC2_sio2_2))
        image_combined_1 = np.concatenate((current_S_1_noisy1, current_S_2_noisy1, current_IC2_12_noisy1, current_IC2_sio2_2_noisy1))
        image_combined_2 = np.concatenate((current_S_1_noisy2, current_S_2_noisy2, current_IC2_12_noisy2, current_IC2_sio2_2_noisy2))
        image_combined_3 = np.concatenate((current_S_1_noisy5, current_S_2_noisy5, current_IC2_12_noisy5, current_IC2_sio2_2_noisy5))
        image_combined_4 = np.concatenate((current_S_1_noisy10, current_S_2_noisy10, current_IC2_12_noisy10, current_IC2_sio2_2_noisy10))
        image_combined_5 = np.concatenate((current_S_1_noisy20, current_S_2_noisy20, current_IC2_12_noisy20, current_IC2_sio2_2_noisy20))
        images_0_[i] = np.asarray(image_combined_0)
        images_1_[i] = np.asarray(image_combined_1)
        images_2_[i] = np.asarray(image_combined_2)
        images_3_[i] = np.asarray(image_combined_3)
        images_4_[i] = np.asarray(image_combined_4)
        images_5_[i] = np.asarray(image_combined_5)
        
        labels___[i] = current_Et
        
    return images_0_,images_1_,images_2_,images_3_,images_4_,images_5_,labels___
        
show_gpu_memory()

#%% new cell (importing dimers 2DES data set)
root_folder_autocorr = os.path.join(os.getcwd(),"data","interf_autocorr_crosscorr_123_0_phase_at_centr_freq_plus_sio2_2mm_unconstrained_random_phi_20000samples-2")
Input_depth = 4
images_autocorr_2,labels_autocorr= importAutocorrTraces(root_folder_autocorr,Input_depth,40,1000)

# print(np.shape(images_autocorr_2),np.shape(labels_autocorr))

# plt.plot(images_autocorr_2[9860].reshape(5,-1)[4])
# plt.plot((labels_autocorr[0].reshape(-1,)[:]))

# import noisy data

# # define the data-folder
# images_autocorr_2_noisy0,\
#     images_autocorr_2_noisy1,\
#         images_autocorr_2_noisy2,\
#             images_autocorr_2_noisy5,\
#                 images_autocorr_2_noisy10,\
#                     images_autocorr_2_noisy20,labels_autocorr  = importAutocorrTracesNoisy(root_folder_autocorr,Input_depth,40,1000)
                    
#%% reassign noisy data to images_autocorr_2
# images_autocorr_2 = images_autocorr_2_noisy20

#%%
# from scipy.stats import norm

# x_all = np.linspace(-5, 5, 778) # entire range of x, both in and out of spec
# # mean = 0, stddev = 1, since Z-transform was calculated
# y2 = norm.pdf(x_all,0.135,1.0)
# y2 = y2/np.max(y2)

# fig, ax = plt.subplots(figsize=(9,6))
# ax.plot(y2)
# plt.show()

# autocorr_mod = (images_autocorr_2[0].reshape(34*34,)[378:]-1)/y2

# plt.plot(images_autocorr_2[0].reshape(34*34,)[378:], linewidth = 1)
# plt.figure()
# plt.plot(autocorr_mod[:], linewidth = 1)

#%% splitting the 2des dimer dataset into train dev and test
n_train = 19000
n_dev = 600
n_test = 400

images_autocorr_2_train = images_autocorr_2[0:n_train]
images_autocorr_2_dev = images_autocorr_2[n_train:n_train+n_dev]
images_autocorr_2_test = images_autocorr_2[n_train+n_dev:n_train+n_dev+n_test]

# # normalizing labels
# labels_autocorr_norm = labels_autocorr
# labels_autocorr_norm = (labels_autocorr_norm - np.mean(labels_autocorr_norm))/np.std(labels_autocorr_norm)
# for i in range(20000):
#     labels_autocorr_norm[i] = (labels_autocorr_norm[i] - labels_autocorr_norm[i][459])
# #     labels_autocorr_norm[i] = ((labels_autocorr_norm[i] + np.pi) % (2 * np.pi) - np.pi)/np.pi # pahse-wrap

labels_autocorr_train = labels_autocorr[0:n_train]#,800-500:800+500]
labels_autocorr_dev = labels_autocorr[n_train:n_train+n_dev]#,800-500:800+500]
labels_autocorr_test = labels_autocorr[n_train+n_dev:n_train+n_dev+n_test]#,800-500:800+500]

print(np.shape(images_autocorr_2_train))
print(np.shape(images_autocorr_2_dev))
print(np.shape(images_autocorr_2_test))

print(np.shape(labels_autocorr_train))
print(np.shape(labels_autocorr_dev))
print(np.shape(labels_autocorr_test))

#%% testing class Dimmers2DESDataset(Dataset)

autocorr_2_data_all = AutocorrDataset(images_autocorr_2, labels_autocorr, n_channels=Input_depth)
autocorr_2_data_train = AutocorrDataset(images_autocorr_2_train, labels_autocorr_train, n_channels=Input_depth)
autocorr_2_data_dev = AutocorrDataset(images_autocorr_2_dev, labels_autocorr_dev, n_channels=Input_depth)
autocorr_2_data_test = AutocorrDataset(images_autocorr_2_test, labels_autocorr_test, n_channels=Input_depth)

sample_number = 78
figure, ax1 = plt.subplots()
ax1.plot(autocorr_2_data_all.__getitem__(sample_number)[1][:], color=u'#1f77b4', label='Spectrum')
ax1.set_ylabel('Spectrum')
plt.figure()
grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((Input_depth,40, 40))[0].T
plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
plt.figure()
grid = autocorr_2_data_all.__getitem__(sample_number)[0][3]
plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
plt.figure()
plt.plot(autocorr_2_data_all.__getitem__(sample_number)[0][3].reshape((-1)))
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((Input_depth,30, 30))[1].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((Input_depth,30, 30))[2].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((Input_depth,30, 30))[3].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((Input_depth,30, 30))[4].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((Input_depth,40, 40))[5].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((6,40, 40))[6].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
# plt.figure()
# grid = autocorr_2_data_all.__getitem__(sample_number)[0][:].reshape((6,40, 40))[7].T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)

print(autocorr_2_data_all.__getitem__(sample_number)[1][:])

#%% plot some samples from development dataset

sample_numbers = [56,78,204,240,251,291,299,317,319,339,383,394,400,414,437,439,440,445,470,474,476,480,482,485,501,502,505,524,535,538,539,542,546,545,551,565,588]

for sample_number in sample_numbers:
    # figure, ax1 = plt.subplots()
    # ax1.plot(autocorr_2_data_dev.__getitem__(sample_number)[1][:], color=u'#1f77b4', label='Spectrum')
    # ax1.set_ylabel('Spectrum')
    # plt.figure()
    # grid = autocorr_2_data_dev.__getitem__(sample_number)[0][:].reshape((Input_depth,40, 40))[0].T
    # plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
    # plt.figure()
    # grid = autocorr_2_data_dev.__getitem__(sample_number)[0][3]
    # plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
    # plt.figure()
    # plt.plot(autocorr_2_data_dev.__getitem__(sample_number)[0][1].reshape((-1)))
    plt.figure()
    plt.plot(autocorr_2_data_dev.__getitem__(sample_number)[1].reshape((-1)))
    plt.title(sample_number)

#%% normalizing data 2des dimers

AutocorrLoader_2 = DataLoader(autocorr_2_data_all, batch_size=len(autocorr_2_data_all), num_workers=0)
data = next(iter(AutocorrLoader_2))
mean_value = data[0].mean() # mean and std can be performed on float, not on Byte
std_value = data[0].std()

print('Mean and STD of overall dataset:',mean_value,std_value)

transform = transforms.Compose(
    [transforms.ToTensor(),
      transforms.Normalize((mean_value,), (std_value,))
])

autocorr_2_data_train_normal = AutocorrDataset(images_autocorr_2_train, labels_autocorr_train, transform, n_channels=Input_depth)
autocorr_2_data_dev_normal = AutocorrDataset(images_autocorr_2_dev, labels_autocorr_dev, transform, n_channels=Input_depth)
autocorr_2_data_test_normal = AutocorrDataset(images_autocorr_2_test, labels_autocorr_test, transform, n_channels=Input_depth)
autocorr_2_data_all_normal = AutocorrDataset(images_autocorr_2, labels_autocorr, transform, n_channels=Input_depth)

AutocorrLoader_2_normal = DataLoader(autocorr_2_data_all_normal, batch_size=len(autocorr_2_data_all_normal), num_workers=0)
data = next(iter(AutocorrLoader_2_normal))
mean_value = data[0].mean() # mean and std can be performed on float, not on Byte
std_value = data[0].std()

print('Mean and STD of normalized overall dataset:',mean_value,std_value)

TrainAutocorrLoader_2_normal = DataLoader(autocorr_2_data_train_normal, batch_size=len(autocorr_2_data_train_normal), num_workers=0)
data = next(iter(TrainAutocorrLoader_2_normal))
mean_value = data[0].mean() # mean and std can be performed on float, not on Byte
std_value = data[0].std()

print('Mean and STD of normalized train dataset:',mean_value,std_value)

#%% new cell (importing FROG data set)
# # importing FROG data set (for FROG)

# images_dataset_file = os.path.join("frog_data","2021031_PG_FROG","frog.csv")
# labels_dataset_file = os.path.join("frog_data","2021031_PG_FROG","ef.csv")

# frogtrace = pd.read_csv(os.path.join(
#     os.getcwd(), 
#     "data",
#     images_dataset_file))
# ef = pd.read_csv(os.path.join(
#     os.getcwd(), 
#     "data",
#     labels_dataset_file)) # ef

# print(np.shape(frogtrace))
# print(np.shape(ef))

# show_gpu_memory()

# ## new cell
# # processing imported FROG dataset

# # if labels are 128-vectors
# n = ef.shape[1]

# n_train = 19500
# n_dev = 300
# n_test = 200

# ind = [*range(0,n)] # for 128-vector of labels
# ind_cut = [*range(40,89)]
# n_cut = len(ind_cut)

# frog_images_all = frogtrace.iloc[:, :]
# train_frog_images = frogtrace.iloc[0:n_train, :]
# dev_frog_images = frogtrace.iloc[n_train:n_train+n_dev, :]
# test_frog_images = frogtrace.iloc[n_train+n_dev:n_train+n_dev+n_test, :]
# frog_labels_all = ef.iloc[:, ind]
# train_frog_labels = ef.iloc[0:n_train, ind] # frogtrace.iloc[:, 0] # ef.iloc[0, n:2*n]
# dev_frog_labels = ef.iloc[n_train:n_train+n_dev, ind]
# test_frog_labels = ef.iloc[n_train+n_dev:n_train+n_dev+n_test, ind]

# frog_labels_all_cut = ef.iloc[:, ind_cut]
# train_frog_labels_cut = ef.iloc[0:n_train, ind_cut] # frogtrace.iloc[:, 0] # ef.iloc[0, n:2*n]
# dev_frog_labels_cut = ef.iloc[n_train:n_train+n_dev, ind_cut]
# test_frog_labels_cut = ef.iloc[n_train+n_dev:n_train+n_dev+n_test, ind_cut]

# print('Overall-data images shape:', str(frog_images_all.shape))
# print('Train-set images shape:', str(train_frog_images.shape))
# print('Dev-set images shape:', str(dev_frog_images.shape))
# print('Test-set images shape:', str(test_frog_images.shape))
# print()
# print('Overall-set labels shape:', str(frog_labels_all.shape))
# print('Train-set labels shape:', str(train_frog_labels.shape))
# print('Dev-set labels shape:', str(dev_frog_labels.shape))
# print('Test-set labels shape:', str(test_frog_labels.shape))
# print()
# print('Overall-set labels shape (cut):', str(frog_labels_all_cut.shape))
# print('Train-set labels shape (cut):', str(train_frog_labels_cut.shape))
# print('Dev-set labels shape (cut):', str(dev_frog_labels_cut.shape))
# print('Test-set labels shape (cut):', str(test_frog_labels_cut.shape))
# # print(train_frog_images.max().max())

# train_frog_data = FROGDataset(train_frog_images, n, n, train_frog_labels, None)
# dev_frog_data = FROGDataset(dev_frog_images, n, n, dev_frog_labels, None)
# test_frog_data = FROGDataset(test_frog_images, n, n, test_frog_labels, None)
# frog_data_all = FROGDataset(frog_images_all, n, n, frog_labels_all, None)

# train_frog_data_cut = FROGDataset(train_frog_images, n, n_cut, train_frog_labels_cut, None)
# dev_frog_data_cut = FROGDataset(dev_frog_images, n, n_cut, dev_frog_labels_cut, None)
# test_frog_data_cut = FROGDataset(test_frog_images, n, n_cut, test_frog_labels_cut, None)
# frog_data_all_cut = FROGDataset(frog_images_all, n, n_cut, frog_labels_all_cut, None)

#%% new cell (plot an image (for FROG))
# # plot an image (for FROG)

# sample_number = 2
# figure, ax1 = plt.subplots()
# ax1.plot(train_frog_data_cut.__getitem__(sample_number)[1][0+1:n+1], color=u'#1f77b4', label='Spectrum')
# ax2 = ax1.twinx()
# ax2.plot(train_frog_data_cut.__getitem__(sample_number)[1][n+1:2*n+1], color=u'#ff7f0e', label='Phase')
# ax1.set_ylabel('Spectrum')
# ax2.set_ylabel('Phase')
# plt.figure()
# grid = train_frog_data_cut.__getitem__(sample_number)[0][:].reshape((n, n)).T
# plt.imshow(grid,interpolation='nearest', cmap=modified_jet)


# print(np.max(frog_data_all_cut.__getitem__(sample_number)[0][:].reshape((n, n)).T))

#%% new cell (Normalizing FROG data)
# # Normalizing data

# AllFROGLoader = DataLoader(frog_data_all, batch_size=len(frog_data_all), num_workers=0)
# data = next(iter(AllFROGLoader))
# mean_value = data[0].mean() # mean and std can be performed on float, not on Byte
# std_value = data[0].std()

# print('Mean and STD of overall dataset:',mean_value,std_value)

# AllFROGLoader_cut = DataLoader(frog_data_all_cut, batch_size=len(frog_data_all_cut), num_workers=0)
# data_cut = next(iter(AllFROGLoader_cut))
# mean_value_cut = data_cut[0].mean() # mean and std can be performed on float, not on Byte
# std_value_cut = data_cut[0].std()

# print('Mean and STD of overall labels-cut dataset:',mean_value_cut,std_value_cut)

# # define transforms for FROG data
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#       transforms.Normalize((mean_value,), (std_value,))
# ])

# train_frog_data_normal = FROGDataset(train_frog_images, n, n, train_frog_labels, transform)
# dev_frog_data_normal = FROGDataset(dev_frog_images, n, n, dev_frog_labels, transform)
# test_frog_data_normal = FROGDataset(test_frog_images, n, n, test_frog_labels, transform)
# frog_data_all_normal = FROGDataset(frog_images_all, n, n, frog_labels_all, transform)

# transform_cut = transforms.Compose(
#     [transforms.ToTensor(),
#       transforms.Normalize((mean_value_cut,), (std_value_cut,))
# ])

# train_frog_data_normal_cut = FROGDataset(train_frog_images, n, n_cut, train_frog_labels_cut, transform_cut)
# dev_frog_data_normal_cut = FROGDataset(dev_frog_images, n, n_cut, dev_frog_labels_cut, transform_cut)
# test_frog_data_normal_cut = FROGDataset(test_frog_images, n, n_cut, test_frog_labels_cut, transform_cut)
# frog_data_all_normal_cut = FROGDataset(frog_images_all, n, n_cut, frog_labels_all_cut, transform_cut)

# print(np.max(frog_data_all_cut.__getitem__(sample_number)[0][:].reshape((n, n)).T))
# print(np.max(frog_data_all_normal_cut.__getitem__(sample_number)[0][:].reshape((n, n)).T))

# AllFROGLoader_normal_cut = DataLoader(frog_data_all_normal_cut, batch_size=len(frog_data_all_normal_cut), num_workers=0)
# data_normal_cut = next(iter(AllFROGLoader_normal_cut))
# mean_value_normal_cut = data_normal_cut[0].mean() # mean and std can be performed on float, not on Byte
# std_value_normal_cut = data_normal_cut[0].std()

# print('\nMean and STD of normalised overall labels-cut dataset:\n',mean_value_normal_cut,std_value_normal_cut)

# train_FROGLoader_normal_cut = DataLoader(train_frog_data_normal_cut, batch_size=len(train_frog_data_normal_cut), num_workers=0)
# train_data_normal_cut = next(iter(train_FROGLoader_normal_cut))
# train_mean_value_normal_cut = train_data_normal_cut[0].mean() # mean and std can be performed on float, not on Byte
# train_std_value_normal_cut = train_data_normal_cut[0].std()

# print('\nMean and STD of normalised train labels-cut dataset:\n',train_mean_value_normal_cut,train_std_value_normal_cut)

# test_FROGLoader_normal_cut = DataLoader(test_frog_data_normal_cut, batch_size=len(test_frog_data_normal_cut), num_workers=0)
# test_data_normal_cut = next(iter(test_FROGLoader_normal_cut))
# test_mean_value_normal_cut = test_data_normal_cut[0].mean() # mean and std can be performed on float, not on Byte
# test_std_value_normal_cut = test_data_normal_cut[0].std()

# print('\nMean and STD of normalised test labels-cut dataset:\n',test_mean_value_normal_cut,test_std_value_normal_cut)

# dev_FROGLoader_normal_cut = DataLoader(dev_frog_data_normal_cut, batch_size=len(dev_frog_data_normal_cut), num_workers=0)
# dev_data_normal_cut = next(iter(dev_FROGLoader_normal_cut))
# dev_mean_value_normal_cut = dev_data_normal_cut[0].mean() # mean and std can be performed on float, not on Byte
# dev_std_value_normal_cut = dev_data_normal_cut[0].std()

# print('\nMean and STD of normalised dev labels-cut dataset:\n',dev_mean_value_normal_cut,dev_std_value_normal_cut)

# print(frog_data_all.__getitem__(3)[1][60])
# print(frog_data_all_normal.__getitem__(3)[1][60])
# print(len(dev_frog_data_normal))
# print(np.shape(dev_frog_data_normal))
# print(len(test_frog_data_normal))
# print(np.shape(test_frog_data_normal))

# plt.plot(train_frog_data.__getitem__(3)[1][:])
# plt.plot(train_frog_data_normal.__getitem__(3)[1][:])

# figure
# plt.plot(train_frog_data_cut.__getitem__(3)[1][:])
# plt.plot(train_frog_data_normal_cut.__getitem__(3)[1][:])

# print(train_frog_data_cut.__getitem__(3))
# print(train_frog_data_normal_cut.__getitem__(3)[1][20])

# show_gpu_memory()


#%% new cell (removing unnecessary variables to clear up some memory (for FROG))
# removing unnecessary variables to clear up some memory (for FROG)

# del grid # for FROG
# del test_frog_labels
# del dev_frog_labels
# del train_frog_labels
# del frog_labels_all
# del ef
# del test_frog_images
# del dev_frog_images
# del train_frog_images
# del frogtrace 
# del frog_images_all


#%% new cell (dump cpu-memory and empty cache of cuda)
dump_cpu_memory_info()
torch.cuda.empty_cache()

#%% new cell (training the network, defining main() function)
# Training the network

def main(trial,
         N_conv__,
         N_fc__,
         Input_size__,
         Input_depth__,
         N_classes__,
         N_trials__,
         N_trial_epochs__,
         batch_norm_1d_0_,
         dropout_1d_0_,
         dropout_prob_1d_0_,
         parameter_ranges_,         
         frozen_network_parameters_,
         conv_activation_functions_,
         fc_activation_functions_,
         trainsets_,
         frozen_network_hyperparameters_,
         hyperparameter_ranges_,
         optim_,
         direction__,
         storage__,
         database__,
         study_name__,
         ):
    torch.cuda.empty_cache()
    
    tic = timeit.default_timer()
    
    m = RunManager(database__,study_name__)
    # to handle Ctrl+C when the program is running
    # This is essentially to properly close log-file
    CtrlC_event_handler(m)
    
    if optim_:
        L = ['\n[OPTIMIZATION MODE]']
    else:
        L = ['\n[TRAINING MODE]']
    m.log_file.writelines(str(line) + '\n' for line in L)
    
    # Parameters
    network_parameters,network_hyperparameters = optuna_suggestNetworkArchitecture(trial,m,
                                                            N_conv__,
                                                            N_fc__,
                                                            batch_norm_1d_0_,
                                                            dropout_1d_0_,
                                                            dropout_prob_1d_0_,
                                                            parameter_ranges_,
                                                            hyperparameter_ranges_,
                                                            network_params_frozen=frozen_network_parameters_,
                                                            network_hyperparams_frozen=frozen_network_hyperparameters_,
                                                            conv_activations=conv_activation_functions_,
                                                            fc_activations=fc_activation_functions_,
                                                            input_size = Input_size__,
                                                            n_classes__=N_classes__)
    
    # conv-layer-related parameters
    conv_layer_out_channels = network_parameters['conv_layer_out_channels']#[6,12]# [5,17,26,26,43,58,67,52,68,86,91]
    N_conv_layers = len(conv_layer_out_channels) # number of convolutional layers
    conv_layer_kernel_sizes = network_parameters['conv_layer_kernel_sizes']#[5,5]#[8,6,5,7,8,5,7,4,7,8,6]
    conv_layer_strides =      network_parameters['conv_layer_strides']#[1,1]#[3,2,2,1,3,1,5,2,5,4,2]
    conv_layer_paddings =     network_parameters['conv_layer_paddings']#[0,0]#[0,3,6,5,10,10,7,8,5,5,10]
    assert len(conv_layer_out_channels) == N_conv_layers, "Double-check the entries of 'conv_layer_out_channels'."
    assert len(conv_layer_kernel_sizes) == N_conv_layers, "Double-check the entries of 'conv_layer_kernel_sizes'."
    assert len(conv_layer_strides) == N_conv_layers, "Double-check the entries of 'conv_layer_strides'."
    assert len(conv_layer_paddings) == N_conv_layers, "Double-check the entries of 'conv_layer_paddings'."
    
    # max_pool-related parameters
    max_pool_layer_numbers =  network_parameters['max_pool_layer_numbers']#[1,1]#[0,1,0,0,0,1,0,1,0,1,0] # layers with maxpool (boolean mask)
    max_pool_kernel_sizes =   network_parameters['max_pool_kernel_sizes']#[2,2]#[4,4,4,4,4,4,4,4,4,4,4] # maxpool kernel sizes corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
    max_pool_strides =        network_parameters['max_pool_strides']#[2,2]#[2,1,2,2,2,2,2,1,2,1,1] # maxpool kernel strides corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
    max_pool_paddings =       network_parameters['max_pool_paddings']#[0,0]#[1,1,2,2,1,2,1,2,2,1,2] # maxpool paddings corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
    assert len(max_pool_layer_numbers) == N_conv_layers, "Double-check the maximum number in 'max_pool_layer_numbers'."
    assert len(max_pool_kernel_sizes) == N_conv_layers, "Double-check the entries of 'max_pool_kernel_sizes'."
    assert len(max_pool_strides) == N_conv_layers, "Double-check the entries of 'max_pool_kernel_strides'."
    assert len(max_pool_paddings) == N_conv_layers, "Double-check the entries of 'max_pool_paddings'."
    
    # batchnorm2d-related parameters
    batch_norm_2d_layer_numbers = network_parameters['batch_norm_2d_layer_numbers']#[0,0]#[1,1,1,0,1,1,1,0,0,1,0] #[2,5,8,10]
    assert len(batch_norm_2d_layer_numbers) == N_conv_layers, "Double-check the entries of 'batch_norm_2d_layer_numbers'."
    
    dropout2d_layer_numbers = network_parameters['dropout2d_layer_numbers']
    dropout2d_probabilities = network_parameters['dropout2d_probabilities']
    assert len(dropout2d_layer_numbers) == N_conv_layers, "Double-check the entries of 'dropout2d_layer_numbers'."
    assert len(dropout2d_probabilities) == N_conv_layers, "Double-check the entries of 'dropout2d_probabilities'."
    
    # linear-layer-related parameters
    linear_layer_out_features = network_parameters['linear_layer_out_features']#[12*4*4,#conv_layer_out_channels[-1],
                                
    N_linear_layers = len(linear_layer_out_features) # number of linear layers
    assert len(linear_layer_out_features) == N_linear_layers, "Double-check the entries of 'linear_layer_out_features'."
    
    total_number_of_layers = N_conv_layers + N_linear_layers # total number of layers
    
    # batchnorm1d-related parameters
    batch_norm_1d_layer_numbers = network_parameters['batch_norm_1d_layer_numbers']#[0,0,0,0]#,0,0,0]#[1,0,1,0]#[3] # later need to distinguish b/w BatchNorm1d and BatchNorm2d (b/w conv and linear layers)
    dropout1d_layer_numbers = network_parameters['dropout1d_layer_numbers']
    dropout1d_probabilities = network_parameters['dropout1d_probabilities']
    assert len(batch_norm_1d_layer_numbers) == N_linear_layers, "Double-check the entries of 'batch_norm_layer_numbers'."
    assert len(dropout1d_layer_numbers) == N_linear_layers, "Double-check the entries of 'dropout1d_layer_numbers'."
    assert len(dropout1d_probabilities) == N_linear_layers, "Double-check the entries of 'dropout1d_probabilities'."

    # network1 = nn.Sequential(layers1)    
   # network2 = nn.Sequential(layers2)
    network3 = Network3(Input_depth__, conv_layer_out_channels, 
                    conv_layer_kernel_sizes, 
                    conv_layer_strides, 
                    conv_layer_paddings, 
                    max_pool_layer_numbers, 
                    max_pool_kernel_sizes, 
                    max_pool_strides, 
                    max_pool_paddings, 
                    batch_norm_2d_layer_numbers,
                    dropout2d_layer_numbers,
                    dropout2d_probabilities, 
                    linear_layer_out_features, 
                    batch_norm_1d_layer_numbers,
                    conv_activation_functions_,
                    fc_activation_functions_,
                    dropout1d_layer_numbers,
                    dropout1d_probabilities
                    )
    # network4 = DenseNet(nr_classes=128).float()

    networks = {
        # 'dnet': network4,
    #     'net1': network1,
        #'net2': network2, 
        'net3': network3
    }

    trainsets = trainsets_

    params = OrderedDict(
        btch_sz = network_hyperparameters['batch_size'] # bacth_size
        , lr = network_hyperparameters['learning_rate']#[.001,.01,.1]
        , lr_decay = network_hyperparameters['learning_rate_decay']#['exp','sqrt']#['cnst','exp','sqrt']
        , d_const = network_hyperparameters['learning_rate_decay_const']
        , b1 = network_hyperparameters['b1'] # for Adam-optimization
        , b2 = network_hyperparameters['b2'] # for Adam-optimization
        , n_wkrs = network_hyperparameters['number_of_workers'] # number of workers
        , device = network_hyperparameters['device']  # ['cuda:0', 'cpu'] # device. Note: 'cuda:0' doesnt allow saving, since ':' is not allowed in filenames
        , shuffle = network_hyperparameters['shuffle']#[True,False]
        # ------------- for FashionMNIST -----------------
        , trainset = network_hyperparameters['trainset'] # 'not_norm', was 'norm' before
        # ------------------------------------------------
        , network = list(networks.keys())
    )
    
    N_epochs = network_hyperparameters['number_of_epochs'][0] # overall number of epochs
    
    # this is to try few initializations of network-parameters for the same run-confiuration,
    # to choose the initialization that gives best cost/accuracy.
    N_trials = N_trials__ # how many different weight initializations to try
    N_trial_epochs = N_trial_epochs__ # after how many epochs the cost/accuracy will be recorded and later compared with that from other weight-initialization trials
    trials_accuracies = [0 for i in range(N_trials)]

    runs = RunBuilder.get_runs(params)

    # loss_function = nn.L1Loss() # labels.long()
    loss_function = nn.MSELoss() # labels.float()
    loss_function_dev = nn.MSELoss() # labels.float()
    # loss_function = nn.SmoothL1Loss() # labels.float()
    
    cnt = 0
    
    start_time = time.time()
    for run in runs:

        gc.collect()

        comment = f'-{run}'

        device = torch.device(run.device)
        
        # dump_cpu_memory_info()
        show_gpu_memory()
        
                # this was used before
        loader = DataLoader(trainsets[run.trainset],batch_size=run.btch_sz,shuffle=run.shuffle,num_workers=run.n_wkrs)
        loader_test_all = DataLoader(autocorr_2_data_test_normal,batch_size=400,shuffle=None,num_workers=0,drop_last = True)
        loader_dev_all = DataLoader(autocorr_2_data_dev_normal,batch_size=600,shuffle=None,num_workers=0,drop_last = True)

        # if there is no learning rate decay needed then use constant learning rate
        if run.lr_decay == 'cnst': lr_ = run.lr

    
        # # this is to account for a right normalisation of loss_dev
        # number_of_iters_per_epoch = len(loader.dataset)//run.btch_sz
        # remainder = len(loader.dataset)%run.btch_sz
        # if remainder > 0:
        #     number_of_iters_per_epoch += 1

    #     optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
        subfolder_ = {} # this is to save info about where weights have been saved in order to later load ones giving the best cost/accuracy
        # try to start few training trials and choose the weights that brought the best cost/accuracy
        for tr in range(N_trials+1):
            
            network3 = Network3(Input_depth__, conv_layer_out_channels, 
                        conv_layer_kernel_sizes, 
                        conv_layer_strides, 
                        conv_layer_paddings, 
                        max_pool_layer_numbers, 
                        max_pool_kernel_sizes, 
                        max_pool_strides, 
                        max_pool_paddings, 
                        batch_norm_2d_layer_numbers,
                        dropout2d_layer_numbers,
                        dropout2d_probabilities, 
                        linear_layer_out_features, 
                        batch_norm_1d_layer_numbers,
                        conv_activation_functions_,
                        fc_activation_functions_,
                        dropout1d_layer_numbers,
                        dropout1d_probabilities
                        )
            
            networks = {
                'net3': network3 
            }
    
            network = networks[run.network].float().to(device)
            print(network)
    
            # weight initialisation for relu nonlinearity    
            # if runs.index(run) == 0: nn.init.kaiming_uniform_(list(network.parameters())[0], nonlinearity = 'relu')
        #     if runs.index(run) == 0: nn.init.kaiming_uniform_(list(network[0].weight, nonlinearity = 'relu')
        
            m.begin_run(run, network, loader, loader_dev_all)
            
            if tr == N_trials:
                N_epochs_current = N_epochs
            else:
                N_epochs_current = N_trial_epochs
                
            subfolder_[str(tr)] = m.save_network_params(network,comment)
            if tr == N_trials:
                if tr != 0:
                    max_accuracy_trial_number = np.argmax(trials_accuracies)
                    m.load_network_params_2( date_=subfolder_[str(max_accuracy_trial_number)], model=network, run=1, epoch=0)
            
            print('\ntrial #',tr,'; N_epochs_current =',N_epochs_current)
            # logging into a file
            L = ['\ntrial #'+str(tr)+'; N_epochs_current ='+str(N_epochs_current)]
            m.log_file.writelines(str(line) + '\n' for line in L)
    
            # print("Before epoch")
            for epoch in range(N_epochs_current):
                print("\nEpoch #",epoch+1)
                L = ["\nEpoch #"+str(epoch+1)]
                m.log_file.writelines(str(line) + '\n' for line in L)
    
                gc.collect()
    
                m.begin_epoch()
    
                # if there is a learning rate decay needed then use this equations for decaying rate
                if run.lr_decay == 'exp': lr_ = (run.d_const/10)**epoch * run.lr # param = 0.95
                if run.lr_decay == 'sqrt': lr_ = (run.d_const/10)/np.sqrt(epoch+1) * run.lr # param = 1
    
                optimizer = optim.Adam(network.parameters(), lr=lr_, betas=(run.b1, run.b2), eps=1e-08, amsgrad=False)        
    
                nnn = 0
                for batch in loader: # Get Batch
                    network.train()                
                                    
                    progressBar(nnn, run.btch_sz, n_train)
                    nnn = nnn + 1
                    
                    gc.collect()
                    images = batch[0].float().to(device)
                    labels = batch[1].float().to(device)
    
                    preds = network(images)
                    loss = loss_function(preds, labels.float()) # this was used before
                            
                    cnt = cnt + 1
                    
                    # After we call the backward() method on our loss tensor,
                    # the gradients will be calculated and added to the grad attributes of our network's parameters.
                    # For this reason, we need to zero out these gradients.
                    optimizer.zero_grad()    
                    loss.backward() # Calculate Gradients    
                    optimizer.step() # Update Weights
        #             loss_dev = loss
        
                    network.eval()
                    with torch.no_grad():
                        loss_dev_ = 0
                        accuracy_ = 0
                        # norm_factor = 0
                        for batch_dev_ in loader_dev_all:
                            images_dev_ = batch_dev_[0].float().to(device)#device)
                            labels_dev_ = batch_dev_[1].float().to(device)#device)
                            preds_dev_ = network(images_dev_)
                            loss_ = loss_function_dev(preds_dev_,labels_dev_.float()) # this was used before
                            loss_dev_ += loss_
                        
                        loss_dev__ = loss_dev_.clone()
                        accuracy_ = loss_dev__.cpu()#loss_dev_
                        if tr < N_trials:
                            trials_accuracies[tr] = accuracy_
                            
        
                    m.track_loss(loss,loss_dev_,accuracy_) # called several times if batch_size is less than the data samples
                    # m.track_num_correct(preds,labels)
                    # # m._get_num_correct(preds=preds,labels=labels)
    
                    # print('\n8\n')
        #             dump_cpu_memory_info()
                
                m.end_epoch(plot_=False)
                m.report_best_cost(m.epoch_accuracies[-1], direction__)  
                
                # if tr == N_trials:
                    # trial.report(m.epoch_accuracies[-1]/m.epoch_costs_dev[-1], step=epoch)
                    # if trial.should_prune():
                    #     raise optuna.TrialPruned()
                        
                # if tr == N_trials:
                #     m.save_network_params(network,comment) # save params for each epoch
                
                if epoch == N_epochs_current-1:
                    m.save_network(network, network_parameters, Input_depth__, conv_activation_functions_, fc_activation_functions_, 'test_save')
    
            # if m.best_cost_changed_state(m.epoch_accuracies[-1], direction__): # save only params for the network that outbeats previous best result
                # m.save_network_params(network,comment) # save params for each epoch
            
            
                # if epoch % 9 == 0:    
                    
                #     # m.save_network_params(network,comment) # save param for the last epoch
            
                    
                    
                #     # plot some example of the prediction
                #     fig, (ax1,ax2) = plt.subplots(2,2)
                    
                #     # ax1.plot(autocorr_2_data_dev.__getitem__(5)[0][4].reshape((-1))[350:-350],'-g')
                #     ax1[0].plot(labels_dev_[24].cpu(),'-b')
                #     ax1[0].plot(preds_dev_[24].cpu(),'-r',linewidth=0.5)
                #     # ax3.legend(["Actual parameters","Predicted parameters"])
                #     # ax1.set_ylim([-1,1])
                #     ax1[0].set_xlim([0,1600-600])
                    
                #     # ax2.plot(autocorr_2_data_dev.__getitem__(20)[0][4].reshape((-1))[350:-350],'-g')
                #     ax1[1].plot(labels_dev_[24].cpu(),'-b')
                #     ax1[1].plot(preds_dev_[24].cpu(),'-r',linewidth=0.5)
                #     # ax3.legend(["Actual parameters","Predicted parameters"])
                #     ax1[1].set_ylim([-0.09,0.09])
                #     ax1[1].set_xlim([600-300,700-280])
                    
                #     # ax3.plot(autocorr_2_data_dev.__getitem__(24)[0][4].reshape((-1))[350:-350],'-g')
                #     ax2[0].plot(labels_dev_[24].cpu(),'-b')
                #     ax2[0].plot(preds_dev_[24].cpu(),'-r',linewidth=0.5)
                #     # ax3.legend(["Actual parameters","Predicted parameters"])
                #     ax2[0].set_ylim([-0.04,0.04])
                #     ax2[0].set_xlim([800-200,1000-200])
                    
                #     # ax3.plot(autocorr_2_data_dev.__getitem__(24)[0][4].reshape((-1))[350:-350],'-g')
                #     ax2[1].plot(labels_dev_[24].cpu(),'-b')
                #     ax2[1].plot(preds_dev_[24].cpu(),'-r',linewidth=0.5)
                #     # ax3.legend(["Actual parameters","Predicted parameters"])
                #     ax2[1].set_ylim([0.8,1.1])
                #     ax2[1].set_xlim([775-300,825-300])
                    
                #     plt.pause(0.0000001)
                    
            m.save_run_data(comment)
            m.end_run(plot_=False) 

    end_time = time.time()
    print('Overall duration: ', end_time-start_time, ' sec')
    L = ['Overall duration: '+ str(end_time-start_time)+ ' sec']
    m.log_file.writelines(str(line) + '\n' for line in L)
    m.save('results_all')
    m.log_file.close()
    toc = timeit.default_timer()
    print()
    print('Time elapsed:', toc - tic)
    if optim_:
        return m.epoch_accuracies[-1]#/m.epoch_costs_dev[-1]
    else:
        return network,network_parameters,network_hyperparameters

#%% new cell (user section; if __name__ == '__main__':)

"""
To try:
    - smaller batch size, so the algorithmm would wander more sideways from the convergence point
    - only real part of the images
    - only imag part of the images
    - complex tensor image
    - crop images to 60x60 (to reduce size and see if it affetcs trainig)
    - split the problem into several networks (e.g. one for energies and another for coordinates)
"""

if __name__ == '__main__':                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    multiprocessing.freeze_support()
    # torch.backends.cudnn.benchmark = False
    
    # define train datasets here
    trainsets = {
        # ----------------------for FROG ----------------------
        'not_norm': autocorr_2_data_train # not normalised
        # ,'not_norm_cut': train_frog_data_cut # normalised
        ,'norm': autocorr_2_data_train_normal # normalised
        # ,'norm_cut': train_frog_data_normal_cut
        # -----------------------------------------------------
        # 'mnist': train_set # add comma in the beginning for keeping it and using FROG
    }
    
    Input_size = 40 # define the lateral size of the images (Input_size x Input_size)
    Input_depth = 4 # define the number of input channels
    N_Classes = 1000 # define the output layer's size
    
    N_Conv = 6 # define the total number of convolutional layers here
    N_Fc = 4 # define the total number of fully-connected layers here
    
    # Define frozen network parameters, that is parameters that optuna will not suggest (and which will not be varied)
    frozen_network_parameters = {}
    # In the following lists, the number of elements should be less or equal to the total number N_Conv of conv-layers.
    # If the number of elements is less than N_Conv, optuna suggests the rest.
    # If a suggestion for a specific element is needed, then provide "-1" (without quotes)
    frozen_network_parameters['conv_layer_out_channels'] = [79, 71, 289, 328, -1, -1]#[79, 60, 92, 328, 227, 93]#[6,12]#[12,20]#,20]
    frozen_network_parameters['conv_layer_kernel_sizes'] = [11, 3, 3, 11, -1, -1]#[11, 7, 17, 11, 3, 17]#[5,5]#[9,11]#,11]
    frozen_network_parameters['conv_layer_strides'] = [1, 1, 2, 4, -1, -1]#[1, 4, 2, 4, 1, 1]#[1,1]#[4,1]#,1]
    frozen_network_parameters['conv_layer_paddings'] = [5, 1, 1, 5, -1, -1]#[5, 1, 7, 5, 1, 8]#[0,0]#[2,4]#,4]
    frozen_network_parameters['max_pool_layer_numbers'] = [0, 0, 0, 0, -1, -1]#[0, 0, 0, 0, 0, 0]#[1,1]#[0,0]#,0] 
    frozen_network_parameters['max_pool_kernel_sizes'] = [0, 0, 0, 0, -1, -1]#[0, 0, 0, 0, 0, 0]#[2,2]#[0,0]#,0]
    frozen_network_parameters['max_pool_strides'] = [0, 0, 0, 0, -1, -1]#[0, 0, 0, 0, 0, 0]#[2,2]#[0,0]#,0]
    frozen_network_parameters['max_pool_paddings'] = [0, 0, 0, 0, -1, -1]#[0, 0, 0, 0, 0, 0]#[0,0]#[0,0]#,0]
    frozen_network_parameters['batch_norm_2d_layer_numbers'] = [1,1,1,1,1,1]#[1, 1, 1, 1, 1, 1]#[0,0]#[0,0]#,0]#[0]
    frozen_network_parameters['dropout2d_layer_numbers'] = []
    frozen_network_parameters['dropout2d_probabilities'] = []
    # In the following lists, the maximum number of elements should be 2 less than the total number (N_fc) of fc-layers
    # This is because the first FC-layer and the output FC-layer are excluded.
    # They are excluded because the output FC-layer is defined by N_Classes,
    # and the number of nodes in the first FC-layer depeneds on the conv-part configuration and is calculated automatically
    frozen_network_parameters['linear_layer_out_features'] = [-1, -1]#[1833, 1356]#[120,60]#[113,82]#[113]#,82]#[113]
    frozen_network_parameters['batch_norm_1d_layer_numbers'] = [1, 1]#[0, 0]#[0,0]#[0,0]#[0]#,0]#[0]
    frozen_network_parameters['dropout1d_layer_numbers'] = []
    frozen_network_parameters['dropout1d_probabilities'] = []
    assert len(frozen_network_parameters['linear_layer_out_features'])<=N_Fc-2, "Double-check the entries of 'frozen_network_parameters['linear_layer_out_features']'."
    assert len(frozen_network_parameters['batch_norm_1d_layer_numbers'])<=N_Fc-2, "Double-check the entries of 'frozen_network_parameters['batch_norm_1d_layer_numbers']'."
    assert len(frozen_network_parameters['dropout1d_layer_numbers'])<=N_Fc-2, "Double-check the entries of 'frozen_network_parameters['dropout1d_layer_numbers']'."
    assert len(frozen_network_parameters['dropout1d_probabilities'])<=N_Fc-2, "Double-check the entries of 'frozen_network_parameters['dropout1d_probabilities']'."
    
    # This variable is used only in the optimization mode (so far) and has to have the value of either [0] or [1]
    # The variable indicates if the batch normalization is performed for the firstest FC-layer
    # (the number of nodes for which is automatically calculated)
    batch_norm_1d_0 = [1] # if empty then optuna suggests it
    dropout_1d_0 = [-1]
    dropout_prob_1d_0 = [-1] # if not empty or -1 then it must be float
    
    # Define activation functions for each of the layers here.
    # Use 'None' if no activation function is required.
    # Otherwise, choose from 'relu', 'lrelu', 'sigmoid', and 'tanh'.
    conv_activation_functions = ['lrelu']*N_Conv # the number of elements must be equal to N_Conv
    assert len(conv_activation_functions) == N_Conv, "Double-check the entries of 'conv_activation_functions'."
    fc_activation_functions =  ['lrelu']*(N_Fc-2)+['None',''] #only the first N_Fc-1 elements are actually used    
    assert len(fc_activation_functions) == N_Fc and fc_activation_functions[N_Fc-1] == '', "Double-check the entries of 'fc_activation_functions'."
    
    # define network hyperparameters (suggestable by optuna) here
    # if [-1] or [] is given then optuna suggests the value
    frozen_network_hyperparameters = {}
    frozen_network_hyperparameters['number_of_epochs'] = [70]#[15]
    frozen_network_hyperparameters['batch_size'] = [81]#[50]
    frozen_network_hyperparameters['learning_rate'] = [0.008149612926988513]#[0.12230988254877717]#[0.12230988254877717]#[0.14230988254877717]#[0.12230988254877717]#[0.001]
    frozen_network_hyperparameters['learning_rate_decay'] = ['sqrt']#['sqrt']#['sqrt']#['cnst'] # 'sqrt','exp'\
    frozen_network_hyperparameters['learning_rate_decay_const'] = [2]#[7]
    frozen_network_hyperparameters['b1'] = [0.883345587855772]#[0.9851682680860913]#[0.9851682680860913]#[0.9]
    frozen_network_hyperparameters['b2'] = [0.9996328747283718]#[0.9987385403153125]#[0.9987385403153125]#[0.999]
    
    # define network hyperparameters (not suggestable by optuna) here
    frozen_network_hyperparameters['number_of_workers'] = [0]
    frozen_network_hyperparameters['device'] = ['cuda'] #'cuda','cpu'
    frozen_network_hyperparameters['shuffle'] = [True]
    frozen_network_hyperparameters['trainset'] = ['norm']
    
    
    # Define parameters' ranges (used only in the optimization mode), within which optuna will suggest values
    parameter_ranges = {}
    parameter_ranges['conv_layer_out_channels'] = [15, 500]
    parameter_ranges['conv_layer_kernel_sizes'] = [3, 19]
    parameter_ranges['conv_layer_strides'] = [1,5]
    parameter_ranges['conv_layer_paddings'] = [1,4]
    parameter_ranges['max_pool_layer_numbers'] = [0,1]
    parameter_ranges['max_pool_kernel_sizes'] = [2,10]
    parameter_ranges['max_pool_strides'] = [1,8]
    parameter_ranges['max_pool_paddings'] = [0,10]
    parameter_ranges['batch_norm_2d_layer_numbers'] = [0,1]
    parameter_ranges['dropout2d_layer_numbers'] = [0,1] # first value n_classes//2 is advisable
    parameter_ranges['dropout2d_probabilities'] = [0.,1.] # must be float
    parameter_ranges['linear_layer_out_features'] = [500, 1000] # first value n_classes//2 is advisable
    parameter_ranges['batch_norm_1d_layer_numbers'] = [0,1]
    parameter_ranges['dropout1d_layer_numbers'] = [0,1] # first value n_classes//2 is advisable
    parameter_ranges['dropout1d_probabilities'] = [0.,1.] # must be float
    
    # Define hyperparameters' ranges (used only in the optimization mode), within which optuna will suggest values
    hyperparameter_ranges = {}
    hyperparameter_ranges['batch_size'] = [10,400]
    hyperparameter_ranges['learning_rate'] = [2e-3, 1e-0]
    hyperparameter_ranges['learning_rate_decay'] = [0,2]
    hyperparameter_ranges['learning_rate_decay_const'] = [1,10]
    hyperparameter_ranges['b1'] = [0.81,0.99]
    hyperparameter_ranges['b2'] = [0.9981,0.9999]
    hyperparameter_ranges['number_of_epochs'] = [1,2000]
    
    # Define the number of trials and the number of trial-epochs
    # If N_Trials > 0, then (in both training and optimisation modes and for the same run-configuration) the neutral network
    # initializes itself and trains N_Trials times. Amongst these trials, those parameters are chosen
    # that give best network performance (cost/accuracy) at the final epoch (defined by N_Trial_epochs)
    # The best parameters are then used in training straight away
    N_Trials = 0 # how many different weight initializations to try ("0" means infinite)
    N_Trial_epochs = 1 # after how many epochs the cost/accuracy will be recorded and later compared with that from other weight-initialization trials
    
    # The flag 'optimize_' tells whether it is going to be the process of neural-network
    # tuning (with optuna), or a training the neural network using given network
    # architecture ('frozen_network_parameters' above)
    optimize_ = True # choose b/w the optimization mode or the training mode
    direction_ = "minimize"
    # choose different names if the set of suggestable parameters is changed
    # (this inconvenience could be solved by using the idea from: https://github.com/optuna/optuna/issues/1855)
    database_ = "i-xcorr-0-phi-at-centr-freq-sio2-et-pred-unrestricted-phase-without-2nd-order-IA-20000-2" # a folder in "logs/" with the same name will be created (if doesnt exists)
    study_name_ = 'architecture-search-15-random-phase-optimisation-with-dropout' # a folder in "logs/[database_]/" with the same name will be created (if doesnt exists)
    # if there is an error "The record doesnt exists" then delete folder [study_name_] and run this cell again
    # Alternatively, one can change the value of [study_name_] to a non-existing folder name.
    # This is a minor bug and is something to be improved.
    storage_ = "sqlite:///" + os.path.join(os.getcwd(),"databases",database_+".db")
    
    # study.optimize(main)
    if optimize_:
        print('\n[OPTIMIZATION MODE]\n')
        study = optuna.create_study(study_name=study_name_, \
                                storage=storage_, \
                                direction=direction_,
                                load_if_exists=True)
        study.optimize(lambda trial: main(trial,
                                          N_Conv,
                                          N_Fc-1,
                                          Input_size,
                                          Input_depth,
                                          N_Classes,
                                          N_Trials,
                                          N_Trial_epochs,
                                          batch_norm_1d_0,
                                          dropout_1d_0,
                                          dropout_prob_1d_0,
                                          parameter_ranges,
                                          frozen_network_parameters,
                                          conv_activation_functions,
                                          fc_activation_functions,
                                          trainsets,
                                          frozen_network_hyperparameters,
                                          hyperparameter_ranges,
                                          optimize_,
                                          direction_,
                                          storage_,
                                          database_,
                                          study_name_
                                          ),
                       callbacks = [optuna_optimize_callback(storage_,database_,study_name_)])
    else:
        print('\n[TRAINING MODE]\n')
        trial = []
        tic = timeit.default_timer()
        network,\
            network_params_,\
                network_hyperparams_\
                    = main(trial,
                        N_Conv,
                        N_Fc-1,
                        Input_size,
                        Input_depth,
                        N_Classes,
                        N_Trials,
                        N_Trial_epochs,
                        batch_norm_1d_0,
                        dropout_1d_0,
                        dropout_prob_1d_0,
                        parameter_ranges,
                        frozen_network_parameters,
                        conv_activation_functions,
                        fc_activation_functions,
                        trainsets,
                        frozen_network_hyperparameters,
                        hyperparameter_ranges,
                        optimize_,
                        direction_,
                        storage_,
                        database_,
                        study_name_)
        toc = timeit.default_timer()
        torch.save(network.state_dict(), os.path.join(os.getcwd(),"_r1" + "_e"+str(frozen_network_hyperparameters['number_of_epochs'][0])+".pt"))
        print()
        print('Time elapsed:', toc - tic)
        print() 

#%% new cell (optuna study statistics)  

dtbs = database_
study_nm = study_name_
drctn = direction_
strg = 'sqlite:///' + os.path.join(os.getcwd(),"databases",dtbs+".db")
study = optuna.create_study(study_name=study_nm, \
                            storage=strg, \
                            direction=drctn,
                            load_if_exists=True)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print(str(study.best_trial))

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
print()
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print(df)
# df.to_csv(r'trials_dataframe.xlsx', index = True, header=True)
# print(df.columns)

#%% new cell (visualisation (optuna))
optuna.visualization.is_available()
fig1 = optuna.visualization.plot_slice(study, 
                                  params=["network_params['batch_norm_1d_layer_numbers'][0]",
                                          "network_params['batch_norm_1d_layer_numbers'][0]"])
fig2 = optuna.visualization.plot_intermediate_values(study)
fig3 = optuna.visualization.plot_optimization_history(study)
fig4 = optuna.visualization.plot_param_importances(study)

fig5 = optuna.visualization.plot_contour(study, params=["network_params['batch_norm_1d_layer_numbers'][0]",
                                          "network_params['batch_norm_1d_layer_numbers'][0]"])
fig6 = optuna.visualization.plot_edf([study, study, study])
fig7 = optuna.visualization.plot_parallel_coordinate(study, params=["network_params['batch_norm_1d_layer_numbers'][0]",
                                          "network_params['batch_norm_1d_layer_numbers'][0]"])
# fig8 = optuna.visualization.plot_pareto_front(study)

# fig.show() # doesnt have any effect in Spyder
fig1.write_image('plot_slice.png') # works in Spyder
fig2.write_image('plot_intermediate_values.png') # works in Spyder
fig3.write_image('plot_optimization_history.png') # works in Spyder
fig4.write_image('plot_param_importances.png') # works in Spyder
fig5.write_image('plot_contour.png') # works in Spyder
fig6.write_image('plot_edf.png') # works in Spyder
fig7.write_image('plot_parallel_coordinate.png') # works in Spyder
# fig8.write_image('plot_pareto_front.png') # works in Spyder
# plt.imshow(fig.to_image())

#%% new cell (visualisation using matplotlib (optuna))

optuna.visualization.matplotlib.is_available()
# optuna.visualization.matplotlib.plot_slice(study, 
#                                   params=["network_params['batch_norm_1d_layer_numbers'][0]",
#                                           "network_params['batch_norm_1d_layer_numbers'][0]"])
optuna.visualization.matplotlib.plot_intermediate_values(study)
optuna.visualization.matplotlib.plot_optimization_history(study)
# optuna.visualization.matplotlib.plot_param_importances(study)

# optuna.visualization.matplotlib.plot_contour(study, params=["network_params['batch_norm_1d_layer_numbers'][0]",
#                                           "network_params['batch_norm_1d_layer_numbers'][0]"])
optuna.visualization.matplotlib.plot_edf([study, study, study])
optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["network_params['batch_norm_1d_layer_numbers'][0]",
                                          "network_params['batch_norm_1d_layer_numbers'][0]"])
# optuna.visualization.matplotlib.plot_pareto_front(study)



#%% new cell (save network parameters from the last training mode)

torch.save(network.state_dict(), os.path.join(os.getcwd(),"_r1" + "_e6000.pt"))

#%% new cell (load network parameters)

# subfolder = "2021-07-26_22-24-40" # noise0
subfolder = "2021-08-11_20-35-35" # noise20
# subfolder = "2021-08-13_22-54-38" # noise10


dtbs = "i-xcorr-0-phi-at-centr-freq-sio2-et-pred-unrestricted-phase-without-2nd-order-IA-20000-2"
study_nm = "architecture-search-13-same-dataset-same-distribution-noise10"

m = RunManager(dtbs,study_nm)

conv_layer_out_channels = [79, 71, 289, 328, 274, 316]#network_params_['conv_layer_out_channels']
# N_conv_layers = len(conv_layer_out_channels) # number of convolutional layers
conv_layer_kernel_sizes = [11, 3, 3, 11, 9, 3]#network_params_['conv_layer_kernel_sizes']
conv_layer_strides =      [1, 1, 2, 4, 5, 3]#network_params_['conv_layer_strides']
conv_layer_paddings =     [5, 1, 1, 5, 4, 1]#network_params_['conv_layer_paddings']

# max_pool-related parameters
max_pool_layer_numbers =  [0, 0, 0, 0, 0, 0]#network_params_['max_pool_layer_numbers'] # layers with maxpool (boolean mask)
max_pool_kernel_sizes =   [0, 0, 0, 0, 0, 0]#network_params_['max_pool_kernel_sizes'] # maxpool kernel sizes corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
max_pool_strides =        [0, 0, 0, 0, 0, 0]#network_params_['max_pool_strides'] # maxpool kernel strides corresponding to maxpools in layers defined by 'max_pool_layer_numbers'
max_pool_paddings =       [0, 0, 0, 0, 0, 0]#network_params_['max_pool_paddings'] # maxpool paddings corresponding to maxpools in layers defined by 'max_pool_layer_numbers'

# batchnorm2d-related parameters
batch_norm_2d_layer_numbers = [1, 1, 1, 1, 1, 1]#network_params_['batch_norm_2d_layer_numbers']

# linear-layer-related parameters
linear_layer_out_features = [316, 922, 884, 1000]#network_params_['linear_layer_out_features']
                            
# N_linear_layers = len(linear_layer_out_features) # number of linear layers

total_number_of_layers = N_conv_layers + N_linear_layers # total number of layers

# batchnorm1d-related parameters
batch_norm_1d_layer_numbers = [1, 1, 1, 0]#network_params_['batch_norm_1d_layer_numbers'] # later need to distinguish b/w BatchNorm1d and BatchNorm2d (b/w conv and linear layers)

conv_activation_functions = ['lrelu', 'lrelu', 'lrelu', 'lrelu', 'lrelu', 'lrelu']
fc_activation_functions = ['lrelu', 'lrelu', 'None', '']

network3 = Network3(Input_depth, conv_layer_out_channels, 
                        conv_layer_kernel_sizes, 
                        conv_layer_strides, 
                        conv_layer_paddings, 
                        max_pool_layer_numbers, 
                        max_pool_kernel_sizes, 
                        max_pool_strides, 
                        max_pool_paddings, 
                        batch_norm_2d_layer_numbers, 
                        linear_layer_out_features, 
                        batch_norm_1d_layer_numbers,
                        conv_activation_functions,
                        fc_activation_functions
                        ).to('cuda')
    # network4 = DenseNet(nr_classes=128).float()

networks = {
    # 'dnet': network4,
#     'net1': network1,
    #'net2': network2, 
    'net3': network3
}

# define train datasets here
trainsets = {
    # ----------------------for FROG ----------------------
    'not_norm': autocorr_2_data_train # not normalised
    # ,'not_norm_cut': train_frog_data_cut # normalised
    ,'norm': autocorr_2_data_train_normal # normalised
    # ,'norm_cut': train_frog_data_normal_cut
    # -----------------------------------------------------
    # 'mnist': train_set # add comma in the beginning for keeping it and using FROG
}

params = OrderedDict(
        btch_sz = [81] # bacth_size
        , lr = [0.008149612926988513]#[.001,.01,.1]
        , lr_decay = ['sqrt']#['exp','sqrt']#['cnst','exp','sqrt']
        , b1 = [0.883345587855772] # for Adam-optimization
        , b2 = [0.9996328747283718] # for Adam-optimization
        , n_wkrs = [0] # number of workers
        , device = ['cuda']  # ['cuda:0', 'cpu'] # device. Note: 'cuda:0' doesnt allow saving, since ':' is not allowed in filenames
        , shuffle = [True]#[True,False]
        # ------------- for FashionMNIST -----------------
        , trainset = ['norm'] # 'not_norm', was 'norm' before
        # ------------------------------------------------
        , network = list(networks.keys())
    )

print("Last used combination of parameters:")
counter = 0
for x in zip(RunBuilder.get_runs(params)):
    counter += 1
    print("Run "+str(counter)+": "+str(x[0]))
print()

m.load_network_params_2( date_=subfolder, model=network3, run=1, epoch=0)

#%% using the new loading function

subfolder = "2021-08-31_15-51-22"
dtbs = "i-xcorr-0-phi-at-centr-freq-sio2-et-pred-unrestricted-phase-without-2nd-order-IA-20000-2"
study_nm = "architecture-search-14-random-phase-dropout2d-test-6"

m = RunManager(dtbs,study_nm)

network3 = m.load_network(subfolder, 1, 12)
network3.eval()
 

#%% new cell
# Some checks

# network = Network3(1, conv_layer_out_channels, 
#                     conv_layer_kernel_sizes, 
#                     conv_layer_strides, 
#                     conv_layer_paddings, 
#                     max_pool_layer_numbers, 
#                     max_pool_kernel_sizes, 
#                     max_pool_strides, 
#                     max_pool_paddings, 
#                     batch_norm_2d_layer_numbers, 
#                     linear_layer_out_features, 
#                     batch_norm_1d_layer_numbers)





with torch.no_grad():
    loader_dev = DataLoader(autocorr_2_data_dev_normal ,batch_size=300,shuffle=None,num_workers=0)
    loader = DataLoader(autocorr_2_data_dev ,batch_size=300,shuffle=None,num_workers=0)
    
    loader_test_all = DataLoader(autocorr_2_data_test_normal ,batch_size=400,shuffle=None,num_workers=0)
    loss_function_test = nn.MSELoss()
    
    loader_train_all = DataLoader(autocorr_2_data_train_normal ,batch_size=300,shuffle=None,num_workers=0)
    loss_function_train = nn.MSELoss()
    
    loss_test_ = 0
    accuracy_ = 0
    # norm_factor = 0
    network4 = network3.to('cuda')
    for batch_test_ in loader_test_all:
        images_test_ = batch_test_[0].float().to('cuda')#device)
        labels_test_ = batch_test_[1].float().to('cuda')#device)
        preds_test_ = network3(images_test_)
        loss_ = loss_function_test(preds_test_,labels_test_.float()) # this was used before
        print('Accuracy over the test datasset: ', loss_)
     
    loss_ = 0
    iiii = 0
    network4 = network3.to('cpu')
    torch.set_printoptions(precision=10)
    for batch_train_ in loader_train_all:
        iiii += 1
        print(iiii,': ',loss_)
        images_train_ = batch_train_[0].float().to('cpu')#device)
        labels_train_ = batch_train_[1].float().to('cpu')#device)
        preds_train_ = network4(images_train_)
        loss_ = loss_ + loss_function_train(preds_train_,labels_train_.float()) # this was used before
        print('Accuracy over the train datasset: ', loss_)
        if iiii==1: break
    
    for batch_dev in loader:
        images_dev = batch_dev[0].float().to('cuda')#device)
        # norm_factor += np.shape(labels_dev)[0]*np.shape(labels_dev)[1]
    
    for batch_dev in loader_dev:
        images_dev_ = batch_dev[0].float().to('cuda')#device)
        labels_dev = batch_dev[1].float().to('cuda')#device)
        # norm_factor += np.shape(labels_dev)[0]*np.shape(labels_dev)[1]


with torch.no_grad():
    
    network3 = network3.to('cuda')
    tic = timeit.default_timer()
    preds_dev = network3(images_dev_)
    toc = timeit.default_timer()
    print('Prdeiction time:',toc-tic,' sec')
    
    # print(labels_dev)
    # print(preds_dev)
    
    nn_ = np.random.randint(300)
    
    nnn = [1,11,16,19,21,33,51,58,65,70,74,82,83,103,114,125,137,139,143,145,159,170,171,174,179,180,182,185,202,201,205,224,246,254,261,274,279,288]
    folder_to_save = os.path.join(os.getcwd(),'plots','_predicted_Et')
    
    for n in [279]:
        # np.save(os.path.join(folder_to_save,str(n)+'_label_noise20_.txt'),labels_test_[n].cpu()[:])
        # np.save(os.path.join(folder_to_save,str(n)+'_prediction_noise20_.txt'),preds_test_[n].cpu()[:])
        
        
        # spectra = images_dev[n].cpu().reshape(-1,)[0:1600]
        # maxind = np.argmax(spectra)
        # fig, (ax1, ax2) = plt.subplots(1,2)
        fig, ax21 = plt.subplots(1,1)
        # # ax1.plot(images_dev[n].cpu().reshape(-1,)[:],'b',linewidth=0.3)
        # # ax2.plot(spectra[800-200:800+200],'b',linewidth=0.3)
        # # ax21 = plt.twinx(ax2)
        ax21.plot(labels_test_[n].cpu()[:],'-b',markersize=6)
        ax21.plot(preds_test_[n].cpu()[:],'-r',markersize=6)
        ax21.set_xlim([200,800])
        # # ax21.set_ylim([-3.14,3.14])
        
        # # plt.ylabel("Energy, cm$^{-1}$")
        # ax21.legend(["Actual parameters","Predicted parameters"])
        # plt.title(str(n))
        # plt.xticks([0,0.5,1,1.5,2],["(E$_1$-11850) cm$^{-1}$","","(E$_2$-11850) cm$^{-1}$","","$(Coupling$+685) cm$^{-1}$"])
        
        # xy_labels = []
        # for i in range(100):
        #     xy_labels.append(10332.17952 + 33*i)
        
        # xylabels = np.linspace(11000,12980,100)
        # n=254
        # plt.figure()
        # grid = np.abs(dimers_data_dev.__getitem__(n)[0][:].reshape((2,100, 100))[0].T + 1j*dimers_data_dev.__getitem__(n)[0][:].reshape((2,100, 100))[1].T)
        # plt.imshow(grid.T,interpolation='nearest', cmap=modified_jet)
        # plt.xlabel("E$_1, cm^{-1}$")
        # plt.ylabel("E$_2, cm^{-1}$")
        # plt.xticks([0,20,40,60,80],[np.round(xy_labels[i]) for i in range(0,100,20)])
        # plt.yticks([0,20,40,60,80],[np.round(xy_labels[i]) for i in range(0,100,20)])
        # plt.figure()
        # grid = dimers_data_dev.__getitem__(n)[0][:].reshape((2,100, 100))[0].T
        # plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
        # plt.figure()
        # grid = dimers_data_dev.__getitem__(n)[0][:].reshape((2,100, 100))[1].T
        # plt.imshow(grid,interpolation='nearest', cmap=modified_jet)
    
# print('Labels:')
# print(labels_dev)
# print('Predictions:')
# print(preds_dev.argmax(dim=1))
# print(np.sum((F.softmax(preds_dev, dim=1).argmax(dim=1)==labels_dev).cpu().numpy())/10000*100)

#%%
print(os.path.dirname(__file__))
print(os.getcwd())
