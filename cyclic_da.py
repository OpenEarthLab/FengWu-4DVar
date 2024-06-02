import os
import io
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import yaml
import time
from collections import OrderedDict
from networks.transformer import LGUnet_all
from petrel_client.client import Client
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import Metrics
import torch.nn.functional as F
from torch_harmonics import *
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time',         type = str,     default = "2018-01-01 00:00:00", )
    parser.add_argument('--end_time',           type = str,     default = "2018-12-31 23:00:00", )
    parser.add_argument('--coeff_dir',          type = str,     default = "dataset/bq_info/", )
    parser.add_argument('--flow_model_dir',     type = str,     default = "world_size16-model-37years-stride1", )
    parser.add_argument('--forecast_model_dir', type = str,     default = "world_size8-model-37years-stride6", )
    parser.add_argument('--da_mode',            type = str,     default = "free_run", )
    parser.add_argument('--da_win',             type = int,     default = 6,     )
    parser.add_argument('--init_lag',           type = int,     default = 8,     )
    parser.add_argument('--Nit',                type = int,     default = 3,     )
    parser.add_argument('--obs_std',            type = float,   default = 0.001, ) 
    parser.add_argument('--obs_type',           type = str,     default = "random_015", )  
    parser.add_argument('--prefix',             type = str, )  
    parser.add_argument('--save_interval',      type = int,     default = 5,     ) 
    parser.add_argument('--save_field',         action = "store_true") 
    parser.add_argument('--save_gt',            action = "store_true") 
    parser.add_argument('--save_obs',           action = "store_true") 

    args = parser.parse_args()
    return args

class data_reader:
    def __init__(self, obs_type, obs_std, model_std, da_win, cycle_time, step_int_time):
        self.client = Client(conf_path="~/petreloss.conf")
        self.device = "cuda"
        self.obs_type = obs_type
        self.da_win   = da_win
        self.cycle_time = cycle_time
        self.step_int_time = step_int_time
        # if not obs_type[:4] == "real":
        obs_var_norm = torch.zeros(69, 128, 256).to(self.device) + obs_std**2
        self.obs_var = obs_var_norm * model_std.reshape(-1, 1, 1)**2

    def get_state(self, tstamp, data_dir="s3://era5_np128x256"):
        state = []
        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z','q', 'u', 'v', 't']
        height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        for vname in single_level_vnames:
            file = os.path.join('single/'+str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            url = f"{data_dir}/{file}-{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                state.append(np.load(f))
        for vname in multi_level_vnames:
            file = os.path.join(str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            for idx in range(13):
                height = height_level[idx]
                url = f"{data_dir}/{file}-{vname}-{height}.0.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    state.append(np.load(f).reshape(1, 128, 256))
        state = np.concatenate(state, 0)
        return torch.from_numpy(state).to(self.device)

    def get_obs_mask(self, tstamp):
        H  = torch.zeros(self.da_win, 69, 128, 256).to(self.device)
        H_file = torch.from_numpy(np.load("dataset/mask_%s.npy"%self.obs_type)).float().to(self.device)
        H = H + H_file

        return H

    def get_obs_gt(self, current_time):
        state = [self.get_state(current_time)]
        for i in range(self.da_win - 1):
            current_time += self.step_int_time
            state.append(self.get_state(current_time))
        gt  = torch.stack(state, 0)
        obs = gt + torch.sqrt(self.obs_var) * torch.randn(self.da_win, 69, 128, 256).to(self.device)
        return obs, gt
    
class cyclic_4dvar:
    def __init__(self, args):
        self.device     = "cuda"
        self.start_time    = pd.Timestamp(args.start_time)
        self.end_time      = pd.Timestamp(args.end_time)
        self.cycle_time    = pd.Timedelta('6H')
        self.step_int_time = pd.Timedelta('1H')
        self.da_mode       = args.da_mode
        self.da_win        = args.da_win
        self.nlon          = 256
        self.nlat          = 128
        self.hpad          = 5
        self.vname_list    = ['z', 'q', 'u', 'v', 't']
        self.geoheight_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.fullname = ['u10', 'v10', 't2m', 'mslp']
        for vname in self.vname_list:
            for geoheight in self.geoheight_list:
                self.fullname.append(vname + str(geoheight))
        self.nlev          = len(self.geoheight_list)
        self.nchannel      = len(self.fullname)
        self.Nit           = args.Nit

        self.model_mean, self.model_std = self.get_model_mean_std()
        self.b_matrix       = self.init_b_matrix(args.coeff_dir)
        self.q_matrix       = self.init_q_matrix(args.coeff_dir)  
        self.flow_model     = self.init_model(args.flow_model_dir)
        self.forecast_model = self.init_model(args.forecast_model_dir)

        self.init_lag = args.init_lag
        self.obs_std  = args.obs_std
        self.obs_type = args.obs_type
        # if self.obs_type[:4] == "real":

        self.name  =  "%s_%s_std%.3f_win%d_lag%d_%s"%(args.prefix, args.obs_type, args.obs_std, args.da_win, args.init_lag, args.end_time)
        print(self.name)

        self.save_field    = args.save_field
        self.save_interval = args.save_interval
        self.save_gt       = args.save_gt
        self.save_obs      = args.save_obs
        self.metric        = Metrics()

        self.init_file_dir()

        self.data_reader = data_reader(args.obs_type, args.obs_std, self.model_std, self.da_win, self.cycle_time, self.step_int_time)
        self.metrics_list = {"bg_wrmse": [], "ana_wrmse": [], "bg_mse": [], "ana_mse": [], "bg_bias": [], "ana_bias": []}
        self.current_time, self.xb = self.get_current_states()
        self.load_eval_ckpts()

        self.static_info   = self.get_static_info() ## for saving redundant calculations

    def init_b_matrix(self, coeff_dir):
        len_scale = torch.from_numpy(np.load(os.path.join(coeff_dir, "len_scale.npy"))).float().to(self.device)
        return {"len_scale": len_scale}

    def init_q_matrix(self, coeff_dir):
        q = []
        for i in range(1, self.da_win):
            q0 = torch.from_numpy(np.load(os.path.join(coeff_dir, "q%d.npy"%i))).to(self.device) / self.model_std.reshape(-1, 1, 1)**2
            q.append(torch.broadcast_to(torch.mean(q0, (1, 2), True), (self.nchannel, self.nlat, self.nlon)))
        q = torch.stack(q, 0)
        print("q", q[:, :, 100, 100])
            
        return q

    def init_model(self, path):
        with open("output/model/%s/training_options.yaml"%(path), 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        model = LGUnet_all(**cfg_params["model"]["network_params"])
        checkpoint_dict = torch.load("output/model/%s/checkpoint_best.pth"%(path))
        checkpoint_model = checkpoint_dict['model']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def init_file_dir(self):
        os.makedirs("da_cycle_results/%s"%(self.name), exist_ok=True)

    def get_static_info(self):
        ### calculating horizontal factor
        x = np.linspace(-self.hpad, self.hpad, 2*self.hpad+1)
        y = np.linspace(-self.hpad, self.hpad, 2*self.hpad+1)
        xx, yy = np.meshgrid(x, y)

        sht = RealSHT(self.nlat, self.nlon, grid="equiangular").to(self.device)
        isht = InverseRealSHT(self.nlat, self.nlon, grid="equiangular").to(self.device)

        kernel = torch.zeros(self.nchannel, self.nlat, self.nlon).to(self.device)
        coeffs_kernel = []
        for layer in range(self.nchannel):
            for i in range(self.hpad):
                kernel[layer, i] = torch.exp(-i**2/(2*self.b_matrix["len_scale"][layer]**2))
            coeffs_kernel.append(sht(kernel[layer]))

        sph_scale = torch.Tensor(np.array(np.broadcast_to(np.arange(0, self.nlat).transpose(), (self.nlat+1, self.nlat)).transpose())).to(self.device)
        sph_scale = 2*np.pi*torch.sqrt(4*np.pi/(2*sph_scale+1))

        ### calculating R
        R = torch.zeros(self.da_win, self.nchannel, self.nlat, self.nlon).to(self.device)
        R[0] = self.data_reader.obs_var / self.model_std.reshape(-1, 1, 1)**2
        for i in range(self.da_win - 1):
            R[i+1] = self.data_reader.obs_var / self.model_std.reshape(-1, 1, 1)**2 + self.q_matrix[i]

        print("R", R[:, :, 100, 100])

        return {"R": R, "sht": sht, "isht": isht, "coeffs_kernel": coeffs_kernel, "sph_scale": sph_scale}

    def get_model_mean_std(self):
        mean_layer = np.load("dataset/layer_mean.npy")
        std_layer  = np.load("dataset/layer_std.npy")

        mean_layer_gpu = torch.from_numpy(mean_layer).float().to(self.device)
        std_layer_gpu  = torch.from_numpy(std_layer).float().to(self.device)
        return mean_layer_gpu, std_layer_gpu

    def get_initial_state(self):
        x0 = self.data_reader.get_state(self.start_time - self.init_lag * pd.Timedelta('6H'))
        xb = self.integrate(x0, self.forecast_model, self.init_lag)
        gt = self.data_reader.get_state(self.start_time)
        rmse = torch.sqrt(torch.mean((gt - xb)**2, (1, 2)))
        print("xb rmse per layer", rmse.cpu().numpy())
        mse  = torch.mean(((gt - xb) / self.model_std.reshape(-1, 1, 1))**2)
        print("xb mse: %.3g"%(mse))
        return xb

    def integrate(self, xa, model, step):
        za = (xa - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
        z = za.unsqueeze(0)
        for i in range(step):
            z = model(z)[:, :self.nchannel].detach()

        return z.reshape(self.nchannel, self.nlat, self.nlon) * self.model_std.reshape(-1, 1, 1) + self.model_mean.reshape(-1, 1, 1)

    def get_current_states(self):
        if os.path.exists("da_cycle_results/%s/current_time.txt"%(self.name)):
            f = open("da_cycle_results/%s/current_time.txt"%self.name, "r")
            self.current_time = pd.Timestamp(f.read())
            state = np.load("da_cycle_results/%s/xb.npy"%self.name)
            self.xb = torch.from_numpy(state).to(self.device)
        else:
            self.current_time = self.start_time
            self.xb = self.get_initial_state()
        
        return self.current_time, self.xb

    def save_eval_result(self, finish=False, gt=None, obs=None):
        for key in self.metrics_list:
            np.save("da_cycle_results/%s/%s"%(self.name, key), self.metrics_list[key])
        print("finish saving results")

        if not finish:
            np.save("da_cycle_results/%s/xb"%self.name, self.xb.cpu().numpy())
            with open("da_cycle_results/%s/current_time.txt"%self.name, 'w') as f:
                f.write(str(self.current_time))
            if self.save_field:
                np.save("da_cycle_results/%s/xb_%s"%(self.name, self.current_time), self.xb.detach().cpu().numpy())
                np.save("da_cycle_results/%s/xa_%s"%(self.name, self.current_time), self.xa.detach().cpu().numpy())
                print("finish saving intermediate fields")
            if self.save_gt:
                np.save("intermediate/ground_truth/gt_%s"%(self.current_time), gt.cpu().numpy())
                print("finish saving ground truth")
            if self.save_obs:
                np.save("intermediate/ground_truth/obs_%s"%(self.current_time), obs.cpu().numpy())
                print("finish saving observations")

    def load_eval_ckpts(self):   
        for key in self.metrics_list:
            if os.path.exists("da_cycle_results/%s/%s.npy"%(self.name, key)):
                self.metrics_list[key] = np.load("da_cycle_results/%s/%s.npy"%(self.name, key)).tolist()

    def get_obs_info(self):
        yo, gt = self.data_reader.get_obs_gt(self.current_time)
        H = self.data_reader.get_obs_mask(self.current_time)
        R = self.static_info["R"]
        
        return yo, H, R, gt

    def transform(self, u, xb):

        field_horizon = []

        for i in range(self.nchannel):
            coeffs_field  = self.static_info["sht"](u[i])
            field_horizon.append(self.static_info["isht"](self.static_info["sph_scale"]*coeffs_field*self.static_info["coeffs_kernel"][i][:, 0].reshape((self.nlat, 1))).unsqueeze(0))
        
        coeff_expand = torch.zeros(69, 128, 256).to(self.device) + 1
        coeff_expand[4] = 0.6
        coeff_expand[5] = 0.6
        coeff_expand[6] = 0.7
        coeff_expand[7] = 0.8
        field_horizon = torch.cat(field_horizon, 0) * coeff_expand #* torch.sqrt(q6_norm_layer)

        return field_horizon + xb

    def one_step_DA(self, gt, xb, yo, H, R, mode):
        if mode == "free_run":
            gt_norm  = (gt[0] - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            xb_norm  = (xb - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W
            WRMSE_bg = self.metric.WRMSE(xb_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std.cpu()).detach()
            bias_bg  = self.metric.Bias(xb_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std.cpu()).detach()
            MSE_bg   = torch.mean((xb_norm - gt_norm)**2).item()

            self.metrics_list["bg_wrmse"].append(WRMSE_bg)
            self.metrics_list["bg_bias"].append(bias_bg)
            self.metrics_list["bg_mse"].append(MSE_bg)

            start_clock = time.time()
            xa = xb
            layer = 11
            print("MSE (total): %.4g RMSE (z500): %.4g Bias (z500): %.4g" % (MSE_bg, WRMSE_bg[layer], bias_bg[layer]), flush=True)
            end_clock   = time.time()

            self.metrics_list["ana_wrmse"].append(WRMSE_bg)
            self.metrics_list["ana_bias"].append(bias_bg)
            self.metrics_list["ana_mse"].append(MSE_bg)

            print("%s DA finished. Time consumed: %d (s)" % (self.current_time, end_clock - start_clock), flush=True)

            return xa

        elif mode == "sc4dvar":
            def cal_loss_bg(x0):
                """
                x0:     C x H x W
                """
                return torch.sum(x0**2) / 2

            def cal_loss_obs(x):
                """
                x0:       C x H x W
                obs:      T x C x H x W
                H:        T x C x H x W
                obs_var:  T x C x H x W
                """
                x_list = [x, ]
                for i in range(self.da_win-1):
                    x = self.integrate(x * self.model_std.reshape(-1, 1, 1) + self.model_mean.reshape(-1, 1, 1), self.flow_model, 1)[:69]
                    x = (x - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
                    x_list.append(x)

                x_pred = torch.stack(x_list, 0)   # T x C x H x W

                return torch.sum( H * (x_pred - yo_norm) ** 2 / R ) / 2

            def loss(w):
                xhat = self.transform(w, xb_norm)
                return cal_loss_bg(w) + cal_loss_obs(xhat)

            def closure():
                lbfgs.zero_grad()
                objective = loss(w)
                objective.backward()
                return objective 

            w = torch.autograd.Variable(torch.zeros(self.nchannel, self.nlat, self.nlon).to(self.device), requires_grad=True)
            gt_norm  = (gt[0] - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
            yo_norm  = (yo - self.model_mean.reshape(1, -1, 1, 1)) / self.model_std.reshape(1, -1, 1, 1)  # T x C x H x W

            lbfgs = optim.LBFGS([w], history_size=10, max_iter=5, line_search_fn="strong_wolfe")
            idx = 11
            start_clock = time.time()           

            kk = 0
            while kk <= self.Nit:
                xb_norm  = (xb - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)  # C x H x W

                xhat_norm = self.transform(w, xb_norm)
                # xhat_norm  = (xhat - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
                WRMSE_GT = self.metric.WRMSE(xhat_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std.cpu()).detach()
                bias_GT = self.metric.Bias(xhat_norm.unsqueeze(0).clone().detach().cpu(), gt_norm.unsqueeze(0).clone().detach().cpu(), None, None, self.model_std.cpu()).detach()
                RMSE_z500_GT = WRMSE_GT[idx].item()
                bias_z500_GT = bias_GT[idx].item()
                MSE_GT = torch.mean((xhat_norm - gt_norm)**2).item()
                loss_total = loss(w).item()
                loss_bg  = cal_loss_bg(w).item()
                loss_obs = cal_loss_obs(xhat_norm).item()

                print("iter: %d, MSE (total): %.4g RMSE (z500): %.4g Bias (z500): %.4g loss: %.4g loss obs: %.4g loss bg: %.4g" % (kk, MSE_GT, RMSE_z500_GT, bias_z500_GT, loss_total, loss_obs, loss_bg), flush=True)
                
                if kk == 0:
                    self.metrics_list["bg_wrmse"].append(WRMSE_GT)
                    self.metrics_list["bg_mse"].append(MSE_GT)
                    self.metrics_list["bg_bias"].append(bias_GT)
                elif kk == self.Nit:
                    self.metrics_list["ana_wrmse"].append(WRMSE_GT)
                    self.metrics_list["ana_mse"].append(MSE_GT)
                    self.metrics_list["ana_bias"].append(bias_GT)

                if kk < self.Nit:
                    lbfgs.step(closure)

                kk = kk + 1
            
            w.detach()
            xhat_norm = self.transform(w, xb_norm)
            end_clock = time.time()
            print("%s DA finished. Time consumed: %d (s)" % (self.current_time, end_clock - start_clock), flush=True)

            return xhat_norm * self.model_std.reshape(-1, 1, 1) + self.model_mean.reshape(-1, 1, 1)

        else:
            raise NotImplementedError("not implemented da mode")

    def run_assimilation(self):
        epoch = 0

        while(self.current_time + self.cycle_time <= self.end_time):
            print("current time:", self.current_time)

            print("obtaining observations...")
            yo, H, R, gt = self.get_obs_info()
            
            print("assimilating...")
            self.xa = self.one_step_DA(gt, self.xb, yo, H, R, self.da_mode)  # [69, 128, 256]

            if epoch % self.save_interval == 0:
                self.save_eval_result(finish=False, gt=gt, obs=yo)

            print("integrating...")
            self.xb = self.integrate(self.xa, self.forecast_model, 1)

            self.current_time = self.current_time + self.cycle_time
            epoch += 1

        print("DA complete")
        self.save_eval_result(finish=True, gt=None)

if __name__ == "__main__":
    args = arg_parser()
    da_agent = cyclic_4dvar(args)
    da_agent.run_assimilation()