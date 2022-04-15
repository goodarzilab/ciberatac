import __main__ as interactive_session
import adabound
from argparse import ArgumentParser
from apex import amp
from datetime import datetime
from model import ResNet1D
from model import ResidualBlock
import gzip
from losses import rankNet
import numpy as np
import os
import pandas as pd
import pyBigWig
from scipy import stats
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import shutil
from functools import partial
from utils import split_tensordict
from utils import printProgressBar
# from utils import prepare_response
from utils import regularize_loss
from utils import get_bce
from utils import get_ce_loss
# from train import assess_performance
from utils import compile_paths
# from utils import motor_log
from utils import make_normalizers
from utils import get_signal_from_bw


device = torch.device("cuda:0")
opt_level = 'O1'


def above_av(vals):
    return vals > 1


def motor_log(epoch, j, dict_perf, lr, tempdir,
              current_loss, net, modelpath, macrobatch,
              regression=False):
    curlogpath = os.path.join(
        tempdir,
        "modelLogUnSharednnn_lr{}_macrobatch{}.tsv".format(
            lr, macrobatch))
    if regression:
        log_model_regression(
            curlogpath,
            epoch, current_loss, j,
            dict_perf)
    return curlogpath


def log_model_regression(logpath, epoch, train_loss,
                         j, dict_perf):
    current_time = str(datetime.now())
    select_vars = [
        "Training.Loss",
        "Training.R", "Tuning.Loss", "Tuning.R",
        "baseline.R", "Training.pMSE", "Tuning.pMSE",
        "Tuning.baseline.pMSE", "Training.bceloss",
        "Tuning.bceloss", "Tuning.baseline.bceloss"]
    if epoch == 0:
        if not os.path.exists(logpath):
            with open(logpath, "w") as loglink:
                adlist = [
                    "Time", "Epoch", "MiniBatch"]
                adlist.extend(select_vars)
                loglink.write("\t".join(adlist) + "\n")
    with open(logpath, "a+") as loglink:
        float_vals = []
        for variable in select_vars:
            float_vals.append(
                dict_perf[variable])
        float_vals = [str(round(each, 5)) for each in float_vals]
        adlist = [current_time, str(epoch), str(j)] + float_vals
        print("\t".join(adlist))
        loglink.write("\t".join(adlist) + "\n")


def merge_torch(dnase_tensor, rna_tensor):
    mat1 = dnase_tensor.reshape(
        1, dnase_tensor.shape[0],
        dnase_tensor.shape[1],
        dnase_tensor.shape[2])
    mat2 = rna_tensor.reshape(
        1, rna_tensor.shape[0],
        rna_tensor.shape[1],
        rna_tensor.shape[2])
    newmat = torch.cat((mat1, mat2))
    return newmat


def find_best_sample(predictions, regions, samplenames):
    out_vals = np.zeros(len(predictions), dtype="|U256")
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)[0]
        preds = predictions[idx_reg]
        idx_best = np.where(preds == max(preds))[0][0]
        out_vals[idx_reg] = np.array(
            [samplenames[idx_reg][idx_best]] * len(preds))
    return out_vals


def assess_performance(net, tensordict_tune, criterion,
                       tensordict_train, num_parts, batch_size,
                       regression=False, loss_scalers=[1, 1, 1, 1],
                       resp_cutoff=0, ordinal=False, respdiff=False):
    dict_perf = {}
    tensorlist = [tensordict_train, tensordict_tune]
    tensornames = ["Training", "Tuning"]
    for i in range(2):
        tensordict = tensorlist[i]
        tensorname = tensornames[i]
        dnase_tensor = torch.from_numpy(
            tensordict["DNase"])
        midvals = tensordict["Averages"][:, 0]
        rna_tensor = torch.from_numpy(
            tensordict["RNA"])
        response_tensor = tensordict["Response"]
        if respdiff:
            response_tensor = tensordict["Resp.Diff"]
        outputs = np.zeros((rna_tensor.shape[0], 1))
        outputs = np.zeros((rna_tensor.shape[0], 2))
        running_loss = 0
        dim_eval = int(dnase_tensor.shape[0] / batch_size) + 1
        if dim_eval > 400:
            dim_eval = 400
        for tidx in range(dim_eval):
            tidx_st = tidx * batch_size
            tidx_end = min(
                [(tidx + 1) * batch_size, rna_tensor.shape[0]])
            if tidx_st >= rna_tensor.shape[0]:
                break
            dnase = dnase_tensor[tidx_st:tidx_end]
            rna = rna_tensor[tidx_st:tidx_end]
            train_tensor = [0]
            dnase = dnase.to(device)
            rna = rna.to(device)
            mave_tensor = torch.from_numpy(
                tensordict["mave"][tidx_st:tidx_end]).float().to(device)
            output, _ = net(
                dnase, rna, mave_tensor)
            del mave_tensor
            del dnase, rna, train_tensor
            loss = criterion(
                output,
                torch.from_numpy(
                    response_tensor[tidx_st:tidx_end]).to(device))
            running_loss += loss.item()
            output_ar = output.cpu().detach().numpy()
            outputs[tidx_st:tidx_end] = output_ar
            del output
            torch.cuda.empty_cache()
            if not hasattr(interactive_session, '__file__'):
                printProgressBar(tidx, dim_eval, suffix=tensornames[i])
        print("\n")
        tuning_loss = running_loss / (tidx + 1)
        resp_full_add = response_tensor[:, 0]
        pred_add = outputs[:, 0]
        tuning_loss = loss.item()
        perfdf = pd.DataFrame(
                {"Prediction": pred_add,
                 "Response": resp_full_add,
                 "Average.DNase": midvals,
                 "Regions": tensordict["Regions"],
                 "Tissues": tensordict["Tissues"]})
        perfdf["BestSample.Response"] = find_best_sample(
            np.array(perfdf["Response"]), np.array(perfdf["Regions"]),
            np.array(perfdf["Tissues"]))
        perfdf["BestSample.Prediction"] = find_best_sample(
            np.array(perfdf["Prediction"]), np.array(perfdf["Regions"]),
            np.array(perfdf["Tissues"]))
        dict_perf["{}.Loss".format(tensorname)] = tuning_loss * loss_scalers[0]
        try:
            dict_perf["{}.pMSE".format(tensorname)] = float(
                get_bce(
                    np.array(perfdf["Response"]),
                    np.array(perfdf["Prediction"]),
                    np.array(perfdf["Regions"]),
                    resp_cutoff, bce=False)) * loss_scalers[2]
            dict_perf["{}.bceloss".format(tensorname)] = float(
                get_region_bce(
                    np.array(perfdf["Response"]),
                    np.array(perfdf["Prediction"]),
                    np.array(perfdf["Regions"]),
                    np.array(perfdf["Tissues"]))) * loss_scalers[3]
            dict_perf["{}.baseline.pMSE".format(tensorname)] = float(
                get_bce(
                    np.array(perfdf["Response"]),
                    np.array(perfdf["Average.DNase"]),
                    np.array(perfdf["Regions"]),
                    resp_cutoff, bce=False)) * loss_scalers[2]
            dict_perf["{}.baseline.bceloss".format(tensorname)] = float(
                get_region_bce(
                    np.array(perfdf["Response"]),
                    np.array(perfdf["Average.DNase"]),
                    np.array(perfdf["Regions"]),
                    np.array(perfdf["Tissues"]))) * loss_scalers[3]
            corval, pval = stats.pearsonr(
                perfdf["Response"], perfdf["Prediction"])
            dict_perf["{}.R".format(tensorname)] = corval
            corval, pval = stats.pearsonr(
                perfdf["Response"], perfdf["Average.DNase"])
            dict_perf["baseline.R"] = corval
        except Exception:
            perfdf.to_csv("Performance_table_causing_issues.tsv",
                          sep="\t")
            print(perfdf["Prediction"])
            raise ValueError("Failed to calculate Pearson R")
    if tensorname == "Tuning":
        for eachkey, eachval in dict_perf.items():
            perfdf[eachkey] = eachval
        for each_key in ["Regions", "Samples", "Tissues"]:
            if each_key in tensordict_tune.keys():
                perfdf[each_key] = [
                    each.decode() for each in
                    tensordict[each_key]]
    return dict_perf, perfdf


def get_signal(pos, ar, window):
    outsig = ar[(pos - window):(pos + window)]
    return np.mean(outsig)


class DataHandler:
    def __init__(self, rnabwpaths, dnasebwpaths,
                 bulkdnasepath,
                 bedpath, seqdir, window=10000,
                 mask_nonpeaks=False,
                 force_tissue_negatives=True,
                 dont_train=False,
                 SCALE_FACTORS=[1, 1],
                 SCALE_OP="identity", arcsinh=False,
                 seed=42, input_normalize="None"):
        self.SCALE_FACTORS = SCALE_FACTORS
        self.SCALE_OP = SCALE_OP
        self.input_normalize = input_normalize
        self.arcsinh = arcsinh
        self.prepared_bigwigs = False
        self.dont_train = dont_train
        self.nucs = np.array(["A", "T", "C", "G"])
        self.window = window
        self.force_tissue_negatives = force_tissue_negatives
        self.seed = seed
        self.sequencedir = seqdir
        self.rnabwpaths = rnabwpaths
        self.dnasebwpaths = dnasebwpaths
        self.bulkdnasepath = bulkdnasepath
        self.bedpath = bedpath
        self.mask_nonpeaks = mask_nonpeaks
        self.make_normalizers()
        self.load_bed()

    def make_normalizers(self, sample_chrom="chr1"):
        bw_paths = [
            self.bulkdnasepath]
        Scaler = make_normalizers(
            bw_paths, self.input_normalize, sample_chrom,
            self.SCALE_FACTORS[1])
        if self.input_normalize == "RobustScaler":
            self.RobustScaler = Scaler
        elif self.input_normalize == "MinMaxScaler":
            self.MinMaxScaler = Scaler

    def mask_dnase(self):
        dnase_ar = self.dnase_signal
        bed = self.bed
        starts = np.array(bed.iloc[:, 1])
        ends = np.array(bed.iloc[:, 2])
        idxs_keep = np.zeros(len(dnase_ar), dtype=bool)
        for i in range(len(starts)):
            idxs_keep[starts[i]:ends[i]] = True
        dnase_ar[np.logical_not(idxs_keep)] = 0
        self.dnase_signal = dnase_ar

    def get_batches(self, start_poses, end_poses,
                    rnabwpath, dnasebwpath, SCALE_FACTORS):
        midpos = np.array(
            start_poses + np.round((end_poses - start_poses) / 2),
            dtype=int)
        rna, _ = self.get_signal_from_bw(
            rnabwpath, midpos, SCALE_FACTORS[0],
            start_poses, end_poses)
        dnase, avg_mid = self.get_signal_from_bw(
            self.bulkdnasepath, midpos, SCALE_FACTORS[1],
            start_poses, end_poses, get_resp=True)
        _, response = self.get_signal_from_bw(
            dnasebwpath, midpos, SCALE_FACTORS[1],
            start_poses, end_poses, get_resp=True)
        return rna, dnase, midpos, avg_mid, response

    def scale_signal(self, signal):
        signal = self.normalize_input(signal)
        if self.SCALE_OP == "identity":
            return signal
        elif self.SCALE_OP == "sqrt":
            return np.sqrt(signal)
        elif self.SCALE_OP == "log2":
            return np.log2(signal + 1)
        else:
            print("Unacceptable SCALE_OP parameter: {}".format(self.SCALE_OP))
            raise ValueError("Unacceptable SCALE_OP")

    def get_signal_from_bw(
            self, bwpath, midpos, SCALE_FACTOR,
            start_poses, end_poses, get_resp=False):
        batchar, avg_vals = get_signal_from_bw(
            bwpath, midpos, SCALE_FACTOR,
            start_poses, end_poses, self.chrom,
            self.chrom_seq, self.window, self.nucs,
            self, get_resp)
        return batchar, avg_vals

    def normalize_input(self, batchar):
        idx_nonzero = np.where(batchar > 0)
        if len(batchar.shape) == 3:
            idx_nonzero = np.where(batchar > 0.1)
        if len(idx_nonzero[0]) > 0:
            curvals = batchar[idx_nonzero].reshape(-1, 1)
            if self.input_normalize == "RobustScaler":
                if not hasattr(self, self.input_normalize):
                    raise ValueError("Scaler not initiated!")
                    # self.RobustScaler = RobustScaler().fit(curvals)
                newvals = self.RobustScaler.transform(curvals)
            elif self.input_normalize == "MinMaxScaler":
                if not hasattr(self, self.input_normalize):
                    raise ValueError("Scaler not initiated!")
                    # self.MinMaxScaler = MinMaxScaler().fit(curvals)
                newvals = self.MinMaxScaler.transform(curvals)
            elif self.input_normalize == "None":
                newvals = curvals
            else:
                print("{} not recognized".format(self.input_normalize))
                raise ValueError("MinMaxScaler not recognizer, check logs")
            if np.min(newvals) < 0.1:
                if len(batchar.shape) == 3:
                    newvals = newvals - min(newvals) + 0.1
                elif np.min(newvals) < 0:
                    newvals = newvals - min(newvals)
            batchar[idx_nonzero] = newvals.reshape(-1)
        return batchar

    def get_region_poses(self, num_variants=700, num_random=400):
        # num_variants = 700  # 600
        # num_random = 400
        # cutoff_0 = np.quantile(self.bed["Bulk.Signal"], 0.05)
        # num_pos_1 = int(max_num / 2)
        if num_variants + num_random > self.bed.shape[0]:
            ratio_regs = float(num_variants) / num_random
            num_variants = int(ratio_regs * self.bed.shape[0])
            num_random = self.bed.shape[0] - num_variants
        cutoff_0 = 0
        tempdf = self.bed.iloc[
            np.where(self.bed["Bulk.Signal"] > cutoff_0)[0],
            np.where(
                self.bed.columns.isin(np.array(self.nonbulk_cols[1:])))[0]]
        # idx_top = np.where(
        #     np.apply_along_axis(
        #         above_av, 1, np.array(tempdf)))[0]
        # tempdf = tempdf.iloc[idx_top, :]
        var_rows = np.apply_along_axis(
            np.var, 1, np.array(tempdf))
        idx_ordered_var = np.argsort(var_rows)
        # idx_mid = np.random.choice(
        #     np.arange(0, var_rows.shape[0] - num_pos_1),
        #     num_pos_1)
        # idx_use = np.concatenate(
        #     [idx_mid, idx_ordered_var[-num_pos_1:]])
        idx_mid = np.random.choice(
                np.arange(0, var_rows.shape[0] - num_variants),
                num_random)
        idx_use = np.concatenate(
            [idx_mid, idx_ordered_var[-num_variants:]])
        # For each cell type, get indices they have the highest value
        # tempar = np.array(tempdf)
        # maxar = np.apply_along_axis(np.max, 1, tempar)
        list_idxs = list(idx_use)
        list_idxs = np.unique(np.array(list_idxs))
        tempdf["Start"] = self.bed.loc[tempdf.index, "start"]
        tempdf["End"] = self.bed.loc[tempdf.index, "end"]
        arr_poses = np.array(list_idxs)
        np.random.seed(self.seed)
        np.random.shuffle(arr_poses)
        out_poses = np.zeros((arr_poses.shape[0], 2), dtype=int)
        out_poses[:, 0] = tempdf.iloc[arr_poses, -2]
        out_poses[:, 1] = tempdf.iloc[arr_poses, -1]
        return out_poses

    def annotate_bed(self, SCALE_FACTOR, quantile_resp=0.1):
        arraynames = ["Bulk"] + [
            os.path.basename(each).replace(".bigWig", "") for
            each in self.dnasebwpaths]
        # cutoff_res = self.scale_signal(0.5 * SCALE_FACTOR)
        bwpaths = [self.bulkdnasepath] + self.dnasebwpaths
        nonbulk_cols = []
        dict_bulk_ars = {}
        width_regions = np.array(self.bed["end"] - self.bed["start"])
        newdf = pd.DataFrame(
            {"start": np.array(
                self.bed["start"] +
                (width_regions / 2), dtype=int)})
        list_resp_cutoffs = []
        for i in range(len(bwpaths)):
            adname = arraynames[i]
            print("Adding {} ATAC-seq signal".format(adname))
            bwObj = pyBigWig.open(bwpaths[i], "rb")
            chromlen = len(self.chrom_seq)
            chrom_temp = self.chrom
            if chrom_temp not in bwObj.chroms().keys():
                chrom_temp = chrom_temp.replace("chr", "")
            adar = bwObj.values(
                chrom_temp, 0, chromlen, numpy=True) * SCALE_FACTOR
            adar[np.isnan(adar)] = 0
            adar = self.scale_signal(adar)
            # adar = self.fix_signal_range(adar)
            dict_bulk_ars[adname] = adar
            motor_get_signal = partial(get_signal, ar=adar, window=100)
            adcolname = "{}.Signal".format(adname)
            nonbulk_cols.append(adcolname)
            self.bed[adcolname] = np.array(
                newdf["start"].map(motor_get_signal))
            resp_cutoff_add = np.quantile(
                np.array(self.bed[adcolname]), quantile_resp)
            list_resp_cutoffs.append(resp_cutoff_add)
            # if i == 0:
            # idx_zero = np.where(self.bed[adcolname] < cutoff_res)[0]
            # self.bed.iloc[idx_zero, -1] = 0
            bwObj.close()
        self.dict_bulk_ars = dict_bulk_ars
        self.nonbulk_cols = nonbulk_cols
        self.cutoff_10p = np.quantile(self.bed["Bulk.Signal"], 0.1)
        final_resp_cutoff = np.mean(np.array(list_resp_cutoffs[1:]))
        return final_resp_cutoff

    def get_batch_nums(self, chrom, batchsize):
        self.batchsize = batchsize
        self.chrom = chrom
        print("Filtering BED for {}".format(chrom))
        self.chrom_seq = self.get_chromseq(chrom)
        self.bed = self.bed[self.bed.iloc[:, 0] == chrom]
        self.num_regions = self.bed.shape[0]
        self.num_batches = int(
            np.round(self.num_regions / self.batchsize))
        # self.process_background()
        return self.num_batches

    def initiate_seq(self, start, end):
        tensor = np.zeros((4, self.window * 2), dtype=float)
        for nucidx in range(len(self.nucs)):
            nuc = self.nucs[nucidx].encode()
            if start > 0:
                j = np.where(self.chrom_seq[start:end] == nuc)[0]
                tensor[nucidx, j] = \
                    tensor[nucidx, j] + 0.1
            else:
                j = np.where(self.chrom_seq[:end] == nuc)[0]
                ad_j = -start
                tensor[nucidx, j + ad_j] = \
                    tensor[nucidx, j + ad_j] + 0.1
        return tensor

    def get_chromseq(self, chrom):
        arpath = os.path.join(
            self.sequencedir,
            "{}_sequence.numpy.gz".format(chrom))
        with gzip.open(arpath, "rb") as arlink:
            npar = np.load(arlink)
        return npar

    def load_bed(self):
        # Chromosomes
        chroms = ["chr{}".format(each_chrom)
                  for each_chrom in
                  list(range(1, 24)) + ["X", "Y", "mt", "EBV"]]
        # Check if gzipped or not
        bed_gzipped = False
        if ".gz" == self.bedpath[-3:]:
            fileobj = gzip.open(self.bedpath, "rb")
            bed_gzipped = True
        else:
            fileobj = open(self.bedpath, "r")
        header = fileobj.readline().decode().rstrip().split("\t")
        print(header)
        fileobj.close()
        # Load BED file
        if header[0] in chroms:
            if bed_gzipped:
                self.bed = pd.read_csv(
                    self.bedpath, sep="\t",
                    compression="gzip", header=None)
            else:
                self.bed = pd.read_csv(
                    self.bedpath, sep="\t", header=None)
            header = list(self.bed.columns)
            header[:3] = ["seqnames", "start", "end"]
            self.bed.columns = header
        else:
            if bed_gzipped:
                self.bed = pd.read_csv(
                    self.bedpath, sep="\t",
                    compression="gzip")
            else:
                self.bed = pd.read_csv(
                    self.bedpath, sep="\t")
        print("Loaded BED file: {}".format(self.bed.head()))


def match_dist_motor(values, minval, maxval):
    unit_vals = (values - min(values)) / (max(values) - min(values))
    print(unit_vals)
    outvals = (unit_vals * (maxval - minval)) + minval
    print("Min of new values: {}".format(min(outvals)))
    print("Max of new values: {}".format(max(outvals)))
    return outvals


def get_optimizer(optname, net, lr):
    if optname == "Adabound":
        optimizer = adabound.AdaBound(
            net.parameters(), lr=lr, final_lr=0.1)
    elif optname == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optname == "Adagrad":
        optimizer = optim.Adagrad(
            net.parameters(), lr=lr*10)
    elif optname == "Adam":
        optimizer = optim.Adam(
            net.parameters(), lr=lr)
    else:
        raise ValueError("optimizer name not recognized")
    return optimizer


def load_model(modelparams, chkpaths, regression=False):
    net = ResNet1D(
        ResidualBlock,
        modelparams["convparam"], dp=modelparams["dropout"],
        inputsize=modelparams["window"],
        filter_rate=modelparams["filter_rate"],
        stride=modelparams["stride"],
        init_conv=int(modelparams["initconv"]),
        kernel_size=int(modelparams["kernel_size"]),
        dilations=modelparams["dilations"],
        pool_type=modelparams["pool_type"],
        pool_dim=int(modelparams["pool_dim"]),
        normtype=modelparams["normtype"],
        activation=modelparams["activation"],
        regression=modelparams["regression"])
    net.to(device)
    optimizer = get_optimizer(
        modelparams["optimizer"], net,
        modelparams["lr"])
    net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
    for eachpath in chkpaths:
        if os.path.exists(eachpath):
            net, optimizer = load_model_from_file(eachpath, net, optimizer)
            print("Loaded from {}".format(eachpath))
    if torch.cuda.device_count() > 1:
        print("Will use {} GPUs!".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)
    return net, optimizer


def make_default_args():
    # from defaults import *
    maxepoch = 100
    modelparams = {
        "filter_rate": 1.25,
        "optimize": "train",
        "dropout": 0.5,
        "lr": 0.001,
        "kernel_size": 20,
        "convparam": [1, 1, 1],
        "dilations": [1, 4, 8],
        "dilation": 1,
        "initconv": 64,
        "pool_dim": 40,
        "pool_type": "Average",
        "activation": "LeakyReLU",
        "optimizer": "Adam",
        "window": 10000,
        "ltype": 3,
        "regression": True,
        "normtype": "BatchNorm",
        "regularize": True,
        "stride": 1,
        "lambda_param": 0.01,
        "augmentations": [],
        "RESP_THRESH": 0.2,
        "LOSS_SCALERS": [10.0, 0.0, 1.0, 1.0],
        "SCALE_OP": "identity",
        "SCALE": [float(1), float(1)]}
    mavepath = "/scratch/hdd001/home/mkarimza/ciberAtac/" +\
        "10x/scviOutput/scVI-LVS-average.tsv"
    datadir = "/scratch/hdd001/home/mkarimza"
    indir = "/scratch/ssd001/home/mkarimza/data/ciberatac/pbmc10x/"
    outdir = "/scratch/ssd001/home/mkarimza/data/" +\
        "ciberatac/pbmc10x/trainedModels/" +\
        "20201228-scviAndContrastive_test"
    rnabwpaths = [
        indir + "rna/Natural_killer_rna.bigWig",
        indir + "rna/B-cells_rna.bigWig"]
    dnasebwpaths = [
        indir + 'atac/B-cells_treat_pileup.bigWig',
        indir + 'atac/CD14+_Mono_treat_pileup.bigWig',
        indir + 'atac/CD8+_T_treat_pileup.bigWig',
        indir + 'atac/DC_treat_pileup.bigWig',
        indir + 'atac/Memory_CD4+_treat_pileup.bigWig',
        indir + 'atac/Naive_CD4+_T_treat_pileup.bigWig',
        indir + 'atac/Natural_killer_treat_pileup.bigWig']
    rnabwpaths = [
        indir + 'rna/B-cells_rna.bigWig',
        indir + 'rna/CD14+_Mono_rna.bigWig',
        indir + 'rna/CD8+_T_rna.bigWig',
        indir + 'rna/DC_rna.bigWig',
        indir + 'rna/Memory_CD4+_rna.bigWig',
        indir + 'rna/Naive_CD4+_T_rna.bigWig',
        indir + 'rna/Natural_killer_rna.bigWig']
    bulkdnasepath = "/scratch/ssd001/home/mkarimza/data" +\
        "/ciberatac/pbmc10x/atac/all_cells_treat_pileup.bigWig"
    chrom = "chr10"
    bedpath = datadir +\
        "/meulemanData/signalMatrix/DHS_" +\
        "Index_and_Vocabulary_hg38_WM20190703.txt.gz"
    seqdir = datadir +\
        "/refData/genomeData/hg38/np"
    batchsize = 24
    window = 10000
    regression = True
    mask_nonpeaks = False
    train_chroms = ["chr{}".format(chrom) for chrom in range(1, 20)
                    if chrom not in [5, 6, 7]]
    list_args = [outdir, rnabwpaths, dnasebwpaths,
                 bulkdnasepath, mavepath,
                 chrom, bedpath, modelparams,
                 seqdir, batchsize, train_chroms,
                 regression, window, mask_nonpeaks, maxepoch]
    return list_args


def get_remaining(logdir, train_chroms, maxepoch, lr=0.001):
    # maxepoch = 15
    idxchrom = 0
    rm_epochs = list(range(maxepoch))
    rm_chroms = []
    used_chroms = []
    perfpaths = [
        each for each in os.listdir(logdir)
        if "Model_at_" in each]
    for each in perfpaths:
        modstr = each.split(".pt")[0]
        modstr = modstr.replace("Model_at_", "")
        used_chroms.extend(modstr.split("_"))
    rm_chroms = list(set(train_chroms) - set(used_chroms))
    rm_chroms.sort()

    # Not get the last epoch
    for i in range(len(train_chroms)):
        logpath = os.path.join(
            logdir,
            "modelLogUnSharednnn_lr{}_macrobatch{}.tsv".format(lr, i))
        if not os.path.exists(logpath):
            j = i - 1
            if j >= 0:
                adstr = "modelLogUnSharednnn_lr{}".format(lr)
                logpath = os.path.join(
                    logdir,
                    "{}_macrobatch{}.tsv".format(
                        adstr, j))
                if os.path.exists(logpath):
                    print(logpath)
                    logdf = pd.read_csv(logpath, sep="\t")
                    last_epoch = max(logdf["Epoch"])
                    rm_epochs = list(
                        range(last_epoch, maxepoch))
                    idxchrom = i
                    if last_epoch > 30:
                        rm_epochs = list(range(maxepoch))
                        idxchrom = i + 1
    return rm_chroms, rm_epochs, idxchrom


def load_model_from_file(chkpath, net, optimizer):
    from collections import OrderedDict
    checkpoint = torch.load(chkpath)
    print("Successfully loaded {}".format(chkpath))
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    new_state_dict2 = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
        k2 = k.replace("scvi", "mave")
        new_state_dict2[k2] = v
    try:
        net.load_state_dict(new_state_dict)
    except Exception:
        try:
            net.load_state_dict(new_state_dict2)
        except Exception:
            print("Check model parameter names")
            raise ValueError("Failed loading model")
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    print("Successfully loaded the model")
    return net, optimizer


def load_mavedf(mavepath, num_sampling=32):
    mavedf = pd.read_csv(mavepath, sep="\t", index_col=0)
    metapath = os.path.join(
        os.path.dirname(mavepath),
        "metadata.tsv.gz")
    mave = {}
    if os.path.exists(metapath) and mavedf.shape[0] > 100:
        metadf = pd.read_csv(metapath, sep="\t", index_col=0)
        for celltype in pd.unique(metadf["CellType"]):
            tempdf = metadf[metadf["CellType"] == celltype]
            newname = celltype.replace(" ", "_")
            idx_select = np.array(tempdf["Barcode.1"])
            idx_select = np.intersect1d(idx_select, mavedf.index)
            select_df = mavedf.loc[idx_select, ]
            medvals = np.apply_along_axis(
                np.mean, 0,
                np.array(select_df.iloc[:, :-1]))
            mave[newname] = medvals
            num_cells = int(tempdf.shape[0] / 4)
            for i in range(num_sampling):
                idx_select = np.random.choice(
                    np.array(tempdf["Barcode.1"]),
                    num_cells, True)
                idx_select = np.intersect1d(
                    idx_select, mavedf.index)
                select_df = mavedf.loc[idx_select, ]
                medvals = np.apply_along_axis(
                    np.mean, 0,
                    np.array(select_df.iloc[:, :-1]))
                adname = newname + ".{}".format(i)
                mave[adname] = medvals
    else:
        for celltype in mavedf.index:
            newname = celltype.replace(" ", "_")
            values = np.array(mavedf.loc[celltype])
            mave[newname] = values
    return mave


def rank_vals(array):
    order = array.argsort()
    ranks = order.argsort()
    return ranks


def get_rloss(response_tensor, predictions, regions):
    '''
    Ranking loss for each genomic region
    '''
    num_regs = 0
    num_regs = len(np.unique(regions))
    num_samples = sum(regions == regions[0])
    pred_rank_ar = np.zeros((num_regs, num_samples), dtype=float)
    resp_rank_ar = np.zeros((num_regs, num_samples), dtype=int)
    true_resp = np.zeros((num_regs, num_samples), dtype=float)
    k = 0
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)[0]
        rank_resp = rank_vals(response_tensor[idx_reg].reshape(-1))
        if len(rank_resp) == pred_rank_ar.shape[1]:
            true_resp[k] = response_tensor[idx_reg].reshape(-1)
            pred_rank_ar[k] = predictions[idx_reg].reshape(
                -1).cpu().detach().numpy()
            resp_rank_ar[k] = rank_resp
        k += 1
    y_pred = torch.from_numpy(pred_rank_ar[:k]).float().to(device)
    # y_pred = torch.nn.functional.softmax(y_pred, 1)
    y_true = torch.from_numpy(resp_rank_ar[:k]).float().to(device)
    rloss = rankNet(y_pred, y_true)
    return rloss


def get_region_bce(response_tensor, predictions, regions, celltypes):
    '''
    Cross entropy loss to determine which cell type
    has the highest accessibility in the region of
    interest
    '''
    from scipy.special import softmax
    unique_celltypes = np.sort(np.unique(celltypes))
    num_regs = len(np.unique(regions))
    num_samples = len(unique_celltypes)
    pred_ar = np.zeros((num_regs, num_samples), dtype=float)
    resp_ar = np.zeros(num_regs, dtype=int)
    k = 0
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)[0]
        if len(idx_reg) == num_samples:
            resp_temp = response_tensor[idx_reg]
            pred_temp = predictions[idx_reg]
            cur_celltypes = celltypes[idx_reg]
            idx_celltypes = np.argsort(cur_celltypes)
            resp_temp = resp_temp[idx_celltypes]
            pred_temp = pred_temp[idx_celltypes]
            idx_max = np.where(resp_temp == max(resp_temp))[0][0]
            resp_ar[k] = idx_max
            pred_ar[k, ] = softmax(pred_temp)
            k += 1
    bce_out = metrics.log_loss(
        resp_ar[:k], pred_ar[:k, ],
        labels=np.arange(unique_celltypes.shape[0]))
    return bce_out


def get_region_ce_torch(criterion_ce, response_tensor, predictions,
                        regions, celltypes):
    '''
    Cross entropy loss to determine which cell type
    has the highest accessibility in the region of
    interest
    '''
    from scipy.special import softmax
    unique_celltypes = np.sort(np.unique(celltypes))
    num_regs = len(np.unique(regions))
    num_samples = len(unique_celltypes)
    pred_ar = np.zeros((num_regs, num_samples), dtype=float)
    resp_ar = np.zeros(num_regs, dtype=int)
    k = 0
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)[0]
        if len(idx_reg) >= num_samples:
            if len(idx_reg) > num_samples:
                idx_reg = idx_reg[:num_samples]
            resp_temp = response_tensor[idx_reg, 0]
            pred_temp = predictions[idx_reg, 0].detach().cpu().numpy()
            cur_celltypes = celltypes[idx_reg]
            idx_celltypes = np.argsort(cur_celltypes)
            resp_temp = resp_temp[idx_celltypes]
            pred_temp = pred_temp[idx_celltypes]
            idx_max = np.where(resp_temp == max(resp_temp))[0][0]
            resp_ar[k] = idx_max
            pred_ar[k, ] = softmax(pred_temp)
            k += 1
    resp_tens = torch.from_numpy(resp_ar[:k]).to(device)
    pred_tens = torch.from_numpy(pred_ar[:k]).to(device)
    bce_out = criterion_ce(pred_tens, resp_tens)
    del resp_tens, pred_tens
    return bce_out


def get_embed_loss(criterion_ss, rna_embed, response_tensor,
                   regions, resp_top):
    loss_ss = torch.zeros(1).to(device)[0]
    num_regs = 0
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)[0]
        embed_temp = rna_embed[idx_reg]
        positive = embed_temp
        negative = embed_temp
        best_resp = max(response_tensor[idx_reg])
        worst_resp = min(response_tensor[idx_reg])
        idx_best = np.where(
            response_tensor[idx_reg] == best_resp)[0][0]
        idx_worst = np.where(
            response_tensor[idx_reg] == worst_resp)[0][0]
        positive = embed_temp[[idx_best] * len(idx_reg)]
        negative = embed_temp[[idx_worst] * len(idx_reg)]
        # Negative examples are already excluded
        idx_top = np.where(
            np.logical_and(
                response_tensor[idx_reg] > resp_top,
                response_tensor[idx_reg] != idx_best))[0]
        if len(idx_top) > 0:
            ad_loss = criterion_ss(
                embed_temp[idx_top],
                positive[idx_top],
                negative[idx_top])
            loss_ss += ad_loss
            num_regs += 1
            # print("{} regions SS loss: {} Responses {}".format(
            #   len(idx_top), ad_loss, response_tensor[idx_reg][idx_top]))
    if num_regs > 0:
        loss_ss = loss_ss / num_regs
    # print("Average across regoins ss loss: {}".format(loss_ss))
    return loss_ss


def train_motor(tensordict, net, optimizer, tidx,
                MINIBATCH, loss_scalers,
                criterion, criterion_ss, running_loss,
                resp_top, modelparams, epoch,
                criterion_direction, criterion_ce,
                use_ssn=False, resp_cutoff=0, respdiff=False):
    tidx_st = tidx * MINIBATCH
    tidx_end = (tidx + 1) * MINIBATCH
    dnase_tensor = torch.from_numpy(
        tensordict["DNase"][
            tidx_st:tidx_end]).to(device)
    rna_tensor = torch.from_numpy(
        tensordict["RNA"][
            tidx_st:tidx_end]).to(device)
    mave_tensor = torch.from_numpy(
        tensordict["mave"][tidx_st:tidx_end]).float().to(device)
    response_tensor = tensordict["Response"][tidx_st:tidx_end]
    if respdiff:
        response_tensor = tensordict["Resp.Diff"][tidx_st:tidx_end]
    optimizer.zero_grad()
    model_init, rna_embed = net(
        dnase_tensor,
        rna_tensor, mave_tensor)
    ce_loss = torch.zeros(1)[0]
    ss_loss = torch.zeros(1)[0]
    bceloss = torch.zeros(1)[0]
    if use_ssn:
        # rloss = get_rloss(
        #     response_tensor, model_init,
        #     tensordict["Regions"][tidx_st:tidx_end])
        bceloss = get_region_ce_torch(
            criterion_ce, response_tensor, model_init,
            tensordict["Regions"][tidx_st:tidx_end],
            tensordict["Tissues"][tidx_st:tidx_end])
        ss_loss = get_embed_loss(
            criterion_ss, rna_embed, response_tensor,
            tensordict["Regions"][tidx_st:tidx_end],
            resp_top)
        ce_loss = get_ce_loss(
            criterion_direction, response_tensor,
            model_init, tensordict["Regions"][tidx_st:tidx_end],
            resp_cutoff=resp_cutoff, bce=False)
        if torch.isnan(ss_loss):
            print("Triplet margin loss: {}".format(ss_loss))
            import joblib
            out_dict = {"rna_embed": rna_embed.detach().cpu(),
                        "response_tensor": response_tensor,
                        "regions": tensordict["Regions"][tidx_st:tidx_end],
                        "resp_top": resp_top}
            joblib.dump(out_dict,
                        "Troublesom_triplet-margin-data.pickle",
                        compress=9)
            raise ValueError("Triplet margin loss failed")
    reg_loss = criterion(
        model_init,
        torch.from_numpy(
            response_tensor).to(device))
    if torch.isnan(reg_loss):
        print("L1 smooth loss: {}".format(reg_loss))
        print(model_init)
        print(response_tensor)
        raise ValueError("L1 smooth loss failed")
    if tidx == 0 and epoch % 25 == 0:
        print("L1 smooth loss: {}".format(reg_loss))
        if use_ssn:
            print("Pairwise MSE loss: {}".format(ce_loss))
    if use_ssn:
        loss = (
            (reg_loss * torch.tensor(loss_scalers[0])) +
            (ss_loss * torch.tensor(loss_scalers[1])) +
            (ce_loss * torch.tensor(loss_scalers[2])) +
            (bceloss * torch.tensor(loss_scalers[3])))
    if modelparams["regularize"]:
        loss = regularize_loss(modelparams, net, loss)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    current_loss = reg_loss.item()
    running_loss += current_loss
    del dnase_tensor, rna_tensor, mave_tensor
    # del labels, label_idx
    torch.cuda.empty_cache()
    return running_loss, optimizer, net, ce_loss, bceloss


def to_dev(cur_ar):
    out_tensor = torch.from_numpy(cur_ar).reshape(-1, 1).to(device)
    return out_tensor


def apply_model(tensordict, net, chrom,
                logdir, idxchrom, criterion,
                batch_size=16, SCALE_FACTOR=1,
                resp_cutoff=0.5, loss_scalers=[1, 1, 1]):
    tensorname = "Test"
    regions = tensordict["Regions"]
    dict_perf = {}
    dnase_tensor = torch.from_numpy(
            tensordict["DNase"])
    midvals = tensordict["Averages"][:, 0]
    rna_tensor = torch.from_numpy(
        tensordict["RNA"])
    response_tensor = tensordict["Response"]
    outputs = np.zeros((rna_tensor.shape[0], 1))
    running_loss = 0
    dim_eval = int(dnase_tensor.shape[0] / batch_size)
    num_parts = 0
    for tidx in range(dim_eval):
        tidx_st = tidx * batch_size
        tidx_end = min(
            [(tidx + 1) * batch_size, rna_tensor.shape[0]])
        if tidx_st >= rna_tensor.shape[0]:
            break
        dnase = dnase_tensor[tidx_st:tidx_end]
        rna = rna_tensor[tidx_st:tidx_end]
        if "mave" in tensordict.keys():
            train_tensor = [0]
            dnase = dnase.to(device)
            rna = rna.to(device)
            mave_tensor = torch.from_numpy(
                tensordict["mave"][tidx_st:tidx_end]).float().to(device)
            output, _ = net(
                dnase, rna, mave_tensor)
            del mave_tensor
        else:
            train_tensor = [0]
            dnase = dnase.to(device)
            rna = rna.to(device)
            output, _ = net(
                dnase,
                rna)
        del dnase, rna, train_tensor
        loss = criterion(
            output,
            torch.from_numpy(
                response_tensor[tidx_st:tidx_end]).to(device))
        running_loss += loss.item()
        output_ar = output.cpu().detach().numpy()
        outputs[tidx_st:tidx_end] = output_ar
        num_parts += 1
        del output
        torch.cuda.empty_cache()
    tuning_loss = running_loss / num_parts
    pred_add = outputs[:, 0]
    resp_add = response_tensor[:, 0]
    tuning_loss = loss.item()
    perfdf = pd.DataFrame(
        {"Prediction": pred_add,
         "Response": resp_add,
         "Average.DNase": midvals,
         "Regions": regions})
    dict_perf["{}.Loss".format(tensorname)] = tuning_loss
    dict_perf["{}.MSE".format(tensorname)] = \
        metrics.mean_squared_error(
            perfdf["Response"] / max(perfdf["Response"]),
            perfdf["Prediction"] / max(perfdf["Prediction"]))
    corval, pval = stats.pearsonr(
        perfdf["Response"], perfdf["Prediction"])
    dict_perf["{}.R".format(tensorname)] = corval
    corval, pval = stats.pearsonr(
        perfdf["Response"], perfdf["Average.DNase"])
    dict_perf["baseline.R"] = corval
    dict_perf["baseline.MSE"] = metrics.mean_squared_error(
        perfdf["Response"] / max(perfdf["Response"]),
        perfdf["Average.DNase"] / max(perfdf["Average.DNase"]))
    try:
        dict_perf["pairwiseMSE"] = get_bce(
            np.array(perfdf["Response"]),
            np.array(perfdf["Prediction"]),
            np.array(perfdf["Regions"]),
            resp_cutoff, bce=False) * loss_scalers[2]
        dict_perf["baseline.pairwiseMSE"] = get_bce(
            np.array(perfdf["Response"]),
            np.array(perfdf["Average.DNase"]),
            np.array(perfdf["Regions"]),
            resp_cutoff, bce=False) * loss_scalers[2]
    except Exception:
        print("oops! Failed at BCE!")
        import joblib
        joblib.dump(
            perfdf, "{}_trouble-causing_perfdf.joblib".format(chrom),
            compress=9)
    for eachkey, eachval in dict_perf.items():
        perfdf[eachkey] = eachval
    for each_key in ["Regions", "Samples", "Tissues"]:
        if each_key in tensordict.keys():
            perfdf[each_key] = [
                each.decode() for each in
                tensordict[each_key]]
    perfpath = os.path.join(
        logdir,
        "{}_testSetPredictions.tsv.gz".format(chrom))
    perfdf.to_csv(
        perfpath,
        sep="\t", compression="gzip")
    sns_plot = sns.relplot(
        y="Response", x="Prediction", hue="Average.DNase",
        size="Average.DNase", data=perfdf, height=6, aspect=1.5)
    sns_plot.savefig(
        perfpath.replace(".tsv.gz", ".pdf"))
    sns_plot.savefig(
        perfpath.replace(".tsv.gz", ".png"))
    return perfdf


def get_augmentation_dicts(tensordict, augmentations):
    '''
    tensordict: a dictionary containing keys DNase,
                RNA, and response each being a numpy array
    augmentaitons: a list containing one or more of
                   reverse_complement, mask_background,
                   and mask_signal
    '''
    from utils import Augmentator
    outdict = {}
    ars = [tensordict["DNase"], tensordict["RNA"]]
    AugClass = Augmentator(ars, tensordict["Response"])
    for each_aug in augmentations:
        outdict[each_aug] = tensordict.copy()
        if each_aug == "reverse_complement":
            new_ars, newresp = AugClass.reverse_complement()
            outdict[each_aug]["DNase"] = new_ars[0]
            outdict[each_aug]["RNA"] = new_ars[1]
            outdict[each_aug]["Response"] = newresp
    return outdict


def train_step(tensordict_all, net, optimizer, chrom, rm_epochs,
               logdir, criterion, criterion_ss, regression,
               modelpath_bestloss, chkpaths, modelpath,
               idxchrom, maxepoch, modelparams, criterion_direction,
               criterion_ce, use_ssn=True, loss_scalers=[1, .1, 0, 10],
               resp_cutoff=0, respdiff=False, augmentations=[]):
    curlogpath = "NA"
    tensordict, tensordict_tune = split_tensordict(
            tensordict_all, ratio=0.8)
    dict_augs = get_augmentation_dicts(tensordict, augmentations)
    resp_top = np.quantile(
        tensordict["Response"], 0.75)
    # MINIBATCH = 40 * torch.cuda.device_count()
    MINIBATCH = len(np.unique(tensordict_all["Samples"])) *\
        6 * torch.cuda.device_count()
    dim_train = int((tensordict["DNase"].shape[0]) / MINIBATCH)
    base_r = -1
    base_loss = 1000
    bad_epochs = 0
    dict_perf, perfdf = assess_performance(
        net, tensordict_tune, criterion,
        tensordict, int(dim_train / 4), int(MINIBATCH / 2),
        regression=regression, loss_scalers=loss_scalers,
        resp_cutoff=resp_cutoff, respdiff=respdiff)
    _ = motor_log(0, dim_train - 1, dict_perf, modelparams["lr"],
                  logdir, 1, net, modelpath,
                  idxchrom, regression=regression)
    # loss_scalers = [10, 1, 0.1]
    for epoch in rm_epochs:
        running_bceloss = 0
        running_ce_loss = 0
        running_loss = 0
        for tidx in range(dim_train):
            running_loss, optimizer, net, ce_loss, bceloss = train_motor(
                tensordict, net, optimizer, tidx, MINIBATCH, loss_scalers,
                criterion, criterion_ss, running_loss,
                resp_top, modelparams, epoch, criterion_direction,
                criterion_ce, use_ssn, resp_cutoff=resp_cutoff,
                respdiff=respdiff)
            for each_aug in augmentations:
                aug_loss, optimizer, net, augce_loss, augbceloss = \
                    train_motor(dict_augs[each_aug], net, optimizer, tidx,
                                MINIBATCH, loss_scalers,
                                criterion, criterion_ss, running_loss,
                                resp_top, modelparams, epoch,
                                criterion_direction, criterion_ce,
                                use_ssn, resp_cutoff=resp_cutoff,
                                respdiff=respdiff)
                running_loss += aug_loss
                ce_loss += augce_loss
                bceloss += augbceloss
            # print("{}/{}\ttotal: {}\tpme: {}\trank: {}".format(
            #         tidx, dim_train, running_loss, ce_loss, rloss))
            running_ce_loss += ce_loss
            running_bceloss += bceloss
        current_loss = running_loss / dim_train
        current_ce_loss = running_ce_loss / dim_train
        current_bceloss = running_bceloss / dim_train
        print("Epoch {}/100 loss: {}\nMSE: {}\nbceloss: {}".format(
                epoch, current_loss, current_ce_loss, current_bceloss))
        dict_perf, perfdf = assess_performance(
            net, tensordict_tune, criterion,
            tensordict, int(dim_train / 4), int(MINIBATCH / 2),
            regression=regression, loss_scalers=loss_scalers,
            resp_cutoff=resp_cutoff)
        cur_r = dict_perf["Tuning.R"]
        tune_loss = dict_perf["Tuning.pMSE"] +\
            dict_perf["Tuning.bceloss"] + dict_perf["Tuning.Loss"]
        if tune_loss < base_loss or cur_r > base_r:
            bad_epochs = 0
            base_r = cur_r
            base_loss = tune_loss
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                "modelparams": modelparams
            }
            torch.save(
                checkpoint,
                modelpath_bestloss.replace(".pt", "-bestRmodel.pt"))
        elif epoch > 60:
            bad_epochs += 1
            if bad_epochs > 5 and base_r > 0.1:
                print("Exiting batch after loading best")
                bestpath = modelpath_bestloss.replace(
                    ".pt", "-bestRmodel.pt")
                del net, optimizer
                torch.cuda.empty_cache()
                net, optimizer = load_model(
                    modelparams, [bestpath], regression)
                break
        if epoch % 5 == 0 and epoch > 0:
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                "modelparams": modelparams
            }
            torch.save(checkpoint, modelpath_bestloss)
            for chkpath in chkpaths:
                shutil.copyfile(modelpath_bestloss, chkpath)
            perfdf["Epoch"] = epoch
            perfdf["Macrobatch"] = chrom
            perfdf.to_csv(
                os.path.join(
                    logdir,
                    "BestPerformanceOnTuningSet_{}.tsv.gz".format(chrom)),
                sep='\t', compression="gzip", index=None)
            if epoch > 9:
                try:
                    plot_epoch_perf(curlogpath)
                except Exception:
                    print(curlogpath)
        curlogpath = motor_log(
            epoch + 1, tidx, dict_perf, modelparams["lr"],
            logdir, current_loss, net, modelpath,
            idxchrom, regression=regression)
    modelpath_chorm = os.path.join(
        logdir, "Model_at_{}.pt".format(chrom))
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict(),
        "modelparams": modelparams
    }
    torch.save(checkpoint, modelpath_chorm)
    idxchrom += 1
    del tensordict, tensordict_tune
    torch.cuda.empty_cache()
    plot_epoch_perf(curlogpath)
    rm_epochs = list(range(maxepoch))
    return net, optimizer, rm_epochs


def plot_epoch_perf(curlogpath):
    if not os.path.exists(curlogpath):
        return
    imgpath = curlogpath.replace(".tsv", ".png")
    pdfpath = curlogpath.replace(".tsv", ".pdf")
    logdf = pd.read_csv(curlogpath, sep="\t")
    cur_colnames = list(logdf.columns)
    cur_colnames = [each.replace("g.MSE", "g.pMSE") for
                    each in cur_colnames]
    logdf.columns = cur_colnames
    losses = list(logdf["Training.Loss"]) +\
        list(logdf["Tuning.Loss"]) +\
        list(logdf["Training.pMSE"]) +\
        list(logdf["Tuning.pMSE"]) +\
        list(logdf["Training.bceloss"]) +\
        list(logdf["Tuning.bceloss"]) +\
        list(logdf["Training.R"]) +\
        list(logdf["Tuning.R"])
    datasets = (list(["Training"] * logdf.shape[0]) +
                list(["Tuning"]) * logdf.shape[0]) * 4
    losstypes = list(
        ["Smooth L1"] * logdf.shape[0] * 2) +\
        list(["Pairwise MSE"] * logdf.shape[0] * 2) +\
        list(["RankNet"] * logdf.shape[0] * 2) +\
        list(["Peason R"] * logdf.shape[0] * 2)
    newdict = {
        "Epoch": list(logdf["Epoch"]) * 8,
        "Loss": losses,
        "Dataset": datasets,
        "Loss type": losstypes}
    newdf = pd.DataFrame(newdict)
    sns_plot = sns.relplot(
        data=newdf,
        x="Epoch", y="Loss",
        kind="line",
        col="Loss type",
        hue="Dataset",
        facet_kws=dict(sharey=False),
        style="Dataset")
    sns_plot.axes[0, 3].set_ylim(0, 1)
    sns_plot.savefig(imgpath)
    sns_plot.savefig(pdfpath)


def merge_batches(list_tensordicts, list_resp_cutoffs):
    new_dict = list_tensordicts[0]
    for i in range(1, len(list_tensordicts)):
        for key, value in new_dict.items():
            adval = list_tensordicts[i][key]
            newvals = np.concatenate((value, adval))
            new_dict[key] = newvals
    regions = new_dict["Regions"]
    idx_sort_regions = np.zeros(len(regions), dtype=int)
    unique_regions = np.unique(regions)
    np.random.shuffle(unique_regions)
    i = 0
    for region in unique_regions:
        idxs = np.where(regions == region)[0]
        j = i + len(idxs)
        idx_sort_regions[i:j] = idxs
        i = i + len(idxs)
    tensordict = {}
    for key, val in new_dict.items():
        tensordict[key] = val[idx_sort_regions]
    out_av = np.mean(np.array(list_resp_cutoffs))
    return tensordict, out_av


def main(outdir, mavepath, rnabwpaths, dnasebwpaths,
         bulkdnasepath, bedpath, batchsize,
         seqdir, mask_nonpeaks, train_chroms,
         modelparams, window=10000, regression=False, maxepoch=100,
         dont_train=False, adname_apply="CiberATAC",
         SCALE_FACTORS=[1, 1], scale_operation="identity",
         train_general=False, loss_scalers=[1.0, 1.0, 1.0, 1.0],
         main_loss="SmoothL1", pretrained_path="NA",
         vae_names="NA", arcsinh=False, respdiff=False,
         augmentations=[], num_contrastive_regions=[700, 400],
         num_chroms_per_batch=4, input_normalize="None"):
    if vae_names == "NA":
        vae_names = process_vae_names(["NA"], rnabwpaths)
    dictpaths = compile_paths(outdir, modelparams)
    chkpaths = dictpaths["chkpaths"]
    adname = dictpaths["adname"]
    modelpath = dictpaths["modelpath"]
    modelpath_bestloss = dictpaths["modelpath_bestloss"]
    resp_thresh = modelparams["RESP_THRESH"]
    check_paths = [modelpath, modelpath_bestloss] + chkpaths
    mave = load_mavedf(mavepath)
    if os.path.exists(pretrained_path):
        check_paths = [pretrained_path]
        print("Will not load check points, will use pre-trained")
        print("Pretrained model: {}".format(pretrained_path))
    net, optimizer = load_model(
        modelparams, check_paths, regression)
    logdir = os.path.join(
        outdir, "modelLog", adname)
    os.makedirs(logdir, exist_ok=True)
    rm_chroms, rm_epochs, idxchrom = get_remaining(
        logdir, train_chroms, maxepoch, modelparams["lr"])
    rm_chroms.sort()
    if main_loss == "SmoothL1":
        criterion = torch.nn.SmoothL1Loss().to(device)
    else:
        criterion = torch.nn.MSELoss().to(device)
    criterion_ss = nn.TripletMarginLoss()
    criterion_direction = nn.MSELoss().to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    if dont_train:
        rm_chroms = train_chroms
    num_chrom_parts = int(len(rm_chroms) / num_chroms_per_batch)
    for idx_chrom in range(num_chrom_parts):
        idx_chrom_st = idx_chrom * num_chroms_per_batch
        idx_chrom_end = min(
            [len(rm_chroms), (idx_chrom + 1) * num_chroms_per_batch])
        cur_chroms = rm_chroms[idx_chrom_st:idx_chrom_end]
        list_tensordicts = []
        list_resp_cutoffs = []
        for chrom in cur_chroms:
            tensordict_chrom, resp_cutoff, bed_temp = get_batch(
                rnabwpaths, dnasebwpaths,
                bulkdnasepath, bedpath, batchsize,
                seqdir, mask_nonpeaks, chrom,
                mave, regression=regression,
                force_tissue_negatives=True,
                dont_train=dont_train,
                SCALE_FACTORS=SCALE_FACTORS,
                SCALE_OP=scale_operation,
                RESP_THRESH=resp_thresh,
                vae_names=vae_names, arcsinh=arcsinh,
                num_contrastive_regions=num_contrastive_regions,
                input_normalize=input_normalize)
            list_tensordicts.append(tensordict_chrom)
            list_resp_cutoffs.append(resp_cutoff)
            bed_temp.to_csv(
                os.path.join(outdir, "{}_bed.tsv.gz".format(chrom)),
                sep="\t", compression="gzip")
        tensordict_all, resp_cutoff = merge_batches(
            list_tensordicts, list_resp_cutoffs)
        del list_tensordicts, list_resp_cutoffs
        chrom_str = "_".join(cur_chroms)
        if dont_train:
            adname_char = "{}_{}".format(
                adname_apply, chrom)
            apply_model(
                tensordict_all, net, adname_char,
                logdir, idxchrom, criterion,
                resp_cutoff=resp_cutoff)
        else:
            net, optimizer, rm_epochs = train_step(
                tensordict_all, net, optimizer, chrom_str,
                rm_epochs, logdir, criterion, criterion_ss, regression,
                modelpath_bestloss, chkpaths, modelpath,
                idxchrom, maxepoch, modelparams, criterion_direction,
                criterion_ce, use_ssn=True, loss_scalers=loss_scalers,
                resp_cutoff=resp_cutoff, respdiff=respdiff,
                augmentations=augmentations)
            adname_char = "{}_{}".format(
                "TrainingPerformance_contrastive", chrom_str)
            _ = apply_model(
                tensordict_all, net, adname_char,
                logdir, idxchrom, criterion,
                resp_cutoff=resp_cutoff)
            from predict import main as main_predict
            outpath_pred = os.path.join(
                logdir, "{}_fullChromPerf_{}.tsv.gz".format(
                    chrom_str, cur_chroms[0]))
            main_predict(outpath_pred, rnabwpaths[0], bulkdnasepath,
                         bedpath, bulkdnasepath, rnabwpaths[0],
                         modelpath_bestloss, 16, seqdir,
                         False, mavepath, vae_names[0],
                         dnasebwpaths[0], SCALE_FACTORS, scale_operation,
                         chrom=cur_chroms[0], input_normalize=input_normalize)
        idxchrom += 1
        del tensordict_all
        torch.cuda.empty_cache()


def get_scale_factors(bwpaths, chrom):
    SCALE_FACTORS = []
    for eachpath in bwpaths:
        AD_SCALE = 1
        bwobj = pyBigWig.open(
            eachpath, "rb")
        chromsize = bwobj.chroms()[chrom]
        values = bwobj.values(chrom, 0, chromsize, numpy=True)
        values[np.isnan(values)] = 0
        max_val = np.max(values)
        AD_SCALE = 100 / float(max_val)
        SCALE_FACTORS.append(AD_SCALE)
        bwobj.close()
        print("Max of {} was {} and changed to {}".format(
                eachpath, max_val, max_val * AD_SCALE))
    return SCALE_FACTORS


def get_batch(rnabwpaths,
              dnasebwpaths, bulkdnasepath, bedpath,
              batchsize, seqdir, mask_nonpeaks,
              chrom, mave,
              window=10000, regression=False,
              force_tissue_negatives=False,
              dont_train=False,
              SCALE_FACTORS=[1, 100],
              SCALE_OP="identity",
              RESP_THRESH=0.2,
              vae_names=["NA"],
              num_mave_samples=32, arcsinh=False,
              num_contrastive_regions=[700, 400],
              input_normalize="None"):
    if vae_names[0] == "NA":
        vae_names = process_vae_names(vae_names, rnabwpaths)
    # SCALE_FACTORS = get_scale_factors(
    #     [rnabwpaths[0], dnasebwpaths[0]], chrom)
    # SCALE_FACTORS = [1, 100]
    DataObj = DataHandler(
        rnabwpaths, dnasebwpaths,
        bulkdnasepath, bedpath, seqdir,
        mask_nonpeaks=mask_nonpeaks,
        force_tissue_negatives=force_tissue_negatives,
        dont_train=dont_train,
        SCALE_FACTORS=SCALE_FACTORS,
        SCALE_OP=SCALE_OP, arcsinh=arcsinh,
        input_normalize=input_normalize)
    DataObj.get_batch_nums(chrom, batchsize)
    resp_cutoff = DataObj.annotate_bed(SCALE_FACTORS[1])
    regions_to_use = DataObj.get_region_poses(
        num_contrastive_regions[0],
        num_contrastive_regions[1])
    num_batches = int(regions_to_use.shape[0] / batchsize / 1)
    TOTBATCHIDX = num_batches * len(rnabwpaths) * batchsize
    NUMPARTS = int(regions_to_use.shape[0] / batchsize)
    if num_batches > NUMPARTS:
        num_batches = NUMPARTS
        TOTBATCHIDX = len(rnabwpaths) * batchsize * num_batches
    train1 = np.zeros(
        (TOTBATCHIDX, 4,
         window * 2), dtype=np.float32)
    train2 = np.zeros(
        (TOTBATCHIDX,
         4, window * 2), dtype=np.float32)
    response = np.zeros(
        (TOTBATCHIDX, 1),
        dtype=int)
    mavemat = np.zeros(
        (TOTBATCHIDX,
         list(mave.values())[0].shape[0]))
    if regression:
        response = np.zeros(
            (TOTBATCHIDX, 1),
            dtype=np.float32)
    averages = np.zeros(
        (TOTBATCHIDX, 1),
        dtype=np.float32)
    regions = np.zeros(
        (TOTBATCHIDX),
        dtype="|S32")
    samples = np.zeros(
        (TOTBATCHIDX),
        dtype="|S32")
    i_st = 0
    i_end = 0
    # num_batches = batchsize * 200
    dict_temp = {}
    for j in range(len(rnabwpaths)):
        print("Loading signal {}/{}".format(j, len(rnabwpaths)))
        rna, dnase, positions, avg_dnase, resp = DataObj.get_batches(
            regions_to_use[:, 0], regions_to_use[:, 1], rnabwpaths[j],
            dnasebwpaths[j], SCALE_FACTORS)
        dict_temp[j] = {"rna": rna, "dnase": dnase, "positions": positions,
                        "avg_dnase": avg_dnase, "resp": resp}
    for i in range(num_batches):
        idx_st = i * batchsize
        idx_end = (i + 1) * batchsize
        if idx_end > regions_to_use.shape[0]:
            idx_end = regions_to_use.shape[0]
        start_poses = regions_to_use[idx_st:idx_end, 0]
        # end_poses = regions_to_use[idx_st:idx_end, 1]
        curbatchsize = idx_end - idx_st
        for j in range(len(rnabwpaths)):
            i_end = i_st + len(start_poses)
            if i_end > train1.shape[0]:
                i_end = train1.shape[0]
            # adname = os.path.basename(rnabwpaths[j])
            adname = vae_names[j]
            rna = dict_temp[j]["rna"][idx_st:idx_end]
            dnase = dict_temp[j]["dnase"][idx_st:idx_end]
            positions = dict_temp[j]["positions"][idx_st:idx_end]
            avg_dnase = dict_temp[j]["avg_dnase"][idx_st:idx_end]
            resp = dict_temp[j]["resp"][idx_st:idx_end]
            # rna, dnase, positions, avg_dnase, resp = DataObj.get_batches(
            #     start_poses, end_poses, rnabwpaths[j],
            #     dnasebwpaths[j], SCALE_FACTORS)
            # resp_cutoff = 0.5 * SCALE_FACTORS[1]
            # resp[resp < resp_cutoff] = 0
            try:
                train1[i_st:i_end] = dnase[:curbatchsize]
            except Exception:
                print("Train shape is {}".format(train1.shape))
                print("DNase shape os {}".format(dnase.shape))
                print("Error at {}:{} to {}".format(i_st, i_end, curbatchsize))
                print("Batch {}, {} to {}, {} to {}".format(
                        i, idx_st, idx_end, i_st, i_end))
                raise ValueError("")
            train2[i_st:i_end] = rna[:curbatchsize]
            if regression:
                response[i_st:i_end, 0] = resp[:curbatchsize]
            else:
                response[i_st:i_end, 0] = np.array(
                    resp[:curbatchsize] > DataObj.cutoff_10p, dtype=int)
            try:
                regions[i_st:i_end] = np.core.defchararray.add(
                    np.array([chrom + "."] * curbatchsize),
                    np.array(positions[:curbatchsize], dtype="U32"))
            except Exception:
                print("Train shape is {}".format(train1.shape))
                print("DNase shape is {}".format(dnase.shape))
                print("Region shape is {}".format(regions.shape))
                print("Positions shape is {}".format(positions.shape))
                print("Error at {}:{} to {}".format(i_st, i_end, curbatchsize))
                print("Batch {}, {} to {}, {} to {}".format(
                        i, idx_st, idx_end, i_st, i_end))
                raise ValueError("")
            averages[i_st:i_end, 0] = avg_dnase[:curbatchsize]
            samples[i_st:i_end] = np.array([adname] * curbatchsize)
            if len(list(mave.keys())) > 100:
                for l in range(i_st, i_end):
                    rand_num = np.random.choice(
                        np.arange(num_mave_samples), 1)[0]
                    adname_temp = adname + ".{}".format(rand_num)
                    mavemat[l, :] = np.array(
                        mave[adname_temp])
            else:
                mavemat[i_st:i_end, :] = np.array(
                    mave[adname])
            i_st = i_end
        if i % 10 == 0:
            print("{}/{} regions added".format(i_st, TOTBATCHIDX))
    # Use regions
    idx_sort_regions = np.zeros(len(regions), dtype=int)
    unique_regions = np.unique(regions)
    np.random.shuffle(unique_regions)
    i = 0
    for region in unique_regions:
        idxs = np.where(regions == region)[0]
        j = i + len(idxs)
        idx_sort_regions[i:j] = idxs
        i = i + len(idxs)
    tensordict = {
        "DNase": train1[idx_sort_regions],
        "RNA": train2[idx_sort_regions],
        "Averages": averages[idx_sort_regions],
        "Regions": regions[idx_sort_regions],
        "Samples": samples[idx_sort_regions],
        "Tissues": samples[idx_sort_regions],
        "Response": response[idx_sort_regions],
        "mave": mavemat[idx_sort_regions]}
    tensordict["Resp.Diff"] = \
        tensordict["Response"] - tensordict["Averages"]
    return tensordict, resp_cutoff, DataObj.bed


def process_vae_names(vae_names, rna_paths):
    if len(vae_names) == len(rna_paths) and vae_names[0] != "NA":
        return vae_names
    else:
        rna_names = [
            os.path.basename(each).replace("_rna.bigWig", "")
            for each in rna_paths]
        print("Will use {}".format(rna_names))
        return rna_names


def adjust_model_params(args):
    modelparams = {
        "optimize": "train",
        "dropout": args.dropout,
        "regression": args.regression,
        "lr": args.lr,
        "ltype": args.ltype,
        "kernel_size": args.kernel_size,
        "convparam": args.convparam,
        "dilations": args.dilations,
        "initconv": args.initconv,
        "stride": args.stride,
        "filter_rate": args.filter_rate,
        "pool_dim": args.pool_dim,
        "pool_type": args.pool_type,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "window": args.window,
        "normtype": args.normtype,
        "regularize": args.regularize,
        "lambda_param": args.lambda_param,
        "SCALE": args.scalers,
        "SCALE_OP": args.scale_operation,
        "LOSS_SCALERS": args.loss_scalers,
        "RESP_THRESH": args.resp_thresh,
        "arcsinh": args.arcsinh,
        "respdiff": args.respdiff,
        "augmentations": args.augmentations,
        "input_normalize": args.input_normalize}
    return modelparams


if __name__ == "__main__":
    def_chroms = ["chr{}".format(chrom) for chrom in
                  list(range(1, 5)) +
                  list(range(8, 15))]
    parser = ArgumentParser(
        description="Predict enhancer activity "
        "using the CiberATAC model. Requires "
        "a bigWig file for the transcriptome "
        "and a bigWig file for the chromatin "
        "accessibility. It also requires "
        "a BED file for list of potential "
        "enhancer to predict on.")
    parser.add_argument(
        "outdir",
        help="Path to directory for saving "
        "model training logs")
    parser.add_argument(
        "mavepath",
        help="Path to the matrix of SCVI averaged data")
    parser.add_argument(
        "bedpath",
        help="Path to a BED file for "
        "list of regions to predict on")
    parser.add_argument(
        "--rnabwpaths",
        nargs="*",
        help="Path to bigWig files of the "
        "transcriptome measures in each cluster")
    parser.add_argument(
        "--dnasebwpaths",
        nargs="*",
        help="Path to bigWig files of chromatin "
        "accessibility (same order as --rnabwpaths)")
    parser.add_argument(
        "--bulkdnasepath",
        help="Bulk DNase-seq path")
    parser.add_argument(
        "--batchsize",
        default=100,
        type=int,
        help="Number of simultaneous batches to"
        "generate and feed into the GPU")
    parser.add_argument(
        "--seqdir",
        required=True,
        help="Path to directory with files named "
        "as <chromosome>_sequence.numpy.gz")
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Specify if using regression instead")
    parser.add_argument(
        "--chroms",
        default=def_chroms,
        nargs="*",
        help="Space-separated list of chroms to use. "
        "default to: {}".format(def_chroms))
    parser.add_argument(
        "--mask-nonpeaks",
        action="store_true",
        help="If specified, will limit the DNase/ATAC-seq "
        "signal to regions within the BED file.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate")
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout probability")
    parser.add_argument(
        "--optimize",
        default="train",
        choices=["train", "tune"],
        help="either train or tune for setting the number "
        "of epochs without improvement")
    parser.add_argument(
        "--convparam",
        nargs="*",
        type=int,
        default=[2, 2, 2],
        help="Convolution parameters. Defaults to "
        "--convparam 1 1 1")
    parser.add_argument(
        "--initconv",
        default=16,
        type=int,
        help="Number of initial convolutional filters in ResNet")
    parser.add_argument(
        "--kernel-size",
        default=20,
        type=int,
        help="Kernel size of ResNet. Defaults to 3")
    parser.add_argument(
        "--dilations",
        nargs="*",
        default=[1, 1, 1, 1],
        type=int,
        help="Space-separated list of dilation "
        "for each of the convolutional layers")
    parser.add_argument(
        "--pool-type",
        default="Average",
        choices=["Average", "Max"],
        help="Pooling parameter")
    parser.add_argument(
        "--pool-dim",
        default=20,
        type=int,
        help="Dimension of pooling parameter")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Convolution stride")
    parser.add_argument(
        "--activation",
        choices=["ReLU", "LeakyReLU", "GELU"],
        default="LeakyReLU",
        help="Activateion function: LeakyReLU, ReLU, or GELU")
    parser.add_argument(
        "--optimizer",
        choices=["SGD", "Adabound", "Adagrad", "Adam"],
        default="Adabound",
        help="One of SGD, Adabound, Adagrad, or Adam.")
    parser.add_argument(
        "--augmentations",
        nargs="*",
        required=False,
        default=[],
        help="Space separated list of one or more of the "
        "augmentation options reverse_complement, "
        "mask_background, and mask_signal")
    parser.add_argument(
        "--regularize",
        action="store_true",
        help="Will perform either L1, L2, or gradient clip "
        "depending on --ltype values.")
    parser.add_argument(
        "--ltype",
        type=int,
        default=3,
        help="If 1 or 2, L1 or L2. If 3, "
        "then clip norming. If 4, L1 and L2.")
    parser.add_argument(
        "--lambda-param",
        type=float,
        help="Lambda regularization parameter",
        default=1)
    parser.add_argument(
        "--window",
        type=int,
        help="Genomic region size. Def. 10000",
        default=10000)
    parser.add_argument(
        "--filter-rate",
        type=float,
        default=2,
        help="Rate of changing number of filters")
    parser.add_argument(
        "--normtype",
        default="BatchNorm",
        help="BatchNorm or LayerNorm",
        choices=["BatchNorm", "LayerNorm"])
    parser.add_argument(
        "--dont-train",
        action="store_true",
        help="If specified, will generate all-chromosome "
        "batches and apply the model on each chromosome "
        "and save the data")
    parser.add_argument(
        "--adname",
        default="CiberATAC",
        help="Character to add to name of the file "
        "when --dont-train is applied")
    parser.add_argument(
        "--scalers",
        nargs="*",
        type=float,
        default=[1, 100],
        help="Scaling factors for RNA and ATAC-seq")
    parser.add_argument(
        "--resp-thresh",
        default=0.1,
        help="Quantile of response threshold to ignore "
        "when calculating the pairwise MSE difference")
    parser.add_argument(
        "--scale-operation",
        default="identity",
        choices=["identity", "log2", "sqrt"],
        help="Specify if you want to apply one of "
        "sqrt or log2 on input values. In case of"
        "log2 it will perform (log2(non-zero-values + 1))")
    parser.add_argument(
        "--train-general",
        action="store_true",
        help="If specified, will also train on only "
        "two samples with non-contrastive examples "
        "to boost chromosome-wide performance")
    parser.add_argument(
        "--loss-scalers",
        nargs="*",
        type=float,
        default=[0.1, 0, 10, 1],
        help="Specify loss scalers for L1 smooth loss, "
        "Triplet margin loss, and MSE loss")
    parser.add_argument(
        "--main-loss",
        default="SmoothL1",
        choices=["MSE", "SmoothL1"],
        help="Specify either MSE or SmoothL1. Defaults "
        "to Smooth L1 loss")
    parser.add_argument(
        "--pretrained-path",
        default="NA",
        help="Path to pre-trained model if exists")
    parser.add_argument(
        "--vae-names",
        default=["NA"],
        nargs="*",
        help="Space-separated name of cell types in the "
        "VAE matrix. If not provided, will use "
        "the basename of rna paths excluding _rna.bigWig")
    parser.add_argument(
        "--arcsinh",
        action="store_true",
        help="If specified, will apply the function on "
        "all of the input/output values")
    parser.add_argument(
        "--respdiff",
        action="store_true",
        help="If specified, train on difference from bulk "
        "instead of the actual response.")
    parser.add_argument(
        "--num-contrastive-regions",
        default=[700, 400],
        type=int,
        nargs="*",
        help="Two integers; first one the number of regions "
        "to sample from the most variant regions, and the "
        "second one as the number of regions to sample from "
        "other genomic regions")
    parser.add_argument(
        "--maxepoch",
        type=int,
        default=100,
        help="Maximum epochs")
    parser.add_argument(
        "--num-chroms-per-batch",
        help="Number of chromosomes to use for obtaining "
        "data of one batch; large numbers may result "
        "in memory crash. 1--4 suggested",
        default=4,
        type=int)
    parser.add_argument(
        "--input-normalize",
        default="None",
        choices=["None", "RobustScaler", "MinMaxScaler", "arcsinh"],
        help="One of None, RobustScaler, or MinMaxScaler.")
    args = parser.parse_args()
    if args.input_normalize == "arcsinh":
        args.acsinh = True
        args.input_normalize = "None"
    print(args)
    vae_names = process_vae_names(args.vae_names, args.rnabwpaths)
    print(args)
    # model parameters
    modelparams = adjust_model_params(args)
    print(modelparams)
    # Check file existance
    required_paths = args.rnabwpaths + args.dnasebwpaths +\
        [args.bulkdnasepath, args.bedpath, args.seqdir]
    for eachpath in required_paths:
        if not os.path.exists(eachpath):
            print("{} doesn't exist!".format(eachpath))
            raise ValueError("Check LOG! Can't access file")
    main(args.outdir, args.mavepath, args.rnabwpaths,
         args.dnasebwpaths, args.bulkdnasepath, args.bedpath,
         args.batchsize, args.seqdir,
         args.mask_nonpeaks, args.chroms,
         modelparams, regression=args.regression,
         dont_train=args.dont_train, adname_apply=args.adname,
         SCALE_FACTORS=args.scalers,
         scale_operation=args.scale_operation,
         train_general=args.train_general,
         loss_scalers=args.loss_scalers,
         main_loss=args.main_loss,
         pretrained_path=args.pretrained_path,
         vae_names=vae_names, arcsinh=args.arcsinh,
         respdiff=args.respdiff, maxepoch=args.maxepoch,
         augmentations=args.augmentations,
         num_contrastive_regions=args.num_contrastive_regions,
         num_chroms_per_batch=args.num_chroms_per_batch,
         input_normalize=args.input_normalize)
