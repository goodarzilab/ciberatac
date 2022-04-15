import adabound
from argparse import ArgumentParser
from apex import amp
# from ciberatac_train import get_scale_factors
from collections import OrderedDict
from model import ResNet1D
from model import ResidualBlock
import gzip
import numpy as np
import os
import pandas as pd
import pyBigWig
from train import load_mavedf
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from utils import extract_weights
from utils import make_normalizers
from utils import get_signal_from_bw


device = torch.device("cuda:0")
opt_level = 'O1'


class DataHandler:
    def __init__(self, rnabwpath, dnasebwpath,
                 bedpath, seqdir, refdnasebwpath="NA",
                 refrnabwpath="NA", window=10000,
                 mask_nonpeaks=False, SCALE_FACTORS=[1, 1],
                 SCALE_OP="identity", arcsinh=False,
                 input_normalize="None"):
        self.prepared_bigwigs = False
        self.arcsinh = arcsinh
        self.input_normalize = input_normalize
        self.SCALE_FACTORS = SCALE_FACTORS
        self.SCALE_OP = SCALE_OP
        self.nucs = np.array(["A", "T", "C", "G"])
        self.window = window
        self.sequencedir = seqdir
        self.rnabwpath = rnabwpath
        self.dnasebwpath = dnasebwpath
        self.bedpath = bedpath
        self.refdnasebwpath = refdnasebwpath
        self.refrnabwpath = refrnabwpath
        self.mask_nonpeaks = mask_nonpeaks
        self.make_normalizers()
        self.load_bed()

    def make_normalizers(self, sample_chrom="chr1"):
        bw_paths = [
            self.dnasebwpath]
        if os.path.exists(self.refdnasebwpath):
            bw_paths = [self.refdnasebwpath]
        Scaler = make_normalizers(
            bw_paths, self.input_normalize, sample_chrom,
            self.SCALE_FACTORS[1])
        if self.input_normalize == "RobustScaler":
            self.RobustScaler = Scaler
        elif self.input_normalize == "MinMaxScaler":
            self.MinMaxScaler = Scaler

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

    def scale_signal(self, vals):
        vals = self.normalize_input(vals)
        if self.SCALE_OP == "identity":
            return vals
        elif self.SCALE_OP == "log2":
            return np.log2(vals + 1)
        elif self.SCALE_OP == "sqrt":
            return np.sqrt(vals)
        else:
            print("Check self.SCALE_OP: {}".format(self.SCALE_OP))
            raise ValueError("Unrecognized SCALE_OP")

    def get_batches(self, i=-1):
        # Get indices
        # idxs = [i * self.batchsize,
        #         (i + 1) * self.batchsize]
        # if idxs[1] > self.num_regions:
        #     idxs[1] = self.num_regions
        # # Get middle position
        # range_idxs = np.arange(idxs[0], idxs[1])
        # start = np.array(
        #     self.bed.iloc[range_idxs, 1])
        # end = np.array(
        #     self.bed.iloc[range_idxs, 2])
        # # print("Processing regions {}:{} with starts as {}".format(
        # #         idxs[0], idxs[1], start))
        # midpos = np.array(
        #     start + np.round((end - start) / 2), dtype=int)
        if i == -1:
            print("Extracting all together")
            start = np.array(self.bed.iloc[:, 1])
            end = np.array(self.bed.iloc[:, 2])
        else:
            idxs = [i * self.batchsize,
                    (i + 1) * self.batchsize]
            idxs[1] = min([idxs[1], self.num_regions])
            range_idxs = np.arange(idxs[0], idxs[1])
            start = np.array(self.bed.iloc[range_idxs, 1])
            end = np.array(self.bed.iloc[range_idxs, 2])
        midpos = np.array(
            start + np.round((end - start) / 2), dtype=int)
        rna, _ = self.get_signal_from_bw(
            self.rnabwpath, midpos, self.SCALE_FACTORS[0],
            start, end)
        dnase, avg_mid = self.get_signal_from_bw(
            self.dnasebwpath, midpos, self.SCALE_FACTORS[1],
            start, end, get_resp=True)
        return rna, dnase, midpos, avg_mid

    def get_signal_from_bw(
            self, bwpath, midpos, SCALE_FACTOR,
            starts, ends, get_resp=False):
        batchar, avg_vals = get_signal_from_bw(
            bwpath, midpos, SCALE_FACTOR,
            starts, ends, self.chrom,
            self.chrom_seq, self.window, self.nucs,
            self, get_resp=get_resp)
        return batchar, avg_vals

    def get_batch_nums(self, chrom, batchsize):
        self.batchsize = batchsize
        self.chrom = chrom
        print("Filtering BED for {}".format(chrom))
        self.chrom_seq = self.get_chromseq(chrom)
        self.bed = self.bed[self.bed.iloc[:, 0] == chrom]
        self.num_regions = self.bed.shape[0]
        self.num_batches = int(
            np.round(self.num_regions / self.batchsize))
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


def load_model(modelpath, regression=True):
    checkpoint = torch.load(modelpath)
    modelparams = {
        "optimize": "train",
        "dropout": 0.5,
        "regression": regression,
        "lr": 0.001,
        "ltype": 3,
        "kernel_size": 20,
        "convparam": [1, 1, 1],
        "dilations": [1, 4, 8],
        "initconv": 64,
        "stride": 1,
        "filter_rate": 1.25,
        "pool_dim": 40,
        "pool_type": "Average",
        "activation": "LeakyReLU",
        "optimizer": "Adam",
        "window": 10000,
        "normtype": "BatchNorm",
        "regularize": True,
        "lambda_param": 0.01,
        "augmentations": []}
    modelparams_loaded = checkpoint.get("modelparams", {})
    if "convparam" in modelparams_loaded.keys():
        print("Loading model params from dictionary")
        modelparams = modelparams_loaded
        print(modelparams)
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
    param_df_before = extract_weights(net)
    param_df_before["State"] = "Initialized"
    optimizer = get_optimizer(
        modelparams["optimizer"], net,
        modelparams["lr"])
    net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    net.eval()
    param_df_after = extract_weights(net)
    param_df_after["State"] = "Pre-trained"
    param_df = pd.concat([param_df_before, param_df_after])
    return net, param_df


def make_default_args():
    datadir = "/scratch/hdd001/home/mkarimza"
    outpath = datadir +\
        "/mkarimza/johnny/A06/ciberAtacPredictions/" +\
        "20210129_ciberAtacPredionctions_LVM2.tsv.gz"
    rnabwpath = datadir +\
        "/johnny/A06/bigWigs/scRnaBigWigs/" +\
        "scRNA-seq_SW480_LVM.bigWig"
    dnasebwpath = datadir +\
        "/johnny/A06/bigWigs/mergedBulkAtac/" +\
        "SW480_P-LVM1-LVM2.bigWig"
    dnasebwpath = datadir +\
        "/johnny/A06/bigWigs/pseudoBulkBigWigs/" +\
        "parentalAndLvm2_treat_pileup.bigWig"
    chrom = "chr10"
    bedpath = "/scratch/hdd001/home/mkarimza" +\
        "/johnny/A06/bigWigs/pseudoBulkBigWigs/" +\
        "parentalAndLvm2_peaks.narrowPeak.gz"
    # bedpath = datadir +\
    #    "/meulemanData/signalMatrix/DHS_" +\
    #    "Index_and_Vocabulary_hg38_WM20190703.txt.gz"
    modelpath = "/scratch/ssd001/home/mkarimza/data" +\
        "/ciberatac/models/2020_05_11/modelLog/20200430_" +\
        "ConvBlock_1_1_1_ConvInit_8_Kernel_60_Dilation_1_" +\
        "Pool_Average_20_Activation_LeakyReLU_lr_0.001_Opt" +\
        "imizer_Adabound_Augmentation_reverse_complement-" +\
        "mask_background-mask_signal_Dropout_0.5/BulkTrai" +\
        "nedModels/Sat_May_30_12-31-00_EDT_2020_20200430_" +\
        "ConvBlock_1_1_1_ConvInit_8_Kernel_60_Dilation_1_" +\
        "Pool_Average_20_Activation_LeakyReLU_lr_0.001_" +\
        "Optimizer_Adabound_Augmentation_reverse_comple" +\
        "ment-mask_background-mask_signal_Dropout_0.5_" +\
        "currentmodel.pt"
    modelpath = "/scratch/ssd001/home/mkarimza/data/" +\
        "ciberatac/pbmc10x/trainedModels/20201109/" +\
        "modelLog/202010_ConvBlock_2_2_2_ConvInit_" +\
        "16x2_Kernel_20_Dilation_1_Pool_Average_20" +\
        "_Activation_LeakyReLU_lr_0.001stride_1_" +\
        "Optimizer_Adabound_Augmentation__Dropout" +\
        "_0.1_gradient_clipping_at_1_regression/" +\
        "20201126_backup.pt"
    modelpath = "/scratch/ssd001/home/mkarimza/" +\
        "data/ciberatac/pbmc10x/trainedModels/" +\
        "20210127-Training-PBMC/modelLog/202010" +\
        "_ConvBlock_1_1_1_ConvInit_64x1.25_Kernel" +\
        "_60_Dilation_1_Pool_Average_40_Activation" +\
        "_LeakyReLU_lr_0.001stride_1_Optimizer_" +\
        "Adabound_Augmentation__Dropout_0.5_" +\
        "gradient_clipping_at_0.01_regression/" +\
        "round1/model_at_chr4.pt"
    seqdir = datadir +\
        "/refData/genomeData/hg38/np"
    batchsize = 24
    mave_name = "LVM2"
    refdnasebwpath, refrnabwpath = [
        "/scratch/ssd001/home/mkarimza/data/" +
        "ciberatac/pbmc10x/atac/all_cells_treat_pileup.bigWig",
        "/scratch/ssd001/home/mkarimza/data/" +
        "ciberatac/pbmc10x/rna/allTcells_rna.bigWig"]
    mavepath = "/scratch/ssd001/home/mkarimza/data/" +\
        "ciberatac/models/vae202101/mixPbmcSw/" +\
        "ReTrainedCustomizedScviPbmcAndSw480/" +\
        "multiTaskVAE_medianValues_pbmc10x.tsv"
    regression = True
    window = 20000
    list_args = [outpath, rnabwpath, dnasebwpath,
                 chrom, bedpath, modelpath,
                 seqdir, batchsize, refdnasebwpath,
                 refrnabwpath, mavepath, mave_name,
                 window, regression]
    return list_args


def predict_motor(DataObj, chrom, batchsize, net, mave_tensor,
                  regression):
    num_batches = DataObj.get_batch_nums(chrom, batchsize) + 1
    beddf = DataObj.bed.copy()
    beddf["Central.Position"] = 0
    beddf["Average.DNase"] = 0
    beddf["CiberATAC.Prediction"] = 0
    rna, dnase, positions, avg_dnase = DataObj.get_batches()
    for i in range(num_batches):
        idx_st = i * batchsize
        idx_end = (i + 1) * batchsize
        if idx_st > beddf.shape[0]:
            break
        if idx_end > beddf.shape[0]:
            idx_end = beddf.shape[0]
        curbatchsize = idx_end - idx_st
        if curbatchsize < 2:
            break
        # rna, dnase, positions, avg_dnase = DataObj.get_batches(i)
        input1 = torch.from_numpy(dnase[idx_st:idx_end]).float().to(device)
        input2 = torch.from_numpy(rna[idx_st:idx_end]).float().to(device)
        output, _ = net(input1, input2, mave_tensor[:curbatchsize])
        if regression:
            output_ar = output.cpu().detach().numpy()[:, 0]
        else:
            output = nn.functional.softmax(output, dim=1)
            output_ar = output.cpu().detach().numpy()[:, 1]
        beddf.iloc[idx_st:idx_end, -1] = output_ar
        beddf.iloc[idx_st:idx_end, -2] = avg_dnase[idx_st:idx_end]
        beddf.iloc[idx_st:idx_end, -3] = positions[idx_st:idx_end]
        del input1, input2, output, output_ar
        torch.cuda.empty_cache()
        if i % 100 == 0:
            print("{}/{} regions added".format(i, num_batches))
    del dnase, rna, positions, avg_dnase
    return beddf


def visualize_params(param_df, outdir):
    import seaborn as sns
    import matplotlib.pyplot as plt
    outdir_params = os.path.join(outdir, "modelParams")
    os.makedirs(outdir_params, exist_ok=True)
    sns_plot = plt.figure(figsize=(16, 9))
    sns.distplot(
        np.array(
            param_df[param_df["State"] == "Initialized"]["Values"]),
        label="Initialized")
    sns.distplot(
        np.array(
            param_df[param_df["State"] != "Initialized"]["Values"]),
        label="Pre-trained")
    plt.legend()
    sns_plot.savefig(
        os.path.join(outdir_params, "model_histogram.png"))
    sns_plot.savefig(
        os.path.join(outdir_params, "model_histogram.pdf"))
    param_df.to_csv(
        os.path.join(
            outdir_params, "model_histogram.tsv.gz"),
        sep="\t", compression="gzip")


def main(outpath, rnabwpath, dnasebwpath,
         bedpath, refdnasebwpath,
         refrnabwpath, modelpath, batchsize, seqdir,
         mask_nonpeaks, mavepath, mave_name,
         annotate_bwpath="NA",
         SCALERS=[1, 100], SCALE_OP="identity",
         regression=True, window=10000,
         early_stop=False, valid_chroms=False,
         chrom="all", arcsinh=False,
         input_normalize="None"):
    outdir = os.path.dirname(outpath)
    mave = load_mavedf(mavepath)
    mave_ar = np.zeros((batchsize, 10))
    mave_ar[:, :] = mave[mave_name]
    if mave_name + ".1" in mave.keys():
        mave_ar[:, :] = mave[mave_name + ".1"]
        print("Using {}.1 for mave".format(mave_name))
    mave_tensor = torch.from_numpy(
        mave_ar).float().to(device)
    # print(mave_tensor)
    net, param_df = load_model(modelpath)
    visualize_params(param_df, outdir)
    beddf = pd.read_csv(
        bedpath, compression="gzip", sep="\t", header=None)
    chroms = list(pd.unique(beddf.iloc[:, 0]))
    ref_chroms = ["chr{}".format(each) for each in
                  list(range(1, 23)) + ["X"]]
    if valid_chroms:
        ref_chroms = "chr6 chr7 chr11 chr13 chr15 chr21 chr22 chrX".split(" ")
    chroms = [each for each in chroms if
              each in ref_chroms]
    if chrom != "all":
        ref_chroms = [chrom]
        chroms = [chrom]
    if early_stop:
        chroms = [chroms[0]]
    if not os.path.exists(outpath):
        list_beds = []
        for chrom in chroms:
            DataObj = DataHandler(
                rnabwpath, dnasebwpath, bedpath, seqdir,
                refdnasebwpath, refrnabwpath,
                window=window,
                mask_nonpeaks=mask_nonpeaks,
                SCALE_FACTORS=SCALERS,
                SCALE_OP=SCALE_OP, arcsinh=arcsinh,
                input_normalize=input_normalize)
            beddf = predict_motor(
                DataObj, chrom, batchsize, net, mave_tensor,
                regression)
            list_beds.append(beddf)
            beddf = pd.concat(list_beds)
            beddf.to_csv(
                outpath.replace(".tsv", "_incomplete.tsv"),
                sep="\t", compression="gzip", index=None)
            if chrom != chroms[-1]:
                del DataObj
        beddf = pd.concat(list_beds)
        beddf.to_csv(
            outpath, sep="\t", compression="gzip", index=None)
        os.remove(outpath.replace(".tsv", "_incomplete.tsv"))
    else:
        beddf = pd.read_csv(outpath, sep="\t", compression="gzip")
        chrom = chroms[0]
        DataObj = DataHandler(
                rnabwpath, dnasebwpath, bedpath, seqdir,
                refdnasebwpath, refrnabwpath,
                window=window,
                mask_nonpeaks=mask_nonpeaks,
                SCALE_FACTORS=SCALERS,
                SCALE_OP=SCALE_OP, arcsinh=arcsinh,
                input_normalize=input_normalize)
    if os.path.exists(annotate_bwpath):
        import seaborn as sns
        print("Adding {} to {} rows".format(annotate_bwpath, beddf.shape))
        beddf["Response"] = 0
        starts = np.array(beddf.iloc[:, 1])
        ends = np.array(beddf.iloc[:, 2])
        beddf["Central.Position"] = np.array(
            starts + ((ends - starts) / 2), dtype=int)
        bw = pyBigWig.open(annotate_bwpath)
        for i in range(beddf.shape[0]):
            chrom = beddf.iloc[i, 0]
            chr_str = chrom
            if chr_str not in list(bw.chroms().keys()):
                chr_str = chrom.replace("chr", "")
            chromsize = bw.chroms()[chr_str]
            midpos = beddf["Central.Position"].iloc[i]
            st = midpos - 100
            end = midpos + 100
            if end > chromsize:
                end = chromsize
            signalvals = bw.values(
                chrom, st, end, numpy=True)
            signalvals[np.isnan(signalvals)] = 0
            if i == 0:
                print(signalvals)
            try:
                signalvals = DataObj.scale_signal(signalvals)
            except Exception:
                print("Failed to normalize")
            if arcsinh:
                signalvals = np.arcsinh(signalvals)
            beddf.iloc[i, -1] = np.mean(signalvals * SCALERS[1])
            if i % 10000 == 0:
                print("{}/{} annotations added".format(i, beddf.shape[0]))
        bw.close()
        print(beddf.head())
        tempdf = beddf.dropna()
        print(tempdf.head())
        beddf["R2.predictions"] = metrics.r2_score(
            tempdf["Response"], tempdf["CiberATAC.Prediction"])
        beddf["R2.baseline"] = metrics.r2_score(
            tempdf["Response"], tempdf["Average.DNase"])
        # Average precision at high values of response
        cutoff_resp = np.quantile(tempdf["Response"], 0.75)
        beddf["AP.prediction"] = metrics.average_precision_score(
            tempdf["Response"] > cutoff_resp, tempdf["CiberATAC.Prediction"])
        beddf["AP.baseline"] = metrics.average_precision_score(
            tempdf["Response"] > cutoff_resp, tempdf["Average.DNase"])
        beddf.to_csv(
            outpath, sep="\t", compression="gzip", index=None)
        if SCALE_OP != "log2":
            beddf["Log.Response"] = np.log10(beddf["Response"] + 1)
            beddf["Log.Prediction"] = np.log10(
                beddf["CiberATAC.Prediction"] + 1)
            beddf["Log.Average"] = np.log10(beddf["Average.DNase"] + 1)
            tempdf = beddf.dropna()
            beddf["R2.Log.predictions"] = metrics.r2_score(
                tempdf["Log.Response"], tempdf["Log.Prediction"])
            beddf["R2.log.baseline"] = metrics.r2_score(
                tempdf["Log.Response"], tempdf["Log.Average"])
            beddf.to_csv(
                outpath, sep="\t", compression="gzip", index=None)
            sns_plot = sns.relplot(
                y="Response", x="CiberATAC.Prediction", hue="Average.DNase",
                size="Average.DNase", data=beddf, height=6, aspect=1.5)
            sns_plot.savefig(
                outpath.replace(".tsv.gz", ".pdf"))
            sns_plot.savefig(
                outpath.replace(".tsv.gz", ".png"))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Predict enhancer activity "
        "using the CiberATAC model. Requires "
        "a bigWig file for the transcriptome "
        "and a bigWig file for the chromatin "
        "accessibility. It also requires "
        "a BED file for list of potential "
        "enhancer to predict on.")
    parser.add_argument(
        "outpath",
        help="Path to BED6 file for output "
        "predictions")
    parser.add_argument(
        "mavepath",
        help="Path to the matrix of SCVI averaged data")
    parser.add_argument(
        "rnabwpath",
        help="Path to bigWig file of the "
        "transcriptome measures")
    parser.add_argument(
        "dnasebwpath",
        help="Path to bigWig file of "
        "chromatin accessibility")
    parser.add_argument(
        "bedpath",
        help="Path to a BED file for "
        "list of regions to predict on")
    parser.add_argument(
        "--refdnasebwpath",
        default="NA",
        help="Path to a reference chromatin "
        "accessibility bigWig file for "
        "signal scaling and normalization")
    parser.add_argument(
        "--refrnabwpath",
        default="NA",
        help="Path to a reference RNA "
        "bigWig file for signal scaling "
        "and normalization")
    parser.add_argument(
        "--modelpath",
        required=True,
        help="Path to CiberATAC trained model .pt")
    parser.add_argument(
        "--batchsize",
        default=80,
        type=int,
        help="Number of simultaneous batches to"
        "generate and feed into the GPU")
    parser.add_argument(
        "--seqdir",
        required=True,
        help="Path to directory with files named "
        "as <chromosome>_sequence.numpy.gz")
    parser.add_argument(
        "--mask-nonpeaks",
        action="store_true",
        help="If specified, will limit the DNase/ATAC-seq "
        "signal to regions within the BED file.")
    parser.add_argument(
        "--mave-name",
        help="Name of the key in mave dictionary "
        "for retrieving the LVs")
    parser.add_argument(
        "--annotate-bwpath",
        default="NA",
        help="If provided, it will add the signal "
        "from the provided bigWig path to the final predictions "
        "in the Response column")
    parser.add_argument(
        "--scalers",
        nargs="*",
        type=float,
        default=[1, 100],
        help="Scaling factors for RNA and ATAC-seq")
    parser.add_argument(
        "--scale-operation",
        default="identity",
        choices=["identity", "log2", "sqrt"],
        help="Specify if you want to apply one of "
        "sqrt or log2 on input values. In case of"
        "log2 it will perform (log2(non-zero-values + 1))")
    parser.add_argument(
        "--valid-chroms",
        action="store_true",
        help="If specified, limit to "
        "chr6 chr7 chr11 chr13 chr15 chr21 chr22 chrX")
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="If specified, will only predict on chr1")
    parser.add_argument(
        "--arcsinh",
        action="store_true",
        help="If specified, pass all input/outputs into np.arcsinh")
    parser.add_argument(
        "--input-normalize",
        default="None",
        choices=["None", "RobustScaler", "MinMaxScaler", "arcsinh"],
        help="One of None, RobustScaler, or MinMaxScaler.")
    args = parser.parse_args()
    if args.input_normalize == "arcsinh":
        args.acsinh = True
        args.input_normalize = "None"
    if os.path.exists(args.outpath):
        print("Output exists, re-annotating")
        # raise ValueError("Output exists, exiting!")
    # Check file existance
    required_paths = [
        args.rnabwpath, args.dnasebwpath,
        args.bedpath, args.modelpath, args.seqdir]
    for eachpath in required_paths:
        if not os.path.exists(eachpath):
            print("{} doesn't exist!".format(eachpath))
            raise ValueError("Check LOG! Can't access file")
    main(args.outpath, args.rnabwpath,
         args.dnasebwpath, args.bedpath,
         args.refdnasebwpath, args.refrnabwpath,
         args.modelpath, args.batchsize, args.seqdir,
         args.mask_nonpeaks, args.mavepath,
         args.mave_name, annotate_bwpath=args.annotate_bwpath,
         SCALERS=args.scalers, SCALE_OP=args.scale_operation,
         early_stop=args.early_stop,
         valid_chroms=args.valid_chroms, arcsinh=args.arcsinh,
         input_normalize=args.input_normalize)
