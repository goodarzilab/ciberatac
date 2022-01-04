import adabound
from argparse import ArgumentParser
from apex import amp
# from ciberatac_train import get_scale_factors
from collections import OrderedDict
from train import scale_down
from model import ResNet1D
from model import ResidualBlock
import gzip
import numpy as np
import os
import pandas as pd
import pyBigWig
from train import load_scvidf
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0")
opt_level = 'O1'


class DataHandler:
    def __init__(self, rnabwpath, dnasebwpath,
                 bedpath, seqdir, refdnasebwpath="NA",
                 refrnabwpath="NA", window=10000,
                 mask_nonpeaks=False, SCALE_FACTORS=[1, 1],
                 SCALE_OP="identity", arcsinh=False):
        self.prepared_bigwigs = False
        self.arcsinh = arcsinh
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
        self.load_bed()

    def get_min_max(self, bwpath):
        bw = pyBigWig.open(self.refrnabwpath)
        try:
            chromsize = bw.chroms()[self.chrom]
            bwvals = bw.values(self.chrom, 0, chromsize, numpy=True)
        except Exception:
            chromsize = bw.chroms()[self.chrom.replace("chr", "")]
            bwvals = bw.values(
                self.chrom.replace("chr", ""), 0, chromsize, numpy=True)
        bwvals[np.isnan(bwvals)] = 0
        maxval = np.max(bwvals)
        minval = np.min(bwvals[bwvals > 0])
        meanval = np.mean(bwvals[bwvals > 0])
        return minval, maxval, meanval

    def process_background(self):
        # Generate minimum and maximum values
        # in self.refdnasebwpath and self.refrnabwpath
        self.minrna = 0.2
        self.meanrna = 2
        self.maxrna = 300
        self.mindnase = 0.2
        self.meandnase = 2
        self.maxdnase = 300
        if os.path.exists(self.refrnabwpath):
            self.minrna, self.maxrna, self.meanrna = \
                self.get_min_max(self.refrnabwpath)
        if os.path.exists(self.refrnabwpath):
            self.mindnase, self.maxdnase, self.meandnase = \
                self.get_min_max(self.refdnasebwpath)

    def prepare_bigwigs(self):
        self.rna_signal = self.scale_adjust_bigwig(
            self.rnabwpath, self.minrna, self.maxrna,
            self.meanrna, self.SCALE_FACTORS[0])
        self.dnase_signal = self.scale_adjust_bigwig(
            self.dnasebwpath, self.mindnase, self.maxdnase,
            self.meandnase, self.SCALE_FACTORS[1])
        if self.mask_nonpeaks:
            self.mask_dnase()
        self.prepared_bigwigs = True

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
        if self.SCALE_OP == "identity":
            return vals
        elif self.SCALE_OP == "log2":
            return np.log2(vals + 1)
        elif self.SCALE_OP == "sqrt":
            return np.sqrt(vals)
        else:
            print("Check self.SCALE_OP: {}".format(self.SCALE_OP))
            raise ValueError("Unrecognized SCALE_OP")

    def scale_adjust_bigwig(self, bwpath, minval, maxval,
                            meanval, scale_factor):
        chrom = self.chrom
        bw = pyBigWig.open(bwpath)
        try:
            chromsize = bw.chroms()[chrom]
            chrom_signal = bw.values(chrom, 0, chromsize, numpy=True)
        except Exception:
            chromsize = bw.chroms()[self.chrom.replace("chr", "")]
            chrom_signal = bw.values(
                self.chrom.replace("chr", ""), 0, chromsize, numpy=True)
        chrom_signal[np.isnan(chrom_signal)] = 0
        if os.path.exists(self.refdnasebwpath):
            if os.path.exists(self.refrnabwpath):
                adjust_mean_factor = \
                    meanval / np.mean(chrom_signal[chrom_signal > 0])
                chrom_signal = chrom_signal * adjust_mean_factor
        # idx_nonzero = np.where(chrom_signal > 0)[0]
        # values_adjust = chrom_signal[idx_nonzero]
        # new_values = match_dist_motor(values_adjust, minval, maxval)
        # chrom_signal[idx_nonzero] = new_values
        chrom_signal = chrom_signal * scale_factor
        if self.arcsinh:
            chrom_signal = np.arcsinh(chrom_signal)
        chrom_signal = scale_down(chrom_signal)
        bw.close()
        return chrom_signal

    def get_batches(self, i):
        if not self.prepared_bigwigs:
            self.prepare_bigwigs()
        # Get indices
        idxs = [i * self.batchsize,
                (i + 1) * self.batchsize]
        if idxs[1] > self.num_regions:
            idxs[1] = self.num_regions
        # Get middle position
        range_idxs = np.arange(idxs[0], idxs[1])
        start = np.array(
            self.bed.iloc[range_idxs, 1])
        end = np.array(
            self.bed.iloc[range_idxs, 2])
        # print("Processing regions {}:{} with starts as {}".format(
        #         idxs[0], idxs[1], start))
        midpos = np.array(
            start + np.round((end - start) / 2), dtype=int)
        rna, _ = self.get_signal_from_bw(
            self.rna_signal, midpos, start, end)
        dnase, avg_mid = self.get_signal_from_bw(
            self.dnase_signal, midpos, start, end)
        return rna, dnase, midpos, avg_mid

    def get_signal_from_bw(
            self, chrom_signal, midpos, starts, ends):
        chrom = self.chrom
        chromsize = chrom_signal.shape[0]
        batchar = np.zeros(
            (len(midpos), 4, self.window * 2), dtype=np.float32)
        i = 0
        avg_vals = np.zeros(len(midpos))
        for each_midpos in midpos:
            start = each_midpos - self.window
            end = each_midpos + self.window
            if end > chromsize:
                end = chromsize
            if start < 0:
                signal = np.zeros(end - start)
                adsig = chrom_signal[:end]
                signal[-start:] = adsig
            else:
                try:
                    signal = chrom_signal[start:end]
                except Exception:
                    print("{}:{}-{}/{}".format(chrom, start, end, chromsize))
                    raise ValueError("Index out of bounds")
            signal = self.scale_signal(signal)
            adtensor = self.initiate_seq(
                start, end)
            for nucidx in range(len(self.nucs)):
                nuc = self.nucs[nucidx].encode()
                if start > 0:
                    j = np.where(self.chrom_seq[start:end] == nuc)[0]
                    if len(j) > 0:
                        adtensor[nucidx, j] += signal[j]
                else:
                    j = np.where(self.chrom_seq[:end] == nuc)[0]
                    j_add = -start
                    if len(j) > 0:
                        adtensor[nucidx, j + j_add] += signal[j + j_add]
            batchar[i] = adtensor
            # midvals_cur = chrom_signal[starts[i]:ends[i]]
            avg_vals[i] = np.mean(
                signal[(self.window - 100):(self.window + 100)])
            # midvals_cur[np.isnan(midvals_cur)] = 0
            # avg_vals[i] = np.mean(
            #     midvals_cur)
            i += 1
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
        self.process_background()
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
    else:
        raise ValueError("optimizer name not recognized")
    return optimizer


def load_model(modelpath, regression=True):
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
        "optimizer": "Adabound",
        "window": 20000,
        "normtype": "BatchNorm",
        "regularize": True,
        "lambda_param": 0.01,
        "augmentations": []}
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
    checkpoint = torch.load(modelpath)
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
    return net


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
    scvi_name = "LVM2"
    refdnasebwpath, refrnabwpath = [
        "/scratch/ssd001/home/mkarimza/data/" +
        "ciberatac/pbmc10x/atac/all_cells_treat_pileup.bigWig",
        "/scratch/ssd001/home/mkarimza/data/" +
        "ciberatac/pbmc10x/rna/allTcells_rna.bigWig"]
    scvipath = "/scratch/ssd001/home/mkarimza/data/" +\
        "ciberatac/models/vae202101/mixPbmcSw/" +\
        "ReTrainedCustomizedScviPbmcAndSw480/" +\
        "multiTaskVAE_medianValues_pbmc10x.tsv"
    regression = True
    window = 20000
    list_args = [outpath, rnabwpath, dnasebwpath,
                 chrom, bedpath, modelpath,
                 seqdir, batchsize, refdnasebwpath,
                 refrnabwpath, scvipath, scvi_name,
                 window, regression]
    return list_args


def predict_motor(DataObj, chrom, batchsize, net, scvi_tensor,
                  regression):
    num_batches = DataObj.get_batch_nums(chrom, batchsize)
    beddf = DataObj.bed.copy()
    beddf["Central.Position"] = 0
    beddf["Average.DNase"] = 0
    beddf["CiberATAC.Prediction"] = 0
    for i in range(num_batches):
        idx_st = i * batchsize
        idx_end = (i + 1) * batchsize
        if idx_end > beddf.shape[0]:
            idx_end = beddf.shape[0]
        curbatchsize = idx_end - idx_st
        rna, dnase, positions, avg_dnase = DataObj.get_batches(i)
        input1 = torch.from_numpy(dnase).float().to(device)
        input2 = torch.from_numpy(rna).float().to(device)
        output, _ = net(input1, input2, scvi_tensor[:curbatchsize])
        if regression:
            output_ar = output.cpu().detach().numpy()[:, 0]
        else:
            output = nn.functional.softmax(output, dim=1)
            output_ar = output.cpu().detach().numpy()[:, 1]
        beddf.iloc[idx_st:idx_end, -1] = output_ar
        beddf.iloc[idx_st:idx_end, -2] = avg_dnase
        beddf.iloc[idx_st:idx_end, -3] = positions
        del dnase, rna, positions, avg_dnase
        del input1, input2, output, output_ar
        torch.cuda.empty_cache()
        if i % 100 == 0:
            print("{}/{} regions added".format(i, num_batches))
    return beddf


def main(outpath, rnabwpath, dnasebwpath,
         bedpath, refdnasebwpath,
         refrnabwpath, modelpath, batchsize, seqdir,
         mask_nonpeaks, scvipath, scvi_name,
         annotate_bwpath="NA",
         SCALERS=[1, 100], SCALE_OP="identity",
         regression=True, window=10000,
         early_stop=False, valid_chroms=False,
         chrom="all", arcsinh=False):
    scvi = load_scvidf(scvipath)
    scvi_ar = np.zeros((batchsize, 10))
    scvi_ar[:, :] = scvi[scvi_name]
    if scvi_name + ".1" in scvi.keys():
        scvi_ar[:, :] = scvi[scvi_name + ".1"]
        print("Using {}.1 for scvi".format(scvi_name))
    scvi_tensor = torch.from_numpy(
        scvi_ar).float().to(device)
    # print(scvi_tensor)
    net = load_model(modelpath)
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
    list_beds = []
    for chrom in chroms:
        DataObj = DataHandler(
            rnabwpath, dnasebwpath, bedpath, seqdir,
            refdnasebwpath, refrnabwpath,
            window=window,
            mask_nonpeaks=mask_nonpeaks,
            SCALE_FACTORS=SCALERS,
            SCALE_OP=SCALE_OP, arcsinh=arcsinh)
        beddf = predict_motor(
            DataObj, chrom, batchsize, net, scvi_tensor,
            regression)
        beddf.to_csv(
            outpath.replace(".tsv", "_incomplete.tsv"),
            sep="\t", compression="gzip", index=None)
        list_beds.append(beddf)
        del DataObj
    beddf = pd.concat(list_beds)
    beddf.to_csv(
        outpath, sep="\t", compression="gzip", index=None)
    os.remove(outpath.replace(".tsv", "_incomplete.tsv"))
    if os.path.exists(annotate_bwpath):
        import seaborn as sns
        print("Adding {}".format(annotate_bwpath))
        beddf["Response"] = 0
        bw = pyBigWig.open(annotate_bwpath)
        for i in range(beddf.shape[0]):
            chrom, st, end = [beddf.iloc[i, j] for j in [0, 1, 2]]
            st = int(st)
            end = int(end)
            # midpos = int(st + round((end - st) / 2))
            # midpos = int(round(st + ((end - st) / 2)))
            # Get the middle signal not st:end
            signalvals = bw.values(
                chrom, st, end, numpy=True)
            signalvals[np.isnan(signalvals)] = 0
            if arcsinh:
                signalvals = np.arcsinh(signalvals)
            # signalvals = signalvals[np.logical_not(np.isnan(signalvals))]
            beddf.iloc[i, -1] = np.mean(signalvals) * SCALERS[1]
        tempdf = beddf.dropna()
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
        "scvipath",
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
        "--scvi-name",
        help="Name of the key in scvi dictionary "
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
    args = parser.parse_args()
    if os.path.exists(args.outpath):
        raise ValueError("Output exists, exiting!")
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
         args.mask_nonpeaks, args.scvipath,
         args.scvi_name, annotate_bwpath=args.annotate_bwpath,
         SCALERS=args.scalers, SCALE_OP=args.scale_operation,
         early_stop=args.early_stop,
         valid_chroms=args.valid_chroms, arcsinh=args.arcsinh)
