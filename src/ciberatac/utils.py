from datetime import datetime
import gzip
import joblib
import linecache
import numpy as np
import os
import pandas as pd
import pyBigWig
import time
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_bce(response_ar, pred_vals, regions, resp_cutoff=0, bce=True):
    response_ar = response_ar.reshape(response_ar.shape[0], 1)
    pred_vals = pred_vals.reshape(pred_vals.shape[0], 1)
    # response_ar = response_ar / max(response_ar)
    # pred_vals = pred_vals / max(pred_vals)
    all_ce_losses = 0
    len_used_regs = 0
    loss = torch.nn.MSELoss()
    if bce:
        loss = torch.nn.BCELoss()
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)[0]
        resp_ar = np.sum(
            response_ar[None, idx_reg] - response_ar[idx_reg, None],
            axis=-1)
        pred_ar = np.sum(
            pred_vals[None, idx_reg] - pred_vals[idx_reg, None], axis=-1)
        pred_tensor = torch.triu(
            torch.from_numpy(pred_ar), diagonal=1)
        resp_tensor = torch.from_numpy(
            resp_ar[np.triu_indices(resp_ar.shape[0])]).view(-1, 1)
        pred_tensor = pred_tensor[
            np.triu_indices(pred_tensor.shape[0])].view(-1, 1)
        idx_eval = np.where(
            abs(resp_tensor.cpu().detach().numpy()) > resp_cutoff)[0]
        if len(idx_eval) > 0:
            len_used_regs += 1
            if bce:
                resp_tensor = resp_tensor[idx_eval]
                resp_tensor[resp_tensor > resp_cutoff] = 1
                resp_tensor[resp_tensor < (-1 * resp_cutoff)] = 0
                pred_tensor = torch.sigmoid(pred_tensor[idx_eval])
            else:
                resp_tensor = resp_tensor[idx_eval]
                pred_tensor = pred_tensor[idx_eval]
            try:
                ce_loss = loss(pred_tensor, resp_tensor)
            except Exception:
                try:
                    ce_loss = loss(pred_tensor, resp_tensor.double())
                except Exception:
                    raise ValueError("Failed at BCE")
            all_ce_losses += ce_loss
    if len_used_regs > 0:
        loss = all_ce_losses.cpu().detach().numpy() / len_used_regs
    else:
        loss = np.nan()
    return loss


def get_ce_loss(criterion_direction, response_tensor, model_init, regions,
                resp_cutoff=0, bce=True):
    pred_vals = model_init.detach().cpu().numpy()
    all_ce_losses = 0
    len_used_regs = 0
    for region in np.unique(regions):
        idx_reg = np.where(regions == region)
        resp_ar = np.sum(
            response_tensor[None, idx_reg] - response_tensor[idx_reg, None],
            axis=-1)[0]
        pred_ar = np.sum(
            pred_vals[None, idx_reg] - pred_vals[idx_reg, None], axis=-1)[0]
        pred_tensor = torch.triu(
            torch.from_numpy(pred_ar), diagonal=1)
        resp_tensor = torch.from_numpy(
            resp_ar[np.triu_indices(resp_ar.shape[0])]).view(-1, 1).to(device)
        pred_tensor = pred_tensor[
            np.triu_indices(pred_tensor.shape[0])].view(-1, 1).to(device)
        idx_eval = np.where(
            abs(resp_tensor.cpu().detach().numpy()) > resp_cutoff)[0]
        if len(idx_eval) > 0:
            len_used_regs += 1
            if not bce:
                resp_tensor = resp_tensor[idx_eval]
                pred_tensor = pred_tensor[idx_eval]
            else:
                resp_tensor = resp_tensor[idx_eval]
                resp_tensor[resp_tensor > resp_cutoff] = 1
                resp_tensor[resp_tensor < (-1 * resp_cutoff)] = 0
                pred_tensor = torch.sigmoid(pred_tensor[idx_eval])
            ce_loss = criterion_direction(pred_tensor, resp_tensor)
            all_ce_losses += ce_loss
            del resp_tensor, pred_tensor
    if len_used_regs > 0:
        ce_loss = all_ce_losses / len_used_regs
    else:
        ce_loss = all_ce_losses
    return ce_loss


def regularize_loss(modelparams, net, loss):
    lambda1 = modelparams["lambda_param"]
    ltype = modelparams["ltype"]
    if ltype == 3:
        if torch.cuda.device_count() > 1:
            torch.nn.utils.clip_grad_norm_(
                net.module.conv.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.module.layer1.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.module.layer2.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.module.layer3.parameters(), lambda1)
            if len(modelparams["convparam"]) == 4:
                torch.nn.utils.clip_grad_norm_(
                    net.module.layer4.parameters(), lambda1)
        else:
            torch.nn.utils.clip_grad_norm_(
                net.conv.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.layer1.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.layer2.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.layer3.parameters(), lambda1)
            if len(modelparams["convparam"]) == 4:
                torch.nn.utils.clip_grad_norm_(
                    net.layer4.parameters(), lambda1)
    if torch.cuda.device_count() > 1:
        l0_params = torch.cat(
            [x.view(-1) for x in
             net.module.conv.parameters()])
        l1_params = torch.cat(
            [x.view(-1) for x in
             net.module.layer1.parameters()])
        l2_params = torch.cat(
            [x.view(-1) for x in
             net.module.layer2.parameters()])
        l3_params = torch.cat(
            [x.view(-1) for x in
             net.module.layer3.parameters()])
        if len(modelparams["convparam"]) == 4:
            l4_params = torch.cat(
                [x.view(-1) for x in
                 net.module.layer4.parameters()])
    else:
        l0_params = torch.cat(
            [x.view(-1) for x in net.conv.parameters()])
        l1_params = torch.cat(
            [x.view(-1) for x in net.layer1.parameters()])
        l2_params = torch.cat(
            [x.view(-1) for x in net.layer2.parameters()])
        l3_params = torch.cat(
            [x.view(-1) for x in net.layer3.parameters()])
        if len(modelparams["convparam"]) == 4:
            l4_params = torch.cat(
                [x.view(-1) for x in net.layer4.parameters()])
    if ltype in [1, 2]:
        l1_l0 = lambda1 * torch.norm(l0_params, ltype)
        l1_l1 = lambda1 * torch.norm(l1_params, ltype)
        l1_l2 = lambda1 * torch.norm(l2_params, ltype)
        l1_l3 = lambda1 * torch.norm(l3_params, ltype)
        if len(modelparams["convparam"]) == 4:
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            loss = loss + l1_l0 + l1_l1 + l1_l2 + l1_l3 + l1_l4
        else:
            loss = loss + l1_l0 + l1_l1 + l1_l2 + l1_l3
    elif ltype == 4:
        l1_l0 = lambda1 * torch.norm(l0_params, 1)
        l1_l1 = lambda1 * torch.norm(l1_params, 1)
        l1_l2 = lambda1 * torch.norm(l2_params, 1)
        l1_l3 = lambda1 * torch.norm(l3_params, 1)
        l2_l0 = lambda1 * torch.norm(l0_params, 2)
        l2_l1 = lambda1 * torch.norm(l1_params, 2)
        l2_l2 = lambda1 * torch.norm(l2_params, 2)
        l2_l3 = lambda1 * torch.norm(l3_params, 2)
        if len(modelparams["convparam"]) == 4:
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l2_l4 = lambda1 * torch.norm(l4_params, 2)
            loss = loss + l1_l0 + l1_l1 + l1_l2 +\
                l1_l3 + l1_l4 + l2_l0 + l2_l1 +\
                l2_l2 + l2_l3 + l2_l4
        else:
            loss = loss + l1_l0 + l1_l1 + l1_l2 +\
                l1_l3 + l2_l0 + l2_l1 +\
                l2_l2 + l2_l3
    return loss


def motor_log(epoch, j, dict_perf, lr, tempdir,
              current_loss, net, modelpath, macrobatch,
              regression=False):
    if regression:
        log_model_regression(
            os.path.join(
                tempdir,
                "modelLog_lr{}_macrobatch{}.tsv".format(
                    lr, macrobatch)),
            epoch, current_loss, j,
            dict_perf["Tuning.R2"], dict_perf["Tuning.Loss"],
            dict_perf["Training.R2"],
            dict_perf["averageDNase.R2"])
    else:
        log_model(
            os.path.join(
                tempdir,
                "modelLog_lr{}_macrobatch{}.tsv".format(
                    lr, macrobatch)),
            epoch, current_loss, j,
            dict_perf["Tuning.auROC"], dict_perf["Tuning.Loss"],
            dict_perf["Tuning.AveragePrecision"],
            dict_perf["Training.auROC"],
            dict_perf["Training.AveragePrecision"],
            dict_perf["average.auROC"],
            dict_perf["average.AP"])


def log_model_regression(logpath, epoch, train_loss,
                         j, tune_r2, tune_loss,
                         train_r2, baseline_r2):
    current_time = str(datetime.now())
    if epoch == 0:
        if not os.path.exists(logpath):
            with open(logpath, "w") as loglink:
                adlist = [
                    "Time", "Epoch", "MiniBatch", "Training.Loss",
                    "Training.R2", "Tuning.Loss",
                    "Tuning.R2", "averageDnase.R2"]
                loglink.write("\t".join(adlist) + "\n")
    with open(logpath, "a+") as loglink:
        float_vals = [train_loss,
                      train_r2, tune_loss,
                      tune_r2, baseline_r2]
        float_vals = [str(round(each, 5)) for each in float_vals]
        adlist = [current_time, str(epoch), str(j)] + float_vals
        print("\t".join(adlist))
        loglink.write("\t".join(adlist) + "\n")


def log_model(logpath, epoch, train_loss,
              j, tune_auroc, tuning_loss, tuning_ap,
              train_auroc, train_ap,
              baseline_auroc,
              baseline_ap):
    current_time = str(datetime.now())
    if epoch == 0:
        if not os.path.exists(logpath):
            with open(logpath, "w") as loglink:
                adlist = [
                    "Time", "Epoch", "MiniBatch", "Training.Loss",
                    "Training.auROC", "Training.AveragePrecision",
                    "Tuning.Loss", "Tuning.auROC",
                    "Tuning.AveragePrecision",
                    "AverageDnase.auROC",
                    "AverageDnase.averagePrecision"]
                print("\t".join(adlist))
                loglink.write("\t".join(adlist) + "\n")
    with open(logpath, "a+") as loglink:
        float_vals = [train_loss, train_auroc,
                      train_ap, tuning_loss,
                      tune_auroc, tuning_ap,
                      baseline_auroc, baseline_ap]
        float_vals = [str(round(each, 5)) for each in float_vals]
        adlist = [current_time, str(epoch), str(j)] + float_vals
        print("\t".join(adlist))
        loglink.write("\t".join(adlist) + "\n")


def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill='█',
                     printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
                100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),
          end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def split_tensordict(tensordict_all, ratio=0.8):
    len_items = len(tensordict_all["Response"])
    split_idx = int(len_items * ratio)
    tensordict = {}
    tensordict_tune = {}
    for each_key, each_val in tensordict_all.items():
        tensordict[each_key] = tensordict_all[each_key][:split_idx]
        tensordict_tune[each_key] = tensordict_all[each_key][split_idx:]
    return tensordict, tensordict_tune


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_adname(modelparams):
    augmentations = modelparams["augmentations"]
    adname = "Block_{}_Init_{}x{}_K{}".format(
        "_".join([str(each) for each in
                  modelparams["convparam"]]),
        int(modelparams["initconv"]),
        modelparams["filter_rate"],
        int(modelparams["kernel_size"]))
    adname = adname +\
        "_D_{}_Pool_{}_{}_{}_lr_{}".format(
            "_".join(
                [str(each) for each in modelparams["dilations"]]),
            modelparams["pool_type"],
            modelparams["pool_dim"],
            modelparams["activation"],
            modelparams["lr"])
    adname = adname +\
        "s_{}__{}_Aug_{}".format(
            modelparams["stride"],
            modelparams["optimizer"],
            "-".join(augmentations))
    adname = adname +\
        "_DP_{}".format(modelparams["dropout"])
    if modelparams["normtype"] != "BatchNorm":
        adname = adname + "_{}".format(modelparams["normtype"])
    if modelparams["regularize"]:
        if modelparams["ltype"] in [1, 2]:
            adname = adname +\
                "_l{}_{}".format(
                    modelparams["ltype"],
                    modelparams["lambda_param"])
        elif modelparams["ltype"] == 3:
            adname = adname +\
                "_GC_{}".format(modelparams["lambda_param"])
        elif modelparams["ltype"] == 4:
            adname = adname +\
                "_l1Andl2_{}".format(modelparams["lambda_param"])
        else:
            raise ValueError("--ltype > 4 not supported")
    if modelparams.get("regression", False):
        adname = adname + "_regression"
    if "SCALE" in modelparams.keys():
        scale_str = "-".join(
            [str(each) for each in modelparams["SCALE"]])
        scale_str = scale_str + "-{}".format(modelparams["SCALE_OP"])
        adname = adname + "_{}".format(scale_str)
    if "LOSS_SCALERS" in modelparams.keys():
        scale_str = "-".join(
            [str(each) for each in modelparams["LOSS_SCALERS"]])
        adname = adname + "_{}".format(scale_str)
    if "RESP_THRESH" in modelparams.keys():
        ad_str = "Resp.Quantile.{}".format(
            modelparams["RESP_THRESH"])
        adname = adname + "_{}".format(ad_str)
    if "arcsinh" in modelparams.keys():
        adname = adname + "_{}{}".format(
            "arcsinh", modelparams["arcsinh"])
    return adname


def compile_paths(outdir, modelparams):
    adname = make_adname(modelparams)
    logdir = os.path.join(
        outdir, "modelLog", adname)
    os.makedirs(logdir, exist_ok=True)
    chkdir = os.path.join(
        "/checkpoint/mkarimza",
        os.environ["SLURM_JOB_ID"])
    if not os.path.exists(chkdir):
        chkdir = os.path.join(logdir, "checkpoints")
        os.makedirs(chkdir, exist_ok=True)
    chkpaths = [
        os.path.join(chkdir, "{}_{}.pt".format(adname, each))
        for each in [0, 1]]
    itrpaths = [
        os.path.join(logdir, "iterDetails_{}.pickle".format(each))
        for each in [0, 1]]
    modelpath = os.path.join(
        logdir,
        "{}_currentmodel.pt".format(adname))
    modelpath_bestloss = os.path.join(
        logdir,
        "{}_bestmodel.pt".format(adname))
    tempdir = os.path.join(
        outdir, "tempData")
    os.makedirs(tempdir, exist_ok=True)
    dictpaths = {
        "adname": adname,
        "chkpaths": chkpaths, "logdir": logdir,
        "modelpath": modelpath,
        "modelpath_bestloss": modelpath_bestloss,
        "tempdir": tempdir, "itrpaths": itrpaths}
    return dictpaths


def prepare_response(response_tensor):
    response_tensor[np.where(response_tensor > 0)] = 1
    response_tensor = np.array(response_tensor, dtype=int)
    response_tensor = torch.from_numpy(response_tensor)
    response_onehot = torch.FloatTensor(
        response_tensor.shape[0], 2)
    response_onehot.zero_()
    response_onehot.scatter_(1, response_tensor, 1)
    return response_onehot


def find_minibatch_size_fast(tensordict, net, criterion,
                             optimizer, device, max_memory=9e9,
                             regression=False):
    from apex import amp
    num_devices = torch.cuda.device_count()
    MINIBATCH = 2
    dnase_tensor = torch.from_numpy(
        tensordict["DNase"][:MINIBATCH]).to(device)
    rna_tensor = torch.from_numpy(
        tensordict["RNA"][:MINIBATCH]).to(device)
    if not regression:
        response_tensor = prepare_response(
            tensordict["Response"][:MINIBATCH])
        optimizer.zero_grad()
        labels = response_tensor.long()
        label_idx = torch.max(labels, 1)[1].to(device)
    else:
        label_idx = torch.from_numpy(
            tensordict["Response"][:MINIBATCH]).to(device)
    model_init = net(
        dnase_tensor,
        rna_tensor)
    try:
        loss = criterion(
            model_init, label_idx)
    except Exception:
        loss = criterion(
            model_init[0], label_idx)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    used_memory = torch.cuda.max_memory_allocated(device)
    print("Processing 2 batches needs {} GB of GPU memory".format(
            used_memory / 1e9))
    newbatch = MINIBATCH * int(max_memory / used_memory) * num_devices
    print("Set minibatch size to {}".format(newbatch))
    del dnase_tensor, rna_tensor
    if not regression:
        del labels, label_idx, model_init, response_tensor
    else:
        del label_idx
    torch.cuda.empty_cache()
    MINIBATCH = newbatch
    try:
        dnase_tensor = torch.from_numpy(
            tensordict["DNase"][:MINIBATCH]).to(device)
        rna_tensor = torch.from_numpy(
            tensordict["RNA"][:MINIBATCH]).to(device)
        if not regression:
            response_tensor = prepare_response(
                tensordict["Response"][:MINIBATCH])
            labels = response_tensor.long()
            label_idx = torch.max(labels, 1)[1].to(device)
        else:
            label_idx = torch.from_numpy(
                tensordict["Response"][:MINIBATCH]).to(device)
        optimizer.zero_grad()
        model_init = net(
            dnase_tensor,
            rna_tensor)
        loss = criterion(
            model_init, label_idx)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        del dnase_tensor, rna_tensor
        if not regression:
            del labels, label_idx, model_init, response_tensor
        else:
            del label_idx
    except Exception:
        newbatch = int(MINIBATCH * 0.8)
    torch.cuda.empty_cache()
    return newbatch


class Augmentator:
    def __init__(self, ars, response=[], nucs=["A", "T", "C", "G"]):
        '''
        Accepts a list of arrays to perform a
        variety of augmentations on them

        ars: a list of numpy arrays
        response: response (only needed for .mask_signal)
        nucs: by default set to A T C G order similar
              to DataHandler
        '''
        self.ars = ars
        self.response = response
        self.nucs = nucs
        self.dict_nucs = {"A": "T", "T": "A",
                          "C": "G", "G": "C",
                          "N": "N"}

    def reverse_complement(self):
        '''
        Reverse complements arrays within self.ars
        '''
        list_ars = []
        for ar in self.ars:
            assert len(ar.shape) == 3, "Expects tensor"
            newar = np.zeros(
                ar.shape, dtype=np.float16)
            for i in range(4):
                nuc = self.nucs[i]
                rev_nuc = self.dict_nucs[nuc]
                # i' is the index of the complement nucleotide
                i_prime = np.where(
                    np.array(self.nucs) == rev_nuc)[0]
                idx_nuc = np.where(ar[:, i, :] > 0)
                # for idx_nuc[1] which refers to position, reverse it
                newar[idx_nuc[0], i_prime, newar.shape[2] - idx_nuc[1] - 1] = \
                    ar[idx_nuc[0], i, idx_nuc[1]]
            list_ars.append(newar)
        return list_ars, self.response

    def mask_background(self):
        '''
        For positions without any signal, set the values to 0
        Signal is identified as value > 0.1
        '''
        BGVAL = 0.1  # background value
        list_ars = []
        for ar in self.ars:
            assert len(ar.shape) == 3, "Expects tensor"
            newar = np.zeros(
                ar.shape, dtype=np.float16)
            for i in range(4):
                idx_add = np.where(ar[:, i, :] > BGVAL)
                newar[idx_add[0], i, idx_add[1]] = \
                    ar[idx_add[0], i, idx_add[1]]
            list_ars.append(newar)
        return list_ars, self.response

    def mask_signal(self):
        '''
        Use this type of augmentation to convert a positive
        example to a negative example.
        Requires self.response and will set it to 0.
        '''
        assert len(self.response) > 0, "Expects response"
        BGVAL = 0.1  # background value
        list_ars = []
        for ar in self.ars:
            assert len(ar.shape) == 3, "Expects tensor"
            newar = np.zeros(
                ar.shape, dtype=np.float16)
            for i in range(4):
                idx_add = np.where(ar[:, i, :] == BGVAL)
                newar[idx_add[0], i, idx_add[1]] = \
                    ar[idx_add[0], i, idx_add[1]]
            list_ars.append(newar)
        out_resp = self.response.copy()
        out_resp[out_resp > 0] = 0
        return list_ars, out_resp


class DataHandler:
    def __init__(self, dnasematpath, dnasemetapath,
                 dnaseindexpath, sequencedir,
                 trainchroms, validchroms,
                 validorgans, window=50000,
                 limit=True,
                 select_organs=["default"],
                 select_labels=["default"],
                 tissuespecific=False):
        '''
        Arguments:
            dnasematpath: Path to dat_FDR01_hg38.txt
            dnasemetapath: Path to DNaseMetaDataWithAvailableRnas.tsv
            dnaseindexpath: Path to DHS_Index_and_Vocabulaty...
            sequencedir: Path to hg38/np
            trainchroms: List of chromosomes to use for training
            validchroms: List of chromosomes to use for validation
            validorgans: List of organs to exclude
            window: Window size around each specific peak
            limit: Boolean indicating if should use tissue-specific
                   peaks only. It will ignore trainids and will only
                   use specific tissues outlined in
                   self.select_organs and self.select_labels
            select_organs: list of organs to limit to
            select_labels: list of enhancer labels to limit to
            tissuespecific: If you'd want multiple samples form the
                            same region
        '''
        self.tissuespecific = tissuespecific
        self.nucs = np.array(["A", "T", "C", "G"])
        self.trainchroms = trainchroms
        self.validchroms = validchroms
        # self.trainids = trainids  # ignore trainids
        self.validorgans = validorgans
        self.dnasematpath = dnasematpath
        self.dnasemetapath = dnasemetapath
        self.dnaseindexpath = dnaseindexpath
        self.sequencedir = sequencedir
        self.window = window
        self.limit = limit
        self.dict_indices_train = {}
        self.select_organs = [
            "Brain", "Placenta", "Tongue",
            "Muscle", "Blood", "Spinal Cord",
            "Heart", "Lung"]
        self.select_labels = [
            "Neural", "Lymphoid",
            "Placental / trophoblast",
            "Musculoskeletal",
            "Cardiac",
            "Pulmonary devel."]
        if select_organs[0] != "default":
            self.select_organs = select_organs
        if select_labels[0] != "default":
            self.select_labels = select_labels
        self.dictissue = {
            "Brain": ["Neural"],
            "Placenta": ["Placental / trophoblast"],
            "Tonge": ["Musculoskeletal"],
            "Muscle": ["Musculoskeletal"],
            "Blood": ["Lymphoid"],
            "Spinal Cord": ["Neural"],
            "Heart": ["Cardiac", "Musculoskeletal"],
            "Lung": ["Pulmonary devel."]}
        self.process_data()

    def get_rna(self, idx_region, idx_sample):
        chrom, summit = [
            self.idxdf.iloc[idx_region, j] for j in
            [0, 6]]
        start = summit - self.window
        end = summit + self.window
        chrom_seq = self.get_seq(chrom)
        rnatensor = self.initiate_seq(
            chrom_seq, start, end)
        if start < 0:
            start = 0
        tempdf = self.metadf.iloc[idx_sample, :]
        rnapaths = tempdf["RnaPaths"].split(",")
        list_rnas = []
        for rnapath in rnapaths:
            rnatensorlocal = rnatensor.copy()
            try:
                bw = pyBigWig.open(rnapath)
            except Exception:
                print("Check {}".format(rnapath))
                raise ValueError("BigWig issue, check logs")
            chrom_max = bw.chroms()[chrom]
            if end > chrom_max:
                end = chrom_max
            signal = bw.values(chrom, start, end, numpy=True)
            signal[np.isinf(signal)] = max(
                signal[np.logical_not(np.isinf(signal))])
            for j in range(len(self.nucs)):
                nuc = self.nucs[j].encode()
                i = np.where(
                    np.logical_and(
                        signal > 0,
                        chrom_seq[start:end] == nuc))[0]
                rnatensorlocal[j, i] = rnatensorlocal[j, i] + signal[i]
            list_rnas.append(rnatensorlocal)
        if len(list_rnas) == 1:
            rnatensor = list_rnas[0]
        else:
            rnatensor = np.zeros(rnatensor.shape)
            for each in list_rnas:
                rnatensor = rnatensor + each
            rnatensor = rnatensor / len(list_rnas)
        return rnatensor

    def process_data(self):
        # dnasemat = pd.read_csv(self.dnasematpath, sep="\t")
        self.metadf = pd.read_csv(self.dnasemetapath, sep="\t")
        self.metadf["Index"] = np.arange(self.metadf.shape[0])
        self.idxdf = pd.read_csv(
            self.dnaseindexpath, sep="\t", compression="gzip")
        self.idxdf["Index"] = np.arange(self.idxdf.shape[0])
        if self.limit:
            self.metadf = self.metadf[
                self.metadf["Organ"].isin(self.select_organs)]
            self.idxdf = self.idxdf[
                self.idxdf["component"].isin(self.select_labels)]
        trainiddf = self.metadf[
            np.logical_not(pd.isna(self.metadf["RnaPaths"]))]
        self.trainorgans = [
            each for each in pd.unique(trainiddf["Organ"])
            if each not in self.validorgans]
        print("Will use the following organs: {}".format(
                self.trainorgans))
        self.idx_sample_train = np.where(
            np.logical_and(
                self.metadf["Organ"].isin(self.trainorgans),
                np.logical_not(pd.isna(self.metadf["RnaPaths"]))))[0]
        self.idx_sample_valid = np.where(
            np.logical_and(
                self.metadf["Organ"].isin(self.validorgans),
                np.logical_not(pd.isna(self.metadf["RnaPaths"]))))[0]
        self.idx_train = np.where(
            self.idxdf["seqname"].isin(self.trainchroms))[0]
        self.idx_valid = np.where(
            self.idxdf["seqname"].isin(self.validchroms))[0]

    def get_seq(self, chrom):
        arpath = os.path.join(
            self.sequencedir,
            "{}_sequence.numpy.gz".format(chrom))
        with gzip.open(arpath, "rb") as arlink:
            npar = np.load(arlink)
        return npar

    def get_valid(self, chrom, sampleid, idx_part, all_parts):
        # Find the index of sample
        # tempdf = self.metadf[
        #     self.metadf["DCC Biosample ID"] == sampleid]
        # idx_sample = list(tempdf["Index"])[0]
        idx_sample = np.where(
            self.metadf["DCC Biosample ID"] == sampleid)[0][0]
        # Find row indices of the chromosome
        idxs_chrom = np.where(
            self.idxdf["seqname"] == chrom)[0]
        x_parts = int(len(idxs_chrom) / all_parts)
        idx_st = idx_part * x_parts
        idx_end = (idx_part + 1) * x_parts
        if idx_end > len(idxs_chrom):
            idx_end = len(idxs_chrom)
        num_regions = len(np.arange(idx_st, idx_end))
        dnase_tensor = np.zeros(
            (num_regions, 4, self.window * 2), dtype=float)
        rna_tensor = np.zeros(
            (num_regions, 4, self.window * 2), dtype=float)
        response_tensor = np.zeros((num_regions, 1), dtype=float)
        SAMPLES = np.empty(num_regions, dtype="|S16")
        REGIONS = np.empty(num_regions, dtype="|S16")
        TISSUE = np.empty(num_regions, dtype="|S128")
        i = 0
        for idx_region in range(idx_st, idx_end):
            dnase_tensor[i] = self.get_dnase(idxs_chrom[idx_region])
            rna_tensor[i] = self.get_rna(
                idxs_chrom[idx_region], idx_sample)
            response_tensor[i] = self.get_response(
                idxs_chrom[idx_region], idx_sample)
            SAMPLES[i] = self.metadf.loc[idx_sample, "DCC Biosample ID"]
            chrom = self.idxdf.iloc[idxs_chrom[idx_region], 0]
            summit = self.idxdf.iloc[idxs_chrom[idx_region], 6]
            TISSUE[i] = self.idxdf.iloc[idxs_chrom[idx_region], 9]
            REGIONS[i] = "{}:{}".format(chrom, summit)
            i = i + 1
            if i % 10 == 0:
                print(
                    "Added {}/{} batches at {}".format(
                        i, num_regions,
                        str(datetime.now())))
        dict_tensors = {
            "DNase": dnase_tensor,
            "RNA": rna_tensor,
            "Response": response_tensor,
            "Samples": SAMPLES,
            "Regions": REGIONS,
            "Tissue": TISSUE}
        return dict_tensors

    def sample_idx_regions(self, idx_regions, idx_samples, batchsize,
                           imbalance=0.25):
        dict_indices = {"Active": {},
                        "Inactive": {}}
        i = 0
        num_samples_per_region = int((1 / imbalance) * 5)
        num_pos_per_region = int(
            np.round(num_samples_per_region * imbalance))
        num_neg_per_region = num_samples_per_region -\
            num_pos_per_region
        curtime = time.time()
        minpos = int(batchsize * imbalance)
        minneg = int(batchsize / imbalance)
        # minpos = int(batchsize / 4)
        # minneg = int(batchsize * 10)
        MINPOS = minpos
        MINNEG = minneg
        TOTAL_REGS = len(idx_regions) * len(idx_samples)
        for idx_region in idx_regions:
            all_idxs = []
            region_values = self.get_responses(idx_region, idx_samples)
            # region_values = np.array(
            #     [self.get_response(idx_region, idxtemp) for idxtemp in
            #      range(self.metadf.shape[0])])
            if self.tissuespecific:
                idxs_pos = np.where(region_values > 0)[0]
                idxs_pos = idx_samples[idxs_pos]
                # idxs_pos = np.intersect1d(idxs_pos, idx_samples)
                idxs_neg = np.where(region_values == 0)[0]
                idxs_neg = idx_samples[idxs_neg]
                # idxs_neg = np.intersect1d(idxs_neg, idx_samples)
                if len(idxs_pos) >= num_pos_per_region and len(idxs_neg) > 8:
                    all_idxs = np.union1d(
                        np.random.choice(idxs_pos, num_pos_per_region),
                        np.random.choice(idxs_neg, num_neg_per_region))
            else:
                all_idxs = idx_samples
            for idx_sample in all_idxs:
                response = self.get_response(
                    idx_region, idx_sample)
                ad_dict = {
                    "idxRegion": idx_region,
                    "idxSample": idx_sample}
                if response > 0:
                    dict_indices["Active"][i] = ad_dict
                    i += 1
                    minpos -= 1
                else:
                    dict_indices["Inactive"][i] = ad_dict
                    i += 1
                    minneg -= 1
                if i % 10000 == 0:
                    print("Added {}/{} regions in {}s".format(
                        i, TOTAL_REGS, int(time.time() - curtime)))
                    print("Found {} positive and {} negatives".format(
                        MINPOS - minpos, MINNEG - minneg))
                if minpos <= 0 and minneg <= 0:
                    break
            if minpos <= 0 and minneg <= 0:
                break
        if not self.tissuespecific:
            dict_indices = self.permutate_dict(dict_indices)
        return dict_indices

    def permutate_dict(self, dict_indices):
        newdict = {}
        for each in ["Active", "Inactive"]:
            newdict[each] = {}
            tempdict = dict_indices[each]
            tempkeys = list(tempdict.keys())
            tempkeys = np.random.choice(tempkeys, len(tempkeys))
            newkey = 0
            for eachkey in tempkeys:
                newdict[each][newkey] = tempdict[eachkey]
                newkey += 1
        return newdict

    def initialize_train(self, batchsize=500000, outpath="", imb=0.4):
        outdir = os.path.dirname(self.dnasematpath)
        if(len(outpath) == 0):
            outpath = os.path.join(
                outdir,
                "IndexInfo_Batchsize_{}_{}_Imb_{}.joblib.gz".format(
                    batchsize, self.tissuespecific, imb))
        if not os.path.exists(outpath):
            idx_regions = np.random.choice(
                self.idx_train, self.idx_train.shape[0])
            idx_samples = np.random.choice(
                self.idx_sample_train,
                self.idx_sample_train.shape[0])
            self.dict_indices_train = self.sample_idx_regions(
                idx_regions, idx_samples, batchsize, imb)
            joblib.dump(
                self.dict_indices_train,
                outpath, compress=9)
        else:
            self.dict_indices_train = joblib.load(outpath)
        self.indices_active = np.array(
            list(self.dict_indices_train["Active"].keys()))
        self.indices_inactive = np.array(
            list(self.dict_indices_train["Inactive"].keys()))

    def get_trainbatch_fast(self, batchsize, stidx, endidx, imb=5):
        SAMPLES = np.empty(batchsize, dtype="|S64")
        REGIONS = np.empty(batchsize, dtype="|S64")
        TISSUES = np.empty(batchsize, dtype="|S128")
        dnasetensor = np.zeros((batchsize, 4, self.window * 2))
        rnatensor = np.zeros((batchsize, 4, self.window * 2))
        responsetensor = np.zeros((batchsize, 1))
        dict_indices = self.dict_indices_train
        i = 0
        for k in range(stidx, endidx):
            try:
                curdict = dict_indices["Active"][k]
            except Exception:
                curdict = dict_indices["Inactive"][k]
            idx_region = curdict["idxRegion"]
            idx_sample = curdict["idxSample"]
            dnasetensor[i] = self.get_dnase(
                idx_region)
            rnatensor[i] = self.get_rna(
                idx_region, idx_sample)
            responsetensor[i] = self.get_response(
                idx_region, idx_sample)
            SAMPLES[i] = list(
                self.metadf["DCC Biosample ID"])[idx_sample]
            TISSUES[i] = self.idxdf.iloc[idx_region, 9]
            chrom = self.idxdf.iloc[idx_region, 0]
            summit = self.idxdf.iloc[idx_region, 6]
            REGIONS[i] = "{}:{}".format(chrom, summit)
            i += 1
            if i % 10 == 0:
                print(
                    "Added {}/{} batches at {}".format(
                        i, batchsize,
                        str(datetime.now())))
        dict_tensors = {
            "DNase": dnasetensor,
            "RNA": rnatensor,
            "Response": responsetensor,
            "Samples": SAMPLES,
            "Regions": REGIONS,
            "Tissues": TISSUES}
        return dict_tensors

    def get_trainbatch_notrandom(self, batchsize, jobid, imb=5,
                                 numregs=5000, outpath=""):
        self.initialize_train(numregs, outpath)
        stidx = jobid * batchsize
        endidx = (jobid + 1) * batchsize
        # posidx = int((jobid + 1) * batchsize / 5)
        # negidx = posidx + int(batchsize * 4 / 5)
        dict_tensors = self.get_trainbatch_fast(
            batchsize, stidx, endidx)
        return dict_tensors

    def get_trainbatch(self, batchsize=1000, num_regions=200, imb=5):
        '''
        imbalance parameter the divisible index of positive cases.
        imb=2 default results in imbalance of 0.5.
        imb=3 results in imbalance of 0.33.
        imb=4 results in imbalance of 0.25, etc.
        '''
        # if num_regions > batchsize:
        #     print("Specify both batchsize and num_regions")
        #     num_regions = int(batchsize / 2)
        # num_samples = int(batchsize / num_regions)
        SAMPLES = np.empty(batchsize, dtype="|S64")
        REGIONS = np.empty(batchsize, dtype="|S64")
        TISSUES = np.empty(batchsize, dtype="|S128")
        dnasetensor = np.zeros((batchsize, 4, self.window * 2))
        rnatensor = np.zeros((batchsize, 4, self.window * 2))
        responsetensor = np.zeros((batchsize, 1))
        i = 0
        idx_regions = np.random.choice(
            self.idx_train, self.idx_train.shape[0])
        idx_samples = np.random.choice(
            self.idx_sample_train,
            self.idx_sample_train.shape[0])
        if self.dict_indices_train.get(1, "NA") != "NA":
            dict_indices = self.dict_indices_train
        else:
            dict_indices = self.sample_idx_regions(
                idx_regions, idx_samples, batchsize)
        indices_active = np.array(list(dict_indices["Active"].keys()))
        indices_inactive = np.array(list(dict_indices["Inactive"].keys()))
        np.random.shuffle(indices_active)
        np.random.shuffle(indices_inactive)
        if len(indices_active) < int(batchsize / 5):
            print("Didn't find enough positive training samples")
            raise ValueError("Stopped due to not-enough training samples")
        posidx = 0
        negidx = 0
        for i in range(batchsize):
            if i % imb == 0:
                curdict = dict_indices["Active"][indices_active[posidx]]
                idx_region = curdict["idxRegion"]
                idx_sample = curdict["idxSample"]
                posidx += 1
            else:
                curdict = dict_indices["Inactive"][indices_inactive[negidx]]
                idx_region = curdict["idxRegion"]
                idx_sample = curdict["idxSample"]
                negidx += 1
            dnasetensor[i] = self.get_dnase(
                idx_region)
            rnatensor[i] = self.get_rna(
                idx_region, idx_sample)
            responsetensor[i] = self.get_response(
                idx_region, idx_sample)
            SAMPLES[i] = list(
                self.metadf["DCC Biosample ID"])[idx_sample]
            TISSUES[i] = self.idxdf.iloc[idx_region, 9]
            chrom = self.idxdf.iloc[idx_region, 0]
            summit = self.idxdf.iloc[idx_region, 6]
            REGIONS[i] = "{}:{}".format(chrom, summit)
            if i % 10 == 0:
                print(
                    "Added {}/{} batches at {}".format(
                        i, batchsize,
                        str(datetime.now())))
        dict_tensors = {
            "DNase": dnasetensor,
            "RNA": rnatensor,
            "Response": responsetensor,
            "Samples": SAMPLES,
            "Regions": REGIONS,
            "Tissues": TISSUES}
        return dict_tensors

    def get_responses(self, idx_region, idx_samples):
        idx_file = self.idxdf.iloc[idx_region, :]["Index"]
        curline = linecache.getline(self.dnasematpath, idx_file + 1)
        values = np.array(
            [float(each) for each in curline.rstrip().split("\t")])
        idx_samples_cor = np.array(
            self.metadf.iloc[idx_samples, :]["Index"])
        out_vals = values[idx_samples_cor]
        return out_vals

    def get_response(self, idx_region, idx_sample):
        idx_file = self.idxdf.iloc[idx_region, :]["Index"]
        idx_sample_cor = self.metadf.iloc[idx_sample, :]["Index"]
        curline = linecache.getline(self.dnasematpath, idx_file + 1)
        try:
            values = np.array(
                [float(each) for each in curline.rstrip().split("\t")])
        except Exception:
            print(curline)
            print(
                "Error occured at Region {} Sample {}".format(
                    idx_region, idx_sample))
            raise ValueError("Error with index out of bounds")
        if values[idx_sample_cor] > 0:
            return 1
        else:
            return 0

    def initiate_seq(self, chrom_seq, start, end):
        tensor = np.zeros((4, self.window * 2), dtype=float)
        for nucidx in range(len(self.nucs)):
            nuc = self.nucs[nucidx].encode()
            j = np.where(chrom_seq[start:end] == nuc)[0]
            tensor[nucidx, j] = \
                tensor[nucidx, j] + 0.1
        return tensor

    def get_dnase(self, idx_region):
        chrom = self.idxdf.iloc[idx_region, 0]
        summit = self.idxdf.iloc[idx_region, 6]
        chrom_seq = self.get_seq(chrom)
        start = summit - self.window
        end = summit + self.window
        dnasetensor = self.initiate_seq(
            chrom_seq, start, end)
        # Limit the idxdf to summit +- window
        tempdf = self.idxdf[
            np.logical_and(
                self.idxdf["start"] > start,
                self.idxdf["end"] < end)]
        tempdf = tempdf[tempdf["seqname"] == chrom]
        # For each of the entries, iterate throught their nucleotides
        # and add the average value
        for i in range(tempdf.shape[0]):
            ist, iend = tempdf.iloc[i, 1], tempdf.iloc[i, 2]
            idxst = ist - start
            idxend = iend - start
            idx_cur = list(tempdf["Index"])[i]
            curline = linecache.getline(self.dnasematpath, idx_cur + 1)
            values = np.array(
                [float(each) for each in curline.rstrip().split("\t")])
            av_val = np.mean(values[self.idx_sample_train])
            for nucidx in range(len(self.nucs)):
                nuc = self.nucs[nucidx].encode()
                j = np.where(chrom_seq[start:end] == nuc)[0]
                # Filter to the coordinate of region i in tempdf
                j = j[np.logical_and(j > idxst, j < idxend)]
                dnasetensor[nucidx, j] = \
                    dnasetensor[nucidx, j] + av_val
            # rnatensorlocal[j, i] = rnatensorlocal[j, i] + signal[i]
            # for j in range(ist, iend):
            #     curloc = j - summit + self.window
            #     curnuc = chrom_seq[j].decode()
            #     if curnuc != "N":
            #         nucidx = np.where(self.nucs == curnuc)[0][0]
            #         dnasetensor[nucidx, curloc] = \
            #             dnasetensor[nucidx, curloc] + av_val
        return dnasetensor

    def get_specific_label(self, regions):
        TISSUES = np.empty(len(regions), dtype="|S128")
        i = 0
        for region in regions:
            chrom, summit = region.decode().split(":")
            adtissue = self.idxdf.iloc[
                np.where(
                    np.logical_and(
                        self.idxdf["seqname"] == chrom,
                        self.idxdf["summit"] == int(summit)))[0][0], -2]
            TISSUES[i] = adtissue
            i += 1
        return TISSUES
