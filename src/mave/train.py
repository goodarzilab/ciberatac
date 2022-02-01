# import adabound
from apex import amp
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
import scipy.sparse as sp_sparse
import tables
from itertools import chain
from model import loss_function
from model import VAE
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
# from train_multitask_ccle import read_tsv
import torch


opt_level = 'O1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_tsv(nparpath, genes, outdir, gmtmat, normalize_vals=True):
    h5outpath = os.path.join(
        outdir, "cellByGeneMatrix.npz")
    if "gct" in nparpath:
        rnadf = pd.read_csv(
            nparpath, sep="\t", index_col=0,
            compression="gzip", skiprows=2)
        rnadf.drop_duplicates(subset=["Description"], inplace=True)
        rnadf = rnadf[rnadf["Description"].isin(genes)]
        npar = np.array(rnadf.iloc[:, 1:])
        ar_genes = np.array(rnadf["Description"])
        barcodes = np.array(rnadf.columns[1:])
    else:
        rnadf = pd.read_csv(
            nparpath, sep="\t", index_col=0,
            compression="gzip")
        npar = np.array(rnadf)
        ar_genes = rnadf.index
        barcodes = np.array(rnadf.columns)
    # Divide by max
    # arsum = np.matrix.sum(npar, axis=0)
    if normalize_vals:
        arsum = np.apply_along_axis(np.sum, 0, npar)
        npar = (npar * 1000) / arsum
    _, idx_g1, idx_g2 = np.intersect1d(genes, ar_genes, return_indices=True)
    npar = npar[idx_g2, :]
    gmtmat = gmtmat[idx_g1, :]
    out_genes = genes[idx_g1]
    npar = np.transpose(npar)
    np.savez_compressed(h5outpath, arr=npar, barcodes=barcodes,
                        genes=ar_genes)
    return npar, barcodes, gmtmat, out_genes


def make_plot_umap(mudf, metadf, outdir, numlvs=10):
    metadf.index = metadf["Barcode"]
    import umap
    import seaborn as sns
    mumat = np.array(mudf.iloc[:, :numlvs])
    for n_neighbors in [10, 100]:
        for min_dist in [0.45]:
            adname = "UMAP_dist-{}_nNeigh-{}".format(
                min_dist, n_neighbors)
            print(adname)
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist)
            embedding = reducer.fit_transform(mumat)
            umap_output = pd.DataFrame(embedding)
            umap_output.columns = ["UMAP1", "UMAP2"]
            umap_output["CellType"] = list(metadf.loc[mudf.index, "CellType"])
            umap_output.index = mudf.index
            umap_output.to_csv(
                os.path.join(outdir, adname + ".tsv.gz"),
                sep="\t", compression="gzip")
            sns_plot = sns.relplot(
                x="UMAP1", y="UMAP2", hue="CellType", data=umap_output,
                height=6, aspect=1.5)
            sns_plot.savefig(
                os.path.join(outdir, adname + ".pdf"))
            sns_plot.savefig(
                os.path.join(outdir, adname + ".png"))


def make_args():
    metapaths = [
        "/scratch/hdd001/home/mkarimza/" +
        "ciberAtac/10x/raw/scRNA-seq_10XPBMC" +
        "_metadataWithCellType.tsv",
        "/scratch/ssd001/home/mkarimza/" +
        "data/ciberatac/models/vae202012/" +
        "SW480Files/metadata_for_vae_visualization.tsv"]
    nparpaths = [
        "/scratch/hdd001/home/mkarimza/" +
        "ciberAtac/10x/raw/pbmc_unsorted_10k" +
        "_filtered_feature_bc_matrix.h5",
        "/scratch/hdd001/home/mkarimza/" +
        "johnny/A06/10X/outs/" +
        "filtered_feature_bc_matrix.h5"]
    genepath = "/scratch/ssd001/home/mkarimza/" +\
        "data/ciberatac/models/vae202101/" +\
        "scviVersusCustomized/customizedScvi" +\
        "FullTrainScaled1000/genes.txt"
    gmtpath = "../c3.tft.v7.2.symbols.gmt"
    genepath = "/scratch/ssd001/home/mkarimza/" +\
        "data/ciberatac/models/vae202012/" +\
        "commonGenes/Genes_passing_40p.txt"
    outdir = "/scratch/ssd001/home/mkarimza/" +\
        "data/ciberatac/models/vae202101/" +\
        "customScviAppliedOnPbmcAndSw480"
    numlvs = 10
    os.makedirs(outdir, exist_ok=True)
    existingmodelpath = "/scratch/ssd001/home/mkarimza/" +\
        "data/ciberatac/models/vae202101/" +\
        "scviVersusCustomized/customized" +\
        "ScviFullTrainScaled1000/VAE_10LVS.pt"
    use_connections = True
    loss_scalers = [1, 1, 1]
    predict_celltypes = True
    num_celltypes = 11
    argslist = [gmtpath, nparpaths, outdir,
                numlvs, genepath, metapaths,
                existingmodelpath,
                use_connections,
                loss_scalers,
                predict_celltypes,
                num_celltypes]
    return argslist


def get_matrix_from_h5(filename):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        tag_keys = getattr(feature_group, '_all_tag_keys').read()
        for key in tag_keys:
            feature_ref[key] = getattr(feature_group, key.decode()).read()

        return feature_ref, barcodes, matrix


def read_npz(nparpath, genes, outdir, gmtmat):
    h5outpath = os.path.join(
        outdir, "cellByGeneMatrix.npz")
    npobj = np.load(nparpath, allow_pickle=True)
    npar = npobj["arr"]
    if npar.shape[0] > npar.shape[1]:
        npar = np.transpose(npar)
    ar_genes = npobj["rows"]
    barcodes = npobj["cols"]
    _, idx_g1, idx_g2 = np.intersect1d(genes, ar_genes, return_indices=True)
    # arsum = np.matrix.sum(npar, axis=0)
    # arsum = np.apply_along_axis(np.sum, 0, npar)
    npar = npar[:, idx_g2]
    gmtmat = gmtmat[idx_g1, :]
    out_genes = genes[idx_g1]
    np.savez_compressed(h5outpath, arr=npar, barcodes=barcodes)
    return npar, barcodes, gmtmat, out_genes


def read_h5(h5path, genes, outdir, gmtmat):
    h5outpath = os.path.join(
        outdir, "cellByGeneMatrix.npz")
    # Must be in form of filtered feature matrix
    feature_ref, barcodes, matrix = get_matrix_from_h5(h5path)
    # Limit the array to gene expression
    idx_gexp = np.where(
        np.array(feature_ref["feature_type"] == b'Gene Expression'))[0]
    npar = matrix.toarray()
    npar = np.transpose(npar[idx_gexp, :])
    # Normalize npar by dividing by sum of the reads then multiplying by 1000)
    # arsum = np.apply_along_axis(np.sum, 0, npar)
    # arsum2d = np.zeros((1, npar.shape[1]))
    # arsum2d[0, :] = arsum
    # npar_scaled = (npar / arsum) * 1000
    # tmat = np.transpose(npar_scaled)
    expar = np.zeros((len(barcodes), len(genes)), dtype=float)
    gene_names = np.array(
        feature_ref["name"], dtype="|U64")
    _, idx_g1, idx_g2 = np.intersect1d(genes, gene_names, return_indices=True)
    expar[:, idx_g1] = npar[:, idx_g2]
    np.savez_compressed(h5outpath, arr=npar, barcodes=barcodes, genes=genes)
    # return npar, barcodes
    return expar, barcodes, gmtmat, genes


def get_genes_from_txt(genepath):
    select_genes = np.loadtxt(genepath, dtype="|U64")
    return select_genes


def make_gmtmat(gmtpath, outdir, genepath):
    gmtoutpath = os.path.join(
        outdir, "gmt_conv_matrix.npz")
    if os.path.exists(gmtoutpath):
        npobj = np.load(gmtoutpath)
        npar = npobj["arr"]
        all_tfs = npobj["tfs"]
        all_genes = npobj["genes"]
        return npar, all_tfs, all_genes
    gmtdict = {}
    with open(gmtpath, "r") as gmtlink:
        for gmtline in gmtlink:
            gmtlist = gmtline.rstrip().split("\t")
            gmtdict[gmtlist[0]] = gmtlist[2:]
    all_tfs = np.array(list(gmtdict.keys()))
    all_tfs = np.sort(all_tfs)
    all_genes = list(gmtdict.values())
    all_genes = list(chain.from_iterable(all_genes))
    all_genes = np.unique(all_genes)
    if genepath != "NA" and os.path.exists(genepath):
        select_genes = get_genes_from_txt(genepath)
        print("Limiting to {} genes found in {}".format(
            len(select_genes), genepath))
        all_genes = np.intersect1d(all_genes, select_genes)
    print("Found {} TFs and {} genes in {}".format(
        len(all_tfs), len(all_genes),
        gmtpath))
    npar = np.zeros((len(all_genes), len(all_tfs)), dtype=bool)
    for tf in all_tfs:
        idx_tf = np.where(all_tfs == tf)[0]
        genes = gmtdict[tf]
        # add index and +1 for the array
        for gene in genes:
            idx_gene = np.where(all_genes == gene)[0]
            npar[idx_gene, idx_tf] = True
        if idx_tf % 100 == 0:
            print("{}/{} TFs added".format(idx_tf[0], len(all_tfs)))
    np.savez_compressed(
        gmtoutpath, arr=npar, tfs=all_tfs, genes=all_genes)
    return npar, all_tfs, all_genes


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_paths(outdir, numlvs):
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    modelpath = os.path.join(
        outdir, "VAE_{}LVS.pt".format(numlvs))
    chkdir = os.path.join(
        "/checkpoint/mkarimza",
        os.environ["SLURM_JOB_ID"])
    chkpath = os.path.join(
        chkdir, "VAE_{}LVS.pt".format(numlvs))
    return logdir, modelpath, chkpath


def train_model(vae, optimizer, MINIBATCH, MAXEPOCH, expar, logdir,
                modelpath, chkpath, one_hot, loss_scalers, predict_celltypes,
                celltypes=[]):
    criterion_class = torch.nn.CrossEntropyLoss()
    time_str = str(datetime.now())
    time_str = time_str.replace(" ", "_")
    time_str = time_str.replace(":", "0")
    logpath = os.path.join(
        logdir,
        "training.log.{}.{}".format(
            os.environ["SLURM_JOB_ID"], time_str))
    accpath = logpath + "_accuracy.txt"
    loglink = open(logpath, "w")
    # header = ["Epoch", "Training.Loss", "MiniBatch.ID", "Time.Stamp"]
    header = ["Epoch", "Reconstruction.Loss", "KLD",
              "CE.Loss", "Accuracy", "MiniBatch.ID",
              "Time.Stamp"]
    loglink.write("\t".join(header) + "\n")
    loglink.close()
    if predict_celltypes:
        acclink = open(accpath, "w")
        header_acc = ["Epoch"]
        for celltype in celltypes:
            header_acc.append(celltype + ".acc")
        acclink.write("\t".join(header_acc) + "\n")
        acclink.close()
    TOTBATCHIDX = int(expar.shape[0] / MINIBATCH)
    # loss_scalers = np.array([300, 1, 1])
    for epoch in range(MAXEPOCH):
        running_loss_reconst = 0
        running_kld = 0
        running_ce = 0
        running_loss = 0
        accval = 0
        celltype_resps = np.zeros(
            (int(TOTBATCHIDX * MINIBATCH)))
        celltype_preds = np.zeros(
            (int(TOTBATCHIDX * MINIBATCH)))
        for idxbatch in range(TOTBATCHIDX):
            idxbatch_st = idxbatch * MINIBATCH
            idxbatch_end = (idxbatch + 1) * MINIBATCH
            train1 = torch.from_numpy(
                expar[idxbatch_st:idxbatch_end, :]).to(device).float()
            local_l_mean = np.mean(
                np.apply_along_axis(
                    np.sum, 1, expar[idxbatch_st:idxbatch_end, :]))
            local_l_var = np.var(
                np.apply_along_axis(
                    np.sum, 1, expar[idxbatch_st:idxbatch_end, :]))
            outdict = vae(train1)
            ct_pred = outdict["ctpred"]
            loss_1, loss_2 = loss_function(
                outdict['qz_m'], outdict['qz_v'], train1,
                outdict['px_rate'], outdict['px_r'],
                outdict['px_dropout'], outdict['ql_m'],
                outdict['ql_v'], True,
                local_l_mean, local_l_var)
            loss_1 = torch.mean(loss_1)
            loss_2 = torch.mean(loss_2)
            optimizer.zero_grad()
            if predict_celltypes:
                one_hot_resp = torch.max(
                    one_hot[idxbatch_st:idxbatch_end], 1)[1].to(device).long()
                one_hot_pred = torch.max(
                    ct_pred, 1)[1]
                celltype_resps[idxbatch_st:idxbatch_end] = \
                    one_hot_resp.detach().cpu().numpy()
                celltype_preds[idxbatch_st:idxbatch_end] = \
                    one_hot_pred.detach().cpu().numpy()
                adacc = accuracy_score(
                    one_hot_resp.detach().cpu().numpy(),
                    one_hot_pred.detach().cpu().numpy())
                accval += adacc
                loss_3 = criterion_class(
                    ct_pred, one_hot_resp)
            else:
                loss_3 = 0
            if idxbatch == 0:
                print(loss_1, loss_2, loss_3)
            if idxbatch == -1 and epoch % 25 == 0:
                loss_scalers = np.array(
                    [loss_1.detach().cpu().numpy(),
                     loss_2.detach().cpu().numpy(),
                     loss_3.detach().cpu().numpy()])
                if np.min(loss_scalers) < 0:
                    if loss_2 < 0:
                        loss_2 = loss_2 * -1
                    else:
                        raise ValueError("One of the losses are negative")
                    print(loss_1)
                    print(loss_2)
                    print(loss_3)
                loss_scalers = loss_scalers / np.min(loss_scalers)
            loss = (loss_1 / torch.tensor(loss_scalers[0])) + (
                loss_2 / torch.tensor(loss_scalers[1])) + (
                loss_3 / torch.tensor(loss_scalers[2]))
            if idxbatch == 0:
                print(loss)
            if torch.isnan(loss):
                print("Losses: {} {} {}".format(loss_1, loss_2, loss_3))
                raise ValueError("NA occured in loss")
            # print(loss)
            if torch.cuda.is_available():
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            running_loss_reconst += (loss_1 / loss_scalers[0])
            running_kld += (loss_2 / loss_scalers[1])
            running_ce += (loss_3 / loss_scalers[2])
            running_loss += loss
            del train1, outdict
            # del one_hot_temp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cur_loss = running_loss / TOTBATCHIDX
        cur_loss_reconst = running_loss_reconst / TOTBATCHIDX
        cur_kld = running_kld / TOTBATCHIDX
        cur_ce = running_ce / TOTBATCHIDX
        accval = accval / TOTBATCHIDX
        adlist_cts = [str(epoch)]
        for k in range(len(celltypes)):
            pred_cell = celltype_preds == k
            resp_cell = celltype_resps == k
            cur_acc = accuracy_score(
                resp_cell, pred_cell)
            adlist_cts.append(str(round(cur_acc, 3)))
        if predict_celltypes:
            with open(accpath, "a+") as acclink:
                acclink.write("\t".join(adlist_cts) + "\n")

        print("Epoch {}, Loss {} at {}".format(
            epoch, cur_loss.item(), datetime.now()))

        with open(logpath, "a+") as loglink:
            adlist = [str(epoch), str(cur_loss_reconst.item()),
                      str(cur_kld.item()), str(cur_ce.item()),
                      str(round(accval, 3)),
                      str(idxbatch), str(datetime.now())]
            # adlist = [str(epoch), str(cur_loss.item()),
            #           str(idxbatch), str(datetime.now())]
            loglink.write("\t".join(adlist) + "\n")
        if epoch % 10 == 0:
            checkpoint = {
                'model': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if torch.cuda.is_available():
                checkpoint["amp"] = amp.state_dict()
            for eachpath in [modelpath, chkpath]:
                torch.save(checkpoint, eachpath)
    return vae


def make_labels(metapath, expar, barcodes):
    if "S" in str(barcodes.dtype):
        barcodes = np.array(barcodes, dtype="|U64")
    metadf = pd.read_csv(metapath, sep="\t", index_col=0)
    if "CellType" not in metadf.columns:
        if "Site_Primary" in metadf.columns:
            metadf["CellType"] = metadf["Site_Primary"]
            metadf["Barcode"] = metadf.index
    classes = np.unique(metadf["CellType"])
    classes = np.array(
        [each for each in classes if "Not" not in each])
    classes = np.array(
        [each for each in classes if "nan" not in each])
    metadf = metadf[metadf["CellType"].isin(classes)]
    metadf = metadf[metadf["Barcode"].isin(barcodes)]
    new_barcodes, idx_1, idx_2 = np.intersect1d(
        barcodes, np.array(metadf["Barcode"]),
        return_indices=True)
    outar = expar[idx_1, :]
    outdf = metadf.iloc[idx_2, :]
    out_barcodes = np.array(barcodes, dtype="|U64")[idx_1]
    one_hot = pd.get_dummies(outdf["CellType"])
    one_hot_tensor = torch.from_numpy(np.array(one_hot))
    return outar, outdf, out_barcodes, one_hot_tensor


def load_npar(nparpath, genes, outdir, gmtmat,
              metapath):
    if ".npz" in nparpath:
        expar, barcodes, gmtmat, genes = read_npz(
            nparpath, genes, outdir, gmtmat)
        list_temp = make_labels(metapath, expar, barcodes)
    elif ".gct" in nparpath or ".tsv" in nparpath:
        expar, barcodes, gmtmat, genes = read_tsv(
            nparpath, genes, outdir, gmtmat, False)
        from train_multitask_ccle import make_labels as tmp_fnc
        list_temp = tmp_fnc(
            metapath, expar, barcodes)
    elif ".h5" in nparpath:
        expar, barcodes, gmtmat, genes = read_h5(
            nparpath, genes, outdir, gmtmat)
        list_temp = make_labels(metapath, expar, barcodes)
    expar, metadf, barcodes, _ = list_temp
    return expar, metadf, barcodes, genes, gmtmat


def filter_by_var(expar, genes, gmtmat, num_genes):
    vars_genes = np.apply_along_axis(np.var, 0, expar)
    idx_sorted = np.argsort(vars_genes)[::-1]
    newexp = expar[:, idx_sorted[:num_genes]]
    newgenes = genes[idx_sorted[:num_genes]]
    gmtmat_new = gmtmat[idx_sorted[:num_genes], :]
    return newexp, newgenes, gmtmat_new


def intersect_lists(genes_list):
    genes = np.intersect1d(genes_list[0], genes_list[1])
    for i in range(2, len(genes_list)):
        genes = np.intersect1d(genes, genes_list[i])
    return genes


def load_inputs(nparpaths, gmtmat, outdir,
                genes, metapaths, filter_var=False,
                num_genes=2000):
    GMTMAT = gmtmat
    gmtmat_genes = genes
    metadf_list = []
    expar_list = []
    barcodes_list = []
    genes_list = []
    celltypes_list = []
    num_barcodes = 0
    for i in range(len(nparpaths)):
        print("Loading {}".format(nparpaths[i]))
        expar, metadf, barcodes, genes, gmtmat = load_npar(
            nparpaths[i], genes, outdir, gmtmat, metapaths[i])
        expar_list.append(expar)
        barcodes_list.append(barcodes)
        celltypes_list.append(
            np.array(metadf["CellType"], dtype="|U64"))
        addf = pd.DataFrame(
            dict(OriginalBarcode=barcodes, CellType=celltypes_list[-1]))
        addf["Dataset"] = "File.{}.".format(i + 1)
        addf["Barcode"] = addf["Dataset"] + addf["OriginalBarcode"]
        metadf_list.append(addf)
        genes_list.append(genes)
        num_barcodes += len(barcodes)
    metadf = pd.concat(metadf_list)
    metadf.index = metadf["Barcode"]
    if len(genes_list) > 1:
        genes = intersect_lists(genes_list)
    else:
        genes = genes_list[0]
    # Filter gmtmat
    _, idx_1, idx_2 = np.intersect1d(gmtmat_genes, genes, return_indices=True)
    # gmtmat = gmtmat[idx_1, :]
    gmtmat = GMTMAT[idx_1, :]
    npar = np.zeros((num_barcodes, len(genes)), dtype=int)
    i_st = 0
    i_end = 0
    for k in range(len(expar_list)):
        cur_genes = genes_list[k]
        expar = expar_list[k]
        shared_genes, idx_1, idx_2 = np.intersect1d(
            genes, cur_genes, return_indices=True)
        i_end = i_st + expar.shape[0]
        npar[i_st:i_end, idx_1] = expar[:, idx_2]
        i_st = i_end
    if filter_var:
        print("Filtering by variance")
        npar, genes, gmtmat = filter_by_var(
            npar, genes, gmtmat, num_genes)
    one_hot = pd.get_dummies(metadf["CellType"])
    one_hot_tensor = torch.from_numpy(np.array(one_hot))
    out_dict = dict(
        expar=npar,
        metadf=metadf,
        barcodes=np.array(metadf["Barcode"]),
        genes=genes,
        gmtmat=gmtmat,
        cellTypes=np.array(celltypes_list),
        one_hot=one_hot_tensor)
    return out_dict


def main(gmtpath, nparpaths, outdir, numlvs, metapaths,
         dont_train=False, genepath="NA", existingmodelpath="NA",
         use_connections=True, loss_scalers=[1, 1, 1],
         predict_celltypes=True, num_celltypes=59, filter_var=False,
         num_genes=2000):
    MINIBATCH = 32
    MAXEPOCH = 20
    gmtmat, tfs, genes = make_gmtmat(gmtpath, outdir, genepath)
    # expar, barcodes = read_h5(h5path, genes, outdir)
    dict_inputs = load_inputs(
        nparpaths, gmtmat, outdir, genes, metapaths, filter_var,
        num_genes)
    expar = dict_inputs["expar"]
    metadf = dict_inputs["metadf"]
    gmtmat = dict_inputs["gmtmat"]
    one_hot = dict_inputs["one_hot"]
    barcodes = dict_inputs["barcodes"]
    # celltypes = dict_inputs["cellTypes"]
    celltypes = []
    if predict_celltypes:
        celltypes = list(pd.unique(metadf["CellType"]))
        celltypes.sort()
    # save metadf
    metadf.to_csv(
        os.path.join(outdir, "metadata.tsv.gz"),
        sep="\t", compression="gzip")
    # Save genes
    print("Shape of expar is : {}".format(expar.shape))
    save_genes(genes, outdir)
    print("Max in expar is {}".format(np.max(expar)))
    if use_connections:
        gmttensor = torch.from_numpy(
            np.transpose(gmtmat)).to(device).long()
    else:
        gmttensor = torch.ones(
            gmtmat.shape[1], gmtmat.shape[0]).to(device).long()
    print("Shape of expar is : {}".format(expar.shape))
    logdir, modelpath, chkpath = get_paths(outdir, numlvs)
    if existingmodelpath == "NA":
        existingmodelpath = modelpath
    vae = VAE(expar.shape[1],  # num genes
              gmttensor,
              num_celltypes,
              0,  # batch
              0,  # labels
              gmtmat.shape[1],  # hiddensize
              numlvs)
    n_params = get_n_params(vae)
    print(vae)
    print("VAE has {} parameters".format(n_params))
    vae.to(device)
    # optimizer = adabound.AdaBound(
    #     vae.parameters(), lr=0.001, final_lr=0.1)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=0.002)
    if torch.cuda.is_available():
        vae, optimizer = amp.initialize(
            vae, optimizer, opt_level=opt_level)
    vae, optimizer = load_existing_model(
        existingmodelpath, chkpath, vae, optimizer)
    if not dont_train:
        np.random.seed(42)
        # For 10 times, sample 1000 cells
        for i in range(20):
            # idx_rand = np.random.choice(
            #     np.arange(expar.shape[0]), SAMPLE_IDXS)
            vae = train_model(
                vae, optimizer, MINIBATCH, MAXEPOCH,
                expar, logdir,
                modelpath, chkpath, one_hot,
                loss_scalers, predict_celltypes,
                celltypes)
            reconst, mumat, sd2mat, tf_act = apply_model(
                vae, expar, numlvs, MINIBATCH)
            mudf = pd.DataFrame(mumat)
            mudf.columns = ["LV.mu.{}".format(each)
                            for each in range(numlvs)]
            mudf["Index"] = np.array(
                barcodes, dtype="|U64")
            mudf.index = mudf["Index"]
            mudf.to_csv(
                os.path.join(outdir, "VAE_mu-matrix.tsv.gz"),
                compression="gzip", sep="\t")
            make_plot_umap(mudf, metadf, outdir, numlvs)
    reconst, mumat, sd2mat, tf_act = apply_model(
        vae, expar, numlvs, MINIBATCH)
    tf_act_df = pd.DataFrame(tf_act)
    tf_act_df.index = np.array(
        barcodes, dtype="|U64")
    tf_act_df.columns = tfs
    tf_act_df["Labels"] = metadf.loc[tf_act_df.index]["CellType"]
    tf_act_df.to_csv(
        os.path.join(outdir, "VAE-TF-adjusted-weights_CellxTF.tsv.gz"),
        sep="\t", compression="gzip")
    # zmat = np_reparameterize(mumat, sd2mat)
    zmat = torch_reparameterize(mumat, sd2mat)
    zdf = pd.DataFrame(zmat)
    zdf.columns = ["LV.Z.{}".format(each)
                   for each in range(numlvs)]
    zdf["Index"] = np.array(
        barcodes, dtype="|U64")
    zdf.index = np.array(
        barcodes, dtype="|U64")
    zdf.to_csv(
        os.path.join(outdir, "VAE_Z-matrix.tsv.gz"),
        compression="gzip", sep="\t")
    outdir_full = os.path.join(
        outdir, "fullDatasetZPlot")
    os.makedirs(outdir_full, exist_ok=True)
    make_plot_umap(zdf, metadf, outdir_full, numlvs)
    mudf = pd.DataFrame(mumat)
    mudf.columns = ["LV.mu.{}".format(each)
                    for each in range(numlvs)]
    mudf["Index"] = np.array(
        barcodes, dtype="|U64")
    mudf.index = mudf["Index"]
    mudf.to_csv(
        os.path.join(outdir, "VAE_mu-matrix.tsv.gz"),
        compression="gzip", sep="\t")
    outdir_full = os.path.join(
        outdir, "fullDatasetPlot")
    os.makedirs(outdir_full, exist_ok=True)
    make_plot_umap(mudf, metadf, outdir_full, numlvs)
    sd2df = pd.DataFrame(sd2mat)
    sd2df.columns = [
        "LV.logVAR.{}".format(each)
        for each in range(numlvs)]
    sd2df["Index"] = mudf["Index"]
    sd2df.index = mudf["Index"]
    sd2df.to_csv(
        os.path.join(outdir, "VAE_variance-matrix.tsv.gz"),
        compression="gzip", sep="\t")


def np_reparameterize(mu, logvar):
    mu_tensor = torch.from_numpy(mu)
    logvar_tensor = torch.from_numpy(logvar)
    std_tensor = torch.exp(0.5 * logvar_tensor)
    eps_tensor = torch.randn_like(std_tensor)
    ztensor = mu_tensor + eps_tensor * std_tensor
    zmat = ztensor.numpy()
    return zmat


def load_existing_model(modelpath, chkpath, vae, optimizer):
    for eachpath in [modelpath, chkpath]:
        if os.path.exists(eachpath):
            try:
                checkpoint = torch.load(eachpath)
                state_dict = checkpoint['model']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    new_state_dict[k] = v
                vae.load_state_dict(new_state_dict)
                optimizer.load_state_dict(checkpoint['optimizer'])
                if torch.cuda.is_available():
                    amp.load_state_dict(checkpoint['amp'])
                print("Loaded from {}".format(eachpath))
                return vae, optimizer
            except Exception:
                pass
        print("Didn't load from any")
        return vae, optimizer


def save_genes(genes, outdir):
    outpath = os.path.join(outdir, "genes.txt")
    outlink = open(outpath, "w")
    for gene in genes:
        outlink.write(gene + "\n")
    outlink.close()


def torch_reparameterize(mumat, varmat):
    from torch.distributions import Normal
    mu = torch.from_numpy(mumat)
    var = torch.from_numpy(varmat)
    normtensor = Normal(mu, var.sqrt()).rsample()
    zmat = normtensor.detach().numpy()
    return zmat


def get_hidden_layer(vae, train1):
    weight_mat = vae.z_encoder.encoder.fc_layers[0][0].weights
    connections = vae.z_encoder.encoder.fc_layers[0][0].connections
    enforced_weights = torch.mul(
        weight_mat, connections)
    ew_times_x = torch.mm(train1, enforced_weights.detach().t())
    add_bias = vae.z_encoder.encoder.fc_layers[0][0].bias
    ew_times_x = torch.add(ew_times_x, add_bias)
    output = ew_times_x.cpu().detach().numpy()
    return output


def apply_model(vae, expar, numlvs, MINIBATCH):
    conn_dim = vae.z_encoder.encoder.fc_layers[0][0].connections.shape[0]
    reconst = np.zeros(expar.shape)
    mumat = np.zeros((expar.shape[0], numlvs))
    sd2mat = np.zeros((expar.shape[0], numlvs))
    tf_activation = np.zeros((expar.shape[0], conn_dim))
    TOTBATCHIDX = int(expar.shape[0] / MINIBATCH)
    for idxbatch in range(TOTBATCHIDX):
        idxbatch_st = idxbatch * MINIBATCH
        idxbatch_end = (idxbatch + 1) * MINIBATCH
        train1 = torch.from_numpy(
            expar[idxbatch_st:idxbatch_end, :]).to(device).float()
        outdict = vae(train1)
        reconst[idxbatch_st:idxbatch_end, :] = \
            outdict["px_scale"].cpu().detach().numpy()
        mumat[idxbatch_st:idxbatch_end, :] = \
            outdict["qz_m"].cpu().detach().numpy()
        sd2mat[idxbatch_st:idxbatch_end, :] = \
            outdict["qz_v"].cpu().detach().numpy()
        tf_activation[idxbatch_st:idxbatch_end, :] = \
            get_hidden_layer(vae, train1)
        if idxbatch % 100 == 0:
            print("Applied on {}/{}".format(idxbatch, TOTBATCHIDX))
    return reconst, mumat, sd2mat, tf_activation


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train VAE using "
        "mapping of genes to TFs")
    parser.add_argument(
        "gmtpath",
        help="Path to GMT file mapping "
        "genes to TFs")
    parser.add_argument(
        "outdir",
        help="Path to output directory for "
        "saving the model and log files")
    parser.add_argument(
        "--nparpaths",
        nargs="*",
        help="Space-separated paths to scRNA-seq "
        "file npz containing arr, rows, and cols")
    parser.add_argument(
        "--numlvs",
        type=int,
        default=10,
        help="Number of latent variables")
    parser.add_argument(
        "--dont-train",
        action="store_true",
        help="Specify if you want to apply an existing "
        "model which is stored in outdir")
    parser.add_argument(
        "--genepath",
        default="NA",
        help="Path to .txt file containing "
        "one gene per line to limit the list "
        "of genes we use here")
    parser.add_argument(
        "--modelpath",
        default="NA",
        help="Specify if you don't want the "
        "model existing in <outdir>/VAE_<--numlvs>LVS.pt")
    parser.add_argument(
        "--metapaths",
        nargs="*",
        required=True,
        help="Space-separated path to metadata tsv with "
        "a column named as barcode and a "
        "column named as cell type")
    parser.add_argument(
        "--use-connections",
        action="store_true",
        help="If set, will enforce weights that don't "
        "correspong to TF-gene mappings to be zero")
    parser.add_argument(
        "--loss-scalers",
        nargs="*",
        default=[1, 1, 1],
        type=float,
        help="Specify values to divide "
        "MSE, KLD, and CE losses by: example: "
        "--loss-scalers 100 1 1")
    parser.add_argument(
        "--predict-celltypes",
        action="store_true",
        help="Specify --predict-celltypes to "
        "optimize the cell type prediction task as well")
    parser.add_argument(
        "--num-celltypes",
        default=59,
        type=int,
        help="Number of cell types to predict (must match "
        "the column CellType in metadata file)")
    parser.add_argument(
        "--filter-var",
        action="store_true",
        help="If specified, will filter by top 2000 most "
        "variant genes")
    parser.add_argument(
        "--num-genes",
        default=2000,
        type=int,
        help="Number of genes to filter by highest variance")
    args = parser.parse_args()
    print(args)
    modelpath = args.modelpath
    if modelpath == "NA":
        modelpath = os.path.join(
            args.outdir, "VAE_{}LVS.pt".format(args.numlvs))
    main(args.gmtpath, args.nparpaths,
         args.outdir, args.numlvs, args.metapaths,
         args.dont_train, args.genepath, modelpath,
         args.use_connections, args.loss_scalers,
         args.predict_celltypes, args.num_celltypes,
         args.filter_var, args.num_genes)
