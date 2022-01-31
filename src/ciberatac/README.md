# Example training of CiberATAC model on PBMC data


## Prerequisite

Before executing this example, please run the example under the mave folder to generate the single-cell embedding matrix for PBMC cell types.

CiberATAC requires a path ending in VAE_mu-matrix.tsv.gz as the second positional parameter


## Downloading the data

Download the PBMC pseudobulk bigWig files from https://doi.org/10.5281/zenodo.5865863 named as pbmc.zip and unzip the folder.

## Training CiberATAC

```
MAVEPATH=mave_data/VAE_mu-matrix.tsv.gz
RNADIR=pbmc/rna
ATACDIR=pbmc/atac
OUTDIR=pbmc/trainedCiberAtac
mkdir -p $OUTDIR
CELLTYPES=(B-cells Natural_killer CD14+_Mono DC)
RNAPATHS=()
ATACPATHS=()
for CELLTYPE in ${CELLTYPES[@]}
do
    RNAPATHS+=($RNADIR/$CELLTYPE\_rna.bigWig)
    ATACPATHS+=($ATACDIR/$CELLTYPE\_treat_pileup.bigWig)
done
BEDPATH=pbmc/all_cells_peaks.narrowPeak.gz
SEQDIR=pbmc/np # numpy files corresponding to hg38 chromosomes
BULKATAC=$ATACDIR/all_cells_treat_pileup.bigWig
CONV="1.1.1"
POOL=40
DP=0.5
LTYPE=3
LAMBDA=0.01
KERNEL=20
CONVINIT="64,1.25"
POOL=40
LOSSSCALERS=(20 0 1 1)
QUANTILE=0.2
KERNEL=20
DILATIONPAR="1,4,8"
CONVINIT="64,1.25"
CHROMS=(chr1 chr2)
CONVPARAM=$(echo $CONV | tr "." " ")
CONVIN=$(echo $CONVINIT | tr "," "\n" | head -1 )
FILTERRATE=$(echo $CONVINIT | tr "," "\n" | tail -1)
DILATION=($(echo $DILATIONPAR | tr "," "\n"))
python train.py $OUTDIR $MAVEPATH $BEDPATH --rnabwpaths ${RNAPATHS[@]} --dnasebwpaths ${ATACPATHS[@]} --bulkdnasepath $BULKATAC --batchsize 6 --seqdir $SEQDIR --regression --regularize --lambda-param $LAMBDA --ltype $LTYPE --dropout $DP --kernel-size $KERNEL --pool-dim $POOL --dilations ${DILATION[@]} --initconv $CONVIN --filter-rate $FILTERRATE --convparam ${CONVPARAM[@]} --chroms ${CHROMS[@]} --scalers 1 1 --scale-operation identity --loss-scalers ${LOSSSCALERS[@]} --resp-thresh $QUANTILE --optimizer Adam --num-contrastive-regions 400 400 --num-chroms-per-batch 1
```

## Applying the predictions on held-out cell types

```
MAVEPATH=mave_data/VAE_mu-matrix.tsv.gz
RNADIR=pbmc/rna
ATACDIR=pbmc/dnase
OUTDIR=pbmc/ciberAtacPredictions
mkdir -p $OUTDIR
CELLTYPES=(B-cells CD14+_Mono CD8+_T DC Memory_CD4+ Naive_CD4+_T Natural_killer)
OUTPATHS=()
RNAPATHS=()
ATACPATHS=()
for CELLTYPE in ${CELLTYPES[@]}
do
    OUTPATHS+=($OUTDIR/ciberAtacPredictions_$CELLTYPE.tsv.gz)
    RNAPATHS+=($RNADIR/$CELLTYPE\_rna.bigWig)
    ATACPATHS+=($ATACDIR/$CELLTYPE\_treat_pileup.bigWig)
done
BEDPATH=pbmc/all_cells_peaks.narrowPeak.gz
SEQDIR=pbmc/hg38_np
BULKATAC=$ATACDIR/all_cells_treat_pileup.bigWig
MODELID=$(ls pbmc/trainedCiberAtac/modelLog | grep bestRmodel.pt | head -1)
MODELPATH=pbmc/trainedCiberAtac/modelLog/$MODELID
BATCHSIZE=24
for i in {0..6}
do
    OUTPATH=${OUTPATHS[$i]}
    RNAPATH=${RNAPATHS[$i]}
    SAMPLENAME=${CELLTYPES[$i]}
    python predict.py $OUTPATH $SCVIPATH $RNAPATH $ATAC $BEDPATH --modelpath $MODELPATH --batchsize $BATCHSIZE --seqdir $SEQDIR --scvi-name ${CELLTYPES[$i]} --annotate-bwpath ${ATACPATHS[$i]} --valid-chroms --scalers 1 1
done
```
