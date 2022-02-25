# Example training on PBMC datasets

## Downloading the data

You need to download the zipped folder names mave_data.zip from https://doi.org/10.5281/zenodo.5865863

## Running the script


```
H5PATH=mave_data/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5
METAPATH=mave_data/scRNA-seq_10XPBMC_metadataWithCellType.tsv
GENEPATH=mave_data/Genes_passing_40p.txt
GMTPATH=mave_data/c3.tft.v7.2.symbols.gmt
OUTDIR=mave_data/mave_training
FILEPATHS=($H5PATH $METAPATH $GENEPATH $GMTPATH)
for each in ${FILEPATHS[@]}
do
    if [ ! -f $each ]
    then
        echo "$each did not exist"
    fi
done
mkdir -p $OUTDIR
python train.py $GMTPATH $OUTDIR --nparpaths $H5PATH --numlvs 10 --genepath $GENEPATH --metapaths $METAPATH --num-celltypes 8 --predict-celltypes --use-connections --loss-scalers 1000 1 1
```

## Example with data from different batches


```
MAVEDIR=mave_dir
H5PATHS=($MAVEDIR/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5 $MAVEDIR/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5)
METAPATHS=($MAVEDIR/scRNA-seq_10XPBMC_metadataWithCellType.tsv $MAVEDIR/scRNA-seq_10XLymphoma_metadataWithCellType.tsv)
GENEPATH=$MAVEDIR/Genes_passing_40p.txt
GMTPATH=$MAVEDIR/c3.tft.v7.2.symbols.gmt
OUTDIR=$MAVEDIR/mave_training_pbmc_lymphoma/no_batch_correction
mkdir -p $OUTDIR

python train.py $GMTPATH $OUTDIR --nparpaths ${H5PATHS[@]} --numlvs 10 --genepath $GENEPATH --metapaths ${METAPATHS[@]} --num-celltypes 12 --predict-celltypes --use-connections --loss-scalers 1000 1 1


OUTDIR=$MAVEDIR/mave_training_pbmc_lymphoma/with_batch_correction
mkdir -p $OUTDIR

python train.py $GMTPATH $OUTDIR --nparpaths ${H5PATHS[@]} --numlvs 10 --genepath $GENEPATH --metapaths ${METAPATHS[@]} --num-celltypes 12 --predict-celltypes --use-connections --loss-scalers 1000 1 1 --include-batches
```
