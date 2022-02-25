# CiberATAC

## Method rationale

Widely used single-cell transcriptomic assays identify dysregulated genes, but fail to identify the
cis-regulatory elements regulating those genes. Our method, Cis-regulatory element Identifcation
By Expressed RNA and ATAC-seq, CiberATAC, identifies active cell-type--specific cis-regulatory
elements in scRNA-seq data and ATAC-seq data (bulk or single-cell). Unlike other deconvolution
methods, CiberATAC does not require access to a deconvolution reference, making it applicable to
rare and new cell types. CiberATAC adopts a novel contrastive learning algorithm by using a siamese
residual convolutional neural network to model bulk or single-cell chromatin accessibility as well as
cell-type{speci c transcription within 20 kbp (+/- 10 kbp) of each cis-regulatory element to predict its
cell-type{speci c activity. We developed a semi-supervised variational auto-encodoer (VAE) inspired
by biological connections, as opposed to fully connected layers, to represent global transcriptome
identities as one of the inputs to the CiberATAC model. We inferred the biological connections
according to the genes each transcription factor regulates (as annotated by the molecular signature
database). CiberATAC deconvolved chromatin accessibility signal of held-out chromosomes and
held-out cell types by outperforming the bulk pro le as measured by Pearson correlation, mean
squared error, and the contrastive loss we developed called the pairwise mean-squared error.


## Method inputs and outputs

### scRNA-seq and bulk or single-cell ATAC-seq

A common scenario for running CiberATAC is when you identify a cell type of interest in your scRNA-seq experiment, and would like to infer chromatin accessibility in specific genomic regions in that group of interest.

For example, you can have scRNA-seq from PBMC, and bulk ATAC-seq for all of PBMC and a few resolved populations, such as B-cells and monocytes, but not all subpopulations (while you have ATAC-seq for all PBMC, monocytes, and B-cells, you don't have ATAC-seq resolved for specific subtypes of T-cells).

In this scenario, you can train CiberATAC's auto-encoder, MAVE, on the scRNA-seq data to learn embeddings of all individual cells.
Once MAVE is trained, you can train CiberATAC with these files:

* A bigWig file of average ATAC-seq signal from all PBMC cells

* Multiple bigWig files of pseudbulk RNA-seq signal from the resolved populations (e.g. one for B-cells, one monocytes, etc.)

* For response, you need bigWig files corresponding to each of the pseudobulk RNA-seq in the ATAC-seq space.

* The `VAE_mu-matrix.tsv.gz` from the trained MAVE


At each instance, CiberATAC sees three inputs:

* A bigWig file of average ATAC-seq signal from all PBMC cells (a 20,000 bp window of signal centered at a peak is chosen)

* A bigWig file of scRNA-seq for a specific population (the same 20,000 bp window as ATAC-seq)

* A single frow from `VAE_mu-matrix.tsv.gz` corresponding to one of the cells that the bigWig scRNA-seq is obtained from

In this instance, the model will try to predict the middle 200 bp ATAC-seq signal specific to the known cell type.

The model optimizes three different losses, two of which rely upon the predictions of the model for the same genomic region but for different cell identities:

* L1 smooth loss for regression task of predicting ATAC-seq signal in the cell type of interest

* A pairwise mean squared error loss (see paper for more details)

* A cross entropy loss for identifying the cell type with the most active chromatin at that genomic region (see paper for more details)


## Using CiberATAC with PBMC data

We have provided scripts under README.md folders under mave and ciberatac subfolders.

For ciberatac training, the provided script will take about 3 minutes on 1 GPU for each epoch.
You can choose a smaller number of regions to speed up: change `--num-contrastive-regions 400 400` to `--num-contrastive-regions 100 100`

You can download the necessary data from https://doi.org/10.5281/zenodo.5865863

Note that the larger the number of regions for training, the more generalizable the predictions to all of the chromosome.


# Maintenance

Mehran karimzadeh developed this package during his post-doc under supervision of Hani Goodarzi (UCSF) and Bo Wang (UHN and Vector Institute).

