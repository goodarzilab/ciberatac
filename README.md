# CiberATAC

Widely used single-cell transcriptomic assays identify dysregulated genes, but fail to identify the
cis-regulatory elements regulating those genes. Our method, Cis-regulatory element Identi cation
By Expressed RNA and ATAC-seq, CiberATAC, identi es active cell-type{speci c cis-regulatory
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
squared error, and the contrastive loss we developed called the pairwise mean-squared error. We
applied CiberATAC on an in-house generated dataset of in vivo-selected sequential metastasis in
colorectal cancer and identi ed the role of HMG20B and its regulon. CiberATAC allows integrating
scRNA-seq datasets with existing bulk ATAC-seq datasets to identify the epigenetic features of rare
cell types with signi cant biological interest.
