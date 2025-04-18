% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/caesar_annotation.R
\name{find.sig.genes}
\alias{find.sig.genes}
\title{Identify Signature Genes for Each Cell Type}
\usage{
find.sig.genes(
  seu,
  distce.assay = "distce",
  ident = NULL,
  expr.prop.cutoff = 0.1,
  assay = NULL,
  genes.use = NULL
)
}
\arguments{
\item{seu}{A Seurat object containing gene expression data.}

\item{distce.assay}{A character string specifying the assay that contains the distance matrix or distance-related data. Default is "distce".}

\item{ident}{A character string specifying the column name in the `meta.data` slot of the Seurat object used to define the identities (clusters or cell groups). If `NULL`, the default identities (`Idents(seu)`) will be used. Default is `NULL`.}

\item{expr.prop.cutoff}{A numeric value specifying the minimum proportion of cells that must express a gene for it to be considered. Default is 0.1.}

\item{assay}{A character string specifying the assay to use for expression data. If `NULL`, the default assay of the Seurat object will be used. Default is `NULL`.}

\item{genes.use}{A character vector specifying the genes to use for the analysis. If `NULL`, all genes in the `distce.assay` assay will be used. Default is `NULL`.}
}
\value{
A list where each element corresponds to a cell group and contains a data frame with the following columns:
\item{distance}{The mean distance of the gene across the cells in the group.}
\item{expr.prop}{The proportion of cells in the group expressing the gene.}
\item{expr.prop.others}{The proportion of cells in other groups expressing the gene.}
\item{label}{The identity label of the cell group.}
\item{gene}{The gene name.}
}
\description{
This function identifies signature genes for each cell type or cell group in a Seurat object using a co-embedding distance-based approach. It computes the average expression and distance metrics for each gene across different groups, while also considering expression proportions.
}
\examples{
data(toydata)

seu <- toydata$seu

seu <- ProFAST::pdistance(seu, reduction = "caesar")
sglist <- find.sig.genes(
    seu = seu
)
str(sglist)

}
\seealso{
None
}
