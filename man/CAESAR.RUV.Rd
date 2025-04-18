% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ruv.R
\name{CAESAR.RUV}
\alias{CAESAR.RUV}
\title{Perform Batch Correction and Integration with CAESAR Using Housekeeping Genes}
\usage{
CAESAR.RUV(
  seuList,
  distList,
  verbose = FALSE,
  species = "human",
  custom_housekeep = NULL
)
}
\arguments{
\item{seuList}{A list of Seurat objects to be integrated.}

\item{distList}{A list of distance matrices corresponding to each Seurat object in `seuList`.}

\item{verbose}{Logical, indicating whether to display progress messages. Default is \code{FALSE}.}

\item{species}{A character string specifying the species, either "human" or "mouse". Default is "human".}

\item{custom_housekeep}{A character vector of custom housekeeping genes. If \code{NULL}, default housekeeping genes for the species are used. Default is \code{NULL}.}
}
\value{
A Seurat object that contains the integrated and batch-corrected data in a new assay called "CAESAR".
}
\description{
This function performs batch correction and integration of multiple Seurat objects using housekeeping genes and distance matrices. It supports human and mouse data, and can optionally use custom housekeeping genes provided by the user.
}
\examples{
data(toydata)

seu <- toydata$seu
markers <- toydata$markers

seu <- ProFAST::pdistance(seu, reduction = "caesar")

marker.freq <- markerList2mat(list(markers))
anno_res <- CAESAR.annotation(seu, marker.freq, cal.confidence = FALSE, cal.proportions = FALSE)

seuList <- list(seu, seu)
distList <- list(anno_res$ave.dist, anno_res$ave.dist)
seuInt <- CAESAR.RUV(seuList, distList, species = "human", verbose = TRUE)

}
