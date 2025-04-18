% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/caesar_image.R
\name{cellembedding_image_matrix}
\alias{cellembedding_image_matrix}
\title{Compute Spatial-Aware Cell Embeddings with Image Information}
\usage{
cellembedding_image_matrix(
  X,
  adjm,
  q = 50,
  reduction.name = "caesar",
  maxIter = 30,
  epsELBO = 1e-06,
  approx_Phi = FALSE,
  verbose = TRUE,
  Phi_diag = TRUE,
  seed = 1
)
}
\arguments{
\item{X}{A gene-by-cell matrix (e.g., the `data` slot from a Seurat object) that serves as the input data for dimensional reduction.}

\item{adjm}{A spatial adjacency matrix representing relationships between cells or spots.}

\item{q}{An integer specifying the number of dimensions for the reduced embeddings. Default is 50.}

\item{reduction.name}{A character string specifying the name of the dimensional reduction method. Default is 'caesar'.}

\item{maxIter}{Maximum number of iterations for the optimization algorithm. Default is 30.}

\item{epsELBO}{A small number specifying the convergence threshold for the optimization algorithm. Default is 1e-6.}

\item{approx_Phi}{Logical, indicating whether to use the approximate method for Phi matrix estimation. Default is FALSE.}

\item{verbose}{Logical, indicating whether to print progress messages. Default is TRUE.}

\item{Phi_diag}{Logical, indicating whether to constrain the Phi matrix to be diagonal. Default is TRUE.}

\item{seed}{An integer used to set the random seed for reproducibility. Default is 1.}
}
\value{
A matrix containing the computed cell embeddings. The number of rows corresponds to the number of cells, and the number of columns corresponds to the specified number of dimensions (`q`).
}
\description{
This function computes low-dimensional cell embeddings from a gene-by-cell matrix. The method initializes cell embeddings using approximate PCA and refines them through a linear factor model nested a intrinsical conditional autoregressive model.
}
