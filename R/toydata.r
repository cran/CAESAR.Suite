#' A toy dataset to run examples
#'
#' A list containing the following components:
#'
#' @format A list with five components.
#' \itemize{
#'   \item seu: A Seurat object, which is a subset of Xenium breast cancer section.
#'   \item pos: A data frame containing the location information of spots. Each row corresponds to a spatial spot, with columns for x and y coordinates.
#'   \item pathway_list: A list containing two pathways of intersecting. This list provides information about gene sets or pathways that are relevant to the dataset. Each element in the list is a character vector of gene names.
#'   \item markers: A list of marker genes for each cell type. This list maps cell types to their respective marker genes, which can be used for cell type annotation.
#'   \item imgf: A matrix containing histology image features. Each row corresponds to a spatial spot, and each column represents a different feature extracted from the histology image.
#' }
#'
#' @docType data
#' @keywords datasets
#' @name toydata
#' @usage data(toydata)
NULL