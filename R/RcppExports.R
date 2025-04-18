# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' @keywords internal
#' @noRd
#'
NULL

gene_embed_weight_cpp <- function(X, ce_cell, adj, c = 1.0) {
    .Call(`_CAESAR_Suite_gene_embed_weight_cpp`, X, ce_cell, adj, c)
}

wpcaCpp <- function(X, nPCs, weighted = TRUE) {
    .Call(`_CAESAR_Suite_wpcaCpp`, X, nPCs, weighted)
}

#' @title
#' getneighborhood_fast
#' @description
#' an efficient function to find the neighborhood based on the matrix of position and a pre-defined cutoff
#'
#' @param x is a n-by-2 matrix of position.
#' @param radius is a threashold of Euclidean distance to decide whether a spot is an neighborhood of another spot. For example, if the Euclidean distance between spot A and B is less than cutoff, then A is taken as the neighbourhood of B.
#' @return A sparse matrix containing the neighbourhood
#'
#' @export
getneighborhood_fastcpp <- function(x, radius) {
    .Call(`_CAESAR_Suite_getneighborhood_fastcpp`, x, radius)
}

pdistance_cpp <- function(Ar, Br, eta = 1e-10) {
    .Call(`_CAESAR_Suite_pdistance_cpp`, Ar, Br, eta)
}

gene_embed_cpp <- function(X, ce_cell) {
    .Call(`_CAESAR_Suite_gene_embed_cpp`, X, ce_cell)
}

weightAdj <- function(pos, img_embed, radius, width) {
    .Call(`_CAESAR_Suite_weightAdj`, pos, img_embed, radius, width)
}

getneighbor_weightmat <- function(x, radius, width) {
    .Call(`_CAESAR_Suite_getneighbor_weightmat`, x, radius, width)
}

#' @keywords internal
#' @noRd
#' 
NULL

imFactorCpp <- function(X, weiAdj, w_plus, mu_int, B_int, Lam_int, Phi_int, M_int, R_int, maxIter, epsELBO, verbose, Phi_diag = TRUE) {
    .Call(`_CAESAR_Suite_imFactorCpp`, X, weiAdj, w_plus, mu_int, B_int, Lam_int, Phi_int, M_int, R_int, maxIter, epsELBO, verbose, Phi_diag)
}

approxPhi_imFactorCpp <- function(X, weiAdj, mu_int, B_int, Lam_int, Phi_int, M_int, R_int, maxIter, epsELBO, verbose, Phi_diag = TRUE) {
    .Call(`_CAESAR_Suite_approxPhi_imFactorCpp`, X, weiAdj, mu_int, B_int, Lam_int, Phi_int, M_int, R_int, maxIter, epsELBO, verbose, Phi_diag)
}

