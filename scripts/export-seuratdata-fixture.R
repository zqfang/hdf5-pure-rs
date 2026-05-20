#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[[1]] else "pbmc3k"
output <- if (length(args) >= 2) args[[2]] else ".tmp/seurat_bench/pbmc3k_source.h5"

if (!requireNamespace("SeuratData", quietly = TRUE)) {
  stop("SeuratData is required")
}
if (!requireNamespace("SeuratObject", quietly = TRUE)) {
  stop("SeuratObject is required")
}
if (!requireNamespace("hdf5r", quietly = TRUE)) {
  stop("hdf5r is required")
}

options(SeuratData.manifest.cache = FALSE)

installed <- rownames(installed.packages())
package <- paste0(dataset, ".SeuratData")
if (!package %in% installed) {
  SeuratData::InstallData(dataset)
}

object <- SeuratData::LoadData(dataset)
assay <- SeuratObject::DefaultAssay(object)

counts <- tryCatch(
  SeuratObject::GetAssayData(object, assay = assay, layer = "counts"),
  error = function(...) SeuratObject::GetAssayData(object, assay = assay, slot = "counts")
)
counts <- methods::as(counts, "dgCMatrix")

parent <- dirname(output)
if (!dir.exists(parent)) {
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
}
if (file.exists(output)) {
  unlink(output)
}

write_string_dataset <- function(group, name, values) {
  dtype <- hdf5r::h5types$H5T_STRING$new(size = Inf)
  group$create_dataset(name, robj = as.character(values), dtype = dtype)
}

file <- hdf5r::H5File$new(output, mode = "w")
on.exit(file$close_all(), add = TRUE)

file[["assay"]] <- assay
file[["dataset"]] <- dataset

rna <- file$create_group("rna")
rna[["data"]] <- as.numeric(counts@x)
rna[["indices"]] <- as.integer(counts@i)
rna[["indptr"]] <- as.integer(counts@p)
rna[["shape"]] <- as.integer(dim(counts))
write_string_dataset(rna, "obs_names", colnames(counts))
write_string_dataset(rna, "var_names", rownames(counts))

meta <- object[[]]
obs <- file$create_group("obs")
for (name in colnames(meta)) {
  value <- meta[[name]]
  if (is.numeric(value) || is.integer(value) || is.logical(value)) {
    obs[[name]] <- as.numeric(value)
  } else {
    write_string_dataset(obs, name, as.character(value))
  }
}

cat("dataset=", dataset, "\n", sep = "")
cat("assay=", assay, "\n", sep = "")
cat("output=", output, "\n", sep = "")
cat("genes=", nrow(counts), " cells=", ncol(counts), " nnz=", length(counts@x), "\n", sep = "")
