#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

usage <- function() {
  cat(
    "Usage: scripts/export-bmcite-mudataseurat.R [output.h5mu] [--dataset bmcite] [--no-install] [--no-repair-var-index]\n",
    "\n",
    "Exports a SeuratData dataset through MuDataSeurat::WriteH5MU().\n",
    "Default output: tests/data/real_world/bmcite_mudataseurat.h5mu\n",
    "\n",
    "The repair step makes the global /var/_index unique by prefixing duplicate\n",
    "modality feature names, which avoids MuDataSeurat::ReadH5MU() failing on\n",
    "bmcite RNA/ADT feature-name collisions such as CD14.\n",
    sep = ""
  )
}

dataset <- "bmcite"
output <- "tests/data/real_world/bmcite_mudataseurat.h5mu"
install_missing <- TRUE
repair_var_index <- TRUE

i <- 1
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg == "--help" || arg == "-h") {
    usage()
    quit(status = 0)
  } else if (arg == "--dataset") {
    i <- i + 1
    if (i > length(args)) stop("--dataset requires a value")
    dataset <- args[[i]]
  } else if (arg == "--no-install") {
    install_missing <- FALSE
  } else if (arg == "--no-repair-var-index") {
    repair_var_index <- FALSE
  } else if (startsWith(arg, "--")) {
    stop("unknown option: ", arg)
  } else {
    output <- arg
  }
  i <- i + 1
}

required <- c("Seurat", "SeuratObject", "SeuratData", "MuDataSeurat", "hdf5r")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("missing required R package(s): ", paste(missing, collapse = ", "))
}

options(SeuratData.manifest.cache = FALSE)

data_package <- paste0(dataset, ".SeuratData")
if (!data_package %in% rownames(installed.packages())) {
  if (!install_missing) {
    stop(data_package, " is not installed; rerun without --no-install to call SeuratData::InstallData(\"", dataset, "\")")
  }
  SeuratData::InstallData(dataset)
}

object <- SeuratData::LoadData(dataset)
parent <- dirname(output)
if (!dir.exists(parent)) {
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
}

MuDataSeurat::WriteH5MU(object, output, overwrite = TRUE)

repair_global_var_index <- function(file) {
  h5 <- hdf5r::H5File$new(file, mode = "r+")
  on.exit(h5$close_all(), add = TRUE)

  mods <- h5[["mod"]]$names
  values <- unlist(lapply(mods, function(mod) {
    idx <- h5[["mod"]][[mod]][["var"]][["_index"]]$read()
    ifelse(duplicated(idx) | duplicated(idx, fromLast = TRUE), paste(mod, idx, sep = ":"), idx)
  }), use.names = FALSE)

  var_index <- h5[["var"]][["_index"]]
  if (length(values) != length(var_index$read())) {
    stop("computed /var/_index length does not match existing dataset")
  }
  var_index[] <- values
  invisible(values)
}

if (repair_var_index) {
  repair_global_var_index(output)
}

roundtrip <- MuDataSeurat::ReadH5MU(output)

cat("dataset=", dataset, "\n", sep = "")
cat("output=", output, "\n", sep = "")
cat("modalities=", paste(Seurat::Assays(roundtrip), collapse = ","), "\n", sep = "")
cat("cells=", ncol(roundtrip), "\n", sep = "")
cat("features_default_assay=", nrow(roundtrip), "\n", sep = "")
cat("size_bytes=", file.info(output)$size, "\n", sep = "")
