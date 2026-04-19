#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(anndataR)
  library(Seurat)
  library(monocle3)
  library(igraph)
  library(ggplot2)
})

parse_args <- function() {
  script_path <- normalizePath(commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))], mustWork = FALSE)
  script_dir <- dirname(sub("^--file=", "", script_path[1]))
  figure_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
  project_root <- normalizePath(file.path(figure_root, ".."), mustWork = TRUE)

  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    input_h5ad = file.path(project_root, "artifacts", "clock_artifacts", "musc_annotation", "musc_atlas_annotated.h5ad"),
    outdir = file.path(project_root, "artifacts", "processed_adata", "monocle3"),
    cell_type_col = "annotation",
    cluster_resolution = 0.5,
    use_partition = "TRUE",
    rann_k = NA_character_,
    minimal_branch_len = NA_character_
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key))
    }
    if (i == length(args)) {
      stop(sprintf("Missing value for %s", key))
    }
    val <- args[[i + 1]]
    nm <- sub("^--", "", key)
    if (!nm %in% names(out)) {
      stop(sprintf("Unknown flag: %s", key))
    }
    out[[nm]] <- val
    i <- i + 2
  }

  out$cluster_resolution <- as.numeric(out$cluster_resolution)
  out$use_partition <- toupper(out$use_partition) %in% c("TRUE", "T", "1", "YES", "Y")
  out$rann_k <- if (is.na(out$rann_k) || out$rann_k == "NA") NA_integer_ else as.integer(out$rann_k)
  out$minimal_branch_len <- if (is.na(out$minimal_branch_len) || out$minimal_branch_len == "NA") NA_integer_ else as.integer(out$minimal_branch_len)
  out
}

pick_root_node <- function(cds, cell_type_col) {
  quiescent_cells <- rownames(colData(cds))[grepl("Quiescent", colData(cds)[[cell_type_col]], ignore.case = TRUE)]
  if (length(quiescent_cells) == 0) {
    stop(sprintf("No quiescent cells found in column '%s'", cell_type_col))
  }

  umap_coords <- reducedDims(cds)$UMAP
  quiescent_coords <- umap_coords[quiescent_cells, , drop = FALSE]
  root_cell_id <- rownames(quiescent_coords)[which.max(quiescent_coords[, 2])]

  closest_vertex <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
  root_node_idx <- as.integer(closest_vertex[root_cell_id, 1])
  root_node <- igraph::V(principal_graph(cds)[["UMAP"]])$name[root_node_idx]

  if (is.na(root_node) || !nzchar(root_node)) {
    stop("Failed to identify a valid Monocle3 root node")
  }

  root_node
}

main <- function() {
  args <- parse_args()
  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

  message(sprintf("[1/6] Reading %s", args$input_h5ad))
  seurat_obj <- read_h5ad(args$input_h5ad, as = "Seurat")

  available_layers <- Layers(seurat_obj)
  message(sprintf("Layers: %s", paste(available_layers, collapse = ", ")))
  if ("counts" %in% available_layers) {
    expression_matrix <- GetAssayData(seurat_obj, layer = "counts")
  } else if ("X" %in% available_layers) {
    message("Layer 'counts' not found. Falling back to layer 'X'.")
    expression_matrix <- GetAssayData(seurat_obj, layer = "X")
  } else if ("data" %in% available_layers) {
    message("Layer 'counts' not found. Falling back to layer 'data'.")
    expression_matrix <- GetAssayData(seurat_obj, layer = "data")
  } else {
    stop("No usable expression layer found. Checked: counts, X, data")
  }

  if (any(expression_matrix < 0)) {
    message("Expression matrix contains negative values; truncating them to zero for Monocle3.")
    expression_matrix[expression_matrix < 0] <- 0
  }

  cell_metadata <- seurat_obj@meta.data
  if (!args$cell_type_col %in% colnames(cell_metadata)) {
    stop(sprintf("Cell metadata column '%s' not found", args$cell_type_col))
  }
  gene_annotation <- data.frame(gene_short_name = rownames(seurat_obj), row.names = rownames(seurat_obj))

  message("[2/6] Building CDS")
  cds <- new_cell_data_set(
    expression_matrix,
    cell_metadata = cell_metadata,
    gene_metadata = gene_annotation
  )

  reductions <- names(seurat_obj@reductions)
  message(sprintf("Reductions: %s", paste(reductions, collapse = ", ")))
  if ("X_scVI" %in% reductions) {
    scvi_coords <- Embeddings(seurat_obj, reduction = "X_scVI")
    scvi_coords <- scvi_coords[colnames(cds), , drop = FALSE]
    reducedDims(cds)$PCA <- scvi_coords
  } else {
    stop("Reduction 'X_scVI' not found in imported Seurat object")
  }

  if ("X_umap" %in% reductions) {
    umap_coords <- Embeddings(seurat_obj, reduction = "X_umap")
    umap_coords <- umap_coords[colnames(cds), , drop = FALSE]
    reducedDims(cds)$UMAP <- umap_coords
  } else {
    stop("Reduction 'X_umap' not found in imported Seurat object")
  }

  message("[3/6] Plotting imported UMAP")
  p_celltype <- plot_cells(
    cds,
    color_cells_by = args$cell_type_col,
    label_cell_groups = TRUE,
    group_label_size = 5,
    show_trajectory_graph = FALSE,
    cell_size = 0.8
  ) + ggtitle("MuSC cell states")
  ggsave(file.path(args$outdir, "musc_umap_celltype.png"), plot = p_celltype, width = 10, height = 8, dpi = 300)

  message(sprintf("[4/6] Clustering cells on imported UMAP (resolution=%.2f)", args$cluster_resolution))
  cluster_timing <- system.time({
    cds <- cluster_cells(cds, reduction_method = "UMAP", resolution = args$cluster_resolution)
  })
  message(sprintf(
    "cluster_cells() finished in %.1fs with %d partitions",
    unname(cluster_timing[["elapsed"]]),
    length(unique(partitions(cds)))
  ))

  message(sprintf("[4.5/6] Learning graph (use_partition=%s)", if (args$use_partition) "TRUE" else "FALSE"))
  
  # Initialize with a fixed ncenter to prevent the SimplePPT deadlock on imported embeddings
  graph_control <- list(ncenter = 400) 
  
  if (!is.na(args$rann_k)) {
    graph_control$rann.k <- args$rann_k
  }
  if (!is.na(args$minimal_branch_len)) {
    graph_control$minimal_branch_len <- args$minimal_branch_len
  }
  
  message(sprintf(
    "Using graph controls: %s",
    paste(sprintf("%s=%s", names(graph_control), unlist(graph_control)), collapse = ", ")
  ))
  
  graph_timing <- system.time({
    cds <- learn_graph(
      cds,
      use_partition = args$use_partition,
      close_loop = FALSE,
      learn_graph_control = graph_control,
      verbose = TRUE  # <--- This will print the actual C++ progress!
    )
  })
  message(sprintf("learn_graph() finished in %.1fs", unname(graph_timing[["elapsed"]])))

  root_node <- pick_root_node(cds, args$cell_type_col)
  message(sprintf("[5/6] Ordering cells from root node %s", root_node))
  cds <- order_cells(cds, root_pr_nodes = root_node)

  p_partition <- plot_cells(
    cds,
    color_cells_by = "partition",
    label_cell_groups = FALSE,
    show_trajectory_graph = FALSE,
    cell_size = 0.8
  ) + ggtitle("Monocle3 partitions")
  ggsave(file.path(args$outdir, "musc_umap_partition.png"), plot = p_partition, width = 10, height = 8, dpi = 300)

  p_pseudotime <- plot_cells(
    cds,
    color_cells_by = "pseudotime",
    label_cell_groups = FALSE,
    label_leaves = FALSE,
    label_branch_points = FALSE,
    cell_size = 0.8
  ) + ggtitle("MuSC pseudotime")
  ggsave(file.path(args$outdir, "musc_umap_pseudotime.png"), plot = p_pseudotime, width = 10, height = 8, dpi = 300)

  closest_vertex <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
  closest_vertex <- as.matrix(closest_vertex[colnames(cds), , drop = FALSE])
  output_cells <- data.frame(
    cell_id = colnames(cds),
    annotation = as.character(colData(cds)[[args$cell_type_col]]),
    pseudotime = as.numeric(pseudotime(cds)[colnames(cds)]),
    partition = as.character(partitions(cds))[colnames(cds)],
    principal_graph_vertex = as.character(closest_vertex[, 1]),
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  message("[6/6] Writing outputs")
  write.table(
    output_cells,
    file = file.path(args$outdir, "musc_monocle3_cells.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  saveRDS(cds, file = file.path(args$outdir, "musc_monocle3_cds.rds"))
}

main()
