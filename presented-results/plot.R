library(ggplot2)
library(cowplot)
library(sitools)
library(viridis)
library(dplyr)
library(tidyr)


W <- 5.126
H <- 2
S <- 1
point_size <- 0.8
line_size <- 0.5
linecolors <- scale_color_brewer(palette = "Set1")
theme <- theme_cowplot(font_size = 7)

sisec <- Vectorize(
  function(t) {
    if (is.na(t)) {
      NA
    } else {
      sitools::f2si(t / 10^6, "s")
    }
  }
)

domain_label <- function(x) parse(text = paste0(x, "^3"))

# iterations * substrates * visited voxels
ops <- function(n, s) 100 * s * n**3

{
  data <- read.csv("gpp/baselines.csv", header = TRUE, sep = ",")
  
  data = filter(data, std_dev / time < 0.05)

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()
  data <- data.frame(data)


  data$precision[data$precision == "D"] <- "Double precision"
  data$precision[data$precision == "S"] <- "Single precision"

  data2 <- read.csv("gpp/biofvms.csv", header = TRUE, sep = ",")
  
  data2 = filter(data2, std_dev / time < 0.05)

  data2 <- data2 %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()
  data2 <- data.frame(data2)
  
  data2$precision[data2$precision == "D"] <- "Double precision"
  data2$precision[data2$precision == "S"] <- "Single precision"

  data3 <- c()
  data3 = rbind(data, data2)
  data3 = filter(data3, s == 1 & nx %in% c(50, 100, 150, 200, 250, 300, 400, 450, 500, 550, 600))

  data3$algorithm[data3$algorithm == "lstcss"] <- "temp-local (baseline)"
  data3$algorithm[data3$algorithm == "lstcs"] <- "space-local"

  ggsave("data-local-normalized.pdf",
    device = "pdf", units = "in", scale = S, width = W, height = H,
    ggplot(data3, aes(
      x = nx, y = time / ops(nx, s), color = algorithm, shape = algorithm
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size") +
      ylab("Time per voxel (log-scale)") +
      scale_color_brewer(palette = "Accent") +
      labs(color = "Algorithm", shape = "Algorithm") +
      scale_y_log10(labels = sisec) +
      scale_x_continuous(labels = domain_label) +
      facet_wrap(~factor(precision, levels = c("Single precision", "Double precision")), scales="free") +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}

{
  data <- read.csv("gpp/temporals.csv", header = TRUE, sep = ",")
  data2 <- read.csv("gpp/baselines.csv", header = TRUE, sep = ",")
  data3 <- read.csv("gpp/biofvms.csv", header = TRUE, sep = ",")
  
  data = filter(data, x_tile_size != 10000)

  data$empty = 1
  data2$substrate_step = 1
  data2$x_tile_size = 10000
  data3$substrate_step = 1
  data3$x_tile_size = 1

  data <- rbind(data, data2, data3)

  data$precision[data$precision == "D"] <- "Double precision"
  data$precision[data$precision == "S"] <- "Single precision"
  
  data = filter(data, std_dev / time < 0.05)


  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()
  data <- data.frame(data)

  data = filter(data, s == 1 & nx %in% c( 100, 150, 200, 250, 300, 400, 450, 500, 550, 600))

  data$x_tile_size[data$x_tile_size == "10000"] <- "whole X (space-local)"
  data$x_tile_size[data$x_tile_size == "1"] <- "1 (temp-local)"
  
  data$x_tile_size <- factor(data$x_tile_size, levels = c("16", "32", "48", "64", "whole X (space-local)", "1 (temp-local)"))

  ggsave("temp-local-tile.pdf",
    device = "pdf", units = "in", scale = S, width = W, height = H,
    ggplot(data, aes(
      x = nx, y = time / ops(nx, s),
      color = factor(x_tile_size)
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size") +
      ylab("Time per voxel (log-scale)") +
      scale_color_manual(values = RColorBrewer::brewer.pal(8, "YlGnBu")[2:9]) +
      labs(color = "X tile size") +
      scale_y_log10(labels = sisec) +
      scale_x_continuous(labels = domain_label) +
      facet_wrap(~factor(precision, levels = c("Single precision", "Double precision")), scales="free") +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}

{
  data <- read.csv("gpp/transpose-temporal.csv", header = TRUE, sep = ",")
  data2 <- read.csv("gpp/temporals.csv", header = TRUE, sep = ",")
  data3 <- read.csv("gpp/partial-blocking.csv", header = TRUE, sep = ",")
  data3$x_tile_size = 32

  data = rbind(data, data2, data3)
  
  data = filter(data, x_tile_size %in%  c(32))

  data$precision[data$precision == "D"] <- "Double precision"
  data$precision[data$precision == "S"] <- "Single precision"
  
  data = filter(data, std_dev / time < 0.05)

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()
  data <- data.frame(data)

  data = filter(data, s == 1 & nx %in% c(100, 150, 200, 250, 300, 400, 450, 500, 600))

  data$x_tile_size[data$x_tile_size == "10000"] <- "whole X"
  
  data$algorithm[data$algorithm == "lstcstai"] <- "tiled+transposed"
  data$algorithm[data$algorithm == "lstcsta"] <- "tiled"
  data$algorithm[data$algorithm == "lstmfppai"] <- "planar"

  ggsave("temp-local-tile-transpose.pdf",
    device = "pdf", units = "in", scale = S, width = W, height = H,
    ggplot(data, aes(
      x = nx, y = time / ops(nx, s),
      color = algorithm, shape = algorithm
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size") +
      ylab("Time per voxel (log-scale)") +
      scale_color_brewer(palette = "Accent") +
      labs(color = "Algorithm", shape = "Algorithm") +
      scale_y_log10(labels = sisec) +
      scale_x_continuous(labels = domain_label) +
      facet_wrap(~factor(precision, levels = c("Single precision", "Double precision")), scales="free") +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}

{
  data <- read.csv("gpp/full-blocking-alt-dist.csv", header = TRUE, sep = ",")

  data$precision[data$precision == "D"] <- "Double precision"
  data$precision[data$precision == "S"] <- "Single precision"

  data <- data %>%
    separate(cores_division, into = c("cores_x", "cores_y", "cores_z"), sep = ",", convert = TRUE, remove = FALSE)
    
  data = filter(data, nx / cores_x >= 3 & ny / cores_y >= 3 & nz / cores_z >= 3)

  data = filter(data, std_dev / time < 0.1)

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()
  data <- data.frame(data)

  data = filter(data, !(cores_division %in% c("1,1,112", "1,2,56", "1,56,2")))
  
  data = filter(data, s == 1 & nx %in% c(50, 100, 150, 200, 250, 300, 400, 500, 600))
  
  data$cores_division <- factor(data$cores_division, levels = c("1,1,112", "1,2,56", "1,4,28", "1,7,16", "1,8,14", "1,14,8", "1,16,7", "1,28,4", "1,56,2"))

  data_sync_step = filter(data, s == 1)

  ggsave("full-blocking.pdf",
    device = "pdf", units = "in", scale = S, width = W, height = H,
    ggplot(data_sync_step, aes(
      x = nx, y = time / ops(nx, s),
      color = factor(cores_division)
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size") +
      ylab("Time per voxel (log-scale)") +
      scale_color_manual(values = RColorBrewer::brewer.pal(7, "YlGnBu")[2:9]) +
      labs(color = "Cores division") +
      scale_y_log10(labels = sisec) +
      scale_x_continuous(labels = domain_label) +
      facet_wrap(~factor(precision, levels = c("Single precision", "Double precision")), scales="free") +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}

data_both = c()

for (prefix in c("gpp", "hbm"))
{
  data_baseline <- read.csv(paste0(prefix, "/baselines.csv"), header = TRUE, sep = ",")
  data_baseline <- subset(data_baseline, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  data_baseline["algorithm"] <- "space-local"

  data_temporal <- read.csv(paste0(prefix, "/biofvms.csv"), header = TRUE, sep = ",")
  data_temporal$algorithm <- "temp-local"
  data_temporal <- subset(data_temporal, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))

  data_temporal_tile <- read.csv(paste0(prefix, "/temporals.csv"), header = TRUE, sep = ",")
  data_temporal_tile$algorithm <- "tiled"
  data_temporal_tile = filter(data_temporal_tile, x_tile_size == 32)
  data_temporal_tile <- subset(data_temporal_tile, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))

  partial_blocking <- read.csv(paste0(prefix, "/partial-blocking.csv"), header = TRUE, sep = ",")
  partial_blocking <- subset(partial_blocking, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  partial_blocking["algorithm"] <- "planar"

  full_blocking <- read.csv(paste0(prefix, "/full-blocking-alt-dist.csv"), header = TRUE, sep = ",")
  full_blocking <- full_blocking %>%
    separate(cores_division, into = c("cores_x", "cores_y", "cores_z"), sep = ",", convert = TRUE, remove = FALSE)
  full_blocking = filter(full_blocking, nx / cores_x >= 3 & ny / cores_y >= 3 & nz / cores_z >= 3)
  full_blocking = filter(full_blocking, cores_division == "1,8,14")
  full_blocking <- subset(full_blocking, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  full_blocking["algorithm"] <- "blocked"

  data <- c()
  data <- rbind(data, data_baseline, data_temporal, data_temporal_tile, partial_blocking)

  if (prefix == "gpp")
    data = filter(data, std_dev / time < 0.05)
  else
    data = filter(data, std_dev / time < 0.3)

  data = rbind(data, full_blocking)

  data = filter(data, s == 1 & nx %in% c(50, 100, 150, 200, 250, 300, 400, 500, 600))

  data$algorithm <- factor(data$algorithm, levels = c("temp-local", "space-local", "tiled", "tiled-transposed", "planar", "blocked"))

  data$precision[data$precision == "D"] <- "Double precision"
  data$precision[data$precision == "S"] <- "Single precision"
  
  if (prefix == "gpp")
    data = filter(data, std_dev / time < 0.1)
  else
    data = filter(data, std_dev / time < 0.3)

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()

  data <- data.frame(data)

  if (prefix == "gpp")
    data$machine = "DDR5"
  else
    data$machine = "HBM2"

  data_both = rbind(data_both, data)
}

ggsave("all-best.pdf",
  device = "pdf", units = "in", scale = S, width = W, height = H * 1.5,
  ggplot(data_both, aes(
    x = nx, y = time / ops(nx, s),
    color = algorithm, shape = algorithm
  )) +
    geom_point(size = point_size) +
    geom_line(linewidth = line_size) +
    xlab("Domain size") +
    ylab("Time per voxel (log-scale)") +
    scale_color_brewer(palette = "Set2") +
    labs(color = "Algorithm", shape = "Algorithm") +
    scale_y_log10(labels = sisec) +
    scale_x_continuous(labels = domain_label) +
    facet_grid(machine~factor(precision, levels = c("Single precision", "Double precision")), scales="free") +
    theme +
    background_grid() +
    theme(legend.position = "bottom")
)
