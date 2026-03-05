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
  data <- read.csv("../results/baselines.csv", header = TRUE, sep = ",")
  

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()
  data <- data.frame(data)


  data$precision[data$precision == "D"] <- "Double precision"
  data$precision[data$precision == "S"] <- "Single precision"

  data2 <- read.csv("../results/biofvms.csv", header = TRUE, sep = ",")
  

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
      xlab("Domain size (log-scale)") +
      ylab("Time per voxel (log-scale)") +
      scale_color_brewer(palette = "Accent") +
      labs(color = "Algorithm", shape = "Algorithm") +
      scale_y_log10(labels = sisec) +
      scale_x_log10(labels = domain_label, breaks = c(64, 128, 256, 400, 600)) +
      facet_wrap(~factor(precision, levels = c("Single precision", "Double precision")), scales="free") +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}
