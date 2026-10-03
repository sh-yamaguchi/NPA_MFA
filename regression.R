source("lib/mfa.R")

reaction_list <- c(
  "R1major", "R2major", "R3major",
  "R1minor", "R2minor", "R3minor"
)

for (rxn in reaction_list) {
  message("=== Running MFA for ", rxn, " ===")

  # build filenames
  y_file  <- paste0("target", sub("(major|minor)$", "", rxn), ".txt")
  x_file  <- paste0("descriptor", rxn, ".txt")
  out_dir <- ""

  # read input tables
  y  <- read.table(y_file, row.names = 1)
  x  <- read.table(x_file, header = FALSE)
  xa <- read.table(x_file, header = FALSE)

  # run MFA
  mfa(x, xa, y, 1.0, "off", rxn, out_dir)

  message("→ Finished ", rxn, "\n")
}