# PARSEG: A Computationally Efficient Approach for Statistical Validation

**Based on:** Frigau, L., Conversano, C. & Antoch, J. "PARSEG: a computationally efficient approach for statistical validation of botanical seeds’ images". *Scientific Reports* (2024).

## How to Cite

If you use this method or code in your research, please cite the original paper:

```bibtex
@article{frigau2024parseg,
  title={PARSEG: a computationally efficient approach for statistical validation of botanical seeds’ images},
  author={Frigau, Luca and Conversano, Claudio and Antoch, Jarom{\'i}r},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={6052},
  year={2024},
  publisher={Nature Portfolio},
  doi={10.1038/s41598-024-56228-6}
}
```

## 1. Introduction and Methodological Rationale

Traditional validation of image segmentation or large dataset classification often requires processing millions of pixels, which is computationally expensive. **PARSEG** (PArtitioning, Random Selection, Estimation, and Generalization) addresses this by finding the **minimum effective sample size** ($s^*$) that guarantees statistically valid results without using the entire dataset.

### The Core Principle: "Divide and Conquer"
To avoid overfitting and ensure robust statistical inference, PARSEG follows a strict protocol:

1.  **Partitioning**: The dataset is split into $M$ disjoint subsets. This ensures that the data used to *select* the optimal sample size is completely independent from the data used to *validate* the performance.

2.  **Selection**: We use the first block to monitor the "learning curve" of a classifier. As we increase the sample size, performance stabilizes. The point of stability (convergence) defines $s^*$.

3.  **Generalization**: We apply this $s^*$ to all $M$ blocks, including the one used for selection, in order to obtain a global estimate of the final performance.
  
---

## 2. Setup and Data Loading

```r
# Load necessary libraries
library(data.table)
library(caret)
library(ranger)     # A Fast Implementation of Random Forests
library(zoo)        # For rolling means in convergence
library(truncnorm)  # Required for PARSEG internal sampling

# Load PARSEG core functions
source("parseg_functions.R")

# Load Data (Skin_NonSkin) 
# available at https://archive.ics.uci.edu/static/public/229/skin+segmentation.zip
# Note: In the original paper, this method is applied to botanical images. 
# Here we use pixel data directly.
dat_model <- fread("Skin_NonSkin.txt", header = FALSE)
setnames(dat_model, c("B", "G", "R", "Class"))

# Preprocessing: Convert Class to Binary Factor (0/1)
# PARSEG requires a binary target factor for confusion matrices.
dat_model[, y := factor(ifelse(Class == 1, 1, 0), levels = c("0", "1"))]
dat_model[, Class := NULL] # Remove original label

cat("Dataset loaded. Total pixels N =", nrow(dat_model), "\n")
```

---

## 3. Step 1: Partitioning (The $M$ Disjoint Sets)

> **From the Paper:** "The procedure starts by partitioning the original image (or dataset) into $M$ disjoint subsets... This allows us to perform the selection of the optimal sample size on one subset and validate it on the others, ensuring the independence of the results.".

We split the data into $M$ disjoint blocks. 
* **Block 1**: Used for the **Selection Phase** (finding $s^*$) and also included in the final validation.
* **Blocks $2 \dots M$**: Used exclusively for the **Estimation Phase** (final validation).

```r
set.seed(2001)

M <- 20                
N <- nrow(dat_model)

# Randomly permute indices to create random disjoint partitions
perm_idx <- sample.int(N, size = N, replace = FALSE)
blocks <- split(perm_idx, rep(1:M, length.out = N))

cat("Partitioning complete: Split data into", M, "disjoint blocks.\n")
```

---

## 4. Step 2: Selection Phase (Finding $s^*$)

We use **Block 1** to simulate the classification process with increasing sample sizes. 

> **Convergence Logic:** We assume that as sample size increases, the classifier's performance metric (e.g., Accuracy) improves until it reaches a plateau. PARSEG identifies the "elbow" of this curve using a **derivative-based stopping rule**: we stop when the marginal gain of adding more pixels becomes negligible.

### 4.1 Define Parameters

```r
# Isolate Block 1
dat_block1 <- dat_model[blocks[[1]], ]

# Cardinality of M_1
n1 <- nrow(dat_block1)

# Sequence of sample sizes to test (logarithmic scale is efficient)
max_allowed <- nrow(dat_block1)
sequence <- unique(round(exp(seq(log(100), log(n1), length.out = 12))))

# Settings
B_rep <- 50          # Replications per size to smooth variance
cls_method <- "rf"   # Random Forest (robust and standard in PARSEG)
metric_idx <- 1      # Index 1 = Accuracy in caret output
                     # Metric index mapping (vals output):
                     # 1  = Accuracy
                     # 2  = Sensitivity
                     # 3  = Specificity
                     # 4  = PosPredValue
                     # 5  = NegPredValue
                     # 6  = Precision
                     # 7  = Recall
                     # 8  = F1
                     # 9  = Prevalence
                     # 10 = DetectionRate
                     # 11 = DetectionPrevalence
                     # 12 = BalancedAccuracy
```

### 4.2 Run Iterative Estimation on Block 1

```r
# Initialize 3D Array: [Replications, Metrics, Sample Sizes]
# The structure must strictly follow 'parseg_functions.R' requirements
temp_res <- vals(dat_block1, b=1, cls=cls_method, i=1, sequence=sequence, pie = 4)
n_metrics <- length(temp_res)
metric_names <- names(temp_res)

# The array
dd <- array(NA, dim = c(B_rep, n_metrics, length(sequence)))
dimnames(dd)[[2]] <- metric_names
dimnames(dd)[[3]] <- as.character(sequence)

cat("Step 2: Running Selection Analysis on Block 1...\n")

# A. Test candidate sizes
for (i in seq_along(sequence)) {
  for (b in 1:B_rep) {
    # 'vals' samples 'sequence[i]' pixels from 'dat_block1'
    dd[b, , i] <- vals(dat_block1, b = b, cls = cls_method, i = i, sequence = sequence, pie = 4)
  }
  cat("Finished sample size:", sequence[i], "\n")
}
```

### 4.3 Apply Convergence Rule

The `convergence` function analyzes the slope of the accuracy curve.

```r
conv_res <- convergence(
  dd = dd, 
  metric = metric_idx, 
  start = 3,           # Ignore first 3 very small sizes
  gamma = 2,           # Stability: must hold for 2 steps
  omega = -0.1         # Slope threshold (flatness)
)

if(conv_res$convergence) {
  opt_s <- conv_res$opt_size
  cat(">> Convergence Reached. Optimal Sample Size s* =", opt_s, "\n")
} else {
  opt_s <- tail(sequence, 1)
  cat(">> Convergence not reached. Defaulting to max tested size:", opt_s, "\n")
}
```

---

## 5. Step 3: Estimation Phase (Validation on $M$ Blocks)

> **From the Paper:** "Once $s^*$ is determined, the method validates the results on the remaining $M-1$ subsets. For each subset, we extract exactly $s^*$ pixels... This provides a distribution of performance metrics that generalizes to the whole population.".

We now "freeze" the sample size at $s^*$ and test it on data (Blocks 1 to $M$).

```r
# Prepare matrix to store results from the M-1 validation blocks
results_mat <- matrix(NA, nrow = M, ncol = n_metrics)
colnames(results_mat) <- metric_names[1:n_metrics]
rownames(results_mat) <- paste0("Block_", 1:M)

cat("Step 3: Validating on remaining blocks using fixed s* =", opt_s, "...\n")

for (m in 1:M) {
  # 1. Access the specific disjoint block
  idx_m <- blocks[[m]]
  
  # Check if block is large enough
  if(length(idx_m) < opt_s) stop("Error: Validation block smaller than s*")
  
  # 2. Subset EXACTLY s* pixels
  # This enforces the parsimony principle of PARSEG
  idx_subset <- idx_m[1:opt_s]
  dat_subset <- dat_model[idx_subset, ]
  
  # 3. Evaluate
  # passing sequence = nrow(dat_subset) uses the full s* subset
  res <- vals(
    dat = dat_subset, 
    b = 2000 + m,      # Ensure different RNG seed
    cls = cls_method, 
    i = 1, 
    sequence = nrow(dat_subset),
    pie = 4
  )
  
  results_mat[m, ] <- res
}
```

---

## 6. Final Results & Generalization

We report the **Mean** and **Standard Deviation** of the metrics across the disjoint blocks. This is the final output of the PARSEG procedure.

```r
final_stats <- data.frame(
  Metric = colnames(results_mat),
  Mean = colMeans(results_mat, na.rm = TRUE),
  SD = apply(results_mat, 2, sd, na.rm = TRUE)
)

# Filter for key metrics
print(final_stats[final_stats$Metric %in% c("Accuracy", "Sensitivity", "Specificity", "Time"), ])

# Computational Efficiency Calculation
total_used <- (M-1) * opt_s + n1
reduction <- 1 - (total_used / N)
cat("\n--- Efficiency Summary ---\n")
cat("Original Data Size (N):    ", N, "pixels\n")
cat("Total Pixels Used (valid): ", total_used, "\n")
cat("Data Reduction:            ", round(reduction * 100, 2), "%\n")
```

Data Reduction measures the fraction of unique pixels involved in the PARSEG workflow and should be interpreted as data efficiency rather than raw computational cost.


## 7. Using PARSEG Results for Segmentation Validation

The final output of PARSEG is **not a single segmentation mask**, but a **validated decision framework** that supports segmentation assessment and comparison in a statistically principled way.

This section explains how to interpret and use the results obtained from the previous steps.

---

## 7.1 What PARSEG actually validates

PARSEG validates a segmentation **indirectly**, by evaluating how well a supervised classifier can reproduce the segmentation-induced labels using pixel-level features.

Concretely:

- each segmentation method induces a binary labeling `y` (foreground / background);
- PARSEG evaluates the *learnability* and *stability* of this labeling;
- good segmentations are:
  - easy to learn from few pixels,
  - stable with respect to subsampling,
  - consistent across disjoint pixel blocks.

Thus, PARSEG assesses **segmentation quality via predictive consistency**, not via geometric overlap alone.

---

## 7.2 Outputs available after PARSEG

After running the full workflow, you have:

1. **Optimal sample size**  
   $s^* =$ `opt_s`
   the minimal number of pixels required for stable performance.

2. **Final validation table**  
   A table like:

   | Metric | Mean | SD |
   |------|------|----|
   | Accuracy | 0.94 | 0.01 |
   | Sensitivity | 0.92 | 0.02 |
   | Specificity | 0.96 | 0.01 |
   | BalancedAccuracy | 0.94 | 0.01 |
   | Time | 0.18 | 0.03 |

   computed over the \(M\) disjoint blocks.

3. **Data efficiency indicator**  
   $\text{fraction used} = \frac{(M-1)s^* + n_1}{N}$
   measuring how many pixels are actually needed.

---

## 7.3 Comparing multiple segmentation methods

PARSEG is especially useful when **several segmentation methods** are available (e.g. Otsu, Sauvola, CNN-based masks, manual annotations).

For each candidate segmentation:

1. build a dataset `(features, y_method_k)`;
2. run the full PARSEG pipeline;
3. extract:
   - $s^*_k$,
   - final performance table.

You can then compare methods using criteria such as the highest final metric.

### Example decision table

| Method | Balanced Acc. | 
|------|---------------|
| Method A | 0.95 | 
| Method B | 0.96 | 
| Method C | 0.91 |

In this example, **Method B** may be preferred over Method A and C, because it achieves the highest value in the considered metric.
