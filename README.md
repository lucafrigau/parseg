# parseg
PARSEG — A data-efficient framework for pixel-level image segmentation with convergence-based sample size selection.

PARSEG (PArtitioning, Random Selection, Estimation, and Generalization) is a statistical framework for image segmentation that determines the minimal sufficient number of labeled pixels required to obtain stable and reliable segmentation performance.
The method combines supervised pixel-level classification, progressive subsampling, and a convergence criterion to select an optimal training size, followed by disjoint-block evaluation for unbiased performance estimation.
PARSEG is particularly suited for large-scale image segmentation tasks, where annotation costs are high and data efficiency is critical.
