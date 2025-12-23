#' Evaluate a segmentation-induced labeling via supervised classification (PARSEG core step)
#'
#' @description
#' This function implements one repeated evaluation step used in PARSEG.
#' It first draws a *partial* subset of pixels (size driven by \code{sequence[i]}),
#' then splits that subset into an internal train/test split controlled by \code{pie},
#' fits a classifier chosen via \code{cls}, predicts on the test set, and returns
#' confusion-matrix based performance measures plus runtime.
#'
#' The training fraction is computed as:
#' \deqn{\\text{train\\_per} = \\frac{\\text{pie}}{1 + \\text{pie}}}
#' so that the train:test ratio is approximately \code{pie:1}. For example,
#' \code{pie = 4} yields an 80/20 split.
#'
#' @param dat A data.frame/data.table containing pixel-level data.
#'   Must include a column named \code{y} with binary classes "0"/"1"
#'   (factor recommended). All remaining columns are used as predictors via
#'   formula \code{y ~ .}.
#' @param b Integer replication index (e.g., 1..B). Used to alter the RNG seed so that
#'   each repetition draws a different subset/split.
#' @param cls Character string selecting the classifier (see code; e.g., \code{"rf"},
#'   \code{"rpart"}, \code{"log"}, \code{"nb"}, \code{"knn"}, \code{"svm"}).
#' @param i Integer index selecting the current sample size from \code{sequence}.
#' @param sequence Numeric/integer vector of candidate sample sizes (in number of pixels).
#'   The function uses \code{sequence[i] / nrow(dat)} as the sampling proportion
#'   for \code{caret::createDataPartition}.
#' @param pie Positive numeric scalar controlling the internal train/test split ratio.
#'   The split is defined by \code{train_per = pie/(1+pie)}, implying a train:test ratio
#'   of approximately \code{pie:1}. Default usage in PARSEG often corresponds to
#'   \code{pie = 4} (80/20).
#'
#' @return Numeric vector of confusion-matrix metrics plus elapsed time (minutes),
#'   or NAs if model fitting/prediction fails.
#'
#' @details
#' - The initial subset is drawn using \code{caret::createDataPartition}, which attempts
#'   to preserve class proportions in the selected indices.
#' - The subsequent train/test split is performed *within* the selected subset and is
#'   controlled by \code{pie}.
#' - Predictions are coerced to a factor with levels \code{c("0","1")} so that
#'   \code{caret::confusionMatrix} aligns class ordering consistently.
#'
#' @note
#' Requires packages: \code{caret}, \code{rpart}, \code{ranger}, \code{e1071}, \code{kknn}.
vals <- function(dat, b, cls, i, sequence, pie) {
  time_val0 <- Sys.time()
  set.seed(122 + b)

  int <- caret::createDataPartition(
    y = dat[["y"]],
    p = (sequence[i] / nrow(dat)),
    list = FALSE
  )

  train_per <- pie / (1 + pie)
  train <- sample(int, nrow(int) * train_per)
  test <- int[which(!(int %in% train)), 1]

  pr.fold <- try(
    switch(
      cls,
      rpart = predict(
        rpart::rpart(y ~ ., data = dat[train, ]),
        dat[test, ],
        "class"
      ),
      rf = predict(
        ranger::ranger(y ~ ., data = dat[train, ]),
        dat[test, ],
        type = "response"
      )$predictions,
      log = round(predict(
        glm(y ~ ., data = dat[train, ], family = binomial(logit)),
        dat[test, ],
        "response"
      )),
      nb = predict(
        e1071::naiveBayes(y ~ ., data = dat[train, ]),
        dat[test, ],
        "class"
      ),
      knn = kknn::kknn(y ~ ., dat[train, ], dat[test, ])$fitted.values,
      svm = predict(e1071::svm(y ~ ., data = dat[train, ]), dat[test, ])
    ),
    silent = TRUE
  )

  time_val1 <- Sys.time()

  if (!is(pr.fold, "try-error")) {
    pr.fold <- factor(pr.fold, levels = c("0", "1"))
    cm <- caret::confusionMatrix(pr.fold, dat[test, ][["y"]])
    a <- c(
      cm$overall[1],
      cm$byClass[1:11],
      time = as.numeric(difftime(time_val1, time_val0, units = "min"))
    )
  } else {
    a <- rep(NA, 9)
  }

  return(a)
}


#' Select an "optimal" sample size s* via PARSEG convergence criterion
#'
#' @description
#' Implements the PARSEG selection of the optimal sample size \eqn{s^*} by monitoring
#' how a standardized consistency metric stabilizes as sample size increases.
#'
#' @param dd A 3D array-like object containing evaluation results.
#'        Dimension 3 (with dimnames) must correspond to candidate sample sizes.
#'        The last entry along dim 3 is expected to represent the full-size evaluation.
#' @param metric Character or numeric index selecting the metric to analyze along
#'        the second dimension of \code{dd}.
#' @param start Integer >= 1. Initial number of candidate sizes to consider before expanding.
#' @param gamma Integer >= 1. Convergence window length: stop when the selected size is unchanged
#'        for \code{gamma} consecutive iterations.
#' @param omega Numeric scalar. Target slope for the derivative-based rule.
#'
#' @return A list containing convergence status and the selected optimal sample size.
convergence <- function(dd, metric, start = 4, gamma = 4, omega = -1) {
  ll <- length(dimnames(dd)[[3]]) - 1

  condition <- TRUE
  vc_pos_sel <- c()
  k <- start

  while (condition) {
    der_rf <- as.table(dd[, metric, ])
    der_rf <- data.table(as.data.frame(der_rf))
    der_rf <- der_rf[,
      list(mean(Freq, na.rm = TRUE), sd(Freq, na.rm = TRUE)),
      by = Var2
    ]
    der_rf[, Var2 := as.numeric(as.character(Var2))]

    der_rf_sub <- der_rf[c(1:k, (ll + 1)), ]

    last_idx <- length(der_rf_sub$Var2)
    x <- scale(der_rf_sub$Var2[-last_idx])

    y_pre_mean <- abs(der_rf_sub$V1 - tail(der_rf_sub$V1, 1))[-last_idx]
    y_pre_sd <- abs(der_rf_sub$V2 - tail(der_rf_sub$V2, 1))[-last_idx]

    y <- scale(y_pre_mean * y_pre_sd)
    y <- c(y[1], zoo::rollmean(y, 3), tail(y, 1))

    f_of_x_mean <- splinefun(x, y, method = "fmm")
    deriv1_mean <- f_of_x_mean(x, deriv = 1)
    a_mean <- deriv1_mean - omega

    selected_pos <- max(which(diff(sign(a_mean)) != 0)) + 1

    vc_pos_sel <- c(
      vc_pos_sel,
      setNames(der_rf$Var2[selected_pos], der_rf$Var2[k])
    )

    condition <- sum(
      tail(vc_pos_sel, gamma) == tail(vc_pos_sel, 1),
      na.rm = TRUE
    ) !=
      gamma

    if (k == ll) {
      condition <- FALSE
    } else {
      k <- k + 1
    }
  }

  convergence <- sum(
    tail(vc_pos_sel, gamma) == tail(vc_pos_sel, 1),
    na.rm = TRUE
  ) ==
    gamma
  opt_size <- ifelse(convergence, tail(vc_pos_sel, 1), der_rf$Var2[ll + 1])
  train_size <- as.numeric(tail(names(vc_pos_sel), 1))
  tot_size <- sum(der_rf$Var2[der_rf$Var2 <= train_size]) + tail(der_rf$Var2, 1)

  return(list(
    convergence = convergence,
    train_size = train_size,
    opt_size = opt_size,
    tot_size = tot_size,
    selected_positions = vc_pos_sel,
    metric = metric,
    start = start,
    gamma = gamma,
    omega = omega
  ))
}
