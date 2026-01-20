#installs and load the package "pacman"
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, nlme, survival, JMbayes2, caret, survAUC, pec)
library(dplyr)
library(tidyr)
library(ggplot2)
library(nlme)
library(survival)
library(JMbayes2)
library(caret)
library(survAUC)
library(pec)

# Read onset_longitudinal
onset_data <- read.csv("onset_longitudinal.csv")
onset <- read.csv("onset.csv")

# Show the number of measurements and unique rats
num_measurements <- nrow(onset_data)
num_unique_rats <- n_distinct(onset_data$NR_Name)
cat("Number of measurements:", num_measurements, "\n")
cat("Number of unique rats:", num_unique_rats, "\n")

# Filter out rows with NA values
onset_data <- drop_na(onset_data)

# Get a list of rats where diet is New or Mix
new_diet_rats <- unique(onset_data$NR_Name[onset_data$diet %in% c("New", "Mix")])
# Remove rats
onset_data <- onset_data %>% filter(!onset_data$NR_Name %in% new_diet_rats)

# Make a new table grouped by NR_Name and grabbing count
onset_data_grouped <- onset_data %>%
  group_by(NR_Name) %>%
  summarize(num_measurements = n())

# Grab list of rats with only 1 measurement
single_measurement_rats <-
  unique(onset_data_grouped$NR_Name[onset_data_grouped$num_measurements == 1])
# Remove rats with only 1 measurement
onset_data_filtered <- onset_data %>%
  filter(!onset_data$NR_Name %in% single_measurement_rats)

# Left join with onset
#onset_data_filtered <- left_join(onset_data_filtered, onset, by = "NR_Name")
# Remove rows where diet is not Rod or Rab
onset_data_filtered <- onset_data_filtered %>%
  filter(diet %in% c("Rod", "Rab"))

# Show the number of measurements and unique rats
num_measurements <- nrow(onset_data_filtered)
num_unique_rats <- n_distinct(onset_data_filtered$NR_Name)
cat("Number of measurements:", num_measurements, "\n")
cat("Number of unique rats:", num_unique_rats, "\n")

# Make 2 separate tables, one containing the survival data
# And another containing the longitudinal data
# Remove duplicates for survival data
survival_data <- onset_data_filtered %>%
  group_by(NR_Name) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(NR_Name, Event_time, Event, sex, diet)
longitudinal_data <- onset_data_filtered %>%
  select(NR_Name, week, weight, rbg, diet, sex)


print(table(longitudinal_data$diet))

# Turn chars into factors
longitudinal_data$sex  <- factor(longitudinal_data$sex)
longitudinal_data$diet <- factor(longitudinal_data$diet)

survival_data$sex  <- factor(survival_data$sex,
                             levels = levels(longitudinal_data$sex))
survival_data$diet <- factor(survival_data$diet,
                             levels = levels(longitudinal_data$diet))

# Create 5 folds based on unique NR_Name
set.seed(123)
folds <- createFolds(unique(longitudinal_data$NR_Name), k = 5)

results <- list()
auc <- list()
brier_score <- list()

for (i in seq_along(folds)) {
  print(paste("Processing fold", i))
  test_ids <- unique(longitudinal_data$NR_Name)[folds[[i]]]
  train_ids <- setdiff(unique(longitudinal_data$NR_Name), test_ids)

  train_long <- subset(longitudinal_data, NR_Name %in% train_ids)
  test_long  <- subset(longitudinal_data, NR_Name %in% test_ids)

  train_surv <- subset(survival_data, NR_Name %in% train_ids)
  test_surv  <- subset(survival_data, NR_Name %in% test_ids)

  # Remove diet from test_surv
  test_surv2 <- test_surv %>% select(-diet)
  # Merge test_long and test_surv2 by NR_Name and sex (left join)
  test_overall <- left_join(test_long, test_surv2, by = c("NR_Name", "sex"))
  # Sort test_overall by NR_Name and week
  test_overall <- test_overall %>% arrange(NR_Name, week)

  # Longitudinal mixed effects model
  print(paste("Fitting models for fold", i))
  lme_fit <- lme(rbg ~ weight + week + sex + diet,
                 random = ~ weight + week | NR_Name,
                 data = train_long, control = lmeControl(opt = "optim"))

  cox_fit <- coxph(Surv(Event_time, Event) ~ sex + diet,
                   data = train_surv, x = TRUE)

  print(paste("Fitting joint model for fold", i))
  jm_fit <- jm(cox_fit, lme_fit, time_var = "week")

  # Example: prediction on test set
  print(paste("Predicting for fold", i))
  preds_long <- predict(jm_fit, newdata = test_overall, process = "longitudinal")
  preds_surv <- predict(jm_fit, newdata = test_overall, process = "event")

  # Collect results
  rmse <- sqrt(mean((unlist(test_overall$rbg) - unlist(preds_long$pred))^2, na.rm = TRUE))
  mae  <- mean(abs(unlist(test_overall$rbg) - unlist(preds_long$pred)), na.rm = TRUE)

  print("Calculating AUC and Brier Score")
  # Calculate AUC and Brier Score using pec package
  times <- sort(unique(test_surv$Event_time))
  pec_res <- pec::pec(
    object = list("Cox" = cox_fit),
    formula = Surv(Event_time, Event) ~ sex + diet,
    data = test_surv,
    train_data = train_surv
  )
  print(pec_res)

  print("Calculating C-index")
  cindex_res <- pec::cindex(
    object = list("Cox" = cox_fit),
    formula = Surv(Event_time, Event) ~ sex + diet,
    data = test_surv,
    eval.times = sort(unique(test_surv$Event_time)),
    splitMethod = "none"   # since you're manually looping folds
  )
  print(cindex_res$AppCindex)

  fold_results <- c(i, rmse, mae)
  results[[i]] <- fold_results

}

# Combine all fold results into a single data frame
final_results <- do.call(rbind, results)
print("Cross-Validation Results:")
print(final_results)