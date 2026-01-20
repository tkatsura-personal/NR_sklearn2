# Print model summaries
print(summary(lme_fit))

# Survival Cox model
cox_fit <- coxph(Surv(Event_time, Event) ~ sex + diet + litter_size,
                 data = survival_data, x = TRUE)

#Print model summaries
print(summary(cox_fit))

print(nrow(longitudinal_data))
print(colSums(is.na(longitudinal_data[, c("rbg", "week", "sex", "diet", "NR_Name")])))
print(length(unique(longitudinal_data$NR_Name)))
print(table(longitudinal_data$sex))

joint_fit <- jm(cox_fit, lme_fit, time_var = "week",
                n_iter = 12000L, n_burnin = 2000L, n_thin = 5L,
                seed = 123)

# Print joint model summaries
print(summary(joint_fit))