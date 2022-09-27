library(tidyverse)

combo <- read_csv('bio-decagon-combo.csv') %>% 
    select(-"Side Effect Name") %>%
    rename(node1 = `STITCH 1`,
           node2 = `STITCH 2`,
           se = `Polypharmacy Side Effect`)


drugs <- bind_rows(d1 =combo$node1, d2=combo$node2) %>%
    mutate(node1 = if_else(is.na(d1), d2, d1)) %>%
    distinct(node1) %>%
    pull()


data <- combo %>%
    mutate(neg = sample(x=drugs, size= nrow(data), replace = TRUE)) %>%
    unite(neg_pair, c("node1", "neg"), remove=FALSE) %>%
    unite(pos_pair, c('node1', 'node2'), remove=FALSE) 

pos_pairs <- data$pos_pair

# many times untill check =0 for all
data <- data %>%
    mutate(check = if_else(neg_pair %in% pos_pairs, 1, 0)) %>%
    mutate(neg = if_else(check == 1, sample(x=drugs, size= nrow(data), replace = TRUE), neg)) %>%
    unite(neg_pair, c("node1", "neg"), remove=FALSE)

negative_samples <- data %>%
    select(node1, node2 = neg, se)


positive_samples <- data %>%
    select(node1, node2, se)

write_csv(positive_samples, 'positive_samples.csv')
write_csv(negative_samples, 'negative_samples.csv')
