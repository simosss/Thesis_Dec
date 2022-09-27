library(tidyverse)

# more than 500 appearances
poly_se <- read_csv('se.csv')

combo <- read_csv('data/bio-decagon-combo.csv') %>% 
    select(-"Side Effect Name") %>%
    rename(node1 = `STITCH 1`,
           node2 = `STITCH 2`,
           se = `Polypharmacy Side Effect`) %>%
    filter(se %in% poly_se$se)

drugs <- unique(c(unique(combo$node1), unique(combo$node2)))

inv_combo <- combo %>%
    mutate(node_temp = node1,
           node1 = node2,
           node2 = node_temp) %>%
    select(-node_temp)

combo_1 <- combo %>%
    slice(which(row_number() %% 2 == 1))

combo_2 <- inv_combo %>%
    slice(which(row_number() %% 2 == 0))

data <- bind_rows(combo_1, combo_2)

rm(combo_1, combo_2, inv_combo)


data <- data %>%
    mutate(neg = sample(x=drugs, size = nrow(data), replace = TRUE)) %>%
    unite(neg_pair, c("node1", "neg"), remove=FALSE) %>%
    unite(pos_pair, c('node1', 'node2'), remove=FALSE) %>%
    unite(inv_pos_pair, c('node2', 'node1'), remove=FALSE) 

positive_pairs <- c(data$pos_pair, data$inv_pos_pair)

data <- data %>%
    mutate(check1 = if_else(neg_pair %in% positive_pairs, 1, 0),
           check2 = if_else(node1 == neg, 1, 0)) %>%
    mutate(neg = if_else(check1 == 1 | check2 == 1, sample(x=drugs, size= nrow(data), replace = TRUE), neg)) %>%
    unite(neg_pair, c("node1", "neg"), remove=FALSE)

i <- 0
# many times untill check =0 for all
while(nrow(data %>% filter(check1 == 1 | check2 ==1)) != 0){
    print(i)
    data <- data %>%
        mutate(check1 = if_else(neg_pair %in% positive_pairs, 1, 0),
               check2 = if_else(node1 == neg, 1, 0)) %>%
        mutate(neg = if_else(check1 == 1 | check2 == 1, sample(x=drugs, size= nrow(data), replace = TRUE), neg)) %>%
        unite(neg_pair, c("node1", "neg"), remove=FALSE)
    i <- i+1
    
}

negative_samples <- data %>%
    select(node1, node2 = neg, se)


positive_samples <- data %>%
    select(node1, node2, se)

write_csv(positive_samples, 'positive_samples_uniform.csv')
write_csv(negative_samples, 'negative_samples_uniform.csv')
