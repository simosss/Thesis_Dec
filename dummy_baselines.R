library(tidyverse)

combo <- read_csv('bio-decagon-combo.csv')
train <- read_csv('decagon_train.csv')
test <- read_csv('decagon_test.csv')

relations <- unique(test$relation)

top_pairs <- train %>%
    unite(pair, c("node1", "node2")) %>%
    filter(!relation %in% c("interacts", "targets")) %>%
    group_by(pair) %>%
    summarize(cnt = n()) %>%
    ungroup() %>%
    slice_max(order_by=cnt, n=50) %>%
    mutate(rank = rank(desc(cnt), ties.method = 'first'))



predictions <- test %>%
    unite(pair, c("node1", "node2"), remove=FALSE) %>%
    mutate(correct = if_else(pair %in% top_pairs$pair, 1, 0)) %>%
    group_by(relation) %>%
    summarize(total = n(),
              correct = sum(correct),
              accuracy = correct/50)
    

max_possible_ap50 <- sum(predictions$accuracy) / length(predictions$accuracy)


rare_train <- read_csv('rare_training.csv')
rare_test <- read_csv('rare_testing.csv')

relations <- unique(rare_test$relation)

node1 <- rare_train['node1']
node2 <- rare_train['node2']

top_drugs <- bind_rows(node1,node2) %>%
    mutate(node1 = if_else(is.na(node1), node2, node1)) %>%
    select(node1) %>%
    group_by(node1) %>%
    summarize(cnt = n()) %>%
    slice_max(order_by=cnt, n=10)

predictions <- rare_test %>%
    #unite(pair, c("node1", "node2"), remove=FALSE) %>%
    mutate(correct = if_else(node2 %in% top_drugs$node1, 1, 0))


hits10 = sum(predictions$correct) / length(predictions$correct)
