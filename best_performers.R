library(tidyverse)

names <- read_csv('data/bio-decagon-effectcategories.csv') %>%
    select(se = 'Side Effect',
           se_name = 'Side Effect Name')

combo <- read_csv('data/bio-decagon-combo.csv') %>%
    select('Polypharmacy Side Effect', 'Side Effect Name') %>%
    distinct(`Polypharmacy Side Effect`, .keep_all = T)

combo <- combo %>%
    select(se = `Polypharmacy Side Effect`,
           se_name = `Side Effect Name`)

per_se_transe <- read_csv('per_se_transe_rare.csv') 

per_se_10 <- per_se_transe[1, ]

per_se_50 <- per_se_transe[2, ]

hits_10 <- per_se_10 %>%
    pivot_longer(1:301, names_to = 'se', values_to = 'hits_at_10') %>%
    left_join(combo, by = 'se') %>%
    mutate(rank_at_10 = rank(desc(hits_at_10)))

hits_50 <- per_se_50 %>%
    pivot_longer(1:301, names_to = 'se', values_to = 'hits_at_50') %>%
    left_join(combo, by = 'se')   %>%
    mutate(rank_at_50 = rank(desc(hits_at_50)))       

transe <- left_join(hits_10, hits_50, by = c('se', 'se_name')) %>%
    select(se, se_name, hits_at_10, hits_at_50, rank_at_10, rank_at_50)

write_csv(transe, 'results/transe_per_se.csv')


per_se_transe <- read_csv('per_se_rotate.csv') 

per_se_10 <- per_se_transe[1, ]

per_se_50 <- per_se_transe[2, ]

hits_10 <- per_se_10 %>%
    pivot_longer(1:300, names_to = 'se', values_to = 'hits_at_10') %>%
    left_join(combo, by = 'se') %>%
    mutate(rank_at_10 = rank(desc(hits_at_10)))

hits_50 <- per_se_50 %>%
    pivot_longer(1:300, names_to = 'se', values_to = 'hits_at_50') %>%
    left_join(combo, by = 'se')   %>%
    mutate(rank_at_50 = rank(desc(hits_at_50)))       

rotate <- left_join(hits_10, hits_50, by = c('se', 'se_name')) %>%
    select(se, se_name, hits_at_10, hits_at_50, rank_at_10, rank_at_50)

all <- left_join(transe, rotate, by = c('se', 'se_name'), suffix = c('_transe', '_rotate')) %>%
    mutate(diff_10 = hits_at_10_rotate - hits_at_10_transe,
           diff_50 = hits_at_50_rotate - hits_at_50_transe,
           rank_diff_10 = rank_at_10_rotate - rank_at_10_transe,
           rank_diff_50 = rank_at_50_rotate - rank_at_50_transe)

write_csv(rotate, 'results/rotate_per_se.csv')
write_csv(all, 'results/performance_per_se_final.csv')

ggplot(all, aes(x=diff_50)) +geom_histogram()
ggplot(all, aes(x=diff_10)) +geom_histogram()
