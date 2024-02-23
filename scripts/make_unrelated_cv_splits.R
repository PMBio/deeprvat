
library("ukbtools")
library(dplyr)
library(ggplot2)
library(arrow)
library(stringr)
library(tidyr)



phenotype_df = read_parquet('~/ukbb/exomes/vcf/preprocessed/genotypes_phenotypes.parquet') #uses ids from old application

sample_mapping = read.csv('~/ukbb/metadata/sample_map_ukb44975_ukb673180.csv') %>% select(-X) 

all_samples = phenotype_df %>% pull('samples')

#################################################
##### LOAD Kinship matrix and map to samples 

ukb_relatedness = read.csv('~/ukbb/metadata/ukb_rel_a81358_s488120.dat', sep = ' ')
ukb_relatedness %>% arrange(Kinship)

#map kinship ids from new application (673180) to old application (44975)
kinship_mapped = ukb_relatedness %>% left_join(sample_mapping, by = c('ID1' = 'id_673180'))
kinship_mapped[is.na(kinship_mapped[['id_44975']]),]
kinship_mapped = kinship_mapped %>% select(-ID1) %>% rename('ID1' = 'id_44975') %>%
  left_join(sample_mapping, by = c('ID2' = 'id_673180')) 
kinship_mapped[is.na(kinship_mapped[['id_44975']]),]

kinship_mapped =  kinship_mapped %>% select(-ID2) %>% rename('ID2' = 'id_44975')

#################################################
### get sets of related samples

library(igraph)

# Sample data
df <- ukb_gen_related_with_data(kinship_mapped, ukb_with_data = all_samples, 
                                cutoff = 0.0884) %>% select(ID1, ID2)

# Create a graph from the data
graph <- graph_from_data_frame(df, directed = FALSE)

# Find connected components
connected_sets <- components(graph)

# Get the sets of connected IDs
connected_id_sets <- lapply(connected_sets$membership, function(m) {
  V(graph)$name[m]
})

# Display the connected sets

related_set_df <- tibble(
  ID = names(connected_id_sets),
  set_id = unlist(connected_id_sets),
)

related_set_size = related_set_df %>%
  group_by(set_id) %>%
  summarize(size = n()) %>%
  arrange(desc(size)) %>%
  mutate(set_idx = row_number())
summary(related_set_size)

#each ID should only be assigned to a single set
related_set_df %>%
  group_by(ID) %>%
  summarise(n_dist = n_distinct(set_id)) %>%
  distinct(n_dist)


#################################################

#################################################

### until here everything is fold independent
### from here it gets fold dependent

### distribute the related sets across folds 

library(reticulate)
assignSamplesToFolds = function(n_folds, cv_splits_out_dir){
  ordered_ids = related_set_size[['set_idx']]
  result_lists <- split(ordered_ids, (ordered_ids - 1) %% n_folds + 1)
  
  
  assigned_folds <- data.frame(
    fold = rep(names(result_lists), sapply(result_lists, length)),
    set_idx = unlist(result_lists)
  ) %>%
    left_join(related_set_size %>% select(-size))
  
  # assign sample ids to folds via the set ids
  sample_id_fold_assignment = related_set_df %>% left_join(assigned_folds)
  sample_id_fold_assignment %>% group_by(fold) %>%
    summarize(size = n())
  
  ##### ##### ##### ##### ##### ##### ##### 
  # get remaning ids
  set.seed(123)
  
  assigned_related_samples = sample_id_fold_assignment[['ID']]
  left_samples = sample(setdiff(all_samples, assigned_related_samples))
  n_all_samples = length(all_samples)
  
  stopifnot(length(left_samples) + length(assigned_related_samples) == n_all_samples)
  
  min_fold_size = floor(n_all_samples / n_folds)
  stopifnot(min_fold_size * n_folds <= n_all_samples)
  
  ##### ##### ##### ##### ##### ##### ##### 
  # get current fold sizes (after assigning related samples) and number of samples that are still needed 
  fold_sizes = sample_id_fold_assignment %>% 
    select(fold, ID) %>%
    distinct() %>%
    group_by(fold) %>%
    mutate(fold_size = n()) %>%
    ungroup() %>%
    select(-ID) %>%
    distinct() %>%
    mutate(left_samples = min_fold_size - fold_size,
           fold = as.numeric(fold)) %>%
    arrange(fold)
  
  ##### ##### ##### ##### ##### ##### ##### 
  # assign shuffled, remaning samples (non related samples) 
  folds_left_samples = c()
  dist_samples = c()
  start_idx = 1
  fold = 1
  for (size in fold_sizes[['left_samples']]){
    end_idx = start_idx + size - 1
    cat('size:', size, 'start_idx:', start_idx, 'end_idx:', end_idx, '\n')
    this_samples = left_samples[start_idx:end_idx]
    print(length(this_samples))
    print(length(this_samples) == size)
    start_idx = end_idx + 1
    folds_left_samples[[as.character(fold)]] = this_samples
    dist_samples = append(dist_samples, this_samples)
    fold = fold + 1
  }
  
  ##### ##### ##### ##### ##### ##### ##### 
  # if any sample is not assigned yet, do it know until no sample is left
  left_samples = setdiff(left_samples, dist_samples)
  if (length(left_samples) > 0){
    c = 1
    for (i in left_samples){
      folds_left_samples[[c]] = append(folds_left_samples[[c]], i)
      c = ifelse(c == n_folds, 1, c + 1)
    }
  } 
  folds_left_samples_df = tibble()
  for (i in names(folds_left_samples)){
    t = tibble(fold = as.numeric(i), ID = folds_left_samples[[i]])
    folds_left_samples_df = rbind(folds_left_samples_df, t)
  }
  final_id_fold = rbind(folds_left_samples_df, 
                        sample_id_fold_assignment %>% select(ID, fold)) 
  stopifnot(nrow(final_id_fold) == n_all_samples)
  
  
  ##### ##### ##### ##### ##### ##### ##### 
  ##### write k and k-1 folds as test and train samples for all k
  
  out_dir = file.path(cv_splits_out_dir,sprintf('%d_fold', n_folds))
  
  if (!file.exists(out_dir)) {
    # If it doesn't exist, create the directory
    dir.create(out_dir)
    cat("Directory created:", out_dir, "\n")
  } else {
    cat("Directory already exists:", out_dir, "\n")
  }
  all_folds = unique(final_id_fold[['fold']])
  # get samples 
  for (test_fold in all_folds){
    train_folds = setdiff(all_folds, test_fold)
    test_samples = filter(final_id_fold, fold == test_fold) %>% pull(ID)
    train_samples = filter(final_id_fold, fold %in% train_folds) %>% pull(ID)
    stopifnot(length(test_samples) + length(train_samples) == n_all_samples)
    stopifnot(length(union(test_samples, train_samples)) == n_all_samples)
    
    fold_idx = as.numeric(test_fold) -1 
    train_out_file = sprintf('%s/samples_train%s.pkl', out_dir, fold_idx)
    print(sprintf('Writing train samples to %s', train_out_file))
    py_save_object(train_samples, train_out_file, pickle = "pickle")
    
    test_out_file = sprintf('%s/samples_test%s.pkl', out_dir, fold_idx)
    print(sprintf('Writing test samples to %s', test_out_file))
    py_save_object(test_samples, test_out_file, pickle = "pickle")
  }
  return(final_id_fold)
}


cv_splits_out_dir = '/omics/odcf/analysis/OE0540_projects/ukbb/exomes/vcf/preprocessed/cv_splits_eva/cv_splits_related_in_same_fold'

assignSamplesToFolds(n_folds = 10, cv_splits_out_dir = cv_splits_out_dir)
assignSamplesToFolds(n_folds = 5, cv_splits_out_dir = cv_splits_out_dir)