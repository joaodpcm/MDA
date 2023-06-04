library(readxl)
events_noise<-read_xlsx("C:/Users/Student/dwhelper/MDA assignment/events_noise2.xlsx")
head(events_noise)
nrow(events_noise)
unique(events_noise$tags)


#########################################################################
## Assuming you have a dataset called 'df' with a character variable called 'var' and you want to extract values containing certain words

# Define the words you want to search for
#1
search_words <- "Culinary"

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values


#2
search_words <- "Cantus"

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values


#4
#quiz in intl
search_words <- "Quiz"

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values

#3
search_words <- "International"

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values


search_words <- c("Charity", "Ukraine")

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values


search_words <- c("Workshop", "Education")

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values

#A 105 international orientation
search_words <- c("International", "First-year", "students")
# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values

#B 149 academic
search_words <- c("Workshop", "Education", "Lecture", "Quiz", "Career")
# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values

#C 236 social
search_words <- c("Party", "Sport", "Cantus", "Culture", "Culinary")
# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values

#D 27 others
search_words <- c("Charity", "Ukraine")
# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values

#E 27 others
search_words <- NA

# Extract values containing the search words
matching_values <- events_noise$tags[grepl(paste(search_words, collapse = "|"), events_noise$tags)]
matching_values
###################################################33

###########################
# Assuming you have a dataset called 'df' with a character variable called 'var' containing multiple values

# Combine all the values into a single string
combined_string <- paste(events_noise$tags, collapse = " ")

# Split the combined string into individual words
words <- strsplit(combined_string, "\\s+")

# Find the unique words
unique_words <- unique(unlist(words))
unique_words
###########################





#@#@##@@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#
# Create the new 'tag_category' variable based on the conditions
events_noise$tag_category <- ifelse(grepl("International|First-year|students", events_noise$tags), "International",
                                    ifelse(grepl("Workshop|Education|Lecture|Quiz|Career", events_noise$tags), "Academic",
                                           ifelse(grepl("Party|Sport|Cantus|Culture|Culinary", events_noise$tags), "Social",
                                                  ifelse(grepl("Charity|Ukraine", events_noise$tags), "Others",
                                                         ifelse(events_noise$tags == "", "NA", NA)))))

# Categorize the 'tag_category' variable as a factor
events_noise$tag_category <- factor(events_noise$tag_category, levels = c("International", "Academic", "Social", "Others", "NA"))

#@#@##@@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#

unique(events_noise$tag_category)







#export dataset
library(openxlsx)
# Specify the file path and name for the Excel file
file_path2 <- "C:/Users/Student/dwhelper/MDA assignment/events_noise_tags_final.xlsx"
write.xlsx(events_noise, file = file_path2, row.names = FALSE)
