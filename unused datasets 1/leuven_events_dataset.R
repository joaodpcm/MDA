library(readxl)
events<-read_xlsx("C:/Users/Student/dwhelper/MDA assignment/filter_date_tags_city.xlsx")
head(events)
unique(events$startTime)
nrow(events)
length(events$tags)

remove_last_n <- 6
events$startTime <- substr(events$startTime, 1, nchar(events$startTime) - remove_last_n)

noise<-read.csv("C:/Users/Student/dwhelper/MDA assignment/noise_data_complete.csv",sep=',')
head(noise)

#rename variable of noise
names(noise)[names(noise) == "result_timestamp"] <- "startTime"

#merge both the datasets
events_noise <- merge(events, noise, by = "startTime", all = TRUE)
head(events_noise)



#create a new variable event_yes
events_noise$event_yes <- ifelse(events_noise$tags == "NA", 0, 1)
head(events_noise)
unique(events_noise$event_yes)

# Replace NA values with 0
events_noise$event_yes[is.na(events_noise$event_yes)] <- 0

#tags categorise
unique(events_noise$tags)


#export dataset
library(openxlsx)
# Specify the file path and name for the Excel file
file_path <- "C:/Users/Student/dwhelper/MDA assignment/events_noise2.xlsx"
write.xlsx(events_noise, file = file_path, row.names = FALSE)
