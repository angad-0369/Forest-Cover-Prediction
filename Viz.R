library(tidyverse)
library(skimr)

df <- read.csv("D:\\EDUCATION\\OTHERS\\MACHINE LEARNING\\Forest cover prediction\\covtype.csv")

glimpse(df)
nrow(df) - nrow(distinct(df))
any(is.na(df))
skim(df)

set.seed(1808)

dff <- (df %>%
  group_by(Cover_Type) %>%
  sample_n(size = 1000) %>%
  ungroup() %>%
  gather(Wilderness_Area, Wilderness_Value, 
         Wilderness_Area1:Wilderness_Area4) %>% 
  filter(Wilderness_Value >= 1) %>%
  select(-Wilderness_Value) %>%
  mutate(Wilderness_Area = str_extract(Wilderness_Area, '\\d+'),
         Wilderness_Area = str_replace_all(Wilderness_Area,
                                           c(`1` = 'Rawah',
                                             `2` = 'Neota',
                                             `3` = 'Comanche Peak',
                                             `4` = 'Cache la Poudre')),
         Wilderness_Area = as.factor(Wilderness_Area)) %>%
  gather(Soil_Type, Soil_Value, Soil_Type1:Soil_Type40) %>%
  filter(Soil_Value == 1) %>%
  select(-Soil_Value) %>% 
  mutate(Soil_Type = as.factor(str_extract(Soil_Type, '\\d+'))) %>%
  mutate(Cover_Type = str_replace_all(Cover_Type,
                                      c(`1` = 'Spruce/Fir',
                                        `2` = 'Lodgepole Pine',
                                        `3` = 'Ponderosa Pine',
                                        `4` = 'Cottonwood/Willow',
                                        `5` = 'Aspen',
                                        `6` = 'Douglas Fir',
                                        `7` = 'Krummholz')),
         Cover_Type = as.factor(Cover_Type)) %>%
  select(Cover_Type:Soil_Type, Elevation:Slope,
         Hillshade_9am:Hillshade_3pm, Vertical_Distance_To_Hydrology,
         Horizontal_Distance_To_Hydrology:Horizontal_Distance_To_Fire_Points))

glimpse(dff)
skim(dff)

palette <- c('sienna1', 'chartreuse', 'lightskyblue1', 
             'hotpink', 'mediumturquoise', 'indianred1', 'gold')
ggplot(dff, aes(x = Cover_Type, y = Elevation)) +
  geom_violin(aes(fill = Cover_Type)) + 
  geom_point(alpha = 0.01, size = 0.5) +
  stat_summary(fun = 'median', geom = 'point') +
  labs(x = 'Forest Cover') +
  scale_fill_manual(values = palette) +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.text.x = element_text(angle = 45,
                                   hjust = 1),
        panel.grid.major.x = element_blank())

palette <- c('sienna1', 'chartreuse', 'cadetblue1', 
             'skyblue2', 'lightpink', 'indianred1', 'olivedrab1')
ggplot(dff, aes(x = Cover_Type, fill = Cover_Type)) +
  geom_bar() +
  facet_wrap(~reorder(Soil_Type, sort(as.integer(Soil_Type))), scales = 'free') +
  labs(fill = 'Cover Type', title = 'Cover Type by Soil Type') +
  scale_fill_manual(values = palette) +
  theme_minimal() +
  theme(legend.position = 'bottom', palette <- c('sienna1', 'chartreuse', 'cadetblue1', 
             'skyblue2', 'lightpink', 'indianred1', 'olivedrab1')
ggplot(dff, aes(x = Cover_Type, fill = Cover_Type)) +
  geom_bar() +
  facet_wrap(~reorder(Soil_Type, sort(as.integer(Soil_Type))), scales = 'free') +
  labs(fill = 'Cover Type', title = 'Cover Type by Soil Type') +
  scale_fill_manual(values = palette) +
  theme_minimal() +
  theme(legend.position = 'bottom',
        plot.title = element_text(hjust = 0.5),
        axis.title = element_blank(),
        axis.text = element_blank(),
        panel.grid = element_blank())

ggplot(dff, aes(x = Hillshade_Noon, y = Hillshade_3pm, color = Cover_Type)) + 
  geom_point(alpha = 0.1, size = 0.5) +
  scale_color_manual(values = palette) +
  labs(title = 'Relationship Between Hillshade at Noon and 3 PM') +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')

ggplot(dff, aes(x = Elevation, y = Vertical_Distance_To_Hydrology, color = Cover_Type)) + 
  geom_point(alpha = 0.1, size = 0.5) +
  scale_color_manual(values = palette) +
  labs(title = 'Relationship Between Elevation and Vertical Distance to Hydrology') +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'bottom')