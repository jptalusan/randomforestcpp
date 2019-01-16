cd randomforestcpp  
cmake .  
cd src  
make  

To train:  
./rf_exe train [all|LABELNAME] -t [NUMBER OF TREES] -c [JSON CONFIG FILE] -i [CSV FILE]  

To test:
./rf_exe classify [all|LABELNAME] -i [CSV FILE]  

output:  
CSV file with the following headers  
labelname,[TREE DECISIONS in array]

