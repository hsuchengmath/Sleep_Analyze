#  Sleep Analyze 

### Description
We want to build 'learning to rank model' for all insomnia patients





### Required data format
1. data is .csv 
2. The first column is userId and other columns is feature data
3. For all users ,their experiment day need to be same
4. For example (user1 and user2 , Experiment day == 9day)


![](https://github.com/hsuchengmath/Sleep_Analyze/blob/master/sample_data.png)




### Required packages
The code has been tested running under Python 3.6.6, with the following packages installed (along with their dependencies):

- numpy == 1.16.0
- pandas == 0.23.4
- pyprind == 2.11.2
- tensorflow == 1.12.0
- scikit-learn == 0.20.2

### Running the code
```
$ python main_1.py --If your data exist missing values ,you need to run main_1.py for imputing it 
                     (Notice :  You save impute_data_stanard by .csv)
$ python main_2.py --Loading you just saved csv file or original csv file(not exist missing value) and run main_2.py and                 
                     you'll get user's rank!!

```
