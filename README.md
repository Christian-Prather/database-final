# database-final

This is a DB final project for Database Managment class. The project is an attempt to see if there is a correleation between contry's economic freedom and their birth rate. It utilizes a CNN for model training. The two data sets are pulled as CSV files from keggle.com

## Data Pipeline
### Clean up
The data is not ideal for ML or Database storage so it must first be cleaned up using clean-up.py, the data is loaded from CSV's and saved out as CSV's parsed and ideal for loading

### SQL Loading 
The data is then loaded to a postgresql DB using the data_laod.sql script

### Machine Learning 
Data is fed into tensorflow API using Pandas to load the SQL data into datframes. 

## Training
Once the data is loaded into a DB (if local with default settings)
you shouled be able to run `python3 training.py` to train the model