# Connect to DB and table
import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.tree import DecisionTreeRegressor


import tensorflow as tf
from tensorflow import keras

tree_reg = DecisionTreeRegressor()
encoder = OneHotEncoder()
label_encoder = LabelEncoder()
lin_reg = LinearRegression()


conn = psycopg2.connect(database="postgres", user="postgres", host="localhost", password="postgres")

curs = conn.cursor()
print("Postgres Version:{}".format(curs.execute('SELECT version()')))

db_version = curs.fetchone()
print(db_version)
# curs.execute("SELECT * FROM freedom LIMIT 10;")
# print(curs.fetchall())
# print(curs.fetchall())

select = "SELECT * FROM freedom, birth WHERE freedom.ISO = birth.code"

curs.execute(select)

result = curs.fetchall()
# print(result.shape)
# print(result)
columns = ["year","country", "iso", "ef", "rank", "fr_country", "fr_iso", \
            "f1970","f1971","f1972","f1973","f1974","f1975","f1976","f1977","f1978","f1979","f1980","f1981","f1982", \
            "f1983","f1984","f1985","f1986","f1987","f1988","f1989","f1990","f1991","f1992","f1993","f1994","f1995",\
            "f1996","f1997","f1998","f1999","f2000","f2001","f2002","f2003","f2004","f2005","f2006","f2007","f2008", \
            "f2009","f2010","f2011","f2012","f2013","f2014","f2015","f2016","i1970","i1971","i1972","i1973","i1974",\
            "i1975","i1976","i1977","i1978","i1979","i1980","i1981","i1982","i1983","i1984","i1985","i1986","i1987",\
            "i1988","i1989","i1990","i1991","i1992","i1993","i1994","i1995","i1996","i1997","i1998","i1999","i2000",\
            "i2001","i2002","i2003","i2004","i2005","i2006","i2007","i2008","i2009","i2010","i2011","i2012","i2013","i2014","i2015","i2016"]

print(len(columns))
data = pd.DataFrame(result, columns=columns)
# print(data)
# Remove unwanted columns
formated_columns = ["year", "iso", "country", "ef", "rank", "fertility", "income"]
formated_tuples = []
for row in data.itertuples():
    # print(row)
    # print("////////////////////////////////////////")
    year = row.year
    fertitlity_str = "f" + str(year)
    income_str = "i" + str(year)

    # print(getattr(row, fertitlity_str))  
    formated_tuples.append((row.year, row.iso, row.country, row.ef, row.rank,getattr(row, fertitlity_str), getattr(row, income_str) ))
    # print(year)

formated_data = pd.DataFrame(formated_tuples, columns= formated_columns)
formated_data.drop("year", axis = 1, inplace=True)
formated_data.drop("income", axis = 1, inplace =True)

print(formated_data)

economy_country = formated_data["country"]

formated_data["country"] = label_encoder.fit_transform(economy_country)
formated_data["iso"] = label_encoder.fit_transform(formated_data["iso"])


train_set, test_set = train_test_split(formated_data, test_size = 0.2, random_state = 42)


economy = train_set.drop("fertility", axis = 1)
labels = train_set["fertility"].copy()

test_economy = test_set.drop("fertility", axis = 1)
test_labels = test_set["fertility"].copy()



print(economy)
print(labels)

lin_reg.fit(economy, labels)
# tree_reg.fit(economy, labels)

some_data = economy.iloc[:5]
some_labels = labels.iloc[:5]

print("Predictions:", lin_reg.predict(some_data))
print("Labels:", list(some_labels))

model = keras.models.Sequential([
    keras.layers.Dense(64, input_shape =[len(economy.keys())], activation = "relu", kernel_initializer='normal'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(40,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile (loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=keras.metrics.RootMeanSquaredError())
histroy = model.fit(economy, labels, epochs=3500, validation_split= 0.2, batch_size=20)

model.evaluate(test_economy, test_labels)
x_new = test_economy[:10]
print(x_new)

y_new = test_labels[:10]
result = model.predict(x_new).flatten()

print("Predictions:", result)
print("Labels:", list(y_new))
model.save("Database.h5")


# Get Table data to Pandas dataframe
curs.close()