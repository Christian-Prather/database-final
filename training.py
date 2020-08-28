# Connect to DB and table
import psycopg2

conn = psycopg2.connect(database="postgres", user="postgres", host="localhost", password="postgres")

curs = conn.cursor()
print("Postgres Version:{}".format(curs.execute('SELECT version()')))

db_version = curs.fetchone()
print(db_version)
curs.execute("SELECT * FROM freedom LIMIT 10;")
print(curs.fetchall())
# print(curs.fetchall())


# Get Table data to Pandas dataframe
curs.close()