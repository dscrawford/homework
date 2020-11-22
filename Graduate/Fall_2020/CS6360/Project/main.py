from faker import Faker

import pandas as pd
import sqlalchemy
from os.path import exists
from numpy.random import randint, uniform, normal

fk = Faker()

DB_NAME = 'uber.db'

engine = sqlalchemy.create_engine('sqlite:///' + DB_NAME)
metadata = sqlalchemy.MetaData()

if not exists(DB_NAME):
    with engine.connect() as con:
        sql_commands = open('Generated_SQL_from_EDRPlus.txt').read().split(';')
        for line in sql_commands:
            con.execute(line)

        metadata.create_all(engine)

inspector = sqlalchemy.inspect(engine)


def generate_random_data(col_name, fk, input_time):
    if col_name == 'lisc_plate_no':
        return fk.license_plate()
    elif col_name == 'color':
        return fk.color_name()
    elif col_name == 'make':
        return ['Honda', 'BMW', 'Lexus', 'Mazda', 'Nissan'][randint(0, 5)]
    elif col_name == 'model':
        return ['CX30', 'GTX50', 'V20', 'HyperDrive', 'Stuff'][randint(0, 5)]
    elif col_name == 'vehicle_class':
        return ['Two-Door', 'Four-Door', 'SUV', 'Minivan', 'Sport'][randint(0, 5)]
    elif col_name == 'model_year':
        return randint(1980, 2020)
    elif col_name == 'phone_no':
        s = ''
        for _ in range(10):
            s += str(randint(0, 9))
        return s
    elif col_name == 'credit_no':
        return fk.credit_card_number()
    elif col_name == 'name_on_card':
        return fk.name()
    elif col_name == 'security_no':
        s = ''
        for _ in range(3):
            s += str(randint(0, 9))
        return s
    elif col_name == 'expiration_date':
        if uniform(0, 1) < 0.01:
            return fk.date_between('today', '-1y')
        else:
            return fk.date_between('today', '+10y')
    elif col_name == 'zip_code':
        return fk.zip()
    elif col_name == 'f_name':
        fk.first_name()
    elif col_name == 'l_name':
        fk.last_name()
    elif col_name == 'cost':
        return max(0, normal(10, 5))
    elif col_name == 'tip':
        return max(0, normal(3, 10))
    elif col_name == 'start_time':
        return fk.date_time_between(end_date='now')
    elif col_name == 'end_time':
        return fk.date_time_between(end_date='+15m')
    elif 'ssn' in col_name.lower():
        s = ''
        for _ in range(9):
            s += str(randint(0, 9))
        return s
    elif col_name == 'department_name':
        return ['ML', 'Driver', 'Software', 'Accounting', 'IT'][randint(0,5)]
    elif col_name == 'state':
        return fk.state()
    elif col_name == 'city':
        return fk.city()
    elif col_name == 'company_name':
        return 'Uber'
    elif col_name == 'years_in_uber':
        return randint(0, 10)
    elif col_name == 'salary':
        return normal(80000, 20000)





for table in inspector.get_table_names():
    for column in inspector.get_columns(table):
        generate_random_data(column['name'], fk)

# print(sql_file)
#
# fk = Faker()
#
# print('fk.name()')
