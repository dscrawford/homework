import pandas as pd
import sqlalchemy
import numpy as np
import datetime

from os.path import exists
from os import remove, removedirs, mkdir
from numpy.random import randint, uniform, normal
from faker import Faker

NUMBER_OF_DRIVERS = 300
NUMBER_OF_CUSTOMERS = 1000
NUMBER_OF_EMPLOYEES = 50
# NUMBER_OF_RIDES = 5000
NUMBER_OF_POOLS = 1000
NUMBER_OF_CUSTOMERS_IN_POOLS = 5000

PRIMARY_COMPANY_NAME = 'UBER'



fk = Faker()

DB_NAME = 'uber.db'

engine = sqlalchemy.create_engine('sqlite:///' + DB_NAME)
metadata = sqlalchemy.MetaData()

if exists(DB_NAME):
    remove(DB_NAME)

with engine.connect() as con:
    create_commands = open('create_table.sql').read().split(';')
    for line in create_commands:
        con.execute(line)
    metadata.create_all(engine)

inspector = sqlalchemy.inspect(engine)


def generate_random_data(col_name, fk, input_time=None):
    if col_name == 'lisc_plate_no':
        return fk.license_plate()
    elif col_name == 'color':
        return fk.color_name()
    elif col_name == 'make':
        return ['Honda', 'BMW', 'Lexus', 'Mazda', 'Nissan'][randint(0, 5)]
    elif col_name == 'model':
        return ['CX30', 'GTX50', 'V20', 'HyperDrive', 'Stuff', 'CoolModel', 'ModelX', 'ModelS', 'Mega'][randint(0, 9)]
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
            return fk.date_between('-1y', 'today')
        else:
            return fk.date_between('today', '+10y')
    elif col_name == 'zip_code':
        return fk.zipcode()
    elif col_name == 'f_name':
        return fk.first_name()
    elif col_name == 'l_name':
        return fk.last_name()
    elif col_name == 'cost':
        return max(0, normal(10, 5))
    elif col_name == 'tip':
        return max(0, normal(3, 10))
    elif col_name == 'start_time':
        return fk.date_time_between(start_date='-10y', end_date='now')
    elif col_name == 'end_time':
        return fk.date_time_between(start_date=input_time, end_date='+15m')
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
        return np.round(normal(80000, 20000))
    elif 'review' in col_name:
        return randint(1,6)
    elif col_name == 'language_spoken':
        return fk.language_name()
    else:
        return None


table_dfs = {}
for table in inspector.get_table_names():
    table_dfs[table] = pd.DataFrame(columns=[column['name'] for column in inspector.get_columns(table)])

# Create Company Uber
table_dfs['Company'] = pd.DataFrame([[PRIMARY_COMPANY_NAME, generate_random_data('phone_no', fk), 'Main']],
                                    columns=table_dfs['Company'].columns)
# Create Departments under Uber
table_dfs['Department'] = pd.DataFrame([['ML'] + [generate_random_data(col['name'], fk) for col in inspector.get_columns('Department')[1:-2]] + [PRIMARY_COMPANY_NAME, None],
                                        ['Driver'] + [generate_random_data(col['name'], fk) for col in inspector.get_columns('Department')[1:-2]] + [PRIMARY_COMPANY_NAME, None],
                                        ['Software'] + [generate_random_data(col['name'], fk) for col in inspector.get_columns('Department')[1:-2]] + [PRIMARY_COMPANY_NAME, None],
                                        ['Accounting'] + [generate_random_data(col['name'], fk) for col in inspector.get_columns('Department')[1:-2]] + [PRIMARY_COMPANY_NAME, None],
                                        ['IT'] + [generate_random_data(col['name'], fk) for col in inspector.get_columns('Department')[1:-2]] + [PRIMARY_COMPANY_NAME, None],
                                        ['Main'] + [generate_random_data(col['name'], fk) for col in inspector.get_columns('Department')[1:-2]] + [PRIMARY_COMPANY_NAME, None]],
                                    columns=table_dfs['Department'].columns)

# Create Employees who work for uber
table_dfs['Employee'] = pd.DataFrame(
    np.hstack([[[generate_random_data(col, fk) for col in table_dfs['Employee'].columns[:-1]] for _ in range(NUMBER_OF_EMPLOYEES)],
               [[table_dfs['Department']['department_name'][randint(0, len(table_dfs['Department']))]] for _ in range(NUMBER_OF_EMPLOYEES)]]),
    columns=table_dfs['Employee'].columns
)

# Set Managers of departments
for i, department in enumerate(table_dfs['Department']['department_name'].to_numpy()):
    dept_ssns = table_dfs['Employee'][table_dfs['Employee']['department_name'] == department]['employee_SSN'].to_numpy()
    mgr_ssn = dept_ssns[randint(0, len(dept_ssns))]
    table_dfs['Department']['manager_SSN'][i] = mgr_ssn

# Generate Car Random Data
table_dfs['Car'] = pd.DataFrame([[generate_random_data(col, fk) for col in table_dfs['Car']] for _ in range(NUMBER_OF_DRIVERS)],
                                columns=table_dfs['Car'].columns)

# Generate Driver Random Data
table_dfs['Driver'] = pd.DataFrame(
    [[generate_random_data(col, fk) for col in table_dfs['Driver'].columns[:-2]] + [table_dfs['Car']['lisc_plate_no'][i], 'Driver'] for i in range(NUMBER_OF_DRIVERS)],
    columns=table_dfs['Driver'].columns
)

# Generate Customer Names
customer_names = [[generate_random_data('f_name', fk), generate_random_data('l_name', fk)] for _ in range(NUMBER_OF_CUSTOMERS)]
combined_names = [[name[0] + ' ' + name[1]] for name in customer_names]
credit_nos = [[generate_random_data('credit_no', fk)] for _ in range(NUMBER_OF_CUSTOMERS)]

credit_card_info = np.hstack([
    credit_nos,
    combined_names,
    [[generate_random_data(col, fk) for col in table_dfs['Credit_Card'].columns[2:]] for _ in range(NUMBER_OF_CUSTOMERS)]
])

# Generate Credit Card Random Data
table_dfs['Credit_Card'] = pd.DataFrame(
    credit_card_info,
    columns=table_dfs['Credit_Card'].columns
)

customer_info = np.hstack([
    customer_names,
    [[generate_random_data('phone_no', fk)] for _ in range(NUMBER_OF_CUSTOMERS)],
    credit_nos
])

# Generate Customer Random Data
table_dfs['Customer'] = pd.DataFrame(
    customer_info,
    columns=table_dfs['Customer'].columns
)

# Create Driver Languages
driver_ssns = table_dfs['Driver']['driver_SSN'].to_numpy()
new_language_prob = 0.2
driver_languages = np.transpose([['English' for _ in range(NUMBER_OF_DRIVERS)], driver_ssns])
new_driver_languages = driver_languages
while True:
    new_driver_languages = new_driver_languages[[uniform() < new_language_prob for _ in range(len(new_driver_languages))]]
    if len(new_driver_languages) == 0:
        break
    new_driver_languages[:, :1] = [[generate_random_data('language_spoken', fk)] for _ in range(len(new_driver_languages))]
    driver_languages = np.vstack([driver_languages, new_driver_languages])
driver_languages = np.unique(driver_languages.astype('<U22'), axis=1)

table_dfs['Driver_Languages'] = pd.DataFrame(
    driver_languages,
    columns=table_dfs['Driver_Languages'].columns
)

# Generate Rides
driver_ssns = table_dfs['Driver']['driver_SSN'].to_numpy()
driver_years_in_uber = table_dfs['Driver']['years_in_uber'].to_numpy()
cust_phones = table_dfs['Customer']['phone_no'].to_numpy()

rides = []
for years_in_uber, driver_ssn in zip(driver_years_in_uber, driver_ssns):
    num_drives = max(0, int(normal(20*years_in_uber, 10)))
    end_date = datetime.datetime.today() - datetime.timedelta(days=(365.0 * years_in_uber) + randint(0, 100))
    cost = normal(20, 5)
    tip = max(0, normal(3, 10))
    for i in range(num_drives):
        start_date = end_date + datetime.timedelta(minutes=max(30, normal(30, 20)), days=max(0, normal(2, 10)))
        end_date = start_date + datetime.timedelta(minutes=max(10, normal(30, 20)))
        customer_phone_no = table_dfs['Customer']['phone_no'][randint(0, NUMBER_OF_CUSTOMERS)]
        rides.append([start_date, end_date, cost, tip, driver_ssn, customer_phone_no])

pools = []
for years_in_uber, driver_ssn in zip(driver_years_in_uber, driver_ssns):
    num_drives = max(0, int(normal(10*years_in_uber, 5)))
    end_date = datetime.datetime.today() - datetime.timedelta(days=(365.0 * years_in_uber) + randint(0, 100))
    # cost = max(5, normal(10, 3))
    # tip = max(0, normal(2, 6))
    for i in range(num_drives):
        start_date = end_date + datetime.timedelta(minutes=max(30, normal(30, 20)), days=max(0, normal(2, 10)))
        end_date = start_date + datetime.timedelta(minutes=max(60, normal(60, 30)))
        pools.append([start_date, end_date, driver_ssn])

table_dfs['Ride'] = pd.DataFrame(
    rides,
    columns=table_dfs['Ride'].columns
)

table_dfs['Pool'] = pd.DataFrame(
    pools,
    columns=table_dfs['Pool'].columns
)

customer_pool = []
# Add Customers & Drivers to Pool
for i, ride in table_dfs['Ride'].iterrows():
    number_of_customers = randint(1, 3)
    cust_in_pool = set()
    cust_in_pool.add(None)
    for cust_number in range(number_of_customers):
        cost = max(5, normal(10, 3))
        tip = max(0, normal(2, 6))
        customer_phone_no = None
        while customer_phone_no in cust_in_pool:
            customer_phone_no = table_dfs['Customer']['phone_no'][randint(0, NUMBER_OF_CUSTOMERS)]
        cust_in_pool.add(customer_phone_no)
        customer_pool.append([cost, tip, customer_phone_no, ride['start_time'], ride['driver_SSN']])

table_dfs['Customer_Pool'] = pd.DataFrame(
    customer_pool,
    columns=table_dfs['Customer_Pool'].columns
)

reviews = []
for i, customer in table_dfs['Customer'].iterrows():
    customer_phone_no = customer['phone_no']
    associated_drivers = np.unique(np.hstack([
        table_dfs['Ride'][table_dfs['Ride']['cust_phone_no'] == customer_phone_no]['driver_SSN'].to_numpy(),
        table_dfs['Customer_Pool'][table_dfs['Customer_Pool']['phone_no'] == customer_phone_no]['driver_SSN'].to_numpy()
    ]))
    for driver_ssn in associated_drivers:
        cust_review = randint(1, 6)
        driver_review = randint(1, 6)
        reviews.append([cust_review, driver_review, driver_ssn, customer_phone_no])

table_dfs['Review'] = pd.DataFrame(
    reviews,
    columns=table_dfs['Review'].columns
)

mkdir('data')
for key in table_dfs:
    table_dfs[key].to_csv('./data/' + key + '.csv', index=False)





#
# print(inspector.get_table_names())
#
# fk = Faker()
#
# print('fk.name()')
