CREATE TABLE Car
(
  lisc_plate_no VARCHAR(9) NOT NULL,
  color VARCHAR(20) NOT NULL,
  make VARCHAR(20) NOT NULL,
  model VARCHAR(20) NOT NULL,
  vehicle_class CHAR(20) NOT NULL,
  model_year DATE NOT NULL,
  PRIMARY KEY (lisc_plate_no)
);

CREATE TABLE Credit_Card
(
  credit_no INTEGER NOT NULL,
  name_on_card VARCHAR(40) NOT NULL,
  security_no INTEGER NOT NULL,
  expiration_date DATE NOT NULL,
  zip_code INTEGER NOT NULL,
  PRIMARY KEY (credit_no)
);

CREATE TABLE Customer
(
  f_name VARCHAR(20) NOT NULL,
  l_name VARCHAR(20) NOT NULL,
  phone_no VARCHAR(10) NOT NULL,
  credit_no INTEGER NOT NULL,
  PRIMARY KEY (phone_no)
);

CREATE TABLE Employee
(
  f_name VARCHAR(20) NOT NULL,
  l_name VARCHAR(20) NOT NULL,
  years_in_uber INTEGER NOT NULL,
  employee_SSN INTEGER NOT NULL,
  salary FLOAT NOT NULL,
  department_name VARCHAR(20) NOT NULL,
  PRIMARY KEY (employee_SSN)
);

CREATE TABLE Driver
(
  f_name VARCHAR(20) NOT NULL,
  l_name VARCHAR(20) NOT NULL,
  years_in_uber INTEGER NOT NULL,
  driver_SSN INTEGER NOT NULL,
  has_done_safety_screening CHAR(1) NOT NULL,
  lisc_plate_no VARCHAR(9) NOT NULL,
  department_name VARCHAR(20) NOT NULL,
  PRIMARY KEY (driver_SSN)
);

CREATE TABLE Ride
(
  start_time DATE NOT NULL,
  end_time DATE NOT NULL,
  cost FLOAT NOT NULL,
  tip FLOAT NOT NULL,
  driver_SSN INTEGER NOT NULL,
  cust_phone_no VARCHAR(10) NOT NULL,
  PRIMARY KEY (start_time, driver_SSN)
);

CREATE TABLE Pool
(
  start_time DATE NOT NULL,
  end_time DATE NOT NULL,
  driver_SSN INTEGER NOT NULL,
  PRIMARY KEY (start_time, driver_SSN)
);

CREATE TABLE Department
(
  department_name VARCHAR(20) NOT NULL,
  state CHAR(2) NOT NULL,
  city VARCHAR(20) NOT NULL,
  company_name VARCHAR(20) NOT NULL,
  manager_SSN INTEGER NOT NULL,
  PRIMARY KEY (department_name)
);

CREATE TABLE Company
(
  company_name VARCHAR(20) NOT NULL,
  phone_no VARCHAR(10) NOT NULL,
  main_dept_name VARCHAR(20) NOT NULL,
  PRIMARY KEY (company_name)  
);

CREATE TABLE Driver_Languages
(
  language_spoken VARCHAR(20) NOT NULL,
  driver_SSN INTEGER NOT NULL,
  PRIMARY KEY (language_spoken, driver_SSN)
);

CREATE TABLE Review
(
  cust_review INTEGER NOT NULL,
  driver_review INTEGER NOT NULL,
  driver_SSN INTEGER NOT NULL,
  cust_phone_no VARCHAR(10) NOT NULL,
  PRIMARY KEY (driver_SSN, cust_phone_no)  
);

CREATE TABLE Customer_Pool
(
  cost FLOAT NOT NULL,
  tip FLOAT NOT NULL,
  cust_phone_no VARCHAR(10) NOT NULL,
  start_time DATE NOT NULL,
  driver_SSN INTEGER NOT NULL,
  PRIMARY KEY (cust_phone_no, start_time, driver_SSN)
);