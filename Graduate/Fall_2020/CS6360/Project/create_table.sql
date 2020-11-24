CREATE TABLE Car
(
  lisc_plate_no INT NOT NULL,
  color INT NOT NULL,
  make INT NOT NULL,
  model CHAR NOT NULL,
  vehicle_class CHAR(20) NOT NULL,
  model_year DATE NOT NULL,
  PRIMARY KEY (lisc_plate_no)
);

CREATE TABLE Credit_Card
(
  credit_no INT NOT NULL,
  name_on_card INT NOT NULL,
  security_no INT NOT NULL,
  expiration_date INT NOT NULL,
  zip_code INT NOT NULL,
  PRIMARY KEY (credit_no)
);

CREATE TABLE Customer
(
  f_name INT NOT NULL,
  l_name INT NOT NULL,
  phone_no INT NOT NULL,
  credit_no INT NOT NULL,
  PRIMARY KEY (phone_no),
  FOREIGN KEY (credit_no) REFERENCES Credit_Card(credit_no)
);

CREATE TABLE Employee
(
  f_name VARCHAR(20) NOT NULL,
  l_name VARCHAR(20) NOT NULL,
  years_in_uber INT NOT NULL,
  employee_SSN INT NOT NULL,
  salary INT NOT NULL,
  department_name VARCHAR(20) NOT NULL,
  PRIMARY KEY (employee_SSN),
  FOREIGN KEY (department_name) REFERENCES Department(department_name)
);

CREATE TABLE Driver
(
  f_name INT NOT NULL,
  l_name INT NOT NULL,
  years_in_uber INT NOT NULL,
  driver_SSN INT NOT NULL,
  has_done_safety_screening INT NOT NULL,
  lisc_plate_no INT NOT NULL,
  department_name VARCHAR(20) NOT NULL,
  PRIMARY KEY (driver_SSN),
  FOREIGN KEY (lisc_plate_no) REFERENCES Car(lisc_plate_no),
  FOREIGN KEY (department_name) REFERENCES Department(department_name)
);

CREATE TABLE Ride
(
  start_time INT NOT NULL,
  end_time INT NOT NULL,
  cost FLOAT NOT NULL,
  tip FLOAT NOT NULL,
  driver_SSN INT NOT NULL,
  cust_phone_no INT NOT NULL,
  PRIMARY KEY (start_time, driver_SSN),
  FOREIGN KEY (driver_SSN) REFERENCES Driver(driver_SSN),
  FOREIGN KEY (cust_phone_no) REFERENCES Customer(phone_no)
);

CREATE TABLE Pool
(
  start_time DATE NOT NULL,
  end_time DATE NOT NULL,
  driver_SSN INT NOT NULL,
  PRIMARY KEY (start_time, driver_SSN),
  FOREIGN KEY (driver_SSN) REFERENCES Driver(driver_SSN)
);

CREATE TABLE Department
(
  department_name VARCHAR(20) NOT NULL,
  state CHAR(2) NOT NULL,
  city VARCHAR(20) NOT NULL,
  company_name VARCHAR(20) NOT NULL,
  manager_SSN INT NOT NULL,
  PRIMARY KEY (department_name),
  FOREIGN KEY (company_name) REFERENCES Company(company_name),
  FOREIGN KEY (manager_SSN) REFERENCES Employee(employee_SSN)
);

CREATE TABLE Company
(
  company_name VARCHAR(20) NOT NULL,
  phone_no INT NOT NULL,
  main_dept_name VARCHAR(20) NOT NULL,
  PRIMARY KEY (company_name),
  FOREIGN KEY (main_dept_name) REFERENCES Department(department_name)
);

CREATE TABLE Driver_Languages
(
  language_spoken INT NOT NULL,
  driver_SSN INT NOT NULL,
  PRIMARY KEY (language_spoken, driver_SSN),
  FOREIGN KEY (driver_SSN) REFERENCES Driver(driver_SSN)
);

CREATE TABLE Review
(
  cust_review INT NOT NULL,
  driver_review INT NOT NULL,
  driver_SSN INT NOT NULL,
  cust_phone_no INT NOT NULL,
  PRIMARY KEY (driver_SSN, cust_phone_no),
  FOREIGN KEY (driver_SSN) REFERENCES Driver(driver_SSN),
  FOREIGN KEY (cust_phone_no) REFERENCES Customer(phone_no)
);

CREATE TABLE Customer_Pool
(
  cost FLOAT NOT NULL,
  tip FLOAT NOT NULL,
  phone_no INT NOT NULL,
  start_time DATE NOT NULL,
  driver_SSN INT NOT NULL,
  PRIMARY KEY (phone_no, start_time, driver_SSN),
  FOREIGN KEY (phone_no) REFERENCES Customer(phone_no),
  FOREIGN KEY (start_time, driver_SSN) REFERENCES Pool(start_time, driver_SSN)
);
