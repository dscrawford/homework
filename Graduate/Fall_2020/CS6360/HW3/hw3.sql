/* Create by Daniel Crawford
 * dsc160130
 * CS6364 - Database Systems
 */

/* Book Database */

CREATE TABLE book (
       book_id		integer,
       title		char(50),
       publisher_name	char(50),
       primary key (book_id)
);

CREATE TABLE book_authors (
       book_id		  integer,
       author_name	  char(50),
       primary key (book_id, author_name)
);

CREATE TABLE publisher (
       name  	       char(50),
       address	       char(50),
       phone	       varchar(10),
       primary key (name)
);

CREATE TABLE book_copies (
       book_id		 integer,
       branch_id	 integer,
       no_of_copies	 integer,
       primary key (book_id, branch_id)
);

CREATE TABLE book_loans (
       book_id		integer,
       branch_id	integer,
       card_no		integer,
       date_out		date,
       due_date		date,
       return_date	date DEFAULT NULL,
       primary key (book_id, branch_id, card_no)
);

CREATE TABLE library_branch (
       branch_id	    integer,
       branch_name	    char(50),
       address		    char(50),
       primary key (branch_id)
);

CREATE TABLE borrower (
       card_no	      integer,
       name	      char(50),
       address	      char(50),
       phone	      varchar(10),
       primary key (card_no)
);


ALTER TABLE book ADD CONSTRAINT fkpub FOREIGN KEY(Publisher_name) REFERENCES PUBLISHER(name) ON DELETE SET NULL;
ALTER TABLE book_authors ADD CONSTRAINT  fkbook FOREIGN KEY(book_id) REFERENCES BOOK(book_id) ON DELETE CASCADE;
ALTER TABLE book_copies ADD CONSTRAINT fkbookcopy FOREIGN KEY(book_id) REFERENCES BOOK(book_id) ON DELETE CASCADE;
ALTER TABLE book_copies ADD CONSTRAINT fkbranchcopy FOREIGN KEY(branch_id) REFERENCES LIBRARY_BRANCH(branch_id) ON DELETE CASCADE;
ALTER TABLE book_loans ADD CONSTRAINT fkloanbranch FOREIGN KEY(branch_id) REFERENCES LIBRARY_BRANCH(branch_id) ON DELETE CASCADE;
ALTER TABLE book_loans ADD CONSTRAINT fkloanbook FOREIGN KEY(book_id) REFERENCES BOOK(book_id) ON DELETE CASCADE;
ALTER TABLE book_loans ADD CONSTRAINT fkloancard FOREIGN KEY(card_no) REFERENCES BORROWER(card_no) ON DELETE CASCADE;

/* Question 1-a */
select dname, count(ssn)
from (department D join employee E on D.dno=E.dno)
group by dname
having avg(salary) > 30000;

/* Question 1-b */
select dname, count(ssn)
from (department D join employee E on D.dno=E.dno)
where gender='M'
group by dname
having avg(salary) > 30000;

/* Question 1-c */
select fname, lname
from employee
where dno in (select dno
  			  from employee
  			  where salary in (select max(salary) from employee));

/* Question 1-d */
select fname, lname, salary
from employee
where salary >= (select distinct salary + 10000
  				 from employee
  				 where salary in (select min(salary) from employee));

/* Question 1-e */
select ssn, salary, dno
from employee E
where 1 < (select count(*) from dependent where essn=ssn)
and (dno, salary) in (select dno, min(salary) as salary from employee group by dno);

/* Question 2-a */
create view dpt_mgr as
select dname, fname, lname, salary
from department D, employee E
where D.mgrssn=E.ssn;

/* Question 2-b */
create view dept_mgr_stats as
select f1.dname, f1.fname, f1.lname, employee_count, project_count
from (select D.dno, dname, fname, lname
	  from employee E, department D
	  where D.mgrssn=E.ssn) f1, 
	 (select D.dno, count(E.ssn) as employee_count 
      from employee E, department D 
      where D.dno=E.dno 
      group by D.dno) f2,
	 (select D.dno, count(pno) as project_count
      from (department D join project P on D.dno=P.dno)
	  group by D.dno) f3
where f1.dno=f2.dno and f2.dno=f3.dno;

/* Question 2-c */
create view project_stats as
select pname, dname, employee_count, hours_worked
from project P, department D,
	(select p.pno, count(ssn) as employee_count, sum(hours) as hours_worked
	 from (works_on W join project P on P.pno=W.pno)
	 group by P.pno) f1
where P.dno=D.dno and f1.pno=P.pno;

/* Question 2-d */
create view project_stats_sequel as
select pname, dname, employee_count, hours_worked
from project P, department D,
	(select p.pno, count(ssn) as employee_count, sum(hours) as hours_worked
	 from (works_on W join project P on P.pno=W.pno)
	 group by P.pno) f1
where P.dno=D.dno and f1.pno=P.pno and employee_count > 1;

/* Question 2-e */
create view employee_dept_info as
select E.fname, E.lname, E.salary, dept_salary_avg, dept_mgr_fname, dept_mgr_lname
from employee E, department D, 
  (select D.dno, avg(salary) as dept_salary_avg
  from (employee E join department D on E.dno=D.dno)
  group by D.dno) f1,
  (select fname as dept_mgr_fname, lname as dept_mgr_lname, D.dno
   from employee E, department D
   where E.dno=D.dno and E.ssn=D.mgrssn) f2
where E.dno = D.dno and E.dno=f1.dno and E.dno=f2.dno;
