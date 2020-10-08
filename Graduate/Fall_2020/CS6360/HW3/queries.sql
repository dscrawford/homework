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

/* Quesiton 1-e */
select ssn, salary, dno
from employee E
where 1 < (select count(*) from dependent where essn=ssn)
and (dno, salary) in (select dno, min(salary) as salary from employee group by dno);
