/* Created by Daniel Crawford on 2020-11-01
   CS6360 - Database Systems 
*/

/* Question 1 */
CREATE OR REPLACE TRIGGER CHECK_EMP_SALARY
       FOR UPDATE OF salary OR INSERT ON EMPLOYEE
COMPOUND TRIGGER
	TYPE mgr_salary_t 	IS TABLE OF EMPLOYEE.salary%TYPE;
	mgr_salary			mgr_salary_t;
	TYPE mgr_salaries_t IS TABLE OF EMPLOYEE.salary%TYPE INDEX BY VARCHAR(80);
	mgr_salaries		mgr_salaries_t;
	TYPE mgr_ssn_t		IS TABLE OF EMPLOYEE.superssn%TYPE;
	mgr_ssn		mgr_ssn_t;

    BEFORE STATEMENT IS
	BEGIN
            SELECT NVL(E.ssn, -1), E.salary
            BULK COLLECT INTO mgr_ssn, mgr_salary
            FROM EMPLOYEE E
            WHERE E.ssn in (select superssn from employee);

            FOR j IN 1..mgr_ssn.COUNT() LOOP
              mgr_salaries(mgr_ssn(j)) := mgr_salary(j);
            END LOOP;
    END BEFORE STATEMENT;
          
    BEFORE EACH ROW IS
      	BEGIN
          IF :new.salary > mgr_salaries(:new.superssn) THEN
            Raise_Application_Error(-20000, 'Cannot have a salary higher than managers');
          END IF;
    END BEFORE EACH ROW;    	
END CHECK_EMP_SALARY;

/* Question 2 */
CREATE TABLE OVERDUE (
       book_title       char(50),
       borrower_name    char(50),
       borrower_phone   varchar(10),
       due_date         date,
       primary key (book_title, borrower_name)
);

CREATE OR REPLACE PROCEDURE Insert_All_Overdue_Branch(lb_branch_id IN LIBRARY_BRANCH.branch_id%TYPE) AS

thisOverdue OVERDUE%ROWTYPE;

CURSOR overdue_rows IS
SELECT B.title, BOR.Name, BOR.Phone, BL.due_date
FROM BOOK B, BORROWER BOR, BOOK_LOANS BL, LIBRARY_BRANCH LB
WHERE LB.branch_id = lb_branch_id AND LB.branch_id = BL.branch_id AND BL.book_id = B.book_id
      AND BOR.card_no = BL.card_no AND return_date IS NULL AND due_date < SYSDATE;

BEGIN
OPEN overdue_rows;
LOOP
        FETCH overdue_rows INTO thisOverdue;
        EXIT WHEN (overdue_rows%NOTFOUND);

        INSERT INTO OVERDUE VALUES thisOverdue;
END LOOP;
CLOSE overdue_rows;
END;
