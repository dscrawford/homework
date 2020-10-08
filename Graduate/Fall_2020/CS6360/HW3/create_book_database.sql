/* Book Database */

CREATE TABLE book (
       book_id		integer,
       title		char(50),
       publisher_name	char(50),
       primary key (book_id),
)

CREATE TABLE book_authors (
       book_id		  integer,
       author_name	  char(50),
       primary key (book_id, author_name)
)

CREATE TABLE publisher (
       name  	       char(50),
       address	       char(50),
       phone	       varchar(10),
       primary key (name)
)

CREATE TABLE book_copies (
       book_id		 integer,
       branch_id	 integer,
       no_of_copies	 integer,
       primary key (book_id, branch_id)
)

CREATE TABLE book_loans (
       book_id		integer,
       branch_id	integer,
       card_no		integer,
       date_out		date,
       due_date		date,
       return_date	date,
       primary key (book_id, branch_id, card_no)
)

CREATE TABLE library_branch (
       branch_id	    integer,
       branch_name	    char(50),
       address		    char(50),
       primary key (branch_id)
)

CREATE TABLE borrower (
       card_no	      integer,
       name	      char(50),
       address	      char(50),
       phone	      varchar(10),
       primary key (card_no)
)


ALTER TABLE book ADD CONSTRAINT fkpub FOREIGN KEY(Publisher_name) REFERENCES PUBLISHER(name)
ALTER TABLE book_authors ADD CONSTRAINT  fkbook FOREIGN KEY(book_id) REFERENCES BOOK(book_id)
ALTER TABLE book_copies ADD CONSTRAINT fkbookcopy FOREIGN KEY(book_id) REFERENCES BOOK(book_id)
ALTER TABLE book_copies ADD CONSTRAINT fkbranchcopy FOREIGN KEY(branch_id) REFERENCES LIBRARY_BRANCH(branch_id)
ALTER TABLE book_loans ADD CONSTRAINT fkbranch
