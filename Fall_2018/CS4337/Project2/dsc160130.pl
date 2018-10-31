oddMultOf3(X) :-
    (   integer(X)
    ->  0 is (X mod 3), \+ 0 is (X mod 6) %Is divisible by 3 and is not even(mod 6)
    ;   write('ERROR: The given parameter is not an integer'), nl).


list_prod_helper([Z], Z).
list_prod_helper([X|Y], Product) :-
    list_prod_helper(Y, Z), Product is X * Z.
list_prod([], 0).
list_prod(X, Product) :-
    list_prod_helper(X, Product),
    format('Product = ~p', Product).
