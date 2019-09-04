oddMultOf3(X) :-
    (   integer(X)
    ->  0 is (X mod 3), 1 is (X mod 2) %Is divisible by 3 and is not even
    ;   format('ERROR: The given parameter is not an integer\n'), fail).


list_prod_helper([], 0).
list_prod_helper([Z], Z).
list_prod_helper([X|Y], Product) :-
    list_prod_helper(Y, Z), Product is X * Z.
list_prod(List, Number) :-
    list_prod_helper(List, Number), !.


palindrome(List) :-
    reverse(List,List).

insert(X1, [], [X1]) :- !.
insert(X1, [X2|L], [X1, X2|L]):- X1 =< X2, !.
insert(X1, [X2|L2], [X2|L1]):- insert(X1, L2, L1).

mySort([], []) :- !.
mySort([X|L], S):- mySort(L, S1), insert(X, S1, S).

isListOfOneValue_Helper([], _).
isListOfOneValue_Helper([H|T], X) :-
    (   H = X
    ->  isListOfOneValue_Helper(T, X)
    ;   false).
    
isListOfOneValue([H|T]) :-
    isListOfOneValue_Helper(T, H).
    
getSecondDistinct_Helper([], _, _).
getSecondDistinct_Helper([Min2|T], X, Min2) :-
    (   X = Min2
    ->  getSecondDistinct_Helper(T, X, Min2)
    ;   true).
    
getSecondDistinct([H|T], Min2) :-
    getSecondDistinct_Helper(T, H, Min2).

myNumberListPrint([]).
myNumberListPrint([H|T]) :-
    number(H), myNumberListPrint(T), !.
myNumberListPrint([H|_]) :-
    \+ number(H), write(H).

secondMin(List, Min2) :-
    \+ List = [],
    (   maplist(number, List)
    ->  (   \+ isListOfOneValue(List)
    	->   mySort(List, Sorted), getSecondDistinct(Sorted,Min2)
    	;   format("ERROR: List has fewer than two unique elements.\n"), fail)
    ;   format("ERROR: \""), myNumberListPrint(List), format("\" is not a number"), nl, fail).

classify(List,Even,Odd) :-
    (   length(List,0)
    ->  List = Even, List = Odd
    ;   [H|T] = List,
        (   0 is H mod 2
    	->  [H|T1] = Even, classify(T, T1, Odd)
    	;   [H|T1] = Odd,  classify(T, Even, T1))).
    

bookends(List1, List2, List3) :-
    append(List1,_,List3), %List3 is the concatenation of List1 + Some list
    append(_,List2,List3), %List3 is the concatenation of Some list + List2
    !.

subslice(List1,List2) :-
    length(List1, 0), !;
    [_|T1] = List2,
    (   append(List1,_,List2)
    ->  true
    ;   subslice(List1, T1)).
    
shift(List,Integer,Shifted) :-
    (   Integer < 0
    ->  Z is abs(Integer),
    	length(B, Z),
    	append(A, B, List),
		append(B, A, Shifted)
    ;  	length(B, Integer),
	    append(B, A, List),
		append(A, B, Shifted)), !.


splitSum(Integer, Result) :-
    X is Integer mod 10,
    Y is div(Integer,10),
    Result is X+Y.

sumOfTwo(Integer, Result) :-
    X is Integer mod 10,
    Y is div(Integer,10),
    Z is Y * 2,
    (   Z > 9
    ->  splitSum(Z,R1), Result is R1 + X
    ;   Result is X + Z).

sumOfDigits(0,Result,Result).
sumOfDigits(Integer, Sum, Result) :-
    X is Integer mod 100,
    Y is div(Integer,100),
    sumOfTwo(X, R1),
    Z is Sum + R1,
    sumOfDigits(Y,Z,Result),!.

luhn(Integer) :-
    sumOfDigits(Integer,0,Result),
    R is Result mod 10,
    R = 0.


%Knowledge Base

edge(a,b).
edge(b,c).
edge(c,d).
edge(d,a).
edge(d,e).
edge(b,a).

path_helper(A,B,List) :-
    edge(A,B);
    edge(A,C), \+ member([A,C], List),
    append([ [A,C] ], List, NewList),
    path_helper(A,C,NewList).

path(A,B) :-
    path_helper(A,B,[]), !.

cycle(A) :-
    path(A,A).
