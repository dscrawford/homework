oddMultOf3(X) :-
    (   integer(X)
    ->  0 is (X mod 3), 1 is (X mod 2) %Is divisible by 3 and is not even
    ;   write('ERROR: The given parameter is not an integer\n'), !).


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


/*
classify_helper([], [], []).
classify_helper([H|T], [H|Even], Odd) :-
    0 is H mod 2,
    classify_helper(T, Even, Odd), !.
classify_helper([H|T], Even, [H|Odd]) :-
    1 is H mod 2,
    classify_helper(T, Even, Odd), !.
*/

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

isCompleteSublist([], [_|_]).
isCompleteSublist([H|T],[H|T1]) :-
    isCompleteSublist(T,T1).

subslice(List1,List2) :-
    length(List1, 0), !;
    [_|T1] = List2,
    (   append(List1,_,List2)
    ->  true
    ;   subslice(List1, T1)).


concat([],X,X).
concat([H|T],X,[H|T1]) :-
    concat(T,X,T1).

shift_1([H|T], List) :-
    concat(T, [H], List).

shift_helper(_,0,_).
shift_helper(List,Integer,Shifted) :-
    X is Integer - 1,
    shift_1(List,Shifted),
    shift_helper(Shifted,X,_).
    
shift(List,Integer,Shifted) :-
    (   Integer => 0
    ->  length(List,Len),X is Len - Integer
    ;   X is Integer),
    shift_helper(List,X,Shifted), !.
    
    
