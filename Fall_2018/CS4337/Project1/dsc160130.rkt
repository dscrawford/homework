#lang racket

#|
Made by Daniel Crawford on Friday September 28th, 2018
Email: dsc160130@utdallas.edu
|#

(provide (all-defined-out))

(define divisible-by-x?
  (lambda (x)
    (lambda (y) (zero? (remainder y x)))))

(define function-9
  (lambda (x)
    (x 9)))

(define my-map
  (lambda (x lst)
    (define map-helper
      (lambda (lst acc)
        (if (null? lst)
            (reverse acc)
            (map-helper (cdr lst) (cons (x (car lst)) acc)))))
    (map-helper lst '())))

(define combine
  (lambda (lst lst2)
    (define combine-helper
      (lambda (lst acc lst2)
        (if (or (null? lst) (null? lst2))
          (reverse acc)
          (combine-helper (cdr lst)
                          (cons (cons (car lst) (cons (car lst2) '())) acc)
                          (cdr lst2)))))
    (combine-helper lst '() lst2)))

(define segregate
  (lambda (x lst)
    (define segregate-helper
      (lambda (lst acc acc2)
        (if (null? lst)
            (cons (reverse acc) (list (reverse acc2)))
            (if (x (car lst))
                (segregate-helper (cdr lst) (cons (car lst) acc) acc2)
                (segregate-helper (cdr lst) acc (cons (car lst) acc2))))))
    (segregate-helper lst '() '())))

(define is-member?
  (lambda (x lst)
    (if (null? lst)
        false
        (if (equal? (car lst) x)
            true
            (is-member? x (cdr lst))))))

(define my-sorted?
  (lambda (x lst)
    (if (eq? (length lst) 1) ;if length is 1, return true
        #t
        (if (x (car lst) (car (cdr lst))) ;if current element R next element == true, reiterate
            (my-sorted? x (cdr lst))
            #f))))

(define my-flatten
  (lambda (lst)
    (if (null? lst) ;if null
        '()
        (if(pair? lst) ;if list is a pair or list
           (append (my-flatten (car lst)) (my-flatten (cdr lst))) ;Append the nested list to the previous list
           (list lst))))) ; since using append


(define upper-threshold
  (lambda (lst x)
    (define upper-threshold-helper
      (lambda (lst x acc)
        (if (null? lst) ;if null list, return
            (reverse acc)
            (if (< (car lst) x) ;if lesser than threshold, otherwise continue
                (upper-threshold-helper (cdr lst) x (cons (car lst) acc))
                (upper-threshold-helper (cdr lst) x acc)))))
    (upper-threshold-helper lst x '())))

(define my-list-ref
  (lambda (lst index)
    (define my-list-ref-helper
      (lambda (lst i)
        (if (null? lst)
            (error "ERROR: Index out of bounds")
            (if (eq? i index)
                (car lst)
                (my-list-ref-helper (cdr lst) (add1 i))))))
    (my-list-ref-helper lst 0)))

(define deep-reverse
  (lambda (lst)
    (define deep-reverse-helper
      (lambda (lst acc)
        (if (null? lst)
            acc
            (if(list? (car lst))
               (deep-reverse-helper (cdr lst) (cons (deep-reverse-helper (car lst) '()) acc))
               (deep-reverse-helper (cdr lst) (cons (car lst) acc))))))
    (deep-reverse-helper lst '())))