# Flash Cross-entropy

A small and not really optimized implementation of memory efficient cross-entropy loss.
The overarching idea is similar to that of Flash-Attention and similar methods, which in essence
is the same idea underlying mapreduce, namely monoidal/semigroup-folds. (See the PDF for details)

A jupyter notebook containing some testing, validation, and use is also included. 

This implementation is done directly in pytorch and has been tested on CPUs, GPU kernel implementation
left as an excersise to the reader ;)


