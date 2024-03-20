gcc -O3 -c -Wall -Werror -fpic c.c -lpthread
gcc -shared -o lib.so c.o
rm c.o