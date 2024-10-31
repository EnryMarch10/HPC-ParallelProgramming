/****************************************************************************
 *
 * omp-linked-list-traversal.c - Linked List traversal with OpenMP tasks
 *
 * Copyright (C) 2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ----------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -fopenmp omp-linked-list-traversal.c -o omp-linked-list-traversal
 *
 * Run with:
 *
 *      OMP_NUM_THREADS=4 ./omp-linked-list-traversal
 *
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

/* Recursively create a list with nodes numbered n, n+1, ... m; if
   n>m, returns the empry list. */
ListNode *list_create( int n, int m )
{
    if (n > m) {
        return NULL;
    } else {
        ListNode *result = (ListNode*)malloc( sizeof(ListNode) );
        result->val = n;
        result->next = list_create(n+1, m);
        return result;
    }
}

/* Recursively destroy the list pointed to by `l` */
void list_destroy( ListNode *l )
{
    if (l) {
        list_destroy(l->next);
        free(l);
    }
}

int fib(int n)
{
    if (n<=0)
        return 1;
    else
        return fib(n-1) + fib(n-2);
}

void process( ListNode *l )
{
    assert(l);
    printf("%d -> %d\n", l->val, fib(l->val));
}

void list_traverse( ListNode *head )
{
#pragma omp parallel
    {
#pragma omp single
	{
            ListNode *p = head;
            while (p) {
#pragma omp task
                process(p);
                p = p->next;
            }
	}
    }
}

int main(int argc, char *argv[])
{
    int n = 40;
    
    if (argc > 1)
        n = atoi(argv[1]);

    ListNode *l = list_create(0, n-1);
    list_traverse(l);
    list_destroy(l);
    return EXIT_SUCCESS;
}
