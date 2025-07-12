#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>


/*Reference: Quicksor.pdf, Lab5-lab10, Assignment2-4, chatgpt, deepseek,
Pacheco, P. S. (2011). An introduction to parallel programming. Elsevier/Morgan Kaufmann.
Nugroho, E. D., Ashari, I. F., Nashrullah, M., Algifari, M. H., & Verdiana, M. (2023). Comparative Analysis of OpenMP and MPI Parallel Computing Implementations in Team Sort Algorithm. Journal of Applied Informatics and Computing, 7(2), 141–149. https://doi.org/10.30871/jaic.v7i2.6409
Kil Jae Kim, Seong Jin Cho, & Jae-Wook Jeon. (2011). Parallel quick sort algorithms analysis using OpenMP 3.0 in embedded system. 2011 11th International Conference on Control, Automation and Systems, 757–761.
https://www.geeksforgeeks.org/dsa/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/
https://www.reddit.com/r/opengl/comments/xph0kf/how_does_a_double_buffer_allow_things_to_be_drawn/
*/

typedef struct {
    int* data;
    int size;
} ThreadData;

int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void sort_array(int* arr, int size) {
    qsort(arr, size, sizeof(int), compare);
}

int** global_medians = NULL;
int* global_pivots = NULL;
int* splitpoints = NULL;
int* exchange_sizes = NULL;

//Here I used double buffer system, I learn it from chatgpt and lab6(Temporal locality)
int** thread_temp_buffers = NULL;
int** thread_buffers_A = NULL;
int** thread_buffers_B = NULL;
// This buffer can check which buffer is active, and buffer A is 0, buffer B is 1
int* buffer_track = NULL;
// threadprivate, advoiding race condition
static int* findsplit_local_temp_buffer = NULL;
static int findsplit_local_temp_buffer_size = 0;
#ifdef _OPENMP
#pragma omp threadprivate(findsplit_local_temp_buffer, findsplit_local_temp_buffer_size)
#endif

int select_pivot(ThreadData* threadData, int num_threads, int group_size) {
    int myid = omp_get_thread_num();
    int locid = myid % group_size;
    int group = myid / group_size; 

    int* data = threadData[myid].data;
    int size = threadData[myid].size;

    sort_array(data, size);

    int median;
    if (size == 0){
        median = 0;
    }else if (size == 1) {
        median = data[0];
    }else if (size % 2 == 0) {
        median = (data[size/2 - 1] + data[size/2]) / 2;
    }else{
        median = data[size/2];
    }
   
    global_medians[group][locid] = median;
    #pragma omp barrier

    if (locid == 0) {
        sort_array(global_medians[group], group_size);
        int mid = group_size / 2;
        if (group_size % 2 == 0){
            global_pivots[group]=(global_medians[group][mid-1] + global_medians[group][mid]) / 2;
        }else{
            global_pivots[group]=global_medians[group][mid];
        }
    }
    #pragma omp barrier
    return global_pivots[group];
}

int findsplit(int* data, int size, int pivot) {
    int* temp = findsplit_local_temp_buffer;
    int left = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] <= pivot) {
            temp[left++] = data[i];
        }
    }
    int split = left;
    for (int i = 0; i < size; i++) {
        if (data[i] > pivot) {
            temp[left++] = data[i];
        }
    }
    memcpy(data, temp, size * sizeof(int));
    return split;
}

void merge(int* dest, int* a, int size_a, int* b, int size_b) {
    int i = 0, j = 0, k = 0;
    while (i < size_a && j < size_b)
        dest[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (i < size_a) dest[k++] = a[i++];
    while (j < size_b) dest[k++] = b[j++];
}
/*Here, the data exchange algorithm is inspired MPI 
1.Each thread has a partner thread for data exchange 
2.Data is copied into temporary buffers (thread_temp_buffers), mimicking MPI's send/receive buffers.
3.shared memory for direct buffer access, avoiding the overhead of communication.
*/
void exchange_data(ThreadData* threadData, int group_size) {
    int myid = omp_get_thread_num();
    int locid = myid % group_size;

    int* current_data = threadData[myid].data;
    int current_size = threadData[myid].size;
    int my_split = splitpoints[myid];

    int *keep_src, *send_src;
    int keep_size, send_size;

    // For lower group, keeping the data less than pivot, and send the data larger than pivot
    if (locid < group_size / 2) { 
        keep_src = current_data;
        keep_size = my_split;
        send_src = current_data + my_split;
        send_size = current_size - my_split;
    } else { 
         // For upper group, keeping the data larger than pivot, and sednd the data less than pivot
        keep_src = current_data + my_split;
        keep_size = current_size - my_split;
        send_src = current_data;
        send_size = my_split;
    }

    // Copy data to be sent to temporary buffer
    memcpy(thread_temp_buffers[myid], send_src, send_size * sizeof(int));
    exchange_sizes[myid] = send_size;

    #pragma omp barrier 

    // Partner thread for exchange
    int partner = (locid < group_size / 2) ? myid + group_size / 2 : myid - group_size / 2;
    int recv_size = exchange_sizes[partner];
    int* recv_data_from_partner = thread_temp_buffers[partner]; // Partner's temp buffer

    //target buffer(buffer A or buffer B)
    int target_buffer_idx = 1 - buffer_track[myid];
    int* target_buffer = (target_buffer_idx == 0) ? thread_buffers_A[myid] : thread_buffers_B[myid];

   
    merge(target_buffer, keep_src, keep_size, recv_data_from_partner, recv_size);


    threadData[myid].data = target_buffer;
    threadData[myid].size = keep_size + recv_size;
    buffer_track[myid] = target_buffer_idx; 

    #pragma omp barrier 
}


void global_sort(ThreadData* threadData, int current_group_size) {
    if (current_group_size <= 1) {
        int myid = omp_get_thread_num();
        sort_array(threadData[myid].data, threadData[myid].size);
        return;
    }

    int pivot = select_pivot(threadData, omp_get_num_threads(), current_group_size);
 
    int myid = omp_get_thread_num();
    splitpoints[myid] = findsplit(threadData[myid].data, threadData[myid].size, pivot);
    
    exchange_data(threadData, current_group_size);
    global_sort(threadData, current_group_size / 2);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <array_size> <num_threads>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if ((num_threads & (num_threads - 1)) != 0 || num_threads == 0) {
        printf("Number of threads must be a power of 2 and greater than 0.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    
    int* data = malloc(n * sizeof(int));

    // Fixed seed for reproducibility, learn from chatgpt
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = rand() % 10000;
    }


    ThreadData* threadData = malloc(num_threads * sizeof(ThreadData));

    global_pivots = malloc((num_threads / 2) * sizeof(int));

    global_medians = malloc((num_threads / 2) * sizeof(int*));
 
    for (int i = 0; i < num_threads / 2; ++i) {
        global_medians[i] = malloc(num_threads * sizeof(int)); 
      
    }

    splitpoints = malloc(num_threads * sizeof(int));
    exchange_sizes = malloc(num_threads * sizeof(int));
    thread_temp_buffers = malloc(num_threads * sizeof(int*));
    thread_buffers_A = malloc(num_threads * sizeof(int*));
    thread_buffers_B = malloc(num_threads * sizeof(int*));
    buffer_track = malloc(num_threads * sizeof(int));
  

    double start_time, end_time; // Declare timing variables here

    // Each thread initializes its own buffers and calls global_sort
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_buffers_A[tid] = malloc(n * sizeof(int));
        thread_buffers_B[tid] = malloc(n * sizeof(int));
        thread_temp_buffers[tid] = malloc(n * sizeof(int)); // Max possible size for temp buffer
        findsplit_local_temp_buffer = malloc(n * sizeof(int));
        findsplit_local_temp_buffer_size = n;
        // Start with A as active
        buffer_track[tid] = 0; 

        int chunk_size = n / num_threads;
        int offset = tid * chunk_size;
        int current_thread_actual_size = (tid == num_threads - 1) ? (n - offset) : chunk_size;
        // Copy data to this thread's buffer A
        memcpy(thread_buffers_A[tid], data + offset, current_thread_actual_size * sizeof(int));
        threadData[tid].data = thread_buffers_A[tid];
        threadData[tid].size = current_thread_actual_size;

        #pragma omp barrier 
        // only master thread record runtime
        #pragma omp master
        {
            start_time = omp_get_wtime();
        }

        global_sort(threadData, num_threads); 

        #pragma omp master
        {
            end_time = omp_get_wtime();
        }

        // Each thread frees its own threadprivate buffer, so we need free it after sort function
        if (findsplit_local_temp_buffer != NULL) {
            free(findsplit_local_temp_buffer);
            findsplit_local_temp_buffer = NULL;
        }
    } 

    // Collect results back into the original data
    int current_offset = 0;
    for (int i = 0; i < num_threads; i++) {
        memcpy(data + current_offset, threadData[i].data, threadData[i].size * sizeof(int));
        current_offset += threadData[i].size;
    }

    int is_sorted = 1;
    for (int i = 1; i < n; i++) { 
        if (data[i-1] > data[i]) {
            is_sorted = 0;
            break;
        }
    }

    if (is_sorted) {
        printf("OK! Array is sorted.\n");
        printf("Time: %.4fs\n", end_time - start_time);
    } else {
        printf("Error! Array is not sorted.\n");
        return 1;
    }

   
    for (int i = 0; i < num_threads; i++) {
        free(thread_buffers_A[i]);
        free(thread_buffers_B[i]);
        free(thread_temp_buffers[i]);
    }
    free(thread_buffers_A);
    free(thread_buffers_B);
    free(thread_temp_buffers);

    for (int i = 0; i < num_threads / 2; ++i) {
        free(global_medians[i]);
    }
    free(global_medians);
    free(buffer_track);
    free(splitpoints);
    free(exchange_sizes);
    free(global_pivots);
    free(threadData);
    free(data); 

    return 0;
}