#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "openacc.h"
//#define NUM_THREAD 4
#define false 0
#define true 1

#define DEBUG

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

//macro for input graph4096.txt
//#define NUM_OF_NODES 4096
//#define EDGELIST_SIZE 24576
//macro for input graph1MW_6.txt
//#define NUM_OF_NODES 1000000
//#define EDGELIST_SIZE 5999970
//#ifndef NUM_OF_NODES
//#define NUM_OF_NODES 1000000
//#endif
//#ifndef EDGELIST_SIZE
//#define EDGELIST_SIZE 5999970
//#endif
//#ifndef NUM_OF_NODES
//#define NUM_OF_NODES 4194304
//#endif
//#ifndef EDGELIST_SIZE
//#define EDGELIST_SIZE 25159848
//#endif
#ifndef NUM_OF_NODES
#define NUM_OF_NODES 4096
#endif
#ifndef EDGELIST_SIZE
#define EDGELIST_SIZE 24576
#endif

#ifdef _OPENARC_

#if NUM_OF_NODES == 4096
	#pragma openarc #define NUM_OF_NODES 4096
#elif NUM_OF_NODES == 1000000
	#pragma openarc #define NUM_OF_NODES 1000000
#elif NUM_OF_NODES == 4194304
	#pragma openarc #define NUM_OF_NODES 4194304
#elif NUM_OF_NODES == 16777216
	#pragma openarc #define NUM_OF_NODES 16777216
#endif
#if EDGELIST_SIZE == 24576
	#pragma openarc #define EDGELIST_SIZE 24576
#elif EDGELIST_SIZE == 5999970
	#pragma openarc #define EDGELIST_SIZE 5999970
#elif EDGELIST_SIZE == 25159848
	#pragma openarc #define EDGELIST_SIZE 25159848
#elif EDGELIST_SIZE == 100666228
	#pragma openarc #define EDGELIST_SIZE 100666228
#endif

#endif

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}

double gettime() {
  struct timeval t;
  gettimeofday(&t,0);
  return t.tv_sec+t.tv_usec*1e-6;
}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	double start_time, end_time;
	start_time = gettime();
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
	end_time = gettime();
	printf("Total time = %lf sec \n", end_time - start_time);
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
  char *input_f;
	int	 num_omp_threads;
	int source;
	//struct Node* h_graph_nodes;
	int * h_graph_nodes_starting;
	int * h_graph_nodes_no_of_edges;
	int *h_graph_mask;
	int *h_updating_graph_mask;
	int *h_graph_visited;
	int start, edgeno;   
	int id,cost;
	int* h_graph_edges;
	int k;
	int* h_cost;
	int stop;
	unsigned int i;
	int tid;
	int costSum;
	FILE *fpo;
#ifdef DEBUG
	double start_time, end_time;
#endif
	
	if(argc!=3){
	Usage(argc, argv);
	exit(0);
	}
    
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	source = 0;

	fscanf(fp,"%d",&no_of_nodes);
	printf("no_of_nodes = %d\n", no_of_nodes);
   
	// allocate host memory
	//h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*no_of_nodes);
/*
	h_graph_nodes_starting = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_nodes_no_of_edges = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);
*/
	h_graph_nodes_starting = (int*) acc_create_unified(NULL, sizeof(int)*no_of_nodes);
	h_graph_nodes_no_of_edges = (int*) acc_create_unified(NULL, sizeof(int)*no_of_nodes);
	h_graph_mask = (int*) acc_create_unified(NULL, sizeof(int)*no_of_nodes);
	h_updating_graph_mask = (int*) acc_create_unified(NULL, sizeof(int)*no_of_nodes);
	h_graph_visited = (int*) acc_create_unified(NULL, sizeof(int)*no_of_nodes);

	// initalize the memory
	for( i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes_starting[i] = start;
		h_graph_nodes_no_of_edges[i] = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);
	printf("edge_list_size = %d\n", edge_list_size);

	//h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	h_graph_edges = (int*) acc_create_unified(NULL, sizeof(int)*edge_list_size);
	for(i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
	//h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	h_cost = (int*) acc_create_unified(NULL, sizeof(int)*no_of_nodes);
	for(i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	printf("Start traversing the tree\n");
	
#ifdef DEBUG
	start_time = gettime();
#endif
    
	k = 0;
#pragma acc data \
copyin(h_graph_visited[0:NUM_OF_NODES], h_graph_nodes_starting[0:NUM_OF_NODES], \
h_graph_nodes_no_of_edges[0:NUM_OF_NODES], h_graph_edges[0:EDGELIST_SIZE], \
h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES]) \
copy(h_cost[0:NUM_OF_NODES]) 
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

		#pragma acc kernels loop independent, gang, worker, private(tid,i) \
		present(h_graph_visited[0:NUM_OF_NODES], h_graph_nodes_starting[0:NUM_OF_NODES], \
		h_graph_nodes_no_of_edges[0:NUM_OF_NODES], h_graph_edges[0:EDGELIST_SIZE], \
		h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES], \
		h_cost[0:NUM_OF_NODES]) //async(0)
		for(tid = 0; tid < NUM_OF_NODES; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
				h_graph_mask[tid]=false;
				for(i=h_graph_nodes_starting[tid]; i<(h_graph_nodes_no_of_edges[tid] + h_graph_nodes_starting[tid]); i++)
				{
					int id = h_graph_edges[i];
					if(!h_graph_visited[id])
					{
						h_cost[id]=h_cost[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}
		}
		//#pragma acc wait(0)
		
		#pragma acc kernels loop independent, gang, worker, private(tid) \
		present(h_graph_visited[0:NUM_OF_NODES], \
		h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES]) //async(0)
  		for(tid=0; tid< NUM_OF_NODES ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop= stop | true;
				h_updating_graph_mask[tid]=false;
			}
		}
		//#pragma acc wait(0)
		k++;
	}
	while(stop);

	//Fake computation to measure timing of unified memory.
	costSum = 0;
	for(i=0;i<no_of_nodes;i++) {
		costSum += h_cost[i];
	}
	

	#ifdef DEBUG
	end_time = gettime();
	printf("sum of h_cost: %d\n", costSum);
	printf("Kernel Executed %d times\n",k);
	printf("Accelerator Elapsed time = %lf sec\n", end_time - start_time);
	#endif


	if(VERIFICATION) {
		int * h_graph_nodes_starting_CPU;
		int * h_graph_nodes_no_of_edges_CPU;
		int *h_graph_mask_CPU;
		int *h_updating_graph_mask_CPU;
		int *h_graph_visited_CPU;
		int start, edgeno_CPU;
		int* h_graph_edges_CPU;
		int* h_cost_CPU;
		int good= 0;
	
		fp = fopen(input_f,"r");
		
		source = 0;

		fscanf(fp,"%d",&no_of_nodes);
		 
		// allocate host memory
		//h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*no_of_nodes);
		h_graph_nodes_starting_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_graph_nodes_no_of_edges_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_graph_mask_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_updating_graph_mask_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_graph_visited_CPU = (int*) malloc(sizeof(int)*no_of_nodes);

		// initalize the memory
		for( i = 0; i < no_of_nodes; i++) 
		{
			fscanf(fp,"%d %d",&start,&edgeno);
			h_graph_nodes_starting_CPU[i] = start;
			h_graph_nodes_no_of_edges_CPU[i] = edgeno;
			h_graph_mask_CPU[i]=false;
			h_updating_graph_mask_CPU[i]=false;
			h_graph_visited_CPU[i]=false;
		}

		//read the source node from the file
		fscanf(fp,"%d",&source);
		source=0;

		//set the source node as true in the mask
		h_graph_mask_CPU[source]=true;
		h_graph_visited_CPU[source]=true;

		fscanf(fp,"%d",&edge_list_size);

		h_graph_edges_CPU = (int*) malloc(sizeof(int)*edge_list_size);
		for(i=0; i < edge_list_size ; i++)
		{
			fscanf(fp,"%d",&id);
			fscanf(fp,"%d",&cost);
			h_graph_edges_CPU[i] = id;
		}

		if(fp)
			fclose(fp);    


		// allocate mem for the result on host side
		h_cost_CPU = (int*) malloc( sizeof(int)*no_of_nodes);
		for(i=0;i<no_of_nodes;i++)
			h_cost_CPU[i]=-1;
		h_cost_CPU[source]=0;
	
	
	#ifdef DEBUG
		start_time = gettime();
	#endif
		  
		k = 0;

		do
		{
			//if no thread changes this value then the loop stops
			stop=false;

			for(tid = 0; tid < NUM_OF_NODES; tid++ )
			{
				if (h_graph_mask_CPU[tid] == true){ 
					h_graph_mask_CPU[tid]=false;
					for(i=h_graph_nodes_starting_CPU[tid]; i<(h_graph_nodes_no_of_edges_CPU[tid] + h_graph_nodes_starting_CPU[tid]); i++)
					{
						int id = h_graph_edges_CPU[i];
						if(!h_graph_visited_CPU[id])
						{
							h_cost_CPU[id]=h_cost_CPU[tid]+1;
							h_updating_graph_mask_CPU[id]=true;
						}
					}
				}
			}


			for(tid=0; tid< NUM_OF_NODES ; tid++ )
			{
				if (h_updating_graph_mask_CPU[tid] == true){
					h_graph_mask_CPU[tid]=true;
					h_graph_visited_CPU[tid]=true;
					stop= stop | true;
					h_updating_graph_mask_CPU[tid]=false;
				}
			}
			k++;
		}
		while(stop);

		good=1;
		for(i=0; i<no_of_nodes; i++) {
			if(h_cost[i] != h_cost_CPU[i]) {
				good = 0;	
				break;;
			}
		}
		
		if(!good) 
			printf("Verification: Failed\n");
		else
			printf("Verification: Successful\n");

		free( h_graph_nodes_starting_CPU);
		free( h_graph_nodes_no_of_edges_CPU);
		free( h_graph_edges_CPU);
		free( h_graph_mask_CPU);
		free( h_updating_graph_mask_CPU);
		free( h_graph_visited_CPU);
		free( h_cost_CPU);

	}




	//Store the result into a file
	fpo = fopen("result.txt","w");
	for(i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
/*
	free( h_graph_nodes_starting);
	free( h_graph_nodes_no_of_edges);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
*/
	acc_delete_unified( h_graph_nodes_starting, 0);
	acc_delete_unified( h_graph_nodes_no_of_edges, 0);
	acc_delete_unified( h_graph_edges, 0);
	acc_delete_unified( h_graph_mask, 0);
	acc_delete_unified( h_updating_graph_mask, 0);
	acc_delete_unified( h_graph_visited, 0);
	acc_delete_unified( h_cost, 0);

}

