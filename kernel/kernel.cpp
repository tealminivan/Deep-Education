#include <cassert>
#include <iostream>
#include <limits>
#include <list>
#include <vector>
#include <omp.h>
#include <string.h>

#include "kernel.h"

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;


void _gspmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{
    //cout << "spmm " << op << "reverse = " << reverse << endl;

    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here.   
    vid_t* offset_list = snaph-> offset;
    vid_t* nebrs_list = snaph-> nebrs;
    
    bool* visited = new bool[snaph->v_count];
    for (int i = 0; i < snaph->v_count - 1; i++){
        visited[i] = false;
    } 

    int level[snaph->v_count];
    list<int> queue;
    visited[0] = true;
    level[0] = 0;
    queue.push_back(0);
    

    while (!queue.empty()){
        int currVertex = queue.front();
        queue.pop_front();
        
        if (reverse){
            input.row_normalize(currVertex, snaph->get_degree(currVertex));
        }

        for( int neighbor=offset_list[currVertex]; neighbor<offset_list[currVertex+1]; neighbor++){
            
            if (!visited[nebrs_list[neighbor]]){
                visited[nebrs_list[neighbor]] = true;
                queue.push_back(nebrs_list[neighbor]);
                level[nebrs_list[neighbor]] = level[currVertex]+1;
            }
            output.row_add(input[currVertex],currVertex);
        }

        if (!reverse){
            output.row_normalize(currVertex, snaph->get_degree(currVertex));
        } 
    }
}

void invoke_gspmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
         return _gspmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
         return _gspmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }
}
