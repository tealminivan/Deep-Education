#include <cassert>
#include <iostream>
#include <limits>
#include <list>
#include <vector>
#include <string.h>
#include <omp.h>

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
    
    #pragma omp parallel for
    for (int currVertex=0; currVertex<snaph->v_count; currVertex++){
        
        if (reverse){
            input.row_normalize(currVertex, snaph->get_degree(currVertex));
        }

        output.row_add(input[currVertex],currVertex);
        for( int neighbor=offset_list[currVertex]; neighbor<offset_list[currVertex+1]; neighbor++){
            output.row_add(input[nebrs_list[neighbor]], currVertex);
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
