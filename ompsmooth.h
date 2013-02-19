/*******************************************************************************
*
* Prototypes for parallel functions
*
*******************************************************************************/

#ifndef _h_ompsmooth
#define _h_ompsmooth

#include<stdint.h>

void smoothSerialXY ( int dim, int halfwidth, float * m1, float * m2 );
void smoothSerialYX ( int dim, int halfwidth, float * m1, float * m2 );
void smoothParallelXYFor ( int dim, int halfwidth, float * m1, float * m2 );
void smoothParallelYXFor ( int dim, int halfwidth, float * m1, float * m2 );
void smoothParallelCoalescedFor ( int dim, int halfwidth, float * m1, float * m2 );


#endif
