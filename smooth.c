/*******************************************************************************
*
* Driver routine to operate a kernal smoother.
*
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "ompsmooth.h"
#include "omp.h"

#define MAT_DIM 6000
#define MAT_SIZE MAT_DIM*MAT_DIM

#define KERNEL_HALFWIDTH 2

typedef struct timeval TimePoint;

inline void NonMerged(int dim, int halfwidth, float *m1, float *m1out, float *m2, float *m2out)
{
  smoothParallelYXFor(dim, halfwidth, m1, m1out);
  smoothParallelYXFor(dim, halfwidth, m2, m2out);
}

inline void Merged(int dim, int halfwidth, float *m1, float *m1out, float *m2, float *m2out)
{
  int i, size=dim*dim, x, y;
  #pragma omp parallel for private(x)
  for (y=0; y<dim; y++)
  {
    for (x=0; x<dim; x++)
    { 
      m1out[y*dim+x] = evaluate( dim, halfwidth, x, y, m1 );
      m2out[y*dim+x] = evaluate( dim, halfwidth, x, y, m2 );
    }
  }
  //#pragma omp parallel for //firstprivate(size)
  //for (i=0; i<size; i++)
  //{ 
  //   m1out[i] = evaluate ( dim, halfwidth, i%dim, i/dim, m1 );
  //   m2out[i] = evaluate ( dim, halfwidth, i%dim, i/dim, m2 );
  //}
  return;
}

TimePoint testFunc(void (*PFunc)(int, int, float *, float *), int dim, int halfwidth, float *m1, float *m2, int iter)
{
  /* get initial time */
  TimePoint ta, tb;
  int i;
  gettimeofday ( &ta, NULL );
  for(i = 0; i < iter ; i++ ) 
    PFunc(dim, halfwidth, m1, m2);
  /* get initial time */
  gettimeofday ( &tb, NULL );

  /* Work out the time */
  int s = tb.tv_sec - ta.tv_sec;
  int u;

  //printf("%d %d %d %d\n", (int)ta.tv_sec, (int)ta.tv_usec, (int)tb.tv_sec, (int)tb.tv_usec);
  if ( ta.tv_usec <= tb.tv_usec ) {
    u = tb.tv_usec - ta.tv_usec;
  } else {
    u = 1000000 + tb.tv_usec - ta.tv_usec;
    s = s-1;
  }
  
  int total = (s*1000000+u+0.0)/iter;
  TimePoint temp;
  temp.tv_sec = (time_t)total/1000000;
  temp.tv_usec = (time_t)total%1000000;

  return temp;
}

TimePoint testDoubleFunc(void (*PFunc)(int, int, float *, float *, float *, float *), int dim, int halfwidth, float *m1, float *m1out, float *m2, float *m2out, int iter)
{
  /* get initial time */
  TimePoint ta, tb;
  int i;
  gettimeofday ( &ta, NULL );
  for(i = 0; i < iter ; i++ ) 
    PFunc(dim, halfwidth, m1, m1out, m2, m2out);
  /* get initial time */
  gettimeofday ( &tb, NULL );

  /* Work out the time */
  int s = tb.tv_sec - ta.tv_sec;
  int u;

  //printf("%d %d %d %d\n", (int)ta.tv_sec, (int)ta.tv_usec, (int)tb.tv_sec, (int)tb.tv_usec);
  if ( ta.tv_usec <= tb.tv_usec ) {
    u = tb.tv_usec - ta.tv_usec;
  } else {
    u = 1000000 + tb.tv_usec - ta.tv_usec;
    s = s-1;
  }
  
  int total = (s*1000000+u+0.0)/iter;
  TimePoint temp;
  temp.tv_sec = (time_t)total/1000000;
  temp.tv_usec = (time_t)total%1000000;

  return temp;
}

int main()
{
  /* Variables for timing */
  TimePoint ta, tb;
  int iter = 20, KHalfWidth; 
 
  /* Create two input matrixes */
  float * m1in;
  float * m1out;
  m1in = malloc ( sizeof(float)*MAT_SIZE );
  m1out = malloc ( sizeof(float)*MAT_SIZE );

  float * m2in;
  float * m2out;
  m2in = malloc ( sizeof(float)*MAT_SIZE );
  m2out = malloc ( sizeof(float)*MAT_SIZE );
  
  /* random data for the input */
  // int fd = open ("/dev/urandom", O_RDONLY );
  // read ( fd, m1in, MAT_SIZE*sizeof(float));
  /* instead, this pattern is preserved by the kernel smoother */
  int64_t x, y;
  for (y=0; y<MAT_DIM; y++) {
    for (x=0; x<MAT_DIM; x++) {
      m1in[y*MAT_DIM+x] = (float)(x+y);
      m2in[y*MAT_DIM+x] = (float)(x+y);
    }
  }

  /* zero the output */
  memset ( m1out, 0, MAT_SIZE*sizeof(float) );
  memset ( m2out, 0, MAT_SIZE*sizeof(float) );
  
  FILE *fp;
  fp = fopen("testLog.txt","w+");

  /********* Serial Test **********/
  fprintf(fp, "%s", "\n********* Serial Test **********\n");
  
  for( KHalfWidth = 1; KHalfWidth <=3 ; KHalfWidth++ ) 
  {
    printf("**************************KernelHalfWidth=%d********************************\n",KHalfWidth);
    fprintf(fp, "\n**************************KernelHalfWidth=%d********************************\n",KHalfWidth);
    
    // smoothSerialYX
    TimePoint SerialYX_Time = testFunc(smoothSerialYX, MAT_DIM, KHalfWidth, m1in, m1out, iter);
    printf ("Serial YX smoother took %d seconds and %d microseconds\n",(int)SerialYX_Time.tv_sec, (int)SerialYX_Time.tv_usec);
    fprintf (fp, "Serial YX smoother took %d seconds and %d microseconds\n",(int)SerialYX_Time.tv_sec, (int)SerialYX_Time.tv_usec);
    
    // smoothSerialXY
    TimePoint SerialXY_Time = testFunc(smoothSerialXY, MAT_DIM, KHalfWidth, m1in, m1out, iter);
    printf ("Serial XY smoother took %d seconds and %d microseconds\n",(int)SerialXY_Time.tv_sec, (int)SerialXY_Time.tv_usec);
    fprintf (fp, "Serial XY smoother took %d seconds and %d microseconds\n",(int)SerialXY_Time.tv_sec, (int)SerialXY_Time.tv_usec);
  }
  
  //printf("/*************************************************************************/\n");
  
  /********* Parallel Tests **********/
  fprintf(fp, "%s", "\n********* Parallel Test **********\n");
  int threads;
  for (threads=1; threads <=32; threads<<=1)
  {
    omp_set_num_threads(threads);
    printf("Threads = %d\n", threads);
    fprintf(fp, "Threads = %d\n", threads);
    
    // smoothParallelYXFor
    TimePoint ParllYX_Time = testFunc(smoothParallelYXFor, MAT_DIM, KERNEL_HALFWIDTH, m1in, m1out, iter);
    printf ("ParallelYX smoother took %d seconds and %d microseconds\n",(int)ParllYX_Time.tv_sec, (int)ParllYX_Time.tv_usec);
    fprintf (fp, "ParallelYX smoother took %d seconds and %d microseconds\n",(int)ParllYX_Time.tv_sec, (int)ParllYX_Time.tv_usec);
  
    // smoothParallelXYFor
    TimePoint ParllXY_Time = testFunc(smoothParallelXYFor, MAT_DIM, KERNEL_HALFWIDTH, m1in, m1out, iter);
    printf ("ParallelXY smoother took %d seconds and %d microseconds\n",(int)ParllXY_Time.tv_sec, (int)ParllXY_Time.tv_usec);
    fprintf (fp, "ParallelXY smoother took %d seconds and %d microseconds\n",(int)ParllXY_Time.tv_sec, (int)ParllXY_Time.tv_usec);

    // smoothParallelCoalescedFor
    TimePoint ParllCoalesced_Time = testFunc(smoothParallelCoalescedFor, MAT_DIM, KERNEL_HALFWIDTH, m1in, m1out, iter);
    printf ("ParallelCoalesced smoother took %d seconds and %d microseconds\n",(int)ParllCoalesced_Time.tv_sec, (int)ParllCoalesced_Time.tv_usec);
    fprintf (fp, "ParallelCoalesced smoother took %d seconds and %d microseconds\n",(int)ParllCoalesced_Time.tv_sec, (int)ParllCoalesced_Time.tv_usec);
    
    // ParallelNonMerged
    TimePoint NonMerged_Time = testDoubleFunc(NonMerged, MAT_DIM, KERNEL_HALFWIDTH, m1in, m1out, m2in, m2out, iter);
    printf ("NonMerged DoubleSmoother took %d seconds and %d microseconds\n",(int)NonMerged_Time.tv_sec, (int)NonMerged_Time.tv_usec);
    fprintf (fp, "NonMerged DoubleSmoother took %d seconds and %d microseconds\n",(int)NonMerged_Time.tv_sec, (int)NonMerged_Time.tv_usec);
    
    TimePoint Merged_Time = testDoubleFunc(Merged, MAT_DIM, KERNEL_HALFWIDTH, m1in, m1out, m2in, m2out, iter);
    printf ("Merged DoubleSmoother took %d seconds and %d microseconds\n",(int)Merged_Time.tv_sec, (int)Merged_Time.tv_usec); 
    fprintf (fp, "Merged DoubleSmoother took %d seconds and %d microseconds\n",(int)Merged_Time.tv_sec, (int)Merged_Time.tv_usec);   
  }
  fclose(fp);
}

