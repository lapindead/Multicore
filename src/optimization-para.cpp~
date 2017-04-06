/*
  Branch and bound algorithm to find the minimum of continuous binary 
  functions using interval arithmetic.

  Mpi version

  Author: Araya Montalvo , Casanova Mario

  Warning : -Ne marche que pour un nombre de machine égale à une puissance de 2 (1,2,4,8,etc...)

  v. 1.0, 2016-04-05

*/

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <queue>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"
#include "mpi.h"
#include "omp.h"
using namespace std;


// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
	       interval &xl, interval& xr, interval& yl, interval& yr)
{
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(),xm);
  xr = interval(xm,x.right());
  yl = interval(y.left(),ym);
  yr = interval(ym,y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,  // Function to minimize
	      const interval& x, // Current bounds for 1st dimension
	      const interval& y, // Current bounds for 2nd dimension
	      double threshold,  // Threshold at which we should stop splitting
	      double& min_ub,  // Current minimum upper bound
	      minimizer_list& ml) // List of current minimizers
{
  interval fxy = f(x,y);
  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return ;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
    // Discarding all saved boxes whose minimum lower bound is 
    // greater than the new minimum upper bound
    auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
    ml.erase(discard_begin,ml.end());
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) { 
    // We have potentially a new minimizer
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }
  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);
  /*#pragma omp parallel
  #pragma omp single
  { 
    #pragma omp task
    minimize(f,xl,yl,threshold,min_ub,ml);
    #pragma omp task
    minimize(f,xl,yr,threshold,min_ub,ml);
    #pragma omp task
    minimize(f,xr,yl,threshold,min_ub,ml);
    #pragma omp task
    minimize(f,xr,yr,threshold,min_ub,ml);

  }  */
    minimize(f,xl,yl,threshold,min_ub,ml);
    minimize(f,xl,yr,threshold,min_ub,ml);
    minimize(f,xr,yl,threshold,min_ub,ml);
    minimize(f,xr,yr,threshold,min_ub,ml);


}


int main(int argc, char * argv[])
{
  int rank, numProcs;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cout.precision(16);
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub = numeric_limits<double>::infinity();

  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is 
  // greater than the new current upper bound
  // Threshold at which we should stop splitting a box
  double precision;

  // Name of the function to optimize
  string choice_fun;

  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  
  bool good_choice;

  //Variable MPI
  //interval myslice[1];
  double minUpAll=42.0;
  queue<interval> subBoxes;
  int SZ;
  int localSZ;
  // Asking the user for the name of the function to optimize
  if(rank == 0){
    do {
      good_choice = true;

      cout << "Which function to optimize?\n";
      cout << "Possible choices: ";
      for (auto fname : functions) {
        cout << fname.first << " ";
      }
      cout << endl;
      cin >> choice_fun;
      
      try {
        fun = functions.at(choice_fun);
      } catch (out_of_range) {
        cerr << "Bad choice" << endl;
        good_choice = false;
      }
    } while(!good_choice);

    // Asking for the threshold below which a box is not split further
    cout << "Precision? ";
    cin >> precision;

  	int size = 1;
  	
  	subBoxes.push(fun.x);
	subBoxes.push(fun.y);
	while (size < numProcs) {
		interval xl, xr, yl, yr;
		interval tmp1 = subBoxes.front();
		subBoxes.pop();
		interval tmp2 = subBoxes.front();
		subBoxes.pop();
		// Enlève la boite divisée
		
		split_box(tmp1,tmp2,xl,xr,yl,yr);
		
		// Ajoutte les nouvelles boites
		subBoxes.push(xl);
		subBoxes.push(yl);
		subBoxes.push(xl);
		subBoxes.push(yr);
		subBoxes.push(xr);
		subBoxes.push(yl);
		subBoxes.push(xr);
		subBoxes.push(yr);
		size = subBoxes.size() / 2;
	}
    SZ = subBoxes.size();
  }
  // bradcast de la taille du tableau de boites
  MPI_Bcast(&SZ, 1, MPI_INT, 0, MPI_COMM_WORLD);
  interval subs[SZ];
  int localSZs[numProcs];
  
  if ( rank == 0 ) {
	
	int nbBox = SZ / 2;
	// fabrication d'un tableau d'intervalles plus facilement utilisable qu'une queue.
	for (int i = 0; i < SZ; ++i) {
		subs[i] = subBoxes.front();
		subBoxes.pop();
	}
	// attribution des boites aux machines
	for (int i = numProcs - 1; i >= 0; --i) {
		if (i > numProcs - (nbBox % numProcs)) {
			localSZs[i] = nbBox / numProcs + nbBox % numProcs;
		} else {
			localSZs[i] = nbBox / numProcs;
		}
	}
  }

  MPI_Bcast(&fun, sizeof(opt_fun_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // broadcast de la position des boites dans le tableau que chaque machine va traiter 
  MPI_Bcast(&localSZs, numProcs, MPI_INT, 0, MPI_COMM_WORLD);
  localSZ = localSZs[rank];

  // broadcast du tableau de boites, un scatter ne marche pas si chaque machine
  // n'a pas le même nombre de boites a traiter.
  MPI_Bcast(&subs,sizeof(interval)*SZ,MPI_BYTE,0,MPI_COMM_WORLD);

  minimizer_list minimums;
  double min_ub2 = numeric_limits<double>::infinity();
  //for (int i = (SZ/numProcs)*rank; i < (SZ/numProcs)*rank + localSZ; i = i + 2) {
  for (int i = (nbBox/numProcs)*rank; i < (nbBox/numProcs)*rank + localSZ; ++i) {
  	minimize(fun.f,subs[i],subs[i+1],precision,min_ub2,minimums);
  	if(min_ub>min_ub2) min_ub=min_ub2;
  
  }
  
  
  // On fait un reduc pour trouver le plu petit min trouver par chaque machine
  MPI_Reduce(&min_ub,&minUpAll,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  // Displaying all potential minimizers
  if (rank==0){
    //copy(minimums.begin(),minimums.end(),
    //ostream_iterator<minimizer>(cout,"\n"));    
    //cout << "Number of minimizers: " << minimums.size() << endl;
    cout << setprecision (6)<<"Upper bound for minimum: " << minUpAll << endl;

  }
  

  MPI_Finalize();
}
