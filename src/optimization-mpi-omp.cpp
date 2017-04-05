/*
  Branch and bound algorithm to find the minimum of continuous binary 
  functions using interval arithmetic.

  Mpi + OMP version

  Author: Araya Montalvo , Casanova Mario

  Warning : -Ne marche que pour un nombre de machine égale à une puissance de 2 (1,2,4,8,etc...)
            -L'utilisation de OMP fait bug quand on demande des précision trop grande
  -> précision limite varie selon le problème.
  v. 1.0, 2016-04-05
*/

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
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
  double min_ub2 = numeric_limits<double>::infinity();

  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is 
  // greater than the new current upper bound
  minimizer_list minimums;
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
  interval tabInterval[numProcs];
  interval tabIntervalY[numProcs];
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

    // Calcule pour taille de chaque "tranche" pour découpé les intervalles
    //suivant le nombre de machine
    double widthX = fun.x.width();
    double n= widthX/numProcs;
    double nY=(fun.y.width())/numProcs;

    //lower bound pour savoir ou commencer le découpage
    double lowerBoundX=fun.x.left();
    double lowerBoundY=fun.y.left();
    
    for ( int i=0;i<numProcs;++i ){
      //On découpe l'intervale en X et en Y un nombre de fois
      //égale au nombre de machines
      interval inter (lowerBoundX,lowerBoundX+n);
      interval interY(lowerBoundY,lowerBoundY+nY);
      tabInterval[i]=inter;
      tabIntervalY[i]=interY;
      // lowerBound = upperBound pour passer au prochaine intervale
      lowerBoundX+=n;
      lowerBoundY+=nY;
    }
    
  
  }

  MPI_Bcast(&fun, sizeof(opt_fun_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //BCast des tableau d'intervalle 
  MPI_Bcast(&tabInterval,sizeof(interval)*numProcs,MPI_BYTE,0,MPI_COMM_WORLD);
  MPI_Bcast(&tabIntervalY,sizeof(interval)*numProcs,MPI_BYTE,0,MPI_COMM_WORLD);


  //Une thread en prallel pour chaque appelle de minimize
  // Appelle de minimize par machine = nombre de machine
  #pragma omp parallel 
  #pragma omp for reduction(min : min_ub)
  for(int i=0;i<numProcs;++i){
      //Chaque machine s'occupe d'une "ligne" -> tabInterval[rank]
      // Chaque machine s'occupe un par un des cubes de la ligne -> tabIntervalY[i]
      minimize(fun.f,tabInterval[rank],tabIntervalY[i],precision,min_ub2,minimums);
      //Si min trouver plus petit que le min actuel alors il devient le nouveau min 
      if(min_ub>min_ub2) min_ub=min_ub2;
      
  }
  // On fait un reduc pour trouver le plu petit min trouver par chaque machine
  MPI_Reduce(&min_ub,&minUpAll,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  // Displaying all potential minimizers
  if (rank==0){
    //copy(minimums.begin(),minimums.end(),
    //ostream_iterator<minimizer>(cout,"\n"));    
    //cout << "Number of minimizers: " << minimums.size() << endl;
    cout << "Upper bound for minimum: " << minUpAll << endl;
  }
  

  MPI_Finalize();
}
