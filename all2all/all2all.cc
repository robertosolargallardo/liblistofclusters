#include <listofclusters.hh>
#include <resultslist.hh>
#include <armadillo>
#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <math.h>

typedef arma::rowvec sift_t;
arma::mat DB;

double idistance(sift_t a,sift_t b)
{
    sift_t c=a-b;
    return(sqrt(arma::dot(c,c)));
}
int main(int argc,char** argv)
{
    DB.load(argv[1],arma::csv_ascii);

	 for(size_t i=0;i<DB.n_rows;i++){
	 	for(size_t j=i+1;j<DB.n_rows;j++){
			std::cout << i << " " << j << " " << idistance(DB.row(i),DB.row(j)) << std::endl;
		}		
    }
    return(0);
}
